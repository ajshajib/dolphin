"""Tests for cosmology module."""

from pathlib import Path
from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pytest
from astropy.constants import c as _C_LIGHT
from astropy.cosmology import FlatLambdaCDM
from scipy.stats import norm

import dolphin.analysis.cosmology as cosmology_module
from dolphin.analysis.cosmology import Cosmology
from dolphin.analysis.output import Output

_ROOT_DIR = Path(__file__).resolve().parents[2]
_TEST_IO_DIR = _ROOT_DIR / "io_directory_example"
_TEST_MODEL_ID = "example"
_TEST_LENS_NAME = "lensed_quasar"

_Z_LENS = 0.5
_Z_SOURCE = 2.0
_WALKER_RATIO = 2


# CREATE TEST UTILITIES BEFORE MAIN MODULE
def _make_cosmology(**overrides):
    """Create a Cosmology instance with default test settings."""
    kwargs = {
        "io_directory": str(_TEST_IO_DIR),
        "lens_name": _TEST_LENS_NAME,
        "model_id": _TEST_MODEL_ID,
        "z_lens": _Z_LENS,
        "z_source": _Z_SOURCE,
        "cosmo_fiducial": FlatLambdaCDM(H0=70, Om0=0.3),
        "walker_ratio": _WALKER_RATIO,
    }
    kwargs.update(overrides)
    return Cosmology(**kwargs)


def _fake_sigma(calls):
    """Stand-in for TDCosmography.velocity_dispersion_dimension_less.

    Returns r_eff + theta_E so the output encodes which inputs were used.
    """

    def fake(**kwargs):
        calls.append(kwargs)
        return np.array([kwargs["r_eff"] + kwargs["theta_E"]])

    return fake


def _selected_samples(cosmo, rng_seed, n_samples):
    """Reproduce the chain rows the class will pick on its first RNG call."""
    chain = cosmo.output.get_reshaped_emcee_chain(
        cosmo.lens_name, cosmo.model_id, cosmo.walker_ratio, cosmo.burn_in
    )
    flat = chain.reshape(-1, chain.shape[-1])
    idx = np.random.default_rng(rng_seed).choice(
        flat.shape[0], size=n_samples, replace=False
    )
    return flat[idx]


@pytest.fixture
def cosmo(monkeypatch):
    """Create a test Cosmology instance with monkeypatched
    velocity_dispersion_dimension_less."""
    c = _make_cosmology(n_samples=5, rng_seed=0)
    calls = []
    monkeypatch.setattr(
        c.TDCosmography,
        "velocity_dispersion_dimension_less",
        _fake_sigma(calls),
    )
    c.calls = calls
    return c


class FakePhotometry:
    """Configurable stand-in for the Photometry class."""

    morphology = None  # what load_morphology_chain returns
    raise_not_found = False  # simulate no saved morphology file
    inversion_calls = None

    def __init__(self, **kwargs):
        pass

    def load_morphology_chain(self):
        if self.raise_not_found:
            raise FileNotFoundError
        return self.morphology

    def do_linear_inversion(self, **kwargs):
        FakePhotometry.inversion_calls.append(kwargs)
        return None, self.morphology


@pytest.fixture
def fake_photometry(monkeypatch):
    """Create a fixture for the FakePhotometry class."""
    FakePhotometry.morphology = None
    FakePhotometry.raise_not_found = False
    FakePhotometry.inversion_calls = []
    monkeypatch.setattr(cosmology_module, "Photometry", FakePhotometry)
    return FakePhotometry


def _r_eff_index(cosmo):
    if "R_sersic_lens_light0" not in cosmo.output.params_sampled:
        pytest.skip("example model does not sample R_sersic_lens_light0")
    return cosmo.output.params_sampled.index("R_sersic_lens_light0")


def _expected_ddt(dphi_arcsec2, dt_days):
    """Independent D_dt [Mpc] = c * dt / dphi, with no mass-sheet terms."""
    dphi_rad2 = np.asarray(dphi_arcsec2) * (1 * u.arcsec).to(u.rad).value ** 2
    dt_s = (np.asarray(dt_days) * u.day).to(u.s).value
    return (_C_LIGHT.value * dt_s / dphi_rad2 * u.m).to(u.Mpc).value


def _ddt_of(cosmology, z_lens, z_source):
    """Time-delay distance [Mpc] for an astropy cosmology."""
    dd = cosmology.angular_diameter_distance(z_lens).value
    ds = cosmology.angular_diameter_distance(z_source).value
    dds = cosmology.angular_diameter_distance_z1z2(z_lens, z_source).value
    return (1 + z_lens) * dd * ds / dds


def _expected_h0_and_g(fid, z_lens, z_source, om_samples, ddt):
    """Replay of `calculated_sampled_h0` for a known set of Omega_m draws."""
    ddt = np.asarray(ddt, dtype=float)
    if ddt.ndim == 1:
        ddt = ddt[:, None]
    h0_fid = float(fid.H0.value)
    r_fid = (
        fid.angular_diameter_distance(z_source).value
        / fid.angular_diameter_distance_z1z2(z_lens, z_source).value
    )
    h0 = np.empty_like(ddt)
    g = np.empty(len(om_samples))
    for i, om in enumerate(om_samples):
        c = FlatLambdaCDM(H0=h0_fid, Om0=float(om), Tcmb0=2.725)
        h0[i, :] = h0_fid * _ddt_of(c, z_lens, z_source) / ddt[i, :]
        r = (
            c.angular_diameter_distance(z_source).value
            / c.angular_diameter_distance_z1z2(z_lens, z_source).value
        )
        g[i] = np.sqrt(r / r_fid)
    return h0, g


@pytest.fixture
def cosmo_tcmb():
    """Create a cosmology fixture with a fixed Tcmb0."""
    return _make_cosmology(
        n_samples=5,
        rng_seed=0,
        cosmo_fiducial=FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725),
    )


def _om_draws(seed=0, om_min=0.05, om_max=0.5, n=5):
    """Omega_m draws the method makes on the first RNG call after seeding."""
    return np.random.default_rng(seed).uniform(om_min, om_max, size=n)


def _dd_over_dds(om, z_lens, z_source):
    """Compute Dd/Dds for a given Omega_m and redshifts."""
    c = FlatLambdaCDM(H0=70.0, Om0=float(om), Tcmb0=2.725)
    return (
        c.angular_diameter_distance(z_lens).value
        / c.angular_diameter_distance_z1z2(z_lens, z_source).value
    )


def _expected_loglike(om_samples, sigma_obs, err, J, kappa, g):
    """Replay of `logarithmic_kinematic_likelihood` for a known set of Omega_m draws."""
    c_km_s = _C_LIGHT.to("km/s").value
    out = []
    for i, om in enumerate(om_samples):
        pred = (
            np.sqrt(1 - kappa[i])
            * g[i]
            * c_km_s
            * np.sqrt(_dd_over_dds(om, _Z_LENS, _Z_SOURCE) * J[i])
        )
        out.append(-0.5 * ((sigma_obs - pred) / err) ** 2)
    return np.array(out)


def _set_bic(cosmo, bic):
    """BIC output stub for compute_bic_weight()"""
    cosmo.output = SimpleNamespace(_bic=bic)


def _expected_bic_weight(bic_model, bic_min, s):
    """Closed form of the BIC weight integral."""
    a = 0.5
    z = (bic_min - bic_model) / s
    return norm.cdf(z) + np.exp(
        -a * (bic_model - bic_min) + 0.5 * (a * s) ** 2
    ) * norm.cdf(-(z + a * s))


class TestCosmology:
    def setup_class(self):
        self.output = Output(_TEST_IO_DIR)
        self.cosmo = _make_cosmology(n_samples=5)

    def test_multi_plane_disabled(self):
        """Check that multi-plane is disabled by default."""
        assert self.cosmo.kwargs_model.get("multi_plane", False) is False

    def test_default_n_samples_is_chain_length(self):
        """Check that the number of samples matches chain length if not specified
        directly."""
        cosmo = _make_cosmology(burn_in=0)
        chain = self.output.get_reshaped_emcee_chain(
            _TEST_LENS_NAME, _TEST_MODEL_ID, _WALKER_RATIO, 0
        )
        assert cosmo.n_samples == chain.reshape(-1, chain.shape[-1]).shape[0]

    def test_burn_in_reduces_default_n_samples(self):
        """Check that specifying a burn-in reduces the number of samples."""
        full = _make_cosmology(burn_in=0)
        trimmed = _make_cosmology(burn_in=10)
        assert trimmed.n_samples < full.n_samples

    def test_fermat_potential_shape(self):
        """Check that the shape of the Fermat potential array is as expected given the
        number of images."""
        dphi = self.cosmo.fermat_potential()
        kwargs_ps = self.cosmo.param.args2kwargs(
            self.output.get_reshaped_emcee_chain(
                _TEST_LENS_NAME, _TEST_MODEL_ID, _WALKER_RATIO, 0
            ).reshape(-1, self.cosmo.param.num_param()[0])[0]
        )["kwargs_ps"]
        n_images = len(kwargs_ps[0]["ra_image"])
        expected_cols = 3 if n_images == 4 else 1
        assert dphi.shape == (5, expected_cols)

    def test_fermat_potential_reproducible(self):
        """Check that the Fermat potential is reproducible with the same seed."""
        a = _make_cosmology(n_samples=5, rng_seed=1).fermat_potential()
        b = _make_cosmology(n_samples=5, rng_seed=1).fermat_potential()
        np.testing.assert_array_equal(a, b)

    def test_fermat_potential_seed_changes_selection(self):
        """Check that changing the seed changes the Fermat potential."""
        a = _make_cosmology(n_samples=5, rng_seed=1).fermat_potential()
        b = _make_cosmology(n_samples=5, rng_seed=2).fermat_potential()
        assert not np.array_equal(a, b)

    def test_fermat_potential_matches_direct_lenstronomy(self):
        """Independent check: recompute one sample directly."""
        cosmo = _make_cosmology(n_samples=1, rng_seed=0)
        dphi = cosmo.fermat_potential()

        # reproduce the same random pick the class makes
        chain = cosmo.output.get_reshaped_emcee_chain(
            _TEST_LENS_NAME, _TEST_MODEL_ID, _WALKER_RATIO, 0
        )
        flat = chain.reshape(-1, chain.shape[-1])
        rng = np.random.default_rng(0)
        idx = rng.choice(flat.shape[0], size=1, replace=False)
        kwargs = cosmo.param.args2kwargs(flat[idx[0]])

        phi = cosmo.TDCosmography.fermat_potential(
            kwargs_lens=kwargs["kwargs_lens"],
            kwargs_ps=kwargs["kwargs_ps"],
            original_ps_position=False,
        )
        expected = phi[1:] - phi[0]
        np.testing.assert_allclose(dphi[0], expected)

    # single component lens light models
    def test_single_component_uses_chain_r_eff(self, cosmo):
        """Test that the single component lens light model uses the correct r_eff from
        the chain when computing J."""
        cosmo.kwargs_model["index_lens_light_model_list"] = [[0]]
        idx = _r_eff_index(cosmo)
        samples = _selected_samples(cosmo, rng_seed=0, n_samples=5)
        beta_draws = np.zeros(len(samples))
        out = cosmo.velocity_dispersion_dimension_less(
            beta_draws=beta_draws, kinematics_band="F814W"
        )

        expected_r_eff = samples[:, idx]
        expected_theta_E = np.array(
            [cosmo.param.args2kwargs(s)["kwargs_lens"][0]["theta_E"] for s in samples]
        )
        np.testing.assert_allclose(out, expected_r_eff + expected_theta_E)

        # gamma and theta_E are forwarded from the sample, not recomputed
        for call, s in zip(cosmo.calls, samples):
            lens0 = cosmo.param.args2kwargs(s)["kwargs_lens"][0]
            assert call["gamma"] == lens0["gamma"]
            assert call["theta_E"] == lens0["theta_E"]

    # multi-component lens light
    def test_multi_component_uses_saved_morphology(self, cosmo, fake_photometry):
        """Test when a multi-component lens light model uses the saved morphology."""
        cosmo.kwargs_model["index_lens_light_model_list"] = [[0, 1]]
        band = cosmo.output.model_settings["band"][0]
        r_eff = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        fake_photometry.morphology = {band: {"r_eff": r_eff}}
        beta_draws = np.zeros(len(r_eff))

        out = cosmo.velocity_dispersion_dimension_less(
            beta_draws=beta_draws, kinematics_band="F814W"
        )

        samples = _selected_samples(cosmo, rng_seed=0, n_samples=5)
        theta_E = np.array(
            [cosmo.param.args2kwargs(s)["kwargs_lens"][0]["theta_E"] for s in samples]
        )
        np.testing.assert_allclose(out, r_eff + theta_E)
        assert fake_photometry.inversion_calls == []  # no recompute

    def test_multi_component_falls_back_to_inversion(self, cosmo, fake_photometry):
        """Test when a multi-component lens light model falls back to inversion."""
        cosmo.kwargs_model["index_lens_light_model_list"] = [[0, 1]]
        band = cosmo.output.model_settings["band"][0]
        r_eff = np.full(5, 0.42)
        fake_photometry.raise_not_found = True
        fake_photometry.morphology = {band: {"r_eff": r_eff}}
        band_config = {"dummy": "config"}
        beta_draws = np.zeros(len(r_eff))

        cosmo.velocity_dispersion_dimension_less(
            beta_draws=beta_draws, band_config=band_config, kinematics_band="F814W"
        )

        assert len(fake_photometry.inversion_calls) == 1
        kw = fake_photometry.inversion_calls[0]
        assert kw["band_config"] == band_config
        assert kw["do_morphology"] is True
        assert kw["n_samples"] == 5
        assert kw["rng_seed"] == 0
        assert kw["walker_ratio"] == cosmo.walker_ratio
        assert all(c["r_eff"] == 0.42 for c in cosmo.calls)

    # Test functions for ddt_from_time_delay()
    def test_ddt_shape_double(self, cosmo):
        """One image pair -> (n_samples, 1)"""
        fermat = np.full((5, 1), 0.3)
        out = cosmo.ddt_from_time_delay(fermat, [10.0], [1.0])
        assert out.shape == (5, 1)

    def test_ddt_shape_quad(self, cosmo):
        """Three image pairs -> (n_samples, 3)"""
        fermat = np.tile([0.1, 0.3, 0.6], (7, 1))
        out = cosmo.ddt_from_time_delay(fermat, [3.0, 8.0, 15.0], [1.0, 1.0, 1.0])
        assert out.shape == (7, 3)

    def test_ddt_matches_analytic(self, cosmo):
        """With sigma=0 the draws equal the measured delays, so the result must equal c
        * dt / dphi exactly (kappa terms are zero)."""
        fermat = np.array([[0.1, 0.3, 0.6], [0.12, 0.28, 0.55]])
        dt = [3.0, 8.0, 15.0]
        out = cosmo.ddt_from_time_delay(fermat, dt, [0.0, 0.0, 0.0])
        np.testing.assert_allclose(out, _expected_ddt(fermat, dt), rtol=1e-6)

    def test_ddt_reproducible_with_same_seed(self):
        """Same seed should produce identical results."""
        fermat = np.full((10, 2), 0.3)
        a = _make_cosmology(n_samples=5, rng_seed=4).ddt_from_time_delay(
            fermat, [5.0, 9.0], [1.0, 1.0]
        )
        b = _make_cosmology(n_samples=5, rng_seed=4).ddt_from_time_delay(
            fermat, [5.0, 9.0], [1.0, 1.0]
        )
        np.testing.assert_array_equal(a, b)

    def test_ddt_seed_changes_draws(self):
        """Different seeds should produce different results."""
        fermat = np.full((10, 1), 0.3)
        a = _make_cosmology(n_samples=5, rng_seed=1).ddt_from_time_delay(
            fermat, [10.0], [1.0]
        )
        b = _make_cosmology(n_samples=5, rng_seed=2).ddt_from_time_delay(
            fermat, [10.0], [1.0]
        )
        assert not np.array_equal(a, b)

    # Tests for calculate_sampled_h0()
    def test_h0_output_shapes(self, cosmo):
        ddt = np.full((5, 3), 3000.0)
        h0, om, g = cosmo.calculate_sampled_h0(ddt)
        assert h0.shape == (5, 3)
        assert om.shape == (5,)
        assert g.shape == (5,)

    def test_h0_om_samples_replay_seed_and_bounds(self, cosmo):
        _, om, _ = cosmo.calculate_sampled_h0(np.full((5, 1), 3000.0))
        np.testing.assert_array_equal(
            om, np.random.default_rng(0).uniform(0.05, 0.5, size=5)
        )
        assert np.all((om >= 0.05) & (om <= 0.5))

    def test_h0_matches_replay(self, cosmo_tcmb):
        ddt = np.array(
            [
                [2800.0, 3100.0],
                [3000.0, 3200.0],
                [3300.0, 2900.0],
                [3050.0, 3050.0],
                [2700.0, 3400.0],
            ]
        )
        h0, om, g = cosmo_tcmb.calculate_sampled_h0(ddt)
        exp_h0, exp_g = _expected_h0_and_g(
            cosmo_tcmb.cosmology, _Z_LENS, _Z_SOURCE, om, ddt
        )
        np.testing.assert_allclose(h0, exp_h0, rtol=1e-10)
        np.testing.assert_allclose(g, exp_g, rtol=1e-10)

    def test_h0_same_cosmology_for_every_image_pair(self, cosmo):
        """Columns share Omega_m, so column ratios depend only on D_dt."""
        ddt = np.column_stack([np.full(5, 3000.0), np.full(5, 6000.0)])
        h0, _, _ = cosmo.calculate_sampled_h0(ddt)
        np.testing.assert_allclose(h0[:, 0] / h0[:, 1], 2.0)

    def test_h0_reproducible_with_same_seed(self):
        ddt = np.full((5, 2), 3000.0)
        a = _make_cosmology(n_samples=5, rng_seed=9).calculate_sampled_h0(ddt)
        b = _make_cosmology(n_samples=5, rng_seed=9).calculate_sampled_h0(ddt)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)

    # tests for compute_logarithmic_kinematic_likelihood()
    def test_kin_matches_replay(self, cosmo):
        rng = np.random.default_rng(123)
        J = rng.uniform(0.05, 0.3, 5)
        kappa = rng.uniform(-0.05, 0.15, 5)
        g = rng.uniform(0.9, 1.1, 5)
        out = cosmo.compute_logarithmic_kinematic_likelihood(260.0, 12.0, J, kappa, g)
        expected = _expected_loglike(_om_draws(), 260.0, 12.0, J, kappa, g)
        np.testing.assert_allclose(out, expected, rtol=1e-10)

    # tests for compute_bic_weight()
    @pytest.mark.parametrize("delta", [-10.0, -2.0, 0.0, 2.0, 10.0])
    def test_bic_matches_closed_form(self, cosmo, delta):
        """Exact check against the analytic convolution."""
        bic_all = [100.0, 104.0, 110.0, 120.0]  # BIC_min = 100
        sigma = np.std(bic_all, ddof=1)
        _set_bic(cosmo, 100.0 + delta)
        w = cosmo.compute_bic_weight(bic_all, n_grid=5001)
        np.testing.assert_allclose(
            w, _expected_bic_weight(100.0 + delta, 100.0, sigma), rtol=1e-4
        )
