# -*- coding: utf-8 -*-
"""Tests for photometry module."""

from pathlib import Path
import pytest
import h5py
import numpy as np
import os

from dolphin.analysis.output import Output
from dolphin.analysis.photometry import Photometry

_ROOT_DIR = Path(__file__).resolve().parents[2]
_TEST_IO_DIR = _ROOT_DIR / "io_directory_example"
_TEST_MODEL_ID_F814W = "example"


class TestPhotometry(object):
    def setup_class(self):
        self.output = Output(_TEST_IO_DIR)
        self.loaded_output1 = self.output.load_output(
            "lensed_quasar", _TEST_MODEL_ID_F814W
        )
        self.band_config1 = {
            "F814W": {
                "lens_light_indices": [0],
                "source_indices": [0],
                "exclude_lens_light_indices": [],
            }
        }

        self.photometry1 = Photometry(
            self.output,
            band_config=self.band_config1,
            model_id=_TEST_MODEL_ID_F814W,
            walker_ratio=2,
            burn_in=-1,
        )

    def test_build_band_models(self):
        """Test that _build_band_models properly functions."""

        band_models1 = self.photometry1._build_band_models()

        assert "F814W" in band_models1
        assert "data_class" in band_models1["F814W"]
        assert "psf_class" in band_models1["F814W"]
        assert "kwargs_numerics" in band_models1["F814W"]
        assert band_models1["F814W"]["likelihood_mask"] is None

    def test_load_photometry_jwst(self, monkeypatch):
        """Test the functionality of loading JWST _load_photometry."""

        monkeypatch.setattr(os.path, "exists", lambda x: True)

        class MockH5File:
            def __init__(self, *args, **kwargs):
                self.data = {
                    "PIXAR_SR": np.array(2.29e-14),
                }

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

            def __contains__(self, key):
                return key in self.data

            def __getitem__(self, key):
                return self.data[key]

        monkeypatch.setattr(h5py, "File", MockH5File)

        calib = self.photometry1._load_photometry(filt="F115W")

        assert calib["instrument"] == "JWST"
        assert "pixar_sr" in calib
        assert calib["pixar_sr"] == 2.29e-14

    def test_load_photometry_hst(self):
        """Test the functionality of loading HST _load_photometry."""

        calib = self.photometry1._load_photometry(filt="F814W")

        assert calib["instrument"] == "HST"
        assert calib["photflam"] == 1.52142145e-19
        assert calib["photzpt"] == -21.1
        assert calib["photplam"] == 8034.189

        with pytest.raises(FileNotFoundError):
            _ = self.photometry1._load_photometry(filt="INVALID")

    def test_get_ab_magnitude_jwst(self, monkeypatch):
        """Test JWST branch of _get_abmag."""

        mock_calib = {
            "instrument": "JWST",
            "pixar_sr": 2.29e-14,
        }

        monkeypatch.setattr(
            self.photometry1,
            "_load_photometry",
            lambda filt: mock_calib,
        )

        flux = np.array([5000.0, 23152.0])

        abmag = self.photometry1._get_abmag(flux, filt="F115W")

        # manually compute expected AB magnitudes
        flux_jy = flux * mock_calib["pixar_sr"] * 1e6

        expected_abmag = -2.5 * np.log10(flux_jy / 3631.0)

        np.testing.assert_allclose(
            abmag,
            expected_abmag,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_get_ab_magnitude_hst(self):
        """Test HST branch of _get_abmag."""

        calib = self.photometry1._load_photometry(filt="F814W")

        flux = np.array([5000.0, 23152.0])

        abmag = self.photometry1._get_abmag(flux, filt="F814W")

        # manually compute expected AB magnitudes
        flux_cgs = flux * calib["photflam"]

        stmag = -2.5 * np.log10(flux_cgs) + calib["photzpt"]

        expected_abmag = (
            stmag
            - 5.0 * np.log10(calib["photplam"])
            + 2.5 * np.log10(299792458e10)
            - 27.5
        )

        assert isinstance(abmag, np.ndarray)

        np.testing.assert_allclose(
            abmag,
            expected_abmag,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_evaluate_band(self):
        """Test evaluate_band returns expected structure and finite values."""

        # grab one posterior sample
        sample = self.output._posterior_samples[-1]

        kwargs_out = self.photometry1.param.args2kwargs(sample)

        kwargs_lens = kwargs_out["kwargs_lens"]
        kwargs_lens_light = kwargs_out["kwargs_lens_light"]
        kwargs_source = kwargs_out["kwargs_source"]
        kwargs_ps = kwargs_out["kwargs_ps"]

        result = self.photometry1.evaluate_band(
            filt="F814W",
            kwargs_lens_all=kwargs_lens,
            kwargs_lens_light_all=kwargs_lens_light,
            kwargs_source_all=kwargs_source,
            kwargs_ps_all=kwargs_ps,
        )

        assert isinstance(result, dict)

        assert "fluxes" in result
        assert "morphology" in result

        fluxes = result["fluxes"]

        assert "images" in fluxes
        assert "lens" in fluxes
        assert "host_lensed" in fluxes
        assert "host_intrinsic" in fluxes

        assert isinstance(fluxes["images"], np.ndarray)

        # physical sanity checks
        assert np.isfinite(fluxes["lens"])
        assert np.isfinite(fluxes["host_lensed"])
        assert np.isfinite(fluxes["host_intrinsic"])
        assert np.all(np.isfinite(fluxes["images"]))

        morph = result["morphology"]

        assert "phi" in morph
        assert "q" in morph
        assert "r_eff" in morph

        assert np.isfinite(morph["phi"])
        assert np.isfinite(morph["q"])
        assert np.isfinite(morph["r_eff"])

        # morphology sanity checks
        assert 0 <= morph["phi"] <= 180
        assert 0 < morph["q"] <= 1
        assert morph["r_eff"] > 0

        # Test error messages
        band_config = {
            "F814W": {
                # "lens_light_indices": [],
                # "source_indices": [],
                "exclude_lens_light_indices": [],
            }
        }

        photometry = Photometry(
            self.output,
            band_config=band_config,
            model_id=_TEST_MODEL_ID_F814W,
            walker_ratio=2,
            burn_in=-1,
        )

        with pytest.raises(ValueError):

            result = photometry.evaluate_band(
                filt="F814W",
                kwargs_lens_all=kwargs_lens,
                kwargs_lens_light_all=kwargs_lens_light,
                kwargs_source_all=kwargs_source,
                kwargs_ps_all=kwargs_ps,
            )

        band_config = {
            "F814W": {
                "lens_light_indices": [0],
                # "source_indices": [],
                "exclude_lens_light_indices": [],
            }
        }

        photometry = Photometry(
            self.output,
            band_config=band_config,
            model_id=_TEST_MODEL_ID_F814W,
            walker_ratio=2,
            burn_in=-1,
        )

        with pytest.raises(ValueError):

            result = photometry.evaluate_band(
                filt="F814W",
                kwargs_lens_all=kwargs_lens,
                kwargs_lens_light_all=kwargs_lens_light,
                kwargs_source_all=kwargs_source,
                kwargs_ps_all=kwargs_ps,
            )

    def test_get_flux_and_morphology(self):
        """Test get_flux_and_morphology output structure and shapes."""

        flux_chain, morph_chain = self.photometry1.get_flux_and_morphology()

        assert isinstance(flux_chain, np.ndarray)

        n_images = self.photometry1.n_images
        n_flux_per_filt = n_images + 3
        expected_cols = len(self.photometry1.filters) * n_flux_per_filt

        assert flux_chain.shape[1] == expected_cols
        assert isinstance(morph_chain, dict)

        for filt in self.photometry1.filters:

            assert filt in morph_chain

            morph = morph_chain[filt]

            assert "phi" in morph
            assert "q" in morph
            assert "r_eff" in morph

            phi = np.array(morph["phi"])
            q = np.array(morph["q"])
            r_eff = np.array(morph["r_eff"])

            assert np.all(np.isfinite(phi))
            assert np.all(np.isfinite(q))
            assert np.all(np.isfinite(r_eff))

            assert np.all((phi >= 0) & (phi <= 180))
            assert np.all((q > 0) & (q <= 1))
            assert np.all(r_eff > 0)

    def test_get_ab_magnitude(self):
        """Test get_ab_magnitude shape and consistency."""

        flux_chain, _ = self.photometry1.get_flux_and_morphology()

        mag_chain = self.photometry1.get_ab_magnitude(flux_chain)

        assert isinstance(mag_chain, np.ndarray)

        assert mag_chain.shape == flux_chain.shape

        assert np.all(np.isfinite(mag_chain))

        positive = flux_chain > 0

        reconstructed = np.zeros_like(flux_chain)
        reconstructed[positive] = mag_chain[positive]

        assert np.all(np.isfinite(reconstructed[positive]))

        assert np.all(mag_chain > -50)
        assert np.all(mag_chain < 100)

        n_images = self.photometry1.n_images
        n_flux_per_filt = n_images + 3

        filt = self.photometry1.filters[0]

        flux_block = flux_chain[:, :n_flux_per_filt]

        expected_mag = self.photometry1._get_abmag(flux_block, filt)

        np.testing.assert_allclose(
            mag_chain[:, :n_flux_per_filt],
            expected_mag,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_save_to_hdf5(self):
        """Test save_to_hdf5 writes expected structure."""

        # generate chains
        flux_chain, morph_chain = self.photometry1.get_flux_and_morphology()

        mag_chain = self.photometry1.get_ab_magnitude(flux_chain)

        # save
        self.photometry1.save_to_hdf5(
            flux_chain,
            mag_chain,
            morph_chain=morph_chain,
        )

        filename = (
            f"{self.photometry1.output.io_directory}/outputs/"
            f"photometry_{self.photometry1.system_name}_"
            f"{self.photometry1.model_id}.h5"
        )

        assert os.path.exists(filename)

        with h5py.File(filename, "r") as f:

            assert f.attrs["system_name"] == self.photometry1.system_name

            filters = list(f.attrs["filters"])

            assert filters == self.photometry1.filters

            for filt in self.photometry1.filters:

                assert filt in f

                grp = f[filt]

                expected_labels = [
                    f"Image{i+1}" for i in range(self.photometry1.n_images)
                ] + ["Lens", "Host_lensed", "Host_intrinsic"]

                for label in expected_labels:

                    assert label in grp

                    subgrp = grp[label]

                    assert "flux" in subgrp
                    assert "magnitude" in subgrp

                    flux_data = subgrp["flux"][:]
                    mag_data = subgrp["magnitude"][:]

                    assert np.all(np.isfinite(flux_data))
                    assert np.all(np.isfinite(mag_data))

            assert "lens_light_morphology" in f

            morph_grp = f["lens_light_morphology"]

            for filt in self.photometry1.filters:

                assert filt in morph_grp

                filt_grp = morph_grp[filt]

                assert "phi" in filt_grp
                assert "q" in filt_grp
                assert "r_eff" in filt_grp

                phi = filt_grp["phi"][:]
                q = filt_grp["q"][:]
                r_eff = filt_grp["r_eff"][:]

                assert np.all(np.isfinite(phi))
                assert np.all(np.isfinite(q))
                assert np.all(np.isfinite(r_eff))

    def test_get_flux_chain(self):
        """Test get_flux_chain correctly reloads saved flux chain."""

        flux_chain, morph_chain = self.photometry1.get_flux_and_morphology()

        mag_chain = self.photometry1.get_ab_magnitude(flux_chain)

        self.photometry1.save_to_hdf5(
            flux_chain,
            mag_chain,
            morph_chain=morph_chain,
        )

        loaded_flux_chain = self.photometry1.get_flux_chain()

        assert isinstance(loaded_flux_chain, np.ndarray)

        assert loaded_flux_chain.shape == flux_chain.shape

        assert np.all(np.isfinite(loaded_flux_chain))

        np.testing.assert_allclose(
            loaded_flux_chain,
            flux_chain,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_get_magnitude_chain(self):
        """Test get_magnitude_chain correctly reloads saved magnitude chain."""

        flux_chain, morph_chain = self.photometry1.get_flux_and_morphology()

        mag_chain = self.photometry1.get_ab_magnitude(flux_chain)

        self.photometry1.save_to_hdf5(
            flux_chain,
            mag_chain,
            morph_chain=morph_chain,
        )

        loaded_mag_chain = self.photometry1.get_magnitude_chain()

        assert isinstance(loaded_mag_chain, np.ndarray)

        assert loaded_mag_chain.shape == mag_chain.shape

        assert np.all(np.isfinite(loaded_mag_chain))

        np.testing.assert_allclose(
            loaded_mag_chain,
            mag_chain,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_get_morphology_chain(self):
        """Test get_morphology_chain correctly reloads saved morphology chain."""

        flux_chain, morph_chain = self.photometry1.get_flux_and_morphology()

        mag_chain = self.photometry1.get_ab_magnitude(flux_chain)

        self.photometry1.save_to_hdf5(
            flux_chain,
            mag_chain,
            morph_chain=morph_chain,
        )

        loaded_morph = self.photometry1.get_morphology_chain()

        assert isinstance(loaded_morph, dict)

        for filt in self.photometry1.filters:

            assert filt in loaded_morph

            for key in ["phi", "q", "r_eff"]:

                assert key in loaded_morph[filt]

                original = np.array(morph_chain[filt][key])
                loaded = np.array(loaded_morph[filt][key])

                assert original.shape == loaded.shape

                assert np.all(np.isfinite(loaded))

                np.testing.assert_allclose(
                    loaded,
                    original,
                    rtol=1e-10,
                    atol=1e-10,
                )

        self.photometry1.save_to_hdf5(
            flux_chain,
            mag_chain,
            # morph_chain=morph_chain,
        )

        loaded_morph = self.photometry1.get_morphology_chain()

        assert loaded_morph == None
