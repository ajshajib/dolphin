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

        self.magnitude_config1 = {
            "F814W": {"photflam": 1.52122335e-19, "photplam": 8034.189}
        }

        self.magnitude_config2 = {"F115W": {"pixar_sr": 2.29160304105492e-14}}

        self.photometry1 = Photometry(
            self.output,
            band_config=self.band_config1,
            model_id=_TEST_MODEL_ID_F814W,
            walker_ratio=2,
            burn_in=-1,
            aperture_radius=None,
            aperture_length=None,
            do_morphology=True,
        )

        self.photometry2 = Photometry(
            self.output,
            band_config=self.band_config1,
            model_id=_TEST_MODEL_ID_F814W,
            walker_ratio=2,
            burn_in=-1,
            aperture_radius=None,
            aperture_length=None,
            do_morphology=False,
        )

    def test_build_band_models(self):
        """Test that _build_band_models properly functions."""

        band_models1 = self.photometry1._build_band_models()

        assert "F814W" in band_models1
        assert "data_class" in band_models1["F814W"]
        assert "psf_class" in band_models1["F814W"]
        assert "kwargs_numerics" in band_models1["F814W"]
        assert band_models1["F814W"]["likelihood_mask"] is None

    def test_aperture_mask(self):
        """Test `_aperature_mask` shapes and behavior."""

        data_class = self.photometry1.band_models["F814W"]["data_class"]

        x_grid, y_grid = data_class.pixel_coordinates

        expected_shape = x_grid.shape

        center_x = 0.0
        center_y = 0.0

        # circular aperture
        circular_mask = self.photometry1._aperture_mask(
            data_class=data_class,
            center_x=center_x,
            center_y=center_y,
            radius=1.0,
        )

        assert circular_mask.shape == expected_shape
        assert circular_mask.dtype == bool

        # should contain some True and some False pixels
        assert np.any(circular_mask)
        assert not np.all(circular_mask)

        # square aperture
        square_mask = self.photometry1._aperture_mask(
            data_class=data_class,
            center_x=center_x,
            center_y=center_y,
            length=1.0,
        )

        assert square_mask.shape == expected_shape
        assert square_mask.dtype == bool

        assert np.any(square_mask)
        assert not np.all(square_mask)

        # full image default mask
        full_mask = self.photometry1._aperture_mask(
            data_class=data_class,
            center_x=center_x,
            center_y=center_y,
        )

        assert full_mask.shape == expected_shape
        assert full_mask.dtype == bool
        assert np.all(full_mask)

    def test_load_photometry(self):
        """Test the functionality of `_load_photometry`."""

        test_mag_config = {
            "F814W": {
                "photflam": 1.52122335e-19,
                "photzpt": -21.1,
                # "photplam": 8034.189
            }
        }

        with pytest.raises(ValueError):
            _ = self.photometry1._load_photometry(
                data_band="F814W", magnitude_config=test_mag_config
            )

        calib = self.photometry1._load_photometry(
            data_band="F814W", magnitude_config=self.magnitude_config1
        )

        assert calib["instrument"] == "HST"
        assert calib["photflam"] == 1.52122335e-19
        assert calib["photplam"] == 8034.189

        calib2 = self.photometry1._load_photometry(
            data_band="F115W", magnitude_config=self.magnitude_config2
        )

        assert calib2["instrument"] == "JWST"
        assert "pixar_sr" in calib2
        assert calib2["pixar_sr"] == 2.29160304105492e-14

        with pytest.raises(KeyError):
            _ = self.photometry1._load_photometry(
                data_band="INVALID", magnitude_config=self.magnitude_config1
            )

    def test_get_ab_magnitude_hst(self):
        """Test `_get_abmag`."""

        calib = self.photometry1._load_photometry(
            data_band="F814W", magnitude_config=self.magnitude_config1
        )

        flux = np.array([5000.0, 23152.0])

        abmag = self.photometry1._get_abmag(
            flux, data_band="F814W", magnitude_config=self.magnitude_config1
        )

        # manually compute expected AB magnitudes
        flux_cgs = flux * calib["photflam"]

        stmag = -2.5 * np.log10(flux_cgs) - 21.1

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

        # Test JWST Calculations
        calib2 = self.photometry1._load_photometry(
            data_band="F115W", magnitude_config=self.magnitude_config2
        )
        abmag = self.photometry1._get_abmag(
            flux, data_band="F115W", magnitude_config=self.magnitude_config2
        )

        # manually compute expected AB magnitudes
        flux_jy = flux * calib2["pixar_sr"] * 1e6

        expected_abmag = -2.5 * np.log10(flux_jy / 3631.0)

        np.testing.assert_allclose(
            abmag,
            expected_abmag,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_do_linear_inversion_single_band(self):
        """Test _do_linear_inversion_single_band returns expected structure and finite
        values."""

        # grab one posterior sample
        sample = self.output._posterior_samples[-1]

        kwargs_out = self.photometry1.param.args2kwargs(sample)

        kwargs_lens = kwargs_out["kwargs_lens"]
        kwargs_lens_light = kwargs_out["kwargs_lens_light"]
        kwargs_source = kwargs_out["kwargs_source"]
        kwargs_ps = kwargs_out["kwargs_ps"]

        result = self.photometry1._do_linear_inversion_single_band(
            data_band="F814W",
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
        assert "source_lensed" in fluxes
        assert "source_intrinsic" in fluxes

        assert isinstance(fluxes["images"], np.ndarray)

        # physical sanity checks
        assert np.isfinite(fluxes["lens"])
        assert np.isfinite(fluxes["source_lensed"])
        assert np.isfinite(fluxes["source_intrinsic"])
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

            result = photometry._do_linear_inversion_single_band(
                data_band="F814W",
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

            result = photometry._do_linear_inversion_single_band(
                data_band="F814W",
                kwargs_lens_all=kwargs_lens,
                kwargs_lens_light_all=kwargs_lens_light,
                kwargs_source_all=kwargs_source,
                kwargs_ps_all=kwargs_ps,
            )

    def test_do_linear_inversion(self):
        """Test `do_linear_inversion` output structure and shapes."""

        flux_chain, morphology_chain = self.photometry1.do_linear_inversion()

        assert isinstance(flux_chain, np.ndarray)

        n_images = self.photometry1.n_images
        n_flux_per_filt = n_images + 3
        expected_cols = len(self.photometry1.filters) * n_flux_per_filt

        assert flux_chain.shape[1] == expected_cols
        assert isinstance(morphology_chain, dict)

        for data_band in self.photometry1.filters:

            assert data_band in morphology_chain

            morph = morphology_chain[data_band]

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

    def test_calculate_ab_magnitude(self):
        """Test `calculate_ab_magnitude` shape and consistency."""

        flux_chain, _ = self.photometry1.do_linear_inversion()

        mag_chain = self.photometry1.calculate_ab_magnitude(
            flux_chain=flux_chain, magnitude_config=self.magnitude_config1
        )

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

        data_band = self.photometry1.filters[0]

        flux_block = flux_chain[:, :n_flux_per_filt]

        expected_mag = self.photometry1._get_abmag(
            flux_block, data_band, magnitude_config=self.magnitude_config1
        )

        np.testing.assert_allclose(
            mag_chain[:, :n_flux_per_filt],
            expected_mag,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_save_to_hdf5(self):
        """Test save_to_hdf5 writes expected structure."""

        # generate chains
        flux_chain, morphology_chain = self.photometry1.do_linear_inversion()

        mag_chain = self.photometry1.calculate_ab_magnitude(
            flux_chain, magnitude_config=self.magnitude_config1
        )

        # save
        self.photometry1.save_to_hdf5(
            flux_chain,
            mag_chain,
            morphology_chain=morphology_chain,
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

            for data_band in self.photometry1.filters:

                assert data_band in f

                grp = f[data_band]

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

            for data_band in self.photometry1.filters:

                assert data_band in morph_grp

                filt_grp = morph_grp[data_band]

                assert "phi" in filt_grp
                assert "q" in filt_grp
                assert "r_eff" in filt_grp

                phi = filt_grp["phi"][:]
                q = filt_grp["q"][:]
                r_eff = filt_grp["r_eff"][:]

                assert np.all(np.isfinite(phi))
                assert np.all(np.isfinite(q))
                assert np.all(np.isfinite(r_eff))

    def test_load_flux_chain(self):
        """Test load_flux_chain correctly reloads saved flux chain."""

        flux_chain, morphology_chain = self.photometry1.do_linear_inversion()

        mag_chain = self.photometry1.calculate_ab_magnitude(
            flux_chain, magnitude_config=self.magnitude_config1
        )

        self.photometry1.save_to_hdf5(
            flux_chain,
            mag_chain,
            morphology_chain=morphology_chain,
        )

        loaded_flux_chain = self.photometry1.load_flux_chain()

        assert isinstance(loaded_flux_chain, np.ndarray)

        assert loaded_flux_chain.shape == flux_chain.shape

        assert np.all(np.isfinite(loaded_flux_chain))

        np.testing.assert_allclose(
            loaded_flux_chain,
            flux_chain,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_load_magnitude_chain(self):
        """Test load_magnitude_chain correctly reloads saved magnitude chain."""

        flux_chain, morphology_chain = self.photometry1.do_linear_inversion()

        mag_chain = self.photometry1.calculate_ab_magnitude(
            flux_chain, magnitude_config=self.magnitude_config1
        )

        self.photometry1.save_to_hdf5(
            flux_chain,
            mag_chain,
            morphology_chain=morphology_chain,
        )

        loaded_mag_chain = self.photometry1.load_magnitude_chain()

        assert isinstance(loaded_mag_chain, np.ndarray)

        assert loaded_mag_chain.shape == mag_chain.shape

        assert np.all(np.isfinite(loaded_mag_chain))

        np.testing.assert_allclose(
            loaded_mag_chain,
            mag_chain,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_load_morphology_chain(self):
        """Test load_morphology_chain correctly reloads saved morphology chain."""

        flux_chain, morphology_chain = self.photometry1.do_linear_inversion()

        mag_chain = self.photometry1.calculate_ab_magnitude(
            flux_chain, magnitude_config=self.magnitude_config1
        )

        self.photometry1.save_to_hdf5(
            flux_chain,
            mag_chain,
            morphology_chain,
        )

        loaded_morph = self.photometry1.load_morphology_chain()

        assert isinstance(loaded_morph, dict)

        for data_band in self.photometry1.filters:

            assert data_band in loaded_morph

            for key in ["phi", "q", "r_eff"]:

                assert key in loaded_morph[data_band]

                original = np.array(morphology_chain[data_band][key])
                loaded = np.array(loaded_morph[data_band][key])

                assert original.shape == loaded.shape

                assert np.all(np.isfinite(loaded))

                np.testing.assert_allclose(
                    loaded,
                    original,
                    rtol=1e-10,
                    atol=1e-10,
                )

        # test that morphology dictionary is None if do_morphology
        # is initialized as False
        flux_chain, _ = self.photometry2.do_linear_inversion()

        self.photometry2.save_to_hdf5(
            flux_chain
        )

        loaded_morph = self.photometry2.load_morphology_chain()

        assert loaded_morph is None
