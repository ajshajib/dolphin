# -*- coding: utf-8 -*-
"""Tests for PSF module."""

import numpy as np
import numpy.testing as npt
from pathlib import Path
import h5py

from dolphin.preprocessing.psf import PSF

from astropy.table import Table
from astropy.io import fits

from unittest.mock import patch, MagicMock
import pytest

_ROOT_DIR = Path(__file__).resolve().parents[2]
_TEST_IO_DIR = _ROOT_DIR / "io_directory_example"


class TestPSF(object):
    def setup_class(self):
        self.psf = PSF(
            _TEST_IO_DIR, lens_name="MOCK", data_band="F814W", instrument="HST"
        )

        self.psf2 = PSF(
            _TEST_IO_DIR, lens_name="MOCK", data_band="F814W", instrument="JWST"
        )

    def test_invalid_instrument(self):
        """Test that an invalid instrument raises an error."""
        with pytest.raises(ValueError):
            _ = PSF(
                _TEST_IO_DIR, lens_name="MOCK", data_band="F814W", instrument="INVALID"
            )

    @patch("dolphin.preprocessing.psf.extract_stars")
    @patch("dolphin.preprocessing.psf.find_peaks")
    @patch("dolphin.preprocessing.psf.WCS")
    @patch("dolphin.preprocessing.psf.fits.open")
    @patch("dolphin.preprocessing.preprocessing_util.get_background")
    def test_get_psf_candidates(
        self,
        mock_get_background,
        mock_fits,
        mock_wcs,
        mock_find_peaks,
        mock_extract_stars,
    ):
        """Test PSF candidate extraction."""

        mock_get_background.return_value = (0.0, 1.0)

        mock_hdu = MagicMock()
        mock_hdu.header = {
            "RA_TARG": 0.0,
            "DEC_TARG": 0.0,
        }
        mock_hdu.data = np.ones((1000, 1000))

        mock_wht_hdu = MagicMock()
        mock_wht_hdu.data = np.ones((1000, 1000))

        mock_fits.return_value.__enter__.side_effect = [
            [mock_hdu],  # header
            [mock_hdu],  # science image
            [mock_wht_hdu],  # weight image
        ]

        mock_wcs.return_value.world_to_pixel.return_value = (500, 500)

        peaks = Table()
        peaks["x_peak"] = [500]
        peaks["y_peak"] = [500]
        peaks["peak_value"] = [5000]
        mock_find_peaks.return_value = peaks

        fake_cutout = MagicMock()
        fake_cutout.data = np.ones((51, 51))

        mock_extract_stars.side_effect = [
            [fake_cutout],  # science
            [fake_cutout],  # weights
            [fake_cutout],  # noise
        ]

        with patch.object(self.psf, "plot_psf_candidates"):
            stars, weights, noise = self.psf.get_psf_candidates()

        assert len(stars) == 1
        assert len(weights) == 1
        assert len(noise) == 1

        mock_find_peaks.assert_called_once()
        assert mock_extract_stars.call_count == 3

    @patch("dolphin.preprocessing.psf.extract_stars")
    @patch("dolphin.preprocessing.psf.find_peaks")
    @patch("dolphin.preprocessing.psf.WCS")
    @patch("dolphin.preprocessing.psf.fits.open")
    @patch("dolphin.preprocessing.preprocessing_util.get_background")
    def test_get_psf_candidates_jwst(
        self,
        mock_get_background,
        mock_fits,
        mock_wcs,
        mock_find_peaks,
        mock_extract_stars,
    ):
        """Test JWST PSF candidate extraction."""

        mock_get_background.return_value = (0.0, 1.0)

        mock_hdul = MagicMock()

        mock_primary = MagicMock()
        mock_primary.header = {
            "TARG_RA": 0.0,
            "TARG_DEC": 0.0,
        }

        mock_sci = MagicMock()
        mock_sci.data = np.ones((1000, 1000))

        mock_wht = MagicMock()
        mock_wht.data = np.ones((1000, 1000))

        mock_var_poisson = MagicMock()
        mock_var_poisson.data = np.ones((1000, 1000))

        mock_var_rnoise = MagicMock()
        mock_var_rnoise.data = np.ones((1000, 1000))

        mock_var_flat = MagicMock()
        mock_var_flat.data = np.ones((1000, 1000))

        mock_hdul.__getitem__.side_effect = lambda key: {
            0: mock_primary,
            "SCI": mock_sci,
            "WHT": mock_wht,
            "VAR_POISSON": mock_var_poisson,
            "VAR_RNOISE": mock_var_rnoise,
            "VAR_FLAT": mock_var_flat,
        }[key]

        mock_fits.return_value.__enter__.side_effect = [
            mock_hdul,
            mock_hdul,
            mock_hdul,
            mock_hdul,
        ]

        mock_wcs.return_value.world_to_pixel.return_value = (500, 500)

        peaks = Table()
        peaks["x_peak"] = [500]
        peaks["y_peak"] = [500]
        peaks["peak_value"] = [5000]
        mock_find_peaks.return_value = peaks

        fake_cutout = MagicMock()
        fake_cutout.data = np.ones((51, 51))

        mock_extract_stars.side_effect = [
            [fake_cutout],  # science
            [fake_cutout],  # weights
            [fake_cutout],  # noise
            [fake_cutout],  # repeat entries for saving test
            [fake_cutout],
            [fake_cutout],
        ]

        with patch.object(self.psf2, "plot_psf_candidates"):
            stars, weights, noise = self.psf2.get_psf_candidates()

        assert len(stars) == 1
        assert len(weights) == 1
        assert len(noise) == 1

        mock_find_peaks.assert_called_once()
        assert mock_extract_stars.call_count == 3

        # test saving
        with patch.object(self.psf2.file_system, "save_star_cutouts") as mock_save:
            with patch.object(self.psf2, "plot_psf_candidates"):
                stars, weights, noise = self.psf2.get_psf_candidates(save=True)

        mock_save.assert_called_once_with(
            lens_name=self.psf2.lens_name,
            data_band=self.psf2.data_band,
            star_exposures=stars,
            star_weights=weights,
            noise_maps=noise,
        )

    @patch("dolphin.preprocessing.psf.extract_stars")
    @patch("dolphin.preprocessing.psf.find_peaks")
    @patch("dolphin.preprocessing.psf.WCS")
    @patch("dolphin.preprocessing.psf.fits.open")
    @patch("dolphin.preprocessing.preprocessing_util.get_background")
    def test_get_psf_candidates_include_exclude(
        self,
        mock_get_background,
        mock_fits,
        mock_wcs,
        mock_find_peaks,
        mock_extract_stars,
    ):
        """Test include_specific and exclude_specific selection."""

        mock_get_background.return_value = (0.0, 1.0)

        mock_hdul = MagicMock()

        mock_primary = MagicMock()
        mock_primary.header = {
            "TARG_RA": 0.0,
            "TARG_DEC": 0.0,
        }

        mock_sci = MagicMock()
        mock_sci.data = np.ones((1000, 1000))

        mock_wht = MagicMock()
        mock_wht.data = np.ones((1000, 1000))

        mock_var_poisson = MagicMock()
        mock_var_poisson.data = np.ones((1000, 1000))

        mock_var_rnoise = MagicMock()
        mock_var_rnoise.data = np.ones((1000, 1000))

        mock_var_flat = MagicMock()
        mock_var_flat.data = np.ones((1000, 1000))

        mock_hdul.__getitem__.side_effect = lambda key: {
            0: mock_primary,
            "SCI": mock_sci,
            "WHT": mock_wht,
            "VAR_POISSON": mock_var_poisson,
            "VAR_RNOISE": mock_var_rnoise,
            "VAR_FLAT": mock_var_flat,
        }[key]

        mock_fits.return_value.__enter__.side_effect = [
            mock_hdul,
            mock_hdul,
            mock_hdul,
            mock_hdul,
            mock_hdul,
            mock_hdul,
        ]

        mock_wcs.return_value.world_to_pixel.return_value = (500, 500)

        peaks = Table()
        peaks["x_peak"] = [100, 200, 300, 400]
        peaks["y_peak"] = [100, 200, 300, 400]
        peaks["peak_value"] = [1000, 900, 800, 700]
        mock_find_peaks.return_value = peaks

        def fake_extract(data, stars_table, size):
            _ = data  # placeholder to prevent crashing
            _ = size  # placeholder to prevent crashing
            return [MagicMock() for _ in range(len(stars_table))]

        mock_extract_stars.side_effect = fake_extract

        test_cases = [
            (
                {"exclude_specific": [1, 3]},
                [100, 300],
            ),
            (
                {"include_specific": [1, 3]},
                [200, 400],
            ),
            (
                {"exclude_specific": [0], "include_specific": [2]},
                [300],  # include_specific takes precedence
            ),
        ]

        with patch.object(self.psf2, "plot_psf_candidates"):
            for kwargs, expected in test_cases:
                mock_extract_stars.reset_mock()

                self.psf2.get_psf_candidates(**kwargs)

                stars_table = mock_extract_stars.call_args_list[0].args[1]

                assert list(stars_table["x_peak"]) == expected
                assert len(stars_table) == len(expected)

    @patch("dolphin.preprocessing.psf.plt.tight_layout")
    @patch("dolphin.preprocessing.psf.plt.subplots")
    @patch("dolphin.preprocessing.preprocessing_util.build_mask")
    def test_make_candidate_mask(
        self,
        mock_build_mask,
        mock_subplots,
        mock_tight_layout,
    ):
        """Test that make_candidate_mask constructs and optionally saves a mask."""

        star = np.ones((5, 5))
        weight = np.full((5, 5), 2.0)
        noise = np.full((5, 5), 3.0)

        expected_mask = np.ones((5, 5), dtype=bool)
        mock_build_mask.return_value = expected_mask

        mock_axes = [MagicMock(), MagicMock(), MagicMock()]
        mock_subplots.return_value = (MagicMock(), mock_axes)

        with (
            patch.object(
                self.psf,
                "load_psf_candidate_attributes",
                return_value=([star], None, [weight], [noise]),
            ),
            patch.object(
                self.psf.file_system,
                "save_psf_candidate_mask",
            ) as mock_save,
        ):

            kwargs_mask = [
                {
                    "type": "circle",
                    "center": (2, 2),
                    "radius": 1,
                }
            ]

            self.psf.make_candidate_mask(
                star_num=0,
                kwargs_mask=kwargs_mask,
                save=True,
            )

        mock_build_mask.assert_called_once_with(star.shape, kwargs_mask)

        # verify each panel was displayed
        assert mock_axes[0].imshow.called
        assert mock_axes[1].imshow.called
        assert mock_axes[2].imshow.called

        mock_axes[0].set_title.assert_called_once_with("Candidate Cutout")
        mock_axes[1].set_title.assert_called_once_with("Weight Map")
        mock_axes[2].set_title.assert_called_once_with(r"$\sigma$")

        mock_tight_layout.assert_called_once()

        mock_save.assert_called_once_with(
            self.psf.lens_name,
            self.psf.data_band,
            0,
            expected_mask,
        )

    @patch("dolphin.preprocessing.psf.psfr.psf_error_map")
    @patch("dolphin.preprocessing.psf.psfr.stack_psf")
    def test_make_psf_psfr(
        self,
        mock_stack_psf,
        mock_psf_error_map,
    ):
        """Test that `make_psf_psfr` operates as expected."""
        # mock candidate data
        star_list = [
            np.ones((3, 3)),
            np.ones((3, 3)),
        ]

        mask_list = [
            np.ones((3, 3), dtype=bool),
            np.ones((3, 3), dtype=bool),
        ]

        with patch.object(
            self.psf,
            "load_psf_candidate_attributes",
            return_value=(star_list, mask_list, None, None),
        ):
            # mock PSFr outputs
            psf_guess = np.array(
                [
                    [1e-30, 1.0, 1e-30],
                    [1.0, 2.0, 1.0],
                    [1e-30, 1.0, 1e-30],
                ]
            )

            center_list = [
                [0.0, 0.0],
                [0.1, -0.1],
            ]

            mock_stack_psf.return_value = (
                psf_guess,
                center_list,
            )

            error_map = np.array(
                [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                    [0.7, 0.8, 0.9],
                ]
            )

            mock_psf_error_map.return_value = error_map

            with patch.object(
                self.psf,
                "plot_psf_and_variance_map",
            ) as mock_plot:
                # test saving
                with patch.object(
                    self.psf.file_system, "save_psf_and_variance_map"
                ) as mock_save:
                    final_psf, variance_map = self.psf.make_psf_psfr(
                        cut_threshold=1e-20,
                        save=True,
                    )

        mock_save.assert_called_once_with(
            lens_name=self.psf.lens_name,
            data_band=self.psf.data_band,
            psf_guess=final_psf,
            variance_map=variance_map,
        )

        # expected masking
        expected_psf = np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 2.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )

        expected_variance = np.array(
            [
                [0.0, 0.2, 0.0],
                [0.4, 0.5, 0.6],
                [0.0, 0.8, 0.0],
            ]
        )

        npt.assert_array_equal(final_psf, expected_psf)
        npt.assert_allclose(variance_map, expected_variance)

        mock_stack_psf.assert_called_once()
        mock_psf_error_map.assert_called_once()
        mock_plot.assert_called_once()

    @patch("dolphin.preprocessing.psf.psfr.psf_error_map")
    @patch("dolphin.preprocessing.psf.psfr.stack_psf")
    def test_make_psf_psfr_oversampling(
        self,
        mock_stack_psf,
        mock_psf_error_map,
    ):
        """Test PSFr reconstruction with oversampling > 1."""
        star_list = [
            np.ones((2, 2)),
            np.ones((2, 2)),
        ]

        mask_list = [
            np.ones((2, 2), dtype=bool),
            np.ones((2, 2), dtype=bool),
        ]

        with patch.object(
            self.psf,
            "load_psf_candidate_attributes",
            return_value=(star_list, mask_list, None, None),
        ):
            # 4x4 oversampled PSF (oversampling=2)
            psf_guess = np.array(
                [
                    [1.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 2.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            )

            center_list = [
                [0.0, 0.0],
                [0.1, -0.1],
            ]

            mock_stack_psf.return_value = (
                psf_guess,
                center_list,
            )

            # native resolution variance map
            variance_map = np.array(
                [
                    [10.0, 20.0],
                    [30.0, 40.0],
                ]
            )

            mock_psf_error_map.return_value = variance_map

            with patch.object(
                self.psf,
                "plot_psf_and_variance_map",
            ) as mock_plot:
                final_psf, final_variance = self.psf.make_psf_psfr(
                    oversampling=2,
                    cut_threshold=0.5,
                )

        expected_psf = np.array(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 2.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )

        # every 2x2 block contains at least one pixel > threshold, so every
        # variance pixel should be retained
        expected_variance = np.array(
            [
                [10.0, 20.0],
                [30.0, 40.0],
            ]
        )

        npt.assert_array_equal(final_psf, expected_psf)
        npt.assert_array_equal(final_variance, expected_variance)

        mock_stack_psf.assert_called_once()
        assert mock_stack_psf.call_args.kwargs["oversampling"] == 2
        mock_psf_error_map.assert_called_once()
        mock_plot.assert_called_once()

    @patch("dolphin.preprocessing.psf.propagate_noise")
    @patch("dolphin.preprocessing.psf.Optimizer")
    @patch("dolphin.preprocessing.psf.Loss")
    @patch("dolphin.preprocessing.psf.ParametersPSF")
    @patch("dolphin.preprocessing.psf.STARRED_PSF")
    def test_make_psf_starred(
        self,
        mock_starred_psf,
        mock_parameters,
        mock_loss,
        mock_optimizer,
        mock_propagate_noise,
    ):
        """Test that `make_psf_starred` operates as expected."""
        # Mock candidate data
        star_data = np.ones((5, 5))
        noise_map = np.ones((5, 5)) * 0.1
        mask = np.ones((5, 5), dtype=bool)

        with patch.object(
            self.psf,
            "load_psf_candidate_attributes",
            return_value=(
                [star_data, star_data],  # star_data_list
                [mask, mask],  # mask_data_list
                None,
                np.array([noise_map, noise_map]),
            ),
        ):
            # Mock STARRED model
            model = MagicMock()
            mock_starred_psf.return_value = model

            model.smart_guess.return_value = (
                {"init": 1},
                {"fixed": 1},
                {"up": 1},
                {"down": 1},
            )

            psf_guess = np.array(
                [
                    [1e-30, 1.0, 1e-30],
                    [1.0, 2.0, 1.0],
                    [1e-30, 1.0, 1e-30],
                ]
            )

            error_map = np.array(
                [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                    [0.7, 0.8, 0.9],
                ]
            )

            model.get_full_psf.return_value = psf_guess
            model.get_psf_error_map.return_value = error_map

            # Mock ParametersPSF
            params = MagicMock()
            mock_parameters.return_value = params

            kwargs_partial = {
                "kwargs_moffat": {"C": 1.0},
                "kwargs_distortion": {},
            }

            kwargs_final = {
                "kwargs_moffat": {"C": 1.0},
                "kwargs_distortion": {},
            }

            params.args2kwargs.side_effect = [
                kwargs_partial,
                kwargs_final,
            ]

            # Mock propagate_noise
            mock_propagate_noise.return_value = [np.ones((3, 3))]

            # Mock optimizers
            optimizer_1 = MagicMock()
            optimizer_2 = MagicMock()

            mock_optimizer.side_effect = [
                optimizer_1,
                optimizer_2,
            ]

            optimizer_1.minimize.return_value = (
                np.array([1.0]),
                None,
                {"loss_history": [1.0, 0.5]},
                None,
            )

            optimizer_2.minimize.return_value = (
                np.array([2.0]),
                None,
                {"loss_history": [0.5, 0.1]},
                None,
            )

            with patch.object(
                self.psf,
                "plot_psf_and_variance_map",
            ) as mock_plot:
                # test saving
                with patch.object(
                    self.psf.file_system, "save_psf_and_variance_map"
                ) as _:
                    final_psf, variance_map = self.psf.make_psf_starred(
                        cut_threshold=1e-20,
                        save=True,
                    )

        # Expected outputs
        expected_psf = np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 2.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )

        expected_variance = np.array(
            [
                [0.0, 0.04, 0.0],
                [0.16, 0.25, 0.36],
                [0.0, 0.64, 0.0],
            ]
        )

        npt.assert_array_equal(final_psf, expected_psf)
        npt.assert_allclose(variance_map, expected_variance)
        assert mock_optimizer.call_count == 2
        assert params.args2kwargs.call_count == 2
        mock_propagate_noise.assert_called_once()
        mock_plot.assert_called_once()

    @patch("dolphin.preprocessing.psf.make_axes_locatable")
    @patch("dolphin.preprocessing.psf.plt.show")
    @patch("dolphin.preprocessing.psf.plt.colorbar")
    @patch("dolphin.preprocessing.psf.WCS")
    @patch("dolphin.preprocessing.psf.fits.getdata")
    def test_plot_psf_candidates(
        self,
        mock_getdata,
        mock_wcs,
        mock_colorbar,
        mock_show,
        mock_divider,
    ):
        """Test that `plot_psf_candidates` produces all expected plots."""

        stars_table = Table(
            {
                "peak_value": [1000.0, 900.0],
                "x_peak": [100.0, 200.0],
                "y_peak": [150.0, 250.0],
            }
        )

        star_exposures = []
        for value in [1, 2]:
            star = MagicMock()
            star.data = np.ones((5, 5)) * value
            star.flux = float(value)
            star_exposures.append(star)

        star_weights = []
        for value in [0.1, 0.2]:
            mock = MagicMock()
            mock.data = np.ones((5, 5)) * value
            star_weights.append(mock)

        noise_maps = []
        for value in [0.1, 0.2]:
            mock = MagicMock()
            mock.data = np.ones((5, 5)) * value
            noise_maps.append(mock)

        mock_getdata.return_value = (np.ones((1000, 1000)), {})

        wcs = MagicMock()
        mock_wcs.return_value = wcs
        wcs.all_pix2world.side_effect = lambda x, y, origin: (x, y)
        wcs.all_world2pix.side_effect = lambda x, y, origin: (x, y)

        divider = MagicMock()
        divider.append_axes.return_value = MagicMock()
        mock_divider.return_value = divider

        self.psf.plot_psf_candidates(
            star_exposures=star_exposures,
            star_weights=star_weights,
            noise_maps=noise_maps,
            stars_table=stars_table,
        )

        # 3 image grids + variance plot + full image plot
        assert mock_show.call_count == 5

        mock_getdata.assert_called_once_with(
            self.psf.image_file_name,
            header=True,
        )
        mock_wcs.assert_called_once()
        mock_colorbar.assert_called_once()
        mock_divider.assert_called_once()

    @patch("dolphin.preprocessing.psf.plt.show")
    @patch("dolphin.preprocessing.psf.plt.tight_layout")
    def test_plot_saved_psf_candidates(
        self,
        mock_tight_layout,
        mock_show,
    ):
        """Test that `plot_saved_psf_candidates` returns the expected plots."""
        # fake star cutouts
        star_exposures = [
            np.ones((5, 5)),
            np.ones((5, 5)),
        ]

        mask_data = [
            np.ones((5, 5), dtype=bool),
            np.ones((5, 5), dtype=bool),
        ]

        star_weights = [
            np.ones((5, 5)),
            np.ones((5, 5)),
        ]

        noise_maps = [
            np.ones((5, 5)) * 0.1,
            np.ones((5, 5)) * 0.1,
        ]

        with patch.object(
            self.psf,
            "load_psf_candidate_attributes",
            return_value=(
                star_exposures,
                mask_data,
                star_weights,
                noise_maps,
            ),
        ) as mock_load:

            self.psf.plot_saved_psf_candidates()

        mock_load.assert_called_once()

        # star cutouts
        # weight maps
        # noise maps
        # counts vs. variance plot
        assert mock_show.call_count == 4

    @patch("dolphin.preprocessing.psf.plt.show")
    @patch("dolphin.preprocessing.psf.plt.tight_layout")
    def test_plot_psf_and_variance_map_psfr(
        self,
        mock_tight_layout,
        mock_show,
    ):
        """Test that the PSFr branch of `plot_psf_and_variance_map` operates as
        expected."""
        psf = np.ones((5, 5))
        variance = np.ones((5, 5)) * 0.1

        psf_cut = psf.copy()
        variance_cut = variance.copy()

        PSF.plot_psf_and_variance_map(
            method="PSFr",
            psf_guess=psf,
            variance_map=variance,
            psf_cut=psf_cut,
            variance_map_cut=variance_cut,
        )

        # original + cut version
        assert mock_show.call_count == 2

    @patch("dolphin.preprocessing.psf.pltf.plot_loss")
    @patch("dolphin.preprocessing.psf.plt.show")
    @patch("dolphin.preprocessing.psf.plt.tight_layout")
    def test_plot_psf_and_variance_map_starred(
        self,
        mock_tight_layout,
        mock_show,
        mock_plot_loss,
    ):
        """Test that the STARRED branch of `plot_psf_and_variance_map` operates as
        expected."""
        psf = np.ones((5, 5))
        variance = np.ones((5, 5)) * 0.1

        kwargs_starred = {"extra_fields": {"loss_history": [10, 5, 1]}}

        PSF.plot_psf_and_variance_map(
            method="STARRED",
            psf_guess=psf,
            variance_map=variance,
            psf_cut=psf,
            variance_map_cut=variance,
            kwargs_starred=kwargs_starred,
        )

        mock_plot_loss.assert_called_once_with([10, 5, 1])

        # loss plot + main plot + cut plot
        assert mock_show.call_count == 3

    @patch("dolphin.preprocessing.psf.plt.show")
    def test_plot_psf_and_variance_map_psfr_no_cut(
        self,
        mock_show,
    ):
        """Test that `plot_psf_and_variance_map` creates only one plot if there is no
        variance map."""
        psf = np.ones((5, 5))
        variance = np.ones((5, 5))

        PSF.plot_psf_and_variance_map(
            method="PSFr",
            psf_guess=psf,
            variance_map=variance,
        )

        assert mock_show.call_count == 1

    @patch("dolphin.preprocessing.psf.plt.show")
    def test_load_saved_psf(
        self,
        mock_show,
    ):
        """Test the functionality of `load_saved_psf`."""
        psf_data = np.ones((5, 5))
        variance_map = np.ones((5, 5))

        with patch.object(
            self.psf.file_system,
            "load_saved_psf",
            return_value=(psf_data, variance_map),
        ) as _:

            psf, variance = self.psf.load_saved_psf(plot=True)

        npt.assert_array_equal(psf, psf_data)
        npt.assert_array_equal(variance, variance_map)

        mock_show.assert_called_once()

    @patch("dolphin.preprocessing.psf.plt.show")
    def test_load_saved_psf_no_plot(
        self,
        mock_show,
    ):
        """Test the functionality of `load_saved_psf` with the `plot` flag to False."""
        psf_data = np.ones((5, 5))
        variance_map = np.ones((5, 5))

        with patch.object(
            self.psf.file_system,
            "load_saved_psf",
            return_value=(psf_data, variance_map),
        ):

            self.psf.load_saved_psf(plot=False)

        mock_show.assert_not_called()

    def test_load_psf_candidate_attributes(self):
        """Test that `load_psf_candidate_attributes` returns expected components."""
        expected = (
            ["stars"],
            ["masks"],
            ["weights"],
            ["noise"],
        )

        with patch.object(
            self.psf.file_system,
            "load_psf_candidate_attributes",
            return_value=expected,
        ) as _:

            result = self.psf.load_psf_candidate_attributes()

        assert result == expected

    def test_save_star_cutouts(self):
        """Test saving star cutouts, weights, and noise maps."""

        lens_name = "lens_system1"
        data_band = "F390W"
        psf_temp = PSF(_TEST_IO_DIR, lens_name, data_band, "HST")

        # redirect preprocessing path to temporary directory
        preprocessing_path = Path(psf_temp.file_system.get_preprocessing_path(lens_name))

        star_exposures = [
            MagicMock(data=np.ones((5, 5))),
            MagicMock(data=np.full((5, 5), 2.0)),
        ]

        star_weights = [
            MagicMock(data=np.ones((5, 5)) * 3),
            MagicMock(data=np.ones((5, 5)) * 4),
        ]

        noise_maps = [
            MagicMock(data=np.ones((5, 5)) * 5),
            MagicMock(data=np.ones((5, 5)) * 6),
        ]

        psf_temp.file_system.save_star_cutouts(
            lens_name=lens_name,
            data_band=data_band,
            star_exposures=star_exposures,
            star_weights=star_weights,
            noise_maps=noise_maps,
        )

        star_dir = preprocessing_path / data_band / "stars"
        weight_dir = preprocessing_path / data_band / "weights"
        noise_dir = preprocessing_path / data_band / "noise_maps"

        # check directories exist
        assert star_dir.exists()
        assert weight_dir.exists()
        assert noise_dir.exists()

        # check files exist
        assert len(list(star_dir.glob("*.fits"))) == 2
        assert len(list(weight_dir.glob("*.fits"))) == 2
        assert len(list(noise_dir.glob("*.fits"))) == 2

        # check FITS contents
        with fits.open(star_dir / "star_0.fits") as hdul:
            np.testing.assert_array_equal(
                hdul[0].data,
                np.ones((5, 5)),
            )

        with fits.open(weight_dir / "weight_1.fits") as hdul:
            np.testing.assert_array_equal(
                hdul[0].data,
                np.ones((5, 5)) * 4,
            )

        with fits.open(noise_dir / "noise_map_0.fits") as hdul:
            np.testing.assert_array_equal(
                hdul[0].data,
                np.ones((5, 5)) * 5,
            )
    def test_save_psf_candidate_mask(self):
        """Test saving PSF candidate masks."""

        lens_name = "lens_system1"
        data_band = "F390W"
        star_num = 2

        psf_temp = PSF(_TEST_IO_DIR, lens_name, data_band, "HST")

        preprocessing_path = Path(
            psf_temp.file_system.get_preprocessing_path(lens_name)
        )

        mask = np.array(
            [
                [True, False, True],
                [False, True, False],
                [True, True, False],
            ]
        )

        psf_temp.file_system.save_psf_candidate_mask(
            lens_name=lens_name,
            data_band=data_band,
            star_num=star_num,
            mask=mask,
        )

        mask_dir = preprocessing_path / data_band / "masks"
        mask_file = mask_dir / f"mask_{star_num}.npy"

        # check directory exists
        assert mask_dir.exists()

        # check file exists
        assert mask_file.exists()

        # check saved mask contents
        saved_mask = np.load(mask_file)

        np.testing.assert_array_equal(
            saved_mask,
            mask,
        )

    def test_save_psf_and_variance_map(self):
        """Test saving PSF and variance map to HDF5 format."""

        lens_name = "lens_system1"
        data_band = "F390W"

        psf_temp = PSF(_TEST_IO_DIR, lens_name, data_band, "HST")

        data_directory = Path(psf_temp.file_system.get_data_directory())
        psf_guess = np.ones((21, 21))
        variance_map = np.full((21, 21), 0.5)

        psf_temp.file_system.save_psf_and_variance_map(
            lens_name=lens_name,
            data_band=data_band,
            psf_guess=psf_guess,
            variance_map=variance_map,
        )

        filename = data_directory / lens_name / f"psf_{lens_name}_{data_band}.h5"

        # check file exists
        assert filename.exists()

        # check HDF5 contents
        with h5py.File(filename, "r") as f:
            assert "kernel_point_source" in f
            assert "psf_variance_map" in f

            np.testing.assert_array_equal(
                f["kernel_point_source"][:],
                psf_guess,
            )

            np.testing.assert_array_equal(
                f["psf_variance_map"][:],
                variance_map,
            )

    def test_load_psf_candidate_attributes(self):
        """Test loading PSF candidate stars, masks, weights, and noise maps."""

        lens_name = "lens_system1"
        data_band = "F390W"

        psf_temp = PSF(_TEST_IO_DIR, lens_name, data_band, "HST")

        preprocessing_path = Path(
            psf_temp.file_system.get_preprocessing_path(lens_name)
        )

        star_dir = preprocessing_path / data_band / "stars"
        weight_dir = preprocessing_path / data_band / "weights"
        noise_dir = preprocessing_path / data_band / "noise_maps"
        mask_dir = preprocessing_path / data_band / "masks"

        # create directories
        star_dir.mkdir(parents=True, exist_ok=True)
        weight_dir.mkdir(parents=True, exist_ok=True)
        noise_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        # create star cutouts
        star_0 = np.ones((5, 5))
        star_1 = np.full((5, 5), 2.0)

        fits.PrimaryHDU(star_0).writeto(
            star_dir / "star_0.fits",
            overwrite=True
        )
        fits.PrimaryHDU(star_1).writeto(
            star_dir / "star_1.fits",
            overwrite=True
        )

        # create masks
        mask_0 = np.array(
            [
                [True, False, True, True, False],
                [True, True, False, True, True],
                [False, True, True, True, True],
                [True, True, True, False, True],
                [True, False, True, True, True],
            ]
        )

        np.save(
            mask_dir / "mask_0.npy",
            mask_0,
        )

        # intentionally do not create mask_1.npy
        # to test the default all True behavior

        # create weights
        weight_0 = np.full((5, 5), 3.0)
        weight_1 = np.full((5, 5), 4.0)

        fits.PrimaryHDU(weight_0).writeto(
            weight_dir / "weight_0.fits",
            overwrite=True
        )
        fits.PrimaryHDU(weight_1).writeto(
            weight_dir / "weight_1.fits",
            overwrite=True
        )

        # create noise maps
        noise_0 = np.full((5, 5), 5.0)
        noise_1 = np.full((5, 5), 6.0)

        fits.PrimaryHDU(noise_0).writeto(
            noise_dir / "noise_map_0.fits",
            overwrite=True
        )
        fits.PrimaryHDU(noise_1).writeto(
            noise_dir / "noise_map_1.fits",
            overwrite=True
        )

        (
            stars,
            masks,
            weights,
            noise_maps,
        ) = psf_temp.file_system.load_psf_candidate_attributes(
            lens_name=lens_name,
            data_band=data_band,
        )

        # check shapes
        assert stars.shape == (2, 5, 5)
        assert masks.shape == (2, 5, 5)
        assert weights.shape == (2, 5, 5)
        assert noise_maps.shape == (2, 5, 5)

        # check star values
        np.testing.assert_array_equal(
            stars[0],
            star_0,
        )

        np.testing.assert_array_equal(
            stars[1],
            star_1,
        )

        # check mask matching
        np.testing.assert_array_equal(
            masks[0],
            mask_0,
        )

        # missing mask should default to all True
        np.testing.assert_array_equal(
            masks[1],
            np.ones((5, 5), dtype=bool),
        )

        # check weights
        np.testing.assert_array_equal(
            weights[0],
            weight_0,
        )

        np.testing.assert_array_equal(
            weights[1],
            weight_1,
        )

        # check noise maps
        np.testing.assert_array_equal(
            noise_maps[0],
            noise_0,
        )

        np.testing.assert_array_equal(
            noise_maps[1],
            noise_1,
        )

    def test_load_saved_psf(self):
        """Test loading saved PSF and variance map."""

        lens_name = "lens_system1"
        data_band = "F390W"

        psf_temp = PSF(_TEST_IO_DIR, lens_name, data_band, "HST")

        psf_file = Path(
            psf_temp.file_system.get_psf_file_path(
                lens_name,
                data_band,
            )
        )

        psf_file.parent.mkdir(parents=True, exist_ok=True)

        psf_data = np.ones((21, 21))
        variance_map = np.full((21, 21), 0.25)

        with h5py.File(psf_file, "w") as file:
            file.create_dataset(
                "kernel_point_source",
                data=psf_data,
            )
            file.create_dataset(
                "psf_variance_map",
                data=variance_map,
            )

        loaded_psf, loaded_variance = psf_temp.file_system.load_saved_psf(
            lens_name=lens_name,
            data_band=data_band,
        )

        np.testing.assert_array_equal(
            loaded_psf,
            psf_data,
        )

        np.testing.assert_array_equal(
            loaded_variance,
            variance_map,
        )