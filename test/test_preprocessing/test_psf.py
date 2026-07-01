# -*- coding: utf-8 -*-
"""Tests for PSF module."""

import pytest
import numpy as np
import numpy.testing as npt
from pathlib import Path

from dolphin.preprocessing.psf import PSF

from astropy.io import fits
from unittest.mock import patch, MagicMock

_ROOT_DIR = Path(__file__).resolve().parents[2]
_TEST_IO_DIR = _ROOT_DIR / "io_directory_example"


class TestPSF(object):
    def setup_class(self):
        self.psf = PSF(
            _TEST_IO_DIR, lens_name="MOCK", data_band="F814W", instrument="HST"
        )

    @pytest.fixture
    def mock_catalog(self):
        return fits.BinTableHDU.from_columns(
            [
                fits.Column(
                    name="MAG_BEST", format="E", array=np.array([23.0, 24.0, 25.0])
                ),
                fits.Column(
                    name="FLUX_RADIUS", format="E", array=np.array([2.0, 3.0, 4.0])
                ),
                fits.Column(
                    name="ELLIPTICITY", format="E", array=np.array([0.05, 0.1, 0.3])
                ),
                fits.Column(
                    name="CLASS_STAR", format="E", array=np.array([0.9, 0.8, 0.2])
                ),
                fits.Column(
                    name="X_IMAGE", format="E", array=np.array([500.0, 600.0, 700.0])
                ),
                fits.Column(
                    name="Y_IMAGE", format="E", array=np.array([500.0, 600.0, 700.0])
                ),
                fits.Column(name="FLAGS", format="I", array=np.array([0, 0, 1])),
                fits.Column(name="NUMBER", format="I", array=np.array([1, 2, 3])),
            ]
        )

    @patch("dolphin.preprocessing.psf.os.replace")
    @patch("dolphin.preprocessing.psf.os.makedirs")
    @patch("dolphin.preprocessing.psf.fits.open")
    @patch("dolphin.preprocessing.psf.subprocess.run")
    def test_make_image_catalog(self, mock_run, mock_fits, mock_makedirs, mock_replace):
        """Test that `make_image_catalog` runs and creates the properly named file."""
        catalog = MagicMock()
        mock_fits.return_value = [None, None, catalog]

        result = self.psf.make_image_catalog()

        assert result is catalog
        mock_run.assert_called_once()
        mock_fits.assert_called_once_with("MOCK_F814W.cat")
        mock_makedirs.assert_called_once()
        mock_replace.assert_called_once()

    def test_get_kwargs_cut(self):
        """Test that `get_kwargs_cut` returns the expected keys."""
        catalog = fits.BinTableHDU.from_columns(
            [
                fits.Column(
                    name="MAG_BEST",
                    format="E",
                    array=np.array([20.0, 21.0, 22.0, 23.0, 24.0]),
                ),
                fits.Column(
                    name="FLUX_RADIUS",
                    format="E",
                    array=np.array([2.0, 3.0, 4.0, 5.0, 6.0]),
                ),
            ]
        )

        with patch.object(
            self.psf, "make_image_catalog", return_value=catalog
        ) as mock_make:

            kwargs_cut, returned_catalog = self.psf.get_kwargs_cut()

        mock_make.assert_called_once()
        assert returned_catalog is catalog
        assert kwargs_cut == {
            "MagMinThresh": 20.0,
            "MagMaxThresh": 21.2,
            "SizeMinThresh": 2.0,
            "SizeMaxThresh": 6.0,
            "EllipticityThresh": 0.1,
            "ClassStarMin": 0.5,
            "ClassStarMax": 1.0,
        }

    @patch("dolphin.preprocessing.psf.WCS")
    @patch("dolphin.preprocessing.psf.Analysis")
    @patch("dolphin.preprocessing.psf.StrongLensSystem")
    @patch("dolphin.preprocessing.psf.fits.open")
    def test_get_psf_candidates(
        self,
        mock_fits,
        mock_system,
        mock_analysis,
        mock_wcs,
        mock_catalog,
    ):
        """Test that `get_psf_candidates` returns the expected number of cutouts,
        weights, and noise maps."""
        kwargs_cut = {
            "SizeMaxThresh": 5,
            "SizeMinThresh": 1,
            "EllipticityThresh": 0.2,
            "MagMaxThresh": 26,
            "MagMinThresh": 22,
            "ClassStarMax": 1.0,
            "ClassStarMin": 0.5,
        }

        # Mock FITS file
        mock_hdu = MagicMock()
        mock_hdu.header = {
            "RA_TARG": 0.0,
            "DEC_TARG": 0.0,
        }

        mock_fits.return_value.__enter__.return_value = [mock_hdu]

        # Mock WCS
        mock_wcs.return_value.world_to_pixel.return_value = (0, 0)

        # Mock StrongLensSystem
        mock_system_instance = mock_system.return_value

        mock_system_instance.get_background.return_value = (
            0.0,  # mean background
            1.0,  # sigma background
        )

        mock_system_instance.get_full_image.return_value = np.ones((1000, 1000))

        mock_system_instance.get_full_exposure.return_value = np.ones((1000, 1000))

        # Mock extracted stars
        fake_cutout = np.ones((5, 5))

        mock_analysis.return_value.get_objects_image.return_value = [
            fake_cutout,
            fake_cutout,
        ]

        with patch.object(self.psf, "plot_psf_candidates"):
            stars, weights, noise = self.psf.get_psf_candidates(
                mock_catalog,
                kwargs_cut,
                radius_pix=0,
            )

        # Assertions
        assert len(stars) == 2
        assert len(weights) == 2
        assert len(noise) == 2
        np.testing.assert_array_equal(
            stars[0],
            fake_cutout,
        )

    @patch("dolphin.preprocessing.psf.psfr.psf_error_map")
    @patch("dolphin.preprocessing.psf.psfr.stack_psf")
    def test_make_psf_psfr(
        self,
        mock_stack_psf,
        mock_psf_error_map,
    ):
        """Test that `make_psf_psfr` operates as expected."""
        # Mock candidate data
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
            # Mock PSFr outputs
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

                final_psf, variance_map = self.psf.make_psf_psfr(
                    cut_threshold=1e-20,
                    save=False,
                )

        # Expected masking
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

        mock_stack_psf.assert_called_once()
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

                final_psf, variance_map = self.psf.make_psf_starred(
                    cut_threshold=1e-20,
                    save=False,
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
    @patch("dolphin.preprocessing.psf.fits.open")
    def test_plot_psf_candidates(
        self,
        mock_fits,
        mock_wcs,
        mock_colorbar,
        mock_show,
        mock_divider,
    ):
        """Test that `plot_psf_candidates` returns the proper plots."""
        catalog = fits.BinTableHDU.from_columns(
            [
                fits.Column(
                    name="MAG_BEST",
                    format="E",
                    array=np.array([20.0, 21.0, 22.0]),
                ),
                fits.Column(
                    name="FLUX_RADIUS",
                    format="E",
                    array=np.array([2.0, 3.0, 4.0]),
                ),
                fits.Column(
                    name="NUMBER",
                    format="I",
                    array=np.array([1, 2, 3]),
                ),
                fits.Column(
                    name="X_IMAGE",
                    format="E",
                    array=np.array([100.0, 200.0, 300.0]),
                ),
                fits.Column(
                    name="Y_IMAGE",
                    format="E",
                    array=np.array([100.0, 200.0, 300.0]),
                ),
            ]
        )

        mask = np.array([True, True, False])

        star_exposures = [
            np.ones((5, 5)),
            np.ones((5, 5)),
        ]

        star_weights = [
            np.ones((5, 5)),
            np.ones((5, 5)),
        ]

        noise_maps = [
            np.ones((5, 5)) * 0.1,
            np.ones((5, 5)) * 0.1,
        ]

        fake_hdu = MagicMock()
        fake_hdu.header = {}
        fake_hdu.data = np.ones((1000, 1000))

        mock_fits.return_value.__enter__.return_value = [fake_hdu]

        wcs = MagicMock()
        mock_wcs.return_value = wcs

        wcs.all_pix2world.side_effect = lambda x, y, z: (x, y)

        wcs.all_world2pix.side_effect = lambda x, y, z: (x, y)

        divider = MagicMock()
        mock_divider.return_value = divider
        divider.append_axes.return_value = MagicMock()

        self.psf.plot_psf_candidates(
            mask=mask,
            star_exposures=star_exposures,
            star_weights=star_weights,
            noise_maps=noise_maps,
            catalog=catalog,
        )

        # magnitude plot
        # cutouts
        # weights
        # noise maps
        # counts vs. variance
        # image locations
        assert mock_show.call_count == 6

    @patch("dolphin.preprocessing.psf.plt.show")
    @patch("dolphin.preprocessing.psf.plt.tight_layout")
    def test_plot_saved_psf_candidates(
        self,
        mock_tight_layout,
        mock_show,
    ):
        """Test that `plot_saved_psf_candidates` returns the expected plots."""
        # Fake star cutouts
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
        ) as mock_load:

            psf, variance = self.psf.load_saved_psf(plot=True)

        mock_load.assert_called_once_with(self.psf)

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

    def test_load_catalog_table(self):
        """Test that `load_catalog_table` returns the expected table."""
        catalog = MagicMock()

        with patch.object(
            self.psf.file_system,
            "load_catalog_table",
            return_value=catalog,
        ) as mock_load:

            result = self.psf.load_catalog_table()

        mock_load.assert_called_once_with(self.psf)

        assert result is catalog

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
        ) as mock_load:

            result = self.psf.load_psf_candidate_attributes()

        mock_load.assert_called_once_with(self.psf)

        assert result == expected
