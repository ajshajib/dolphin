"""Tests for preprocessing_util module."""

from unittest.mock import MagicMock, patch

import numpy as np
import numpy.testing as npt
import pytest
from astropy.io import fits

from dolphin.preprocessing import preprocessing_util


class TestPreprocessingUtil:

    @patch("dolphin.preprocessing.preprocessing_util.Background2D")
    @patch("dolphin.preprocessing.preprocessing_util.fits.getdata")
    def test_get_background(
        self,
        mock_getdata,
        mock_background2d,
    ):
        """Test that `get_background` correctly returns the background median and RMS
        from `Background2D`."""

        image = np.ones((100, 100))
        mock_getdata.return_value = image

        mock_background = MagicMock()
        mock_background.background_median = 100.5
        mock_background.background_rms_median = 2.3
        mock_background2d.return_value = mock_background

        mean, rms = preprocessing_util.get_background(image_file_name="fake_image.fits")

        assert mean == 100.5
        assert rms == 2.3

        mock_getdata.assert_called_once()
        mock_background2d.assert_called_once()

    def test_compute_noise_map_jwst(self, tmp_path):
        """Test that `compute_noise_map()` correctly reads the JWST ERR array."""

        image_file = tmp_path / "jwst.fits"

        err = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ]
        )

        primary = fits.PrimaryHDU()
        sci = fits.ImageHDU(data=np.ones((3, 3)), name="SCI")
        err_hdu = fits.ImageHDU(data=err, name="ERR")

        hdul = fits.HDUList([primary, sci, err_hdu])
        hdul.writeto(image_file)

        noise_map = preprocessing_util.compute_noise_map(
            instrument="JWST",
            image_file_name=str(image_file),
        )
        np.testing.assert_array_equal(noise_map, err)

    @patch("dolphin.preprocessing.preprocessing_util.get_background")
    def test_compute_noise_map_hst(
        self,
        mock_get_background,
        tmp_path,
    ):
        """Test that `compute_noise_map` correctly computes the HST noise map."""

        image_file = tmp_path / "hst_image.fits"
        weight_file = tmp_path / "hst_weight.fits"

        data = np.array(
            [
                [4.0, 9.0],
                [16.0, 25.0],
            ]
        )
        weight = np.array(
            [
                [1.0, 3.0],
                [4.0, 5.0],
            ]
        )

        sigma_bkd = 2.0
        mock_get_background.return_value = (0.0, sigma_bkd)

        fits.PrimaryHDU(data=data).writeto(image_file)
        fits.PrimaryHDU(data=weight).writeto(weight_file)

        noise_map = preprocessing_util.compute_noise_map(
            instrument="HST",
            image_file_name=str(image_file),
            weight_file_name=str(weight_file),
        )

        expected = np.sqrt(np.abs(data / weight) + sigma_bkd**2)
        np.testing.assert_allclose(noise_map, expected)

    def test_build_mask_none(self):
        """Test that no kwargs returns an all True mask."""
        mask = preprocessing_util.build_mask((5, 5))

        npt.assert_array_equal(mask, np.ones((5, 5)))

    def test_build_mask_circle(self):
        """Test construction of a circular mask."""
        mask = preprocessing_util.build_mask(
            (5, 5),
            kwargs_mask=[
                {
                    "type": "circle",
                    "center": (2, 2),
                    "radius": 1,
                }
            ],
        )

        expected = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=float,
        )

        npt.assert_array_equal(mask, expected)

    def test_build_mask_square(self):
        """Test construction of a square mask."""
        mask = preprocessing_util.build_mask(
            (5, 5),
            kwargs_mask=[
                {
                    "type": "square",
                    "center": (2, 2),
                    "size": 3,
                }
            ],
        )

        expected = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=float,
        )

        npt.assert_array_equal(mask, expected)

    def test_build_mask_ellipse(self):
        """Test construction of an elliptical mask."""
        mask = preprocessing_util.build_mask(
            (5, 5),
            kwargs_mask=[{"type": "ellipse", "center": (2, 2), "a": 2, "b": 1}],
        )

        expected = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=float,
        )

        npt.assert_array_equal(mask, expected)

    def test_build_mask_invert(self):
        """Test inverted masks remove pixels from the keep mask."""
        mask = preprocessing_util.build_mask(
            (5, 5),
            kwargs_mask=[
                {
                    "type": "circle",
                    "center": (2, 2),
                    "radius": 2,
                },
                {
                    "type": "circle",
                    "center": (2, 2),
                    "radius": 1,
                    "invert": True,
                },
            ],
        )

        expected = np.array(
            [
                [0, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [1, 0, 0, 0, 1],
                [0, 1, 0, 1, 0],
                [0, 0, 1, 0, 0],
            ],
            dtype=float,
        )

        npt.assert_array_equal(mask, expected)

    def test_build_mask_only_inverted_masks(self):
        """Test that only inverted masks remove pixels from a full image."""
        mask = preprocessing_util.build_mask(
            (5, 5),
            kwargs_mask=[
                {
                    "type": "circle",
                    "center": (2, 2),
                    "radius": 1,
                    "invert": True,
                },
            ],
        )

        expected = np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 0, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 0, 1, 1],
                [1, 1, 1, 1, 1],
            ],
            dtype=float,
        )

        npt.assert_array_equal(mask, expected)

    def test_build_mask_invalid_type(self):
        """Test that an invalid mask type raises an error."""
        with pytest.raises(ValueError):
            preprocessing_util.build_mask(
                (5, 5),
                kwargs_mask=[
                    {
                        "type": "triangle",
                    }
                ],
            )
