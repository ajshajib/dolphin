# -*- coding: utf-8 -*-
"""Tests for preprocessing_util module."""

import numpy as np
from pathlib import Path

import dolphin.preprocessing.preprocessing_util as preprocessing_util

from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy.testing as npt
import pytest

_ROOT_DIR = Path(__file__).resolve().parents[2]
_TEST_IO_DIR = _ROOT_DIR / "io_directory_example"


class TestPreprocessingUtil(object):

    @patch("dolphin.preprocessing.preprocessing_util.os.replace")
    @patch("dolphin.preprocessing.preprocessing_util.os.makedirs")
    @patch("dolphin.preprocessing.preprocessing_util.fits.open")
    @patch("dolphin.preprocessing.preprocessing_util.subprocess.run")
    def test_make_image_catalog(
        self,
        mock_run,
        mock_fits,
        mock_makedirs,
        mock_replace,
    ):
        """Test `make_image_catalog` returns correct HDU."""

        catalog_hdu = MagicMock()
        mock_hdul = MagicMock()
        mock_hdul.__getitem__.side_effect = lambda i: (
            catalog_hdu if i == 2 else MagicMock()
        )
        mock_fits.return_value = mock_hdul

        result = preprocessing_util.make_image_catalog(
            _TEST_IO_DIR,
            "MOCK",
            "F814W",
        )

        assert result is catalog_hdu
        mock_run.assert_called_once()
        mock_makedirs.assert_called_once()
        mock_replace.assert_called_once()

    @patch("dolphin.preprocessing.preprocessing_util.fits.open")
    @patch("dolphin.preprocessing.preprocessing_util.os.path.exists")
    def test_get_background(self, mock_exists, mock_fits):
        """Test that `get_background` correctly extracts the background mean and RMS
        from a catalog file."""
        mock_exists.return_value = True

        header_text = ["SEXBKGND 100.5", "SEXBKDEV 2.3"]

        mock_hdul = MagicMock()
        mock_hdul.__enter__.return_value = mock_hdul
        mock_hdul[1].data = np.array([[header_text]], dtype=object)

        mock_fits.return_value = mock_hdul

        mean, rms = preprocessing_util.get_background(
            _TEST_IO_DIR, lens_name="MOCK", data_band="F814W"
        )

        assert mean == 100.5
        assert rms == 2.3
        mock_exists.assert_called_once()
        mock_fits.assert_called_once()

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
