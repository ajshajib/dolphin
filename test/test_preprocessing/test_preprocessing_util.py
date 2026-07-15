# -*- coding: utf-8 -*-
"""Tests for preprocessing_util module."""

import numpy as np
from pathlib import Path

import dolphin.preprocessing.preprocessing_util as preprocessing_util

from unittest.mock import patch, MagicMock
import numpy.testing as npt
import pytest

_ROOT_DIR = Path(__file__).resolve().parents[2]
_TEST_IO_DIR = _ROOT_DIR / "io_directory_example"


class TestPreprocessingUtil(object):

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

        mean, rms = preprocessing_util.get_background(
            _TEST_IO_DIR,
            lens_name="MOCK",
            data_band="F814W",
        )

        assert mean == 100.5
        assert rms == 2.3

        mock_getdata.assert_called_once()
        mock_background2d.assert_called_once()

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