# -*- coding: utf-8 -*-
"""Tests for data module."""

from pathlib import Path

from dolphin.processor.core import Processor
import numpy as np
import numpy.testing as npt

_ROOT_DIR = Path(__file__).resolve().parents[2]
_TEST_IO_DIR = _ROOT_DIR / "io_directory_example"


class TestProcessor(object):
    def setup_class(self):
        self.processor = Processor(_TEST_IO_DIR)

    @classmethod
    def teardown_class(cls):
        pass

    def test_swim(self):
        """Test `swim` method."""
        self.processor.swim("lens_system1", "test")

        self.processor.swim(
            "lens_system1", "test", use_jax=True, recipe_name="galaxy-galaxy"
        )
        self.processor.swim(
            "lensed_quasar", "test", use_jax=True, recipe_name="galaxy-quasar"
        )

    def test_get_kwargs_data_joint(self):
        """Test `get_kwargs_data_joint` method."""
        kwargs_data_joint = self.processor.get_kwargs_data_joint("lens_system1")

        assert kwargs_data_joint["multi_band_type"] == "multi-linear"

        assert len(kwargs_data_joint["multi_band_list"]) == 1
        assert len(kwargs_data_joint["multi_band_list"][0]) == 3

        kwargs_data_joint = self.processor.get_kwargs_data_joint("lens_system5")

        npt.assert_array_equal(
            kwargs_data_joint["time_delays_measured"], [1.0, 1.0, 1.0]
        )
        npt.assert_array_equal(
            kwargs_data_joint["time_delays_uncertainties"],
            [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]],
        )

    def test_get_image_data(self):
        """Test `get_image_data` method."""
        image_data = self.processor.get_image_data("lens_system1", "F390W")
        assert image_data is not None

    def test_get_psf_data(self):
        """Test `get_image_data` method."""
        psf_data = self.processor.get_psf_data("lens_system1", "F390W")
        assert psf_data is not None
