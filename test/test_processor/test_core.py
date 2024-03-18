# -*- coding: utf-8 -*-
"""Tests for data module."""
from pathlib import Path

from dolphin.processor.core import Processor

_ROOT_DIR = Path(__file__).resolve().parents[2]
_TEST_IO_DIR = _ROOT_DIR / "io_directory_example"


class TestProcessor(object):
    def setup_class(self):
        self.processor = Processor(_TEST_IO_DIR)

    @classmethod
    def teardown_class(cls):
        pass

    def test_swim(self):
        """Test `swim` method.

        :return:
        :rtype:
        """
        self.processor.swim("lens_system1", "test")

    def test_get_kwargs_data_joint(self):
        """Test `get_kwargs_data_joint` method.

        :return:
        :rtype:
        """
        kwargs_data_joint = self.processor.get_kwargs_data_joint("lens_system1")

        assert kwargs_data_joint["multi_band_type"] == "multi-linear"

        assert len(kwargs_data_joint["multi_band_list"]) == 1
        assert len(kwargs_data_joint["multi_band_list"][0]) == 3

    def test_get_image_data(self):
        """Test `get_image_data` method.

        :return:
        :rtype:
        """
        image_data = self.processor.get_image_data("lens_system1", "F390W")
        assert image_data is not None

    def test_get_psf_data(self):
        """Test `get_image_data` method.

        :return:
        :rtype:
        """
        psf_data = self.processor.get_psf_data("lens_system1", "F390W")
        assert psf_data is not None
