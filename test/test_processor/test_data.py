# -*- coding: utf-8 -*-
"""Tests for data module."""
from pathlib import Path
import numpy.testing as npt
from dolphin.processor.data import Data
from dolphin.processor.data import ImageData
from dolphin.processor.data import PSFData

_ROOT_DIR = Path(__file__).resolve().parents[2]


class TestData(object):
    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def test_load_from_file(self):
        """Test `load_from_file` method.

        :return:
        :rtype:
        """
        data = Data()

        data_file = (
            _ROOT_DIR
            / "io_directory_example"
            / "data"
            / "lens_system1"
            / "image_lens_system1_F390W.h5"
        )
        data.load_from_file(data_file)


class TestImageData(object):
    def setup_class(self):
        data_file = (
            _ROOT_DIR
            / "io_directory_example"
            / "data"
            / "lens_system1"
            / "image_lens_system1_F390W.h5"
        )
        self.image_data = ImageData(data_file)

    @classmethod
    def teardown_class(cls):
        pass

    def test_kwargs_data(self):
        """Test `kwargs_data` property.

        :return:
        :rtype:
        """
        for key in [
            "image_data",
            "background_rms",
            "exposure_time",
            "ra_at_xy_0",
            "dec_at_xy_0",
            "transform_pix2angle",
        ]:
            assert key in self.image_data.kwargs_data

    def test_get_image(self):
        """Test `get_image` method.

        :return:
        :rtype:
        """
        image = self.image_data.get_image()

        assert len(image.shape) == 2
        assert image.shape == (120, 120)

    def test_get_image_coordinate_system(self):
        """Test `get_image_coordinate_system` method.

        :return:
        :rtype:
        """
        coord_sys = self.image_data.get_image_coordinate_system()

        x0, y0 = coord_sys.map_pix2coord(0, 0)
        assert x0 == self.image_data.kwargs_data["ra_at_xy_0"]
        assert y0 == self.image_data.kwargs_data["dec_at_xy_0"]

    def test_get_image_pixel_number(self):
        """Test `get_image_pixel_number` method.

        :return:
        :rtype:
        """
        assert self.image_data.get_image_size() == 120

    def test_get_image_pixel_scale(self):
        """Test `get_image_pixel_scale` method.

        :return:
        :rtype:
        """
        npt.assert_almost_equal(
            self.image_data.get_image_pixel_scale(), 0.04, decimal=6
        )


class TestPSFData(object):
    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def test_kwargs_psf(self):
        """Test `kwargs_psf` property.

        :return:
        :rtype:
        """
        psf_file = (
            _ROOT_DIR
            / "io_directory_example"
            / "data"
            / "lens_system1"
            / "psf_lens_system1_F390W.h5"
        )

        psf = PSFData(psf_file)

        for key in [
            "psf_type",
            "kernel_point_source",
            "kernel_point_source_init",
            "psf_variance_map",
        ]:
            assert key in psf.kwargs_psf
