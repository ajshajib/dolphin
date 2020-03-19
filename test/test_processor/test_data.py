# -*- coding: utf-8 -*-
"""
Tests for data module.
"""
from pathlib import Path

from dolphin.processor.data import *

_ROOT_DIR = Path(__file__).resolve().parents[2]


class TestData(object):

    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def test_load_from_file(self):
        """
        Test `load_from_file` method.
        :return:
        :rtype:
        """
        data = Data()

        data_file = _ROOT_DIR / 'io_directory_example' \
                    / 'data' / 'lens_system1' \
                    / 'image_lens_system1_F390W.h5'
        data.load_from_file(data_file)


class TestImageData(object):

    def setup_class(self):
        data_file = _ROOT_DIR / 'io_directory_example' \
                    / 'data' / 'lens_system1' \
                    / 'image_lens_system1_F390W.h5'
        self.image_data = ImageData(data_file)

    @classmethod
    def teardown_class(cls):
        pass

    def test_kwargs_data(self):
        """
        Test `kwargs_data` property.
        :return:
        :rtype:
        """
        for key in ['image_data', 'background_rms', 'exposure_time',
                    'ra_at_xy_0', 'dec_at_xy_0', 'transform_pix2angle']:
            assert key in self.image_data.kwargs_data

    def test_get_image(self):
        """
        Test `get_image` method.
        :return:
        :rtype:
        """
        image = self.image_data.get_image()

        assert len(image.shape) == 2
        assert image.shape == (120, 120)



class TestPSFData(object):

    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def test_kwargs_psf(self):
        """
        Test `kwargs_psf` property.
        :return:
        :rtype:
        """
        psf_file =  _ROOT_DIR / 'io_directory_example' \
                    / 'data' / 'lens_system1' \
                    / 'psf_lens_system1_F390W.h5'

        psf = PSFData(psf_file)

        for key in ['psf_type', 'kernel_point_source',
                    'kernel_point_source_init', 'psf_error_map']:
            assert key in psf.kwargs_psf