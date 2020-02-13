# -*- coding: utf-8 -*-
"""
Tests for data module.
"""

import pytest
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

        data_file = _ROOT_DIR / 'test_working_directory' \
                    / 'data' / 'test_system' \
                    / 'image_test_system_f390w.hdf5'
        data.load_from_file(data_file)


class TestImageData(object):

    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def test_kwargs_data(self):
        """
        Test `kwargs_data` property.
        :return:
        :rtype:
        """
        data_file = _ROOT_DIR / 'test_working_directory' \
                    / 'data' / 'test_system' \
                    / 'image_test_system_f390w.hdf5'
        image_data = ImageData(data_file)

        #image_data.kwargs_data

        for key in ['image_data', 'background_rms', 'exposure_time',
                    'ra_at_xy_0', 'dec_at_xy_0', 'transform_pix2angle']:
            assert key in image_data.kwargs_data


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
        psf_file =  _ROOT_DIR / 'test_working_directory' \
                    / 'data' / 'test_system' \
                    / 'psf_test_system_f390w.hdf5'

        psf = PSFData(psf_file)

        for key in ['psf_type', 'kernel_point_source',
                    'kernel_point_source_init', 'psf_error_map']:
            assert key in psf.kwargs_psf


if __name__ == '__main__':
    pytest.main()