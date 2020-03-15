# -*- coding: utf-8 -*-
"""
Tests for data module.
"""

import pytest
from pathlib import Path

from dolphin.processor.core import *

_ROOT_DIR = Path(__file__).resolve().parents[2]
_TEST_WORK_DIR = _ROOT_DIR / 'io_directory_example'


class TestProcessor(object):

    def setup_class(self):
        self.processor = Processor(_TEST_WORK_DIR)

    @classmethod
    def teardown_class(cls):
        pass

    def test_swim(self):
        """
        Test `swim` method.
        :return:
        :rtype:
        """
        self.processor.swim('test_system', 'test')

    def test_get_kwargs_data_joint(self):
        """
        Test `get_kwargs_data_joint` method.
        :return:
        :rtype:
        """
        kwargs_data_joint = self.processor.get_kwargs_data_joint('test_system')

        assert kwargs_data_joint['multi_band_type'] == 'multi-linear'

        assert len(kwargs_data_joint['multi_band_list']) == 1
        assert len(kwargs_data_joint['multi_band_list'][0]) == 3

    def test_get_image_data(self):
        """
        Test `get_image_data` method.
        :return:
        :rtype:
        """
        image_data = self.processor.get_image_data('test_system', 'F390W')
        assert image_data is not None

    def test_get_psf_data(self):
        """
        Test `get_image_data` method.
        :return:
        :rtype:
        """
        psf_data = self.processor.get_psf_data('test_system', 'F390W')
        assert psf_data is not None

    def test_save_load_output(self):
        """
        Test `_save_output` and `load_output` methods.
        :return:
        :rtype:
        """
        save_dict = {
            'kwargs_test': {'0': None, '1': 'str', '2': [3, 4]},
            'array_test': np.array([1.])
        }

        self.processor._save_output('test', 'save_test', save_dict)

        assert self.processor.load_output('test', 'save_test') == save_dict

    def test_numpy_to_json_encoding(self):
        """
        Test `class NumpyEncoder` and `hook_json_to_numpy` function.
        :return:
        :rtype:
        """
        a = np.array([[0, 2], [3, 4]])
        b = {'1': a}
        c = {'0': {'1': a}, '2': [1, 2]}
        d = [{'0': {'1': a}, '2': [1, 2]}, 'string', [a, a]]

        assert np.all(self.processor.decode_numpy_arrays(
            self.processor.encode_numpy_arrays(a)
        ) == a)

        assert np.all(self.processor.decode_numpy_arrays(
            self.processor.encode_numpy_arrays(b['1'])
        ) == b['1'])

        assert np.all(self.processor.decode_numpy_arrays(
            self.processor.encode_numpy_arrays(c['0']['1'])
        ) == c['0']['1'])

        assert np.all(self.processor.decode_numpy_arrays(
            self.processor.encode_numpy_arrays(d[0]['0']['1'])
        ) == d[0]['0']['1'])
        assert np.all(self.processor.decode_numpy_arrays(
            self.processor.encode_numpy_arrays(d[2][1])
        ) == d[2][1])
        assert np.all(self.processor.decode_numpy_arrays(
            self.processor.encode_numpy_arrays(d[2][0])
        ) == d[2][0])
        assert np.all(self.processor.decode_numpy_arrays(
            self.processor.encode_numpy_arrays(d[2][1])
        ) == d[2][1])
