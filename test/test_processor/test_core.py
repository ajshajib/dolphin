# -*- coding: utf-8 -*-
"""
Tests for data module.
"""

import pytest
from pathlib import Path

from dolphin.processor.core import *

_ROOT_DIR = Path(__file__).resolve().parents[2]
_TEST_WORK_DIR = _ROOT_DIR / 'test_working_directory'


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

    def test_save_load_output(self):
        """
        Test `_save_output` and `load_output` methods.
        :return:
        :rtype:
        """
        save_dict = {
            'kwargs_test': None
        }

        self.processor._save_output('test', 'save_test', save_dict)

        assert self.processor.load_output('test', 'save_test') == save_dict