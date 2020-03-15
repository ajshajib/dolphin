# -*- coding: utf-8 -*-
"""
Tests for files module.
"""

import pytest
from pathlib import Path
import os

from dolphin.processor.files import *

_ROOT_DIR = Path(__file__).resolve().parents[2]
_TEST_IO_DIR = _ROOT_DIR / 'io_directory_example'


class TestFileSystem(object):

    def setup_class(self):
        self.file_system = FileSystem(_TEST_IO_DIR)

    @classmethod
    def teardown_class(cls):
        pass

    def test_path2str(self):
        """
        Test `path2str` method.
        :return:
        :rtype:
        """
        assert Path(self.file_system.path2str(_TEST_IO_DIR)) == _TEST_IO_DIR

    def test_get_lens_list_file_path(self):
        """
        Test `get_lens_list_file_path`
        :return:
        :rtype:
        """
        lens_list_file_path = _TEST_IO_DIR / 'lens_list.txt'

        assert Path(self.file_system.get_lens_list_file_path()) == \
            lens_list_file_path

    def test_get_lens_list(self):
        """
        Test `get_lens_list` method.
        :return:
        :rtype:
        """
        lens_list = ['test_system']

        assert self.file_system.get_lens_list() == lens_list

    def test_get_config_file_path(self):
        """
        Test `get_config_file_path` method.
        :return:
        :rtype:
        """
        config_file_path = _TEST_IO_DIR / 'settings' / \
                                            'test_system_config.yml'

        assert Path(self.file_system.get_config_file_path('test_system')) == \
            config_file_path

    def test_get_logs_directory(self):
        """
        Test `get_logs_directory` method.
        :return:
        :rtype:
        """
        logs_directory = _TEST_IO_DIR / 'logs'

        assert Path(self.file_system.get_logs_directory()) == logs_directory

    def test_get_settings_directory(self):
        """
        Test `get_settings_directory` method.
        :return:
        :rtype:
        """
        settings_dir = _TEST_IO_DIR / 'settings'

        assert Path(self.file_system.get_settings_directory()) == settings_dir

    def test_get_outputs_directory(self):
        """
        Test `get_outputs_directory` method.
        :return:
        :rtype:
        """
        outputs_dir = _TEST_IO_DIR / 'outputs'

        assert Path(self.file_system.get_outputs_directory()) == outputs_dir

    def test_get_data_directory(self):
        """
        Test `get_data_directory` method.
        :return:
        :rtype:
        """
        data_dir = _TEST_IO_DIR / 'data'

        assert Path(self.file_system.get_data_directory()) == data_dir

    def test_get_image_file_path(self):
        """
        Test `get_image_file_path` method.
        :return:
        :rtype:
        """
        path = _TEST_IO_DIR / 'data' / 'test_system' / \
            'image_test_system_F390W.hdf5'

        assert Path(self.file_system.get_image_file_path('test_system',
                                                         'F390W')) == path

    def test_get_psf_file_path(self):
        """
        Test `get_psf_file_path` method.
        :return:
        :rtype:
        """
        path = _TEST_IO_DIR / 'data' / 'test_system' / \
            'psf_test_system_F390W.hdf5'

        assert Path(self.file_system.get_psf_file_path('test_system',
                                                       'F390W')) == path

    def test_get_log_file_path(self):
        """
        Test `get_log_file_path` method.
        :return:
        :rtype:
        """
        with open(str(_TEST_IO_DIR.resolve())
                  + '/logs/log_name_test.txt', 'w') as f:
            pass

        path = _TEST_IO_DIR / 'logs' / 'log_name_test.txt'

        assert Path(self.file_system.get_log_file_path('name', 'test')) \
            == path

        os.remove(str(path.resolve()))

    def test_get_output_file_path(self):
        """
        Test `get_output_file_path` method.
        :return:
        :rtype:
        """
        with open(str(_TEST_IO_DIR.resolve())
                  + '/outputs/output_name_test.json', 'w') as f:
            pass

        path = _TEST_IO_DIR / 'outputs' / 'output_name_test.json'

        assert Path(self.file_system.get_output_file_path('name', 'test')) \
            == path

        os.remove(str(path.resolve()))