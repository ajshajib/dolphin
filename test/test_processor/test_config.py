# -*- coding: utf-8 -*-
"""
Tests for config module.
"""
import pytest
from pathlib import Path

from dolphin.processor.config import *

_ROOT_DIR = Path(__file__).resolve().parents[2]

class TestModelConfig(object):

    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def test_load(self):
        """
        Test the `load` method in `class ModelConfig`.
        :return:
        :rtype:
        """
        test_setting_file_dir = _ROOT_DIR / 'test_working_directory' \
                   / 'settings' / 'test_system.yml'
        config = ModelConfig(test_setting_file_dir)

        pass