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

    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def teardown_class(cls):
        pass

