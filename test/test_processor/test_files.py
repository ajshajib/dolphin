# -*- coding: utf-8 -*-
"""Tests for files module."""

import pytest
from pathlib import Path
import os
import numpy as np

from dolphin.processor.files import FileSystem

_ROOT_DIR = Path(__file__).resolve().parents[2]
_TEST_IO_DIR = _ROOT_DIR / "io_directory_example"


class TestFileSystem(object):
    def setup_class(self):
        self.file_system = FileSystem(_TEST_IO_DIR)

    @classmethod
    def teardown_class(cls):
        pass

    def test_path2str(self):
        """Test `path2str` method.

        :return:
        :rtype:
        """
        assert Path(self.file_system.path2str(_TEST_IO_DIR)) == _TEST_IO_DIR

    def test_get_lens_list_file_path(self):
        """Test `get_lens_list_file_path` :return:

        :rtype:
        """
        lens_list_file_path = _TEST_IO_DIR / "lens_list.txt"

        assert Path(self.file_system.get_lens_list_file_path()) == lens_list_file_path

    def test_get_lens_list(self):
        """Test `get_lens_list` method.

        :return:
        :rtype:
        """
        lens_list = ["lens_system1", "lens_system2", "lens_system3"]

        assert self.file_system.get_lens_list() == lens_list

    def test_get_config_file_path(self):
        """Test `get_config_file_path` method.

        :return:
        :rtype:
        """
        config_file_path = _TEST_IO_DIR / "settings" / "lens_system1_config.yml"

        assert (
            Path(self.file_system.get_config_file_path("lens_system1"))
            == config_file_path
        )

    def test_get_logs_directory(self):
        """Test `get_logs_directory` method.

        :return:
        :rtype:
        """
        logs_directory = _TEST_IO_DIR / "logs"

        assert Path(self.file_system.get_logs_directory()) == logs_directory

    def test_get_settings_directory(self):
        """Test `get_settings_directory` method.

        :return:
        :rtype:
        """
        settings_dir = _TEST_IO_DIR / "settings"

        assert Path(self.file_system.get_settings_directory()) == settings_dir

    def test_get_outputs_directory(self):
        """Test `get_outputs_directory` method.

        :return:
        :rtype:
        """
        outputs_dir = _TEST_IO_DIR / "outputs"

        assert Path(self.file_system.get_outputs_directory()) == outputs_dir

    def test_get_data_directory(self):
        """Test `get_data_directory` method.

        :return:
        :rtype:
        """
        data_dir = _TEST_IO_DIR / "data"

        assert Path(self.file_system.get_data_directory()) == data_dir

    def test_get_image_file_path(self):
        """Test `get_image_file_path` method.

        :return:
        :rtype:
        """
        path = _TEST_IO_DIR / "data" / "lens_system1" / "image_lens_system1_F390W.h5"

        assert (
            Path(self.file_system.get_image_file_path("lens_system1", "F390W")) == path
        )

    def test_get_psf_file_path(self):
        """Test `get_psf_file_path` method.

        :return:
        :rtype:
        """
        path = _TEST_IO_DIR / "data" / "lens_system1" / "psf_lens_system1_F390W.h5"

        assert Path(self.file_system.get_psf_file_path("lens_system1", "F390W")) == path

    def test_get_log_file_path(self):
        """Test `get_log_file_path` method.

        :return:
        :rtype:
        """
        with open(str(_TEST_IO_DIR.resolve()) + "/logs/log_name_test.txt", "w"):
            pass

        path = _TEST_IO_DIR / "logs" / "log_name_test.txt"

        assert Path(self.file_system.get_log_file_path("name", "test")) == path

        os.remove(str(path.resolve()))

    def test_get_output_file_path(self):
        """Test `get_output_file_path` method.

        :return:
        :rtype:
        """
        with open(str(_TEST_IO_DIR.resolve()) + "/outputs/output_name_test.json", "w"):
            pass

        path = _TEST_IO_DIR / "outputs" / "output_name_test.json"

        assert Path(self.file_system.get_output_file_path("name", "test")) == path

        os.remove(str(path.resolve()))

    def test_save_load_output(self):
        """Test for the `save_output()` and `load_output()` will be covered by
        `test_save_load_output_json()` and `test_save_load_output_h5()` methods.

        :return:
        :rtype:
        """
        with pytest.raises(ValueError):
            self.file_system.save_output("test", "save_test", {}, file_type="invalid")

        with pytest.raises(ValueError):
            self.file_system.load_output("test", "save_test", file_type="invalid")

    def test_save_load_output_json(self):
        """Test `save_output_json` and `load_output_json` methods.

        :return:
        :rtype:
        """
        save_dict = {
            "kwargs_test": {"0": None, "1": "str", "2": [3, 4]},
            "array_test": np.array([1.0]),
        }

        self.file_system.save_output("test", "save_test", save_dict, file_type="json")

        assert (
            self.file_system.load_output("test", "save_test", file_type="json")
            == save_dict
        )

    def test_save_load_output_h5(self):
        """Test `save_output` and `load_output` methods.

        :return:
        :rtype:
        """
        save_dict = {
            "settings": {"some": ["settings"]},
            "kwargs_result": {"0": 1, "1": "str", "2": [3, 4]},
            "fit_output": [
                [
                    "PSO",
                    [np.ones((1, 50)), np.ones((4, 50)), np.ones((1, 50))],
                    np.array(["{}".format(i) for i in range(4)]),
                ],
                [
                    "EMCEE",
                    np.ones((50, 4)),
                    ["{}".format(i) for i in range(4)],
                    np.ones(50),
                ],
            ],
        }

        self.file_system.save_output("test", "save_test", save_dict, file_type="h5")

        out = self.file_system.load_output("test", "save_test", file_type="h5")

        assert save_dict["settings"] == out["settings"]
        assert save_dict["kwargs_result"] == out["kwargs_result"]

        for i in [0, 2]:
            assert np.all(save_dict["fit_output"][0][i] == out["fit_output"][0][i])
        for i in range(3):
            assert np.all(
                save_dict["fit_output"][0][1][i] == out["fit_output"][0][1][i]
            )

        for i in range(3):
            assert np.all(save_dict["fit_output"][1][i] == out["fit_output"][1][i])

        with pytest.raises(ValueError):
            save_dict["fit_output"].append(
                [
                    "INVALID",
                    np.ones((4, 50)),
                    np.array(["{}".format(i) for i in range(4)]),
                ]
            )
            self.file_system.save_output("test", "save_test", save_dict, file_type="h5")

    def test_numpy_to_json_encoding(self):
        """Test `class NumpyEncoder` and `hook_json_to_numpy` function.

        :return:
        :rtype:
        """
        a = np.array([[0, 2], [3, 4]])
        b = {"1": a}
        c = {"0": {"1": a}, "2": [1, 2]}
        d = [{"0": {"1": a}, "2": [1, 2]}, "string", [a, a]]

        assert np.all(
            self.file_system.decode_numpy_arrays(
                self.file_system.encode_numpy_arrays(a)
            )
            == a
        )

        assert np.all(
            self.file_system.decode_numpy_arrays(
                self.file_system.encode_numpy_arrays(b["1"])
            )
            == b["1"]
        )

        assert np.all(
            self.file_system.decode_numpy_arrays(
                self.file_system.encode_numpy_arrays(c["0"]["1"])
            )
            == c["0"]["1"]
        )

        assert np.all(
            self.file_system.decode_numpy_arrays(
                self.file_system.encode_numpy_arrays(d[0]["0"]["1"])
            )
            == d[0]["0"]["1"]
        )
        assert np.all(
            self.file_system.decode_numpy_arrays(
                self.file_system.encode_numpy_arrays(d[2][1])
            )
            == d[2][1]
        )
        assert np.all(
            self.file_system.decode_numpy_arrays(
                self.file_system.encode_numpy_arrays(d[2][0])
            )
            == d[2][0]
        )
        assert np.all(
            self.file_system.decode_numpy_arrays(
                self.file_system.encode_numpy_arrays(d[2][1])
            )
            == d[2][1]
        )
