# -*- coding: utf-8 -*-
"""Tests for output module."""
from pathlib import Path
import pytest
import numpy as np
import matplotlib.pyplot as plt

from dolphin.processor import Processor
from dolphin.analysis.output import Output

_ROOT_DIR = Path(__file__).resolve().parents[2]
_TEST_IO_DIR = _ROOT_DIR / "io_directory_example"


class TestOutput(object):
    def setup_class(self):
        self.output = Output(_TEST_IO_DIR)
        self.processor = Processor(_TEST_IO_DIR)

    @classmethod
    def teardown_class(cls):
        pass

    def test_swim(self):
        """Test that `swim` method is not accessible.

        :return:
        :rtype:
        """
        with pytest.raises(NotImplementedError):
            self.output.swim()

    def test_properties(self):
        """Test class properties.

        :return:
        :rtype:
        """
        with pytest.raises(ValueError):
            _ = self.output.fit_output

        with pytest.raises(ValueError):
            _ = self.output.kwargs_result

        with pytest.raises(ValueError):
            _ = self.output.model_settings

        assert self.output.samples_mcmc == []
        assert self.output.params_mcmc == []
        assert self.output.num_params_mcmc == 0

        self.output._params_mcmc = ["param1", "param2"]
        assert self.output.num_params_mcmc == 2

        self.output._samples_mcmc = np.ones(10)
        assert np.all(self.output.samples_mcmc == np.ones(10))
        self.output._samples_mcmc = None

        self.output._params_mcmc = ["param1"]
        assert self.output.params_mcmc == ["param1"]
        self.output._params_mcmc = None

    def test_load_output(self):
        """Test that outputs are saved and corresponding class variables are not None.

        :return:
        :rtype:
        """
        save_dict = {
            "settings": {"some": "settings"},
            "kwargs_result": {"0": None, "1": "str", "2": [3, 4]},
            "fit_output": [
                ["emcee", [[2, 2], [3, 3]], ["param1", "param2"], [0.5, 0.2]]
            ],
            "multi_band_list_out": ["band1", "band2"],
        }

        self.processor.file_system.save_output("test", "post_process_load", save_dict)

        self.output.load_output("test", "post_process_load")

        assert np.all(self.output.fit_output[0][1] == save_dict["fit_output"][0][1])
        assert np.all(self.output.fit_output[0][3] == save_dict["fit_output"][0][3])
        assert self.output.kwargs_result == save_dict["kwargs_result"]
        assert self.output._multi_band_list_out == save_dict["multi_band_list_out"]
        assert self.output.model_settings == save_dict["settings"]

    def test_plot_model_overview(self):
        """Test `plot_model_overview` method.

        :return:
        :rtype:
        """
        with pytest.raises(ValueError):
            _ = self.output.plot_model_overview("lens_system1")

        fig = self.output.plot_model_overview("lens_system2", "example")

        plt.close(fig)

        fig2 = self.output.plot_model_overview(
            "lens_system2",
            "example",
            v_min=-3.0,
            v_max=1.0,
            print_results=True,
            show_source_light=True,
        )

        plt.close(fig2)

    def test_plot_model_decomposition(self):
        """Test `plot_model_decomposition` method.

        :return:
        :rtype:
        """
        with pytest.raises(ValueError):
            _ = self.output.plot_model_decomposition("lens_system2")

        fig = self.output.plot_model_decomposition("lens_system2", "example")

        plt.close(fig)

        fig2 = self.output.plot_model_decomposition(
            "lens_system2", "example", v_min=-3.0, v_max=1.0
        )

        plt.close(fig2)

    def test_plot_mcmc_trace(self):
        """Test `plot_mcmc_trace` method.

        :return:
        :rtype:
        """
        fig = self.output.plot_mcmc_trace(
            "lens_system2", "example", 2, verbose=True, burn_in=0
        )

        plt.close(fig)

        fig2 = self.output.plot_mcmc_trace(
            "lens_system2",
            "example",
            2,
            verbose=True,
            burn_in=0,
            parameters_to_plot=["gamma_lens0"],
        )

        plt.close(fig2)

        with pytest.raises(ValueError):
            self.output.plot_mcmc_trace(
                "lens_system2",
                "example",
                2,
                verbose=True,
                burn_in=0,
                parameters_to_plot=["gamma_lens42"],
            )

    def test_get_reshaped_emcee_chain(self):
        """Test `get_reshaped_emcee_chain` method.

        :return:
        :rtype:
        """
        self.output.get_reshaped_emcee_chain("lens_system2", "example", 2, burn_in=1)

    def test_get_param_class(self):
        """Test `get_param_class` method.

        :return:
        :rtype:
        """
        param_class = self.output.get_param_class("lens_system2", "example")

        param_class.num_param()

    def test_get_kwargs_from_args(self):
        """Test `get_kwargs_from_args` method.

        :return:
        :rtype:
        """
        param_class = self.output.get_param_class("lens_system2", "example")

        n, _ = param_class.num_param()

        self.output.get_kwargs_from_args(
            "lens_system2",
            "example",
            self.output.samples_mcmc[0],
            linear_solve=True,
        )

    def test_get_magnification_extended_source(self):
        """Test `get_magnification_extended_source` method.

        :return:
        :rtype:
        """
        lens_name = "lens_system2"
        model_id = "example"

        # Test with kwargs_result provided
        magnification = self.output.get_magnification_extended_source(
            lens_name=lens_name,
            model_id=model_id,
            kwargs_result=None,
            band_index=0,
            plot=True,
        )

        assert isinstance(magnification, float)

        # Test ValueError when neither model_id nor kwargs_result is provided
        with pytest.raises(
            ValueError,
            match="Either the `model_id` or the `kwargs_result` needs to be provided!",
        ):
            self.output.get_magnification_extended_source(
                lens_name=lens_name,
                band_index=0,
                plot=False,
            )

    def test_get_critical_curve(self):
        """Test `get_critical_curve` method."""
        ra_crit, dec_crit = self.output.get_critical_curve(
            lens_name="lens_system2",
            model_id="example",
            kwargs_result=None,
            band_index=0,
        )
        assert isinstance(ra_crit, np.ndarray)
        assert isinstance(dec_crit, np.ndarray)
        assert len(ra_crit) == len(dec_crit)

        # Test ValueError when neither model_id nor kwargs_result is provided
        with pytest.raises(
            ValueError,
            match="Either the `model_id` or the `kwargs_result` needs to be provided!",
        ):
            self.output.get_critical_curve(
                lens_name="lens_system2",
                band_index=0,
            )

    def test_get_magnification_point_source(self):
        """Test `get_magnification_point_source` method.

        :return:
        :rtype:
        """
        lens_name = "lens_system2"
        model_id = "example"

        # Test with kwargs_result provided
        magnification = self.output.get_magnification_point_source(
            lens_name=lens_name,
            model_id=model_id,
            kwargs_result=None,
            band_index=0,
            plot=True,
        )

        assert isinstance(magnification, list)

        # Test ValueError when neither model_id nor kwargs_result is provided
        with pytest.raises(
            ValueError,
            match="Either the `model_id` or the `kwargs_result` needs to be provided!",
        ):
            self.output.get_magnification_point_source(
                lens_name=lens_name,
                band_index=0,
                plot=False,
            )
