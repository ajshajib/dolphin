# -*- coding: utf-8 -*-
"""Tests for config module."""

import pytest
from pathlib import Path
from copy import deepcopy
import numpy as np
import numpy.testing as npt
from dolphin.processor.config import ModelConfig
from dolphin.Util.jax_util import custom_logL_addition_jax
from dolphin.processor.files import FileSystem


_ROOT_DIR = Path(__file__).resolve().parents[2]
IO_DIRECTORY = str((_ROOT_DIR / "io_directory_example").resolve())
FILE_SYSTEM = FileSystem(IO_DIRECTORY)
CONFIG_1 = ModelConfig("lens_system1", FILE_SYSTEM)
CONFIG_3 = ModelConfig("lens_system3", io_directory=IO_DIRECTORY)

def test_custom_logL_addition_jax():
    """Test `custom_logL_addition_jax` method.

    :return:
    :rtype:
    """
    # Mass paramters : (phi_m = 0 deg, q_m = 0.8)
    # Satisfy both priors (phi_L = 10 deg, q_L = 0.8)
    kwargs_lens = [{"e1": 0.111, "e2": 0.0}]
    kwargs_lens_light = [{"e1": 0.166, "e2": 0.060}]

    prior_ref = CONFIG_1.custom_logL_addition(
        kwargs_lens=kwargs_lens,
        kwargs_lens_light=kwargs_lens_light,
    )
    prior = custom_logL_addition_jax(
        ModelConfig=CONFIG_1,
        kwargs_lens=kwargs_lens,
        kwargs_lens_light=kwargs_lens_light,
    )
    npt.assert_allclose(prior, prior_ref, atol=1e-15, rtol=1e-15)

    # qm < qL (phi_L = 0 deg, q_L = 0.9)
    kwargs_lens = [{"e1": 0.111, "e2": 0.0}]
    kwargs_lens_light = [{"e1": 0.0526, "e2": 0.0}]
    prior_ref = CONFIG_1.custom_logL_addition(
        kwargs_lens=kwargs_lens,
        kwargs_lens_light=kwargs_lens_light,
    )
    prior = custom_logL_addition_jax(
        ModelConfig=CONFIG_1,
        kwargs_lens=kwargs_lens,
        kwargs_lens_light=kwargs_lens_light,
    )
    npt.assert_allclose(prior, prior_ref, atol=1e-15, rtol=1e-15)

    # phi_m != phi_L (phi_L = 20 deg, q_L = 0.8)
    kwargs_lens = [{"e1": 0.111, "e2": 0.0}]
    kwargs_lens_light = [{"e1": 0.0851, "e2": 0.0714}]
    prior_ref = CONFIG_1.custom_logL_addition(
        kwargs_lens=kwargs_lens,
        kwargs_lens_light=kwargs_lens_light,
    )
    prior = custom_logL_addition_jax(
        ModelConfig=CONFIG_1,
        kwargs_lens=kwargs_lens,
        kwargs_lens_light=kwargs_lens_light,
    )
    npt.assert_allclose(prior, prior_ref, atol=1e-15, rtol=1e-15)

    # Test logarithmic shapelets prior
    kwargs_lens = [{"e1": 0.111, "e2": 0.0}]
    kwargs_lens_light = [{"e1": 0.166, "e2": 0.060}]
    kwargs_source = [
        {"R_sersic": 1.0},
        {"beta": 0.1},
        {"R_sersic": 1.0},
        {"beta": 0.1},
    ]
    prior_ref = CONFIG_3.custom_logL_addition(
        kwargs_lens=kwargs_lens,
        kwargs_lens_light=kwargs_lens_light,
        kwargs_source=kwargs_source,
    )
    prior = custom_logL_addition_jax(
        ModelConfig=CONFIG_3,
        kwargs_lens=kwargs_lens,
        kwargs_lens_light=kwargs_lens_light,
        kwargs_source=kwargs_source,
    )
    npt.assert_allclose(prior, prior_ref, atol=1e-15, rtol=1e-15)

    # Settings set to False  (phi_L = 20 deg, q_L = 0.9)
    config2 = deepcopy(CONFIG_1)
    config2.settings["lens_option"]["limit_mass_pa_from_light"] = np.inf
    config2.settings["lens_option"]["limit_mass_q_from_light"] = np.inf
    config2.settings["source_light_option"][
        "shapelet_scale_logarithmic_prior"
    ] = False
    kwargs_lens = [{"e1": 0.111, "e2": 0.0}]
    kwargs_lens_light = [{"e1": 0.0403, "e2": 0.0338}]
    prior_ref = config2.custom_logL_addition(
        kwargs_lens=kwargs_lens,
        kwargs_lens_light=kwargs_lens_light,
    )
    prior = custom_logL_addition_jax(
        ModelConfig=config2,
        kwargs_lens=kwargs_lens,
        kwargs_lens_light=kwargs_lens_light,
    )
    npt.assert_allclose(prior, prior_ref, atol=1e-15, rtol=1e-15)

    # Change setting data type (phi_L = 20 deg, q_L = 0.9)
    config3 = deepcopy(CONFIG_1)
    config3.settings["lens_option"]["limit_mass_q_from_light"] = 0.2
    config3.settings["lens_option"]["limit_mass_pa_from_light"] = 5
    kwargs_lens = [{"e1": 0.111, "e2": 0.0}]
    kwargs_lens_light = [{"e1": 0.0403, "e2": 0.0338}]
    prior_ref = config3.custom_logL_addition(
        kwargs_lens=kwargs_lens,
        kwargs_lens_light=kwargs_lens_light,
    )
    prior = custom_logL_addition_jax(
        ModelConfig=config3,
        kwargs_lens=kwargs_lens,
        kwargs_lens_light=kwargs_lens_light,
    )
    npt.assert_allclose(prior, prior_ref, atol=1e-15, rtol=1e-15)

    # Raise error when settings are not bool, int or float
    config4a = deepcopy(CONFIG_1)
    config4a.settings["lens_option"]["limit_mass_pa_from_light"] = "Test"
    with pytest.raises(ValueError):
        custom_logL_addition_jax(
            ModelConfig=config4a,
            kwargs_lens=[{"e1": 0.111, "e2": 0.0}],
            kwargs_lens_light=[{"e1": 0.166, "e2": 0.060}],
        )

    config4b = deepcopy(CONFIG_1)
    config4b.settings["lens_option"]["limit_mass_q_from_light"] = "Test"
    with pytest.raises(ValueError):
        custom_logL_addition_jax(
            ModelConfig=config4b,
            kwargs_lens=[{"e1": 0.111, "e2": 0.0}],
            kwargs_lens_light=[{"e1": 0.166, "e2": 0.060}],
        )