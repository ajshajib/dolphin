# -*- coding: utf-8 -*-
"""This module handles JAX related imports and functions separately from the main dolphin modules."""

from jax import config, numpy as jnp
config.update("jax_enable_x64", True)
from jaxtronomy.Util.param_util import ellipticity2phi_q

def custom_logL_addition_jax(
    ModelConfig,
    kwargs_lens=None,
    kwargs_source=None,
    kwargs_lens_light=None,
    kwargs_ps=None,
    kwargs_special=None,
    kwargs_extinction=None,
    kwargs_tracer_source=None,
):
    """Provide additional likelihood terms to be sent to `JAXtronomy`. This is the
    same function as above but compatible with jax.jit.

    :param ModelConfig: instance of ModelConfig class
    :type ModelConfig: `ModelConfig` class instance
    :param kwargs_lens: dictionary containing lens model keyword arguments
    :type kwargs_lens: `dict`
    :param kwargs_source: dictionary containing source model keyword arguments
    :type kwargs_source: `dict`
    :param kwargs_lens_light: dictionary containing lens light model keyword arguments
    :type kwargs_lens_light: `dict`
    :param kwargs_ps: dictionary containing point source model keyword arguments
    :type kwargs_ps: `dict`
    :param kwargs_special: dictionary containing special model keyword arguments
    :type kwargs_special: `dict`
    :param kwargs_extinction: dictionary containing extinction model keyword arguments
    :type kwargs_extinction: `dict`
    :param kwargs_tracer_source: dictionary containing tracer source model keyword
    :return: prior
    :rtype: float
    """
    prior = 0.0

    # Limit the difference between pa_light and pa_mass for the deflector, where pa is the
    # position angle of the major axis
    if (
        "lens_option" in ModelConfig.settings
        and "limit_mass_pa_from_light" in ModelConfig.settings["lens_option"]
    ):
        max_mass_pa_difference = ModelConfig.settings["lens_option"][
            "limit_mass_pa_from_light"
        ]

        if not isinstance(max_mass_pa_difference, (int, float)):
            raise ValueError(
                "The value for limit_mass_pa_from_light should be a number!"
            )

        pa_mass = (
            ellipticity2phi_q(kwargs_lens[0]["e1"], kwargs_lens[0]["e2"])[0]
            * 180
            / jnp.pi
        )
        pa_light = (
            ellipticity2phi_q(
                kwargs_lens_light[0]["e1"], kwargs_lens_light[0]["e2"]
            )[0]
            * 180
            / jnp.pi
        )

        diff = jnp.minimum(
            jnp.abs(pa_light - pa_mass), 180 - jnp.abs(pa_light - pa_mass)
        )

        prior = jnp.where(
            diff > jnp.abs(max_mass_pa_difference),
            prior - ((diff - jnp.abs(max_mass_pa_difference)) ** 2) / 1e-3,
            prior,
        )

    # Limit the difference between q_light and q_mass for the deflector, where q is the axis
    # ratio of the elliptical profile
    if (
        "lens_option" in ModelConfig.settings
        and "limit_mass_q_from_light" in ModelConfig.settings["lens_option"]
    ):
        max_mass_q_difference = ModelConfig.settings["lens_option"][
            "limit_mass_q_from_light"
        ]

        if not isinstance(max_mass_q_difference, (int, float)):
            raise ValueError(
                "The value for limit_mass_q_from_light should be a number!"
            )

        q_mass = ellipticity2phi_q(kwargs_lens[0]["e1"], kwargs_lens[0]["e2"])[
            1
        ]
        q_light = ellipticity2phi_q(
            kwargs_lens_light[0]["e1"], kwargs_lens_light[0]["e2"]
        )[1]

        q_mass = ellipticity2phi_q(kwargs_lens[0]["e1"], kwargs_lens[0]["e2"])[
            1
        ]
        q_light = ellipticity2phi_q(
            kwargs_lens_light[0]["e1"], kwargs_lens_light[0]["e2"]
        )[1]
        diff = q_light - q_mass
        prior = jnp.where(
            diff > max_mass_q_difference,
            prior - ((diff - max_mass_q_difference) ** 2) / 1e-4,
            prior,
        )

    # Provide logarithmic_prior on the source light profile beta param
    if (
        "source_light_option" in ModelConfig.settings
        and "shapelet_scale_logarithmic_prior"
        in ModelConfig.settings["source_light_option"]
    ):
        if ModelConfig.settings["source_light_option"]["shapelet_scale_logarithmic_prior"]:
            for i, model in enumerate(ModelConfig.get_source_light_model_list()):
                if model == "SHAPELETS":
                    beta = kwargs_source[i]["beta"]
                    prior += -jnp.log(beta)

    return prior