# -*- coding: utf-8 -*-
"""Tests for config module."""

import pytest
from copy import deepcopy
import jax
import numpy as np
import numpy.testing as npt
import os
from pathlib import Path

from dolphin.processor.config import Config
from dolphin.processor.config import ModelConfig
from dolphin.processor.files import FileSystem

jax.config.update("jax_enable_x64", True)

_ROOT_DIR = Path(__file__).resolve().parents[2]


class TestConfig(object):
    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def test_load(self):
        """Test the `load` method in `class ModelConfig`."""
        test_setting_file = (
            _ROOT_DIR / "io_directory_example" / "settings" / "lens_system1_config.yaml"
        )
        config = Config()
        config.load_config_from_yaml(str(test_setting_file.resolve()))


class TestModelConfig(object):
    """"""

    def setup_method(self):
        self.io_directory = str((_ROOT_DIR / "io_directory_example").resolve())
        file_system = FileSystem(self.io_directory)

        self.config_1 = ModelConfig("lens_system1", file_system)
        self.config_3 = ModelConfig("lens_system3", io_directory=self.io_directory)
        self.config_4 = ModelConfig("lens_system4", io_directory=self.io_directory)
        self.config_5 = ModelConfig("lens_system5", file_system)
        self.config_wsat = ModelConfig("lens_system_wsat", file_system)

    def teardown_method(self):
        pass

    def test_init(self):
        settings = self.config_1.settings

        ModelConfig(self.config_1.lens_name, settings=settings)

    def test_pixel_size(self):
        """Test the `pixel_size` property."""
        self.config_3.settings["pixel_size"] = [0.04, 0.08]
        npt.assert_almost_equal(self.config_1.pixel_size, [0.04], decimal=6)
        npt.assert_almost_equal(self.config_3.pixel_size, [0.04, 0.08], decimal=6)

        config = deepcopy(self.config_3)
        npt.assert_almost_equal(config.pixel_size, [0.04, 0.08], decimal=6)

        config.settings["pixel_size"] = 0.04
        npt.assert_almost_equal(config.pixel_size, [0.04, 0.04], decimal=6)

    def test_lens_name(self):
        """Test the `lens_name` property."""
        assert self.config_1.lens_name == "lens_system1"
        assert self.config_5.lens_name == "lens_system5"

    def test_deflector_center_ra(self):
        """Test the `deflector_center_ra` property."""
        assert self.config_1.deflector_center_ra == 0.04
        assert self.config_5.deflector_center_ra == 0.0

    def test_deflector_center_dec(self):
        """Test the `deflector_center_ra` property."""
        assert self.config_1.deflector_center_dec == -0.04
        assert self.config_5.deflector_center_dec == 0.0

    def test_deflector_centroid_bound(self):
        """Test the `deflector_centroid_bound` property."""
        assert self.config_1.deflector_centroid_bound == 0.2
        assert self.config_5.deflector_centroid_bound == 0.5

    def test_number_of_bands(self):
        """Test the `test_band_number` property."""
        assert self.config_1.number_of_bands == 1

    def test_get_kwargs_model(self):
        """Test `get_kwargs_model` method."""
        kwargs_model = {
            "lens_model_list": ["EPL", "SHEAR_GAMMA_PSI"],
            "source_light_model_list": ["SERSIC_ELLIPSE"],
            "lens_light_model_list": ["SERSIC_ELLIPSE"],
            "point_source_model_list": [],
            "index_lens_light_model_list": [[0]],
            "index_source_light_model_list": [[0]],
        }

        assert kwargs_model == self.config_1.get_kwargs_model()

        # No profile_kwargs_list for SERSIC-only configs
        assert "lens_light_profile_kwargs_list" not in self.config_1.get_kwargs_model()

        self.config_5.settings["kwargs_model"] = {
            "key1": "value1",
            "key2": "value2",
        }
        kwargs_model2 = self.config_5.get_kwargs_model()

        assert kwargs_model2["key1"] == "value1"
        assert kwargs_model2["key2"] == "value2"

        del self.config_5.settings["kwargs_model"]

        kwargs_model_4 = self.config_4.get_kwargs_model()
        assert kwargs_model_4["lens_model_list"] == ["EPL", "SHEAR_GAMMA_PSI"]

    def test_get_kwargs_model_mge(self):
        """Test `get_kwargs_model` MGE_SET / MGE_SET_ELLIPSE handling."""
        # MGE_SET_ELLIPSE with explicit n_comp in mge_config
        config_mge = deepcopy(self.config_1)
        config_mge.settings["model"]["lens_light"] = ["MGE_SET_ELLIPSE"]
        config_mge.settings["lens_light_option"] = {"mge_config": {0: {"n_comp": 15}}}
        kwargs = config_mge.get_kwargs_model()
        assert "lens_light_profile_kwargs_list" in kwargs
        assert kwargs["lens_light_profile_kwargs_list"] == [{"n_comp": 15}]

        # Test default n_comp when mge_config is not specified
        config_mge2 = deepcopy(self.config_1)
        config_mge2.settings["model"]["lens_light"] = ["MGE_SET"]
        kwargs2 = config_mge2.get_kwargs_model()
        assert kwargs2["lens_light_profile_kwargs_list"] == [{"n_comp": 20}]

        # Test mixed list: MGE_SET + non-MGE model (hits `else` branch appending {})
        config_mge3 = deepcopy(self.config_3)
        config_mge3.settings["model"]["lens_light"] = ["MGE_SET", "SERSIC_ELLIPSE"]
        kwargs3 = config_mge3.get_kwargs_model()

        # 2 bands × 2 models = [MGE_SET, SERSIC_ELLIPSE, MGE_SET, SERSIC_ELLIPSE]
        assert kwargs3["lens_light_profile_kwargs_list"] == [
            {"n_comp": 20},
            {},
            {"n_comp": 20},
            {},
        ]

    def test_get_kwargs_constraints(self):
        """Test `get_kwargs_constraints` method."""

        kwargs_constraints = {
            "joint_source_with_source": [],
            "joint_lens_light_with_lens_light": [],
            "joint_source_with_point_source": [],
            "joint_lens_with_light": [[0, 0, ["center_x", "center_y"]]],
            "joint_lens_with_lens": [],
        }

        kwargs_constraints_3 = {
            "joint_source_with_source": [
                [0, 1, ["center_x", "center_y"]],
                [0, 2, ["center_x", "center_y"]],
                [0, 3, ["center_x", "center_y"]],
                [0, 2, ["n_sersic", "e1", "e2"]],
            ],
            "joint_lens_light_with_lens_light": [
                [0, 1, ["center_x", "center_y"]],
                [0, 2, ["center_x", "center_y"]],
                [0, 3, ["center_x", "center_y"]],
                [0, 2, ["n_sersic", "e1", "e2"]],
                [1, 3, ["n_sersic", "e1", "e2"]],
            ],
            "joint_source_with_point_source": [],
            "joint_lens_with_light": [[0, 0, ["center_x", "center_y"]]],
            "joint_lens_with_lens": [],
        }

        assert kwargs_constraints == self.config_1.get_kwargs_constraints()

        self.config_5.settings["kwargs_constraints"] = {
            "joint_source_with_source": [[0, 1, ["center_x", "center_y"]]],
            "joint_source_with_point_source": [[0, 0], [0, 1]],
        }
        kwargs_constraints = self.config_5.get_kwargs_constraints()
        assert kwargs_constraints["joint_source_with_source"] == [
            [0, 1, ["center_x", "center_y"]]
        ]
        assert kwargs_constraints["joint_source_with_point_source"] == [[0, 0], [0, 1]]
        del self.config_5.settings["kwargs_constraints"]

        assert kwargs_constraints_3 == self.config_3.get_kwargs_constraints()

        kwargs_constraints_wsat = {
            "joint_source_with_source": [
                [0, 1, ["center_x", "center_y"]],
                [0, 2, ["center_x", "center_y"]],
                [0, 3, ["center_x", "center_y"]],
                [0, 2, ["n_sersic", "e1", "e2"]],
            ],
            "joint_lens_light_with_lens_light": [
                [0, 1, ["center_x", "center_y"]],
                [0, 2, ["center_x", "center_y"]],
                [0, 3, ["center_x", "center_y"]],
                [0, 2, ["n_sersic", "e1", "e2"]],
                [1, 3, ["n_sersic", "e1", "e2"]],
                [4, 6, ["center_x", "center_y", "n_sersic", "e1", "e2"]],
                [5, 7, ["center_x", "center_y", "n_sersic", "e1", "e2"]],
            ],
            "joint_source_with_point_source": [],
            "joint_lens_with_light": [
                [0, 0, ["center_x", "center_y"]],
                [4, 2, ["center_x", "center_y"]],
                [5, 3, ["center_x", "center_y"]],
                [6, 2, ["center_x", "center_y"]],
                [7, 3, ["center_x", "center_y"]],
            ],
            "joint_lens_with_lens": [],
        }

        assert kwargs_constraints_wsat == self.config_wsat.get_kwargs_constraints()

    def test_get_kwargs_likelihood(self):
        """Test `get_kwargs_likelihood` method."""
        test_likelihood = {
            "force_no_add_image": False,
            "source_marg": False,
            # 'point_source_likelihood': True,
            # 'position_uncertainty': 0.00004,
            # 'check_solver': False,
            # 'solver_tolerance': 0.001,
            "check_positive_flux": True,
            "check_bounds": True,
            "bands_compute": [True],
            "prior_lens": [],
            "prior_lens_light": [],
            "prior_ps": [],
            "prior_source": [],
            "custom_logL_addition": self.config_1.custom_logL_addition,
            # 'image_likelihood_mask_list': self.config.get_masks()
        }
        kwargs_likelihood = self.config_1.get_kwargs_likelihood()
        kwargs_likelihood.pop("image_likelihood_mask_list")
        assert kwargs_likelihood == test_likelihood

        kwargs_likelihood2 = self.config_3.get_kwargs_likelihood()
        assert kwargs_likelihood2["prior_lens"] == [
            [0, "gamma", 2.11, 0.03],
            [0, "theta_E", 1.11, 0.13],
        ]
        assert kwargs_likelihood2["prior_lens_light"] == [[0, "R_sersic", 0.21, 0.15]]
        assert kwargs_likelihood2["prior_source"] == [[0, "beta", 0.15, 0.05]]

        config = deepcopy(self.config_3)
        config.settings["point_source_option"] = {
            "gaussian_prior": {0: [["ra_image", 0.21, 0.15]]}
        }

        kwargs_likelihood3 = config.get_kwargs_likelihood()
        assert kwargs_likelihood3["prior_ps"] == [[0, "ra_image", 0.21, 0.15]]

    def test_custom_logL_addition(self):
        """Test `custom_logL_addition` method."""
        # Mass paramters : (phi_m = 0 deg, q_m = 0.8)
        # Satisfy both priors (phi_L = 10 deg, q_L = 0.8)
        prior = self.config_1.custom_logL_addition(
            kwargs_lens=[{"e1": 0.111, "e2": 0.0}],
            kwargs_lens_light=[{"e1": 0.166, "e2": 0.060}],
        )
        assert prior == 0

        # qm < qL (phi_L = 0 deg, q_L = 0.9)
        prior = self.config_1.custom_logL_addition(
            kwargs_lens=[{"e1": 0.111, "e2": 0.0}],
            kwargs_lens_light=[{"e1": 0.0526, "e2": 0.0}],
        )
        assert round(prior) == round(-((0.1 - 0.0) ** 2) / (1e-4))

        # phi_m != phi_L (phi_L = 20 deg, q_L = 0.8)
        prior = self.config_1.custom_logL_addition(
            kwargs_lens=[{"e1": 0.111, "e2": 0.0}],
            kwargs_lens_light=[{"e1": 0.0851, "e2": 0.0714}],
        )
        assert round(prior, -3) == -((20 - 15) ** 2) / (1e-3)

        # Test logarithmic shapelets prior
        prior = self.config_3.custom_logL_addition(
            kwargs_lens=[{"e1": 0.111, "e2": 0.0}],
            kwargs_lens_light=[{"e1": 0.166, "e2": 0.060}],
            kwargs_source=[
                {"R_sersic": 1.0},
                {"beta": 0.1},
                {"R_sersic": 1.0},
                {"beta": 0.1},
            ],
        )
        assert round(prior, 2) == round(-2 * np.log(0.1), 2)

        # Settings set to False  (phi_L = 20 deg, q_L = 0.9)
        config2 = deepcopy(self.config_1)
        config2.settings["lens_option"]["limit_mass_pa_from_light"] = np.inf
        config2.settings["lens_option"]["limit_mass_q_from_light"] = np.inf
        config2.settings["source_light_option"][
            "shapelet_scale_logarithmic_prior"
        ] = False
        prior = config2.custom_logL_addition(
            kwargs_lens=[{"e1": 0.111, "e2": 0.0}],
            kwargs_lens_light=[{"e1": 0.0403, "e2": 0.0338}],
        )
        assert prior == 0

        # Change setting data type (phi_L = 20 deg, q_L = 0.9)
        config3 = deepcopy(self.config_1)
        config3.settings["lens_option"]["limit_mass_q_from_light"] = 0.2
        config3.settings["lens_option"]["limit_mass_pa_from_light"] = 5
        prior = config3.custom_logL_addition(
            kwargs_lens=[{"e1": 0.111, "e2": 0.0}],
            kwargs_lens_light=[{"e1": 0.0403, "e2": 0.0338}],
        )
        assert round(prior, -3) == -((20 - 5) ** 2) / (1e-3)

        # Raise error when settings are not bool, int or float
        config4a = deepcopy(self.config_1)
        config4a.settings["lens_option"]["limit_mass_pa_from_light"] = "Test"
        with pytest.raises(ValueError):
            config4a.custom_logL_addition(
                kwargs_lens=[{"e1": 0.111, "e2": 0.0}],
                kwargs_lens_light=[{"e1": 0.166, "e2": 0.060}],
            )

        config4b = deepcopy(self.config_1)
        config4b.settings["lens_option"]["limit_mass_q_from_light"] = "Test"
        with pytest.raises(ValueError):
            config4b.custom_logL_addition(
                kwargs_lens=[{"e1": 0.111, "e2": 0.0}],
                kwargs_lens_light=[{"e1": 0.166, "e2": 0.060}],
            )

    def test_custom_logL_addition_JAX(self):
        """Test `custom_logL_addition_JAX` method.

        :return:
        :rtype:
        """
        # Mass paramters : (phi_m = 0 deg, q_m = 0.8)
        # Satisfy both priors (phi_L = 10 deg, q_L = 0.8)
        kwargs_lens = [{"e1": 0.111, "e2": 0.0}]
        kwargs_lens_light = [{"e1": 0.166, "e2": 0.060}]

        prior_ref = self.config_1.custom_logL_addition(
            kwargs_lens=kwargs_lens,
            kwargs_lens_light=kwargs_lens_light,
        )
        prior = self.config_1.custom_logL_addition_JAX(
            kwargs_lens=kwargs_lens,
            kwargs_lens_light=kwargs_lens_light,
        )
        npt.assert_allclose(prior, prior_ref, atol=1e-15, rtol=1e-15)

        # qm < qL (phi_L = 0 deg, q_L = 0.9)
        kwargs_lens = [{"e1": 0.111, "e2": 0.0}]
        kwargs_lens_light = [{"e1": 0.0526, "e2": 0.0}]
        prior_ref = self.config_1.custom_logL_addition(
            kwargs_lens=kwargs_lens,
            kwargs_lens_light=kwargs_lens_light,
        )
        prior = self.config_1.custom_logL_addition_JAX(
            kwargs_lens=kwargs_lens,
            kwargs_lens_light=kwargs_lens_light,
        )
        npt.assert_allclose(prior, prior_ref, atol=1e-15, rtol=1e-15)

        # phi_m != phi_L (phi_L = 20 deg, q_L = 0.8)
        kwargs_lens = [{"e1": 0.111, "e2": 0.0}]
        kwargs_lens_light = [{"e1": 0.0851, "e2": 0.0714}]
        prior_ref = self.config_1.custom_logL_addition(
            kwargs_lens=kwargs_lens,
            kwargs_lens_light=kwargs_lens_light,
        )
        prior = self.config_1.custom_logL_addition_JAX(
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
        prior_ref = self.config_3.custom_logL_addition(
            kwargs_lens=kwargs_lens,
            kwargs_lens_light=kwargs_lens_light,
            kwargs_source=kwargs_source,
        )
        prior = self.config_3.custom_logL_addition_JAX(
            kwargs_lens=kwargs_lens,
            kwargs_lens_light=kwargs_lens_light,
            kwargs_source=kwargs_source,
        )
        npt.assert_allclose(prior, prior_ref, atol=1e-15, rtol=1e-15)

        # Settings set to False  (phi_L = 20 deg, q_L = 0.9)
        config2 = deepcopy(self.config_1)
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
        prior = config2.custom_logL_addition_JAX(
            kwargs_lens=kwargs_lens,
            kwargs_lens_light=kwargs_lens_light,
        )
        npt.assert_allclose(prior, prior_ref, atol=1e-15, rtol=1e-15)

        # Change setting data type (phi_L = 20 deg, q_L = 0.9)
        config3 = deepcopy(self.config_1)
        config3.settings["lens_option"]["limit_mass_q_from_light"] = 0.2
        config3.settings["lens_option"]["limit_mass_pa_from_light"] = 5
        kwargs_lens = [{"e1": 0.111, "e2": 0.0}]
        kwargs_lens_light = [{"e1": 0.0403, "e2": 0.0338}]
        prior_ref = config3.custom_logL_addition(
            kwargs_lens=kwargs_lens,
            kwargs_lens_light=kwargs_lens_light,
        )
        prior = config3.custom_logL_addition_JAX(
            kwargs_lens=kwargs_lens,
            kwargs_lens_light=kwargs_lens_light,
        )
        npt.assert_allclose(prior, prior_ref, atol=1e-15, rtol=1e-15)

        # Raise error when settings are not bool, int or float
        config4a = deepcopy(self.config_1)
        config4a.settings["lens_option"]["limit_mass_pa_from_light"] = "Test"
        with pytest.raises(ValueError):
            config4a.custom_logL_addition(
                kwargs_lens=[{"e1": 0.111, "e2": 0.0}],
                kwargs_lens_light=[{"e1": 0.166, "e2": 0.060}],
            )

        config4b = deepcopy(self.config_1)
        config4b.settings["lens_option"]["limit_mass_q_from_light"] = "Test"
        with pytest.raises(ValueError):
            config4b.custom_logL_addition(
                kwargs_lens=[{"e1": 0.111, "e2": 0.0}],
                kwargs_lens_light=[{"e1": 0.166, "e2": 0.060}],
            )

    def test_load_mask(self):
        """Test `load_mask` static method."""
        mask_array = np.zeros((20, 20))
        mask_file = os.path.join(self.io_directory, "temp_test_mask.npy")
        np.save(mask_file, mask_array)

        loaded_mask = ModelConfig.load_mask(mask_file)
        npt.assert_array_equal(loaded_mask, mask_array)

        # Clean up the temporary file
        os.remove(mask_file)

    def test_get_masks(self):
        """Test `get_masks` method."""
        masks = self.config_1.get_masks()

        assert len(masks) == self.config_1.number_of_bands

        for n in range(self.config_1.number_of_bands):
            assert masks[n].shape == (120, 120)

        masks2 = self.config_5.get_masks()
        assert masks2 is None

        self.config_5.settings["mask"] = None
        assert self.config_5.get_masks() is None

        masks3 = self.config_3.get_masks()
        # Test mask_edge_pixel (2 pixels border)
        assert masks3[1][5, 0:6].tolist() == [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        assert masks3[1][5, -6:].tolist() == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]

        # test elliptical mask
        config_elliptial_mask = deepcopy(self.config_1)
        del config_elliptial_mask.settings["mask"]["radius"]

        with pytest.raises(ValueError):
            config_elliptial_mask.get_masks()

        config_elliptial_mask.settings["mask"]["a"] = [1]

        with pytest.raises(ValueError):
            config_elliptial_mask.get_masks()

        config_elliptial_mask.settings["mask"]["b"] = [0.5]

        with pytest.raises(ValueError):
            config_elliptial_mask.get_masks()

        config_elliptial_mask.settings["mask"]["angle"] = [np.pi / 4.0]

        masks_elliptical = config_elliptial_mask.get_masks()
        assert len(masks_elliptical) == config_elliptial_mask.number_of_bands
        for n in range(config_elliptial_mask.number_of_bands):
            assert masks_elliptical[n].shape == (
                120,
                120,
            )

        # test mask provided
        config_mask_provided = deepcopy(self.config_1)
        config_mask_provided.settings["lens_name"] = "lensed_quasar"
        config_mask_provided.settings["band"] = ["F814W"]
        config_mask_provided.settings["mask"]["provided"] = True
        mask = config_mask_provided.get_masks()
        assert mask[0].shape == (120, 120)

        # test extra regions
        config_extra_regions = deepcopy(self.config_1)
        config_extra_regions.settings["mask"]["extra_regions"] = [[[0, 0, 2.0]]]
        mask = config_extra_regions.get_masks()
        assert np.sum(mask[0]) == 0

    def test_get_kwargs_psf_iteration(self):
        """Test `get_psf_iteration` method."""
        assert self.config_1.get_kwargs_psf_iteration() == {}

        kwargs_psf_iteration = self.config_5.get_kwargs_psf_iteration()

        assert kwargs_psf_iteration == {
            "stacking_method": "median",
            "keep_psf_variance_map": True,
            "psf_symmetry": 4,
            "block_center_neighbour": 0.5,
            "num_iter": 20,
            "psf_iter_factor": 0.5,
        }

    def test_get_kwargs_params(self):
        """Test `get_kwargs_params` method."""
        for key in [
            "lens_model",
            "source_model",
            "lens_light_model",
            "point_source_model",
        ]:
            kwargs_params = self.config_1.get_kwargs_params()

            assert key in kwargs_params

            assert len(kwargs_params[key]) == 5

    def test_get_kwargs_numerics(self):
        """Test `get_kwargs_numerics` method."""
        test_numerics = [
            {
                "supersampling_factor": 3,
                "supersampling_convolution": False,
                "supersampling_kernel_size": 3,
                "flux_evaluate_indexes": None,
                "point_source_supersampling_factor": 1,
                "compute_mode": "regular",
            }
        ]

        assert test_numerics == self.config_1.get_kwargs_numerics()

        self.config_5.settings["band"] = ["F390W"]
        if "numeric_option" in self.config_5.settings:
            del self.config_5.settings["numeric_option"]
        assert test_numerics == self.config_5.get_kwargs_numerics()

        config = deepcopy(self.config_1)
        config.settings["numeric_option"]["supersampling_factor"] = None
        kwargs_numerics = config.get_kwargs_numerics()
        for kwargs_numerics_band in kwargs_numerics:
            assert kwargs_numerics_band["supersampling_factor"] == 3

    def test_get_point_source_params(self):
        """Test `get_point_source_params` method."""
        ps_params = self.config_1.get_point_source_params()
        assert ps_params == [[]] * 5

        ps_params = self.config_5.get_point_source_params()
        assert np.all(ps_params[0][0]["ra_image"] == [1.0, 0.0, 1.0, 0.0])
        assert np.all(ps_params[0][0]["dec_image"] == [0.0, 1.0, 0.0, -1.0])

    def test_get_lens_model_list(self):
        """Test `get_lens_model_list` method."""
        assert self.config_5.get_lens_model_list() == ["EPL", "SHEAR_GAMMA_PSI"]

        config = deepcopy(self.config_5)
        del config.settings["model"]["lens"]
        assert config.get_lens_model_list() == []

        assert self.config_wsat.get_lens_model_list() == [
            "EPL",
            "SHEAR_GAMMA_PSI",
            "SIE",
            "SIS",
        ]

        config = deepcopy(self.config_wsat)
        del config.settings["satellites"]["is_elliptical"]
        print(config.get_lens_model_list())
        assert config.get_lens_model_list() == [
            "EPL",
            "SHEAR_GAMMA_PSI",
            "SIS",
            "SIS",
        ]

    def test_get_source_light_model_list(self):
        """Test `get_source_light_model_list` method."""
        config = deepcopy(self.config_5)
        del config.settings["model"]["source_light"]
        assert config.get_source_light_model_list() == []

        config2 = deepcopy(self.config_3)
        assert config2.get_source_light_model_list() == [
            "SERSIC_ELLIPSE",
            "SHAPELETS",
            "SERSIC_ELLIPSE",
            "SHAPELETS",
        ]

    def test_get_lens_light_model_list(self):
        """Test `get_lens_light_model_list` method."""
        config = deepcopy(self.config_5)
        del config.settings["model"]["lens_light"]
        assert config.get_lens_light_model_list() == []

        config2 = deepcopy(self.config_3)
        assert config2.get_lens_light_model_list() == [
            "SERSIC_ELLIPSE",
            "SERSIC_ELLIPSE",
            "SERSIC_ELLIPSE",
            "SERSIC_ELLIPSE",
        ]

        assert self.config_wsat.get_lens_light_model_list() == [
            "SERSIC_ELLIPSE",
            "SERSIC_ELLIPSE",
            "SERSIC_ELLIPSE",
            "SERSIC_ELLIPSE",
            "SERSIC_ELLIPSE",
            "SERSIC",
            "SERSIC_ELLIPSE",
            "SERSIC",
        ]

        config = deepcopy(self.config_wsat)
        del config.settings["satellites"]["is_elliptical"]
        assert config.get_lens_light_model_list() == [
            "SERSIC_ELLIPSE",
            "SERSIC_ELLIPSE",
            "SERSIC_ELLIPSE",
            "SERSIC_ELLIPSE",
            "SERSIC",
            "SERSIC",
            "SERSIC",
            "SERSIC",
        ]

    def test_get_point_source_model_list(self):
        """Test `get_point_source_model_list` method."""
        config = deepcopy(self.config_5)
        assert config.get_point_source_model_list() == ["LENSED_POSITION"]

    def test_get_special_list(self):
        """Test `get_special_list` method."""

        # Test 1: ensure consistency of special models
        # if specified in config file
        config = deepcopy(self.config_5)
        config.settings["model"]["special"] = ["astrometric_uncertainty"]
        assert config.get_special_list() == ["astrometric_uncertainty"]

        # Test 2: ensure special list is empty if not
        # specified in the config file
        config = deepcopy(self.config_1)
        assert config.get_special_list() == []

        # Test 3: ensure error message prints if special
        # type is not supported
        config = deepcopy(self.config_3)
        config.settings["model"]["special"] = ["INVALID"]
        with pytest.raises(ValueError):
            config.get_special_list()

    def test_get_lens_model_params(self):
        """Test `get_lens_model_params` method."""
        self.config_5.settings["model"]["lens"] = ["INVALID"]
        with pytest.raises(ValueError):
            self.config_5.get_lens_model_params()

        self.config_5.settings["model"]["lens"] = ["SPEP"]
        self.config_5.get_lens_model_params()

        params = self.config_4.get_lens_model_params()
        assert self.config_4.settings["model"]["lens"][0] == "SIE"
        assert params[2][0] == {"gamma": 2.0}

        params = self.config_wsat.get_lens_model_params()
        for i in range(5):
            assert len(params[i]) == 4
        assert params[0][2] == {
            "center_x": 1,
            "center_y": 1,
            "e1": 0,
            "e2": 0,
            "gamma": 2.0,
            "theta_E": 0.1,
        }
        assert params[0][3] == {
            "center_x": 1.5,
            "center_y": 1.5,
            "theta_E": 0.1,
        }

    def test_get_lens_light_model_params(self):
        """Test `get_lens_light_model_params` method."""
        config = deepcopy(self.config_5)
        config.settings["model"]["lens_light"] = ["INVALID"]
        with pytest.raises(ValueError):
            config.get_lens_light_model_params()

        params = self.config_wsat.get_lens_light_model_params()
        for i in range(5):
            assert len(params[i])
        assert params[0][4] == {
            "center_x": 1,
            "center_y": 1,
            "R_sersic": 0.2,
            "amp": 1.0,
            "e1": 0.0,
            "e2": 0.0,
            "n_sersic": 4.0,
        }
        assert params[0][5] == {
            "center_x": 1.5,
            "center_y": 1.5,
            "R_sersic": 0.2,
            "amp": 1.0,
            "n_sersic": 4.0,
        }
        assert params[0][6] == {
            "center_x": 1,
            "center_y": 1,
            "R_sersic": 0.2,
            "amp": 1.0,
            "e1": 0.0,
            "e2": 0.0,
            "n_sersic": 4.0,
        }
        assert params[0][7] == {
            "center_x": 1.5,
            "center_y": 1.5,
            "R_sersic": 0.2,
            "amp": 1.0,
            "n_sersic": 4.0,
        }

    def test_get_source_light_model_params(self):
        """Test `get_source_light_model_params` method."""
        config = deepcopy(self.config_1)
        config.settings["model"]["source_light"] = ["INVALID"]
        with pytest.raises(ValueError):
            config.get_source_light_model_params()

        config2 = deepcopy(self.config_3)
        config2.get_source_light_model_params()
        assert config2.settings["source_light_option"]["n_max"] == [2, 4]

        config3 = deepcopy(self.config_3)
        config3.settings["source_light_option"]["n_max"] = 2
        config3.get_source_light_model_params()
        assert config3.settings["source_light_option"]["n_max"] == [2, 2]

    def test_get_special_params(self):
        """Test `get_special_params` method."""

        # Test 1: ensure consistency of astrometric uncertainty params
        # if specified in config file
        config = deepcopy(self.config_5)

        config.settings["model"]["special"] = ["astrometric_uncertainty"]
        config.settings["special_option"] = {
            "delta_x_image": [0.004, 0.004, 0.004, 0.004],
            "delta_y_image": [0.004, 0.004, 0.004, 0.004],
            "delta_image_lower": -0.004,
            "delta_image_upper": 0.004,
        }

        params = config.get_special_params()

        init, sigma, fixed, lower, upper = params

        npt.assert_array_equal(
            init["delta_x_image"], np.array([0.004, 0.004, 0.004, 0.004])
        )
        npt.assert_array_equal(
            init["delta_y_image"], np.array([0.004, 0.004, 0.004, 0.004])
        )

        assert np.all(sigma["delta_x_image"] == 0.004)
        assert np.all(sigma["delta_y_image"] == 0.004)
        assert len(sigma["delta_x_image"]) == 4
        assert len(sigma["delta_y_image"]) == 4

        assert np.all(lower["delta_x_image"] == -0.004)
        assert np.all(upper["delta_x_image"] == 0.004)

        assert np.all(lower["delta_y_image"] == -0.004)
        assert np.all(upper["delta_y_image"] == 0.004)

        assert fixed == {}

        # Test 2: ensure special params is empty list of dictionaries
        # if not specified in the config file
        config = deepcopy(self.config_1)

        params = config.get_special_params()

        assert params == [{}, {}, {}, {}, {}]

    def test_fill_in_fixed_from_settings(self):
        """Test `fill_in_fixed_from_settings` method."""
        fixed = [{}]
        fixed = self.config_1.fill_in_fixed_from_settings("lens_light", fixed)
        assert fixed == [{"n_sersic": 4.0}]

        fixed = [{}, {}, {}, {}]
        fixed = self.config_3.fill_in_fixed_from_settings("lens_light", fixed)
        assert fixed == [{"n_sersic": 4.0}, {}, {"n_sersic": 4.0}, {}]

        fixed = [{}, {}]
        fixed = self.config_3.fill_in_fixed_from_settings("lens", fixed)
        assert fixed == [{"gamma": 2.0}, {}]

        with pytest.raises(AssertionError):
            self.config_3.fill_in_fixed_from_settings("invalid", fixed)

    def test_get_psf_supersampling_factor(self):
        """Test `get_psf_supersampling_factor` method."""
        assert self.config_1.get_psf_supersampled_factor() == 1

        self.config_1.settings["psf_supersampled_factor"] = 3
        assert self.config_1.get_psf_supersampled_factor() == 3

    def test_get_index_lens_light_model_list(self):
        """Test `get_index_lens_light_model_list` method."""
        assert self.config_1.get_index_lens_light_model_list() == [[0]]
        assert self.config_3.get_index_lens_light_model_list() == [[0, 1], [2, 3]]
        config = deepcopy(self.config_5)
        del config.settings["model"]["lens_light"]
        assert config.get_index_lens_light_model_list() == []

        config2 = deepcopy(self.config_5)
        assert len(config2.get_index_lens_light_model_list()) == 1
        config2.settings["band"] = ["F390W", "F555W"]
        assert len(config2.get_index_lens_light_model_list()) == 2

        assert self.config_wsat.get_index_lens_light_model_list() == [
            [0, 1, 4, 6],
            [2, 3, 5, 7],
        ]

    def test_get_index_source_light_model_list(self):
        """Test `get_index_source_light_model_list` method."""
        assert self.config_1.get_index_source_light_model_list() == [[0]]
        assert self.config_3.get_index_source_light_model_list() == [[0, 1], [2, 3]]

        config = deepcopy(self.config_5)
        del config.settings["model"]["source_light"]
        assert config.get_index_source_light_model_list() == []

        config2 = deepcopy(self.config_5)
        assert len(config2.get_index_source_light_model_list()) == 1
        config2.settings["band"] = ["F390W", "F555W"]
        assert len(config2.get_index_source_light_model_list()) == 2

    def test_get_lens_light_model_params_mge(self):
        """Test `get_lens_light_model_params` MGE_SET / MGE_SET_ELLIPSE branch."""
        # MGE_SET: sigma_min/sigma_width/center/amp present, no ellipticity
        config_mge = deepcopy(self.config_1)
        config_mge.settings["model"]["lens_light"] = ["MGE_SET"]
        params = config_mge.get_lens_light_model_params()
        assert len(params) == 5  # fixed, init, sigma, lower, upper
        assert "sigma_min" in params[0][0]
        assert "sigma_width" in params[0][0]
        assert "center_x" in params[0][0]
        assert "center_y" in params[0][0]
        assert "amp" in params[0][0]
        assert "e1" not in params[0][0]

        # MGE_SET_ELLIPSE: additionally has e1/e2
        config_mge_e = deepcopy(self.config_1)
        config_mge_e.settings["model"]["lens_light"] = ["MGE_SET_ELLIPSE"]
        params_e = config_mge_e.get_lens_light_model_params()
        assert "e1" in params_e[0][0]
        assert "e2" in params_e[0][0]
        assert "sigma_min" in params_e[0][0]

    def test_get_kwargs_constraints_mge(self):
        """Test MGE_SET / MGE_SET_ELLIPSE joining in `get_kwargs_constraints`."""
        config = deepcopy(self.config_3)
        config.settings["model"]["lens_light"] = ["MGE_SET_ELLIPSE"]
        constraints = config.get_joint_lens_light_with_lens_light()
        # Should join sigma_min, sigma_width, e1, e2 across 2 bands
        found_mge_join = False
        for entry in constraints:
            if "sigma_min" in entry[2]:
                found_mge_join = True
                assert "sigma_width" in entry[2]
                assert "e1" in entry[2]
                assert "e2" in entry[2]
        assert found_mge_join

        # MGE_SET (non-ellipse) should not join e1, e2
        config2 = deepcopy(self.config_3)
        config2.settings["model"]["lens_light"] = ["MGE_SET"]
        constraints2 = config2.get_joint_lens_light_with_lens_light()
        for entry in constraints2:
            if "sigma_min" in entry[2]:
                assert "e1" not in entry[2]
                assert "e2" not in entry[2]

    def test_custom_logL_addition_mge_set(self):
        """Test that custom_logL_addition skips ellipticity priors for MGE_SET."""
        config = deepcopy(self.config_1)
        config.settings["model"]["lens_light"] = ["MGE_SET"]
        # config_1 has both limit_mass_pa_from_light and limit_mass_q_from_light
        # set. Without the first-model-has-ellipticity guard, these would try
        # to read e1/e2 from MGE kwargs and crash. Prior must return 0.
        prior = config.custom_logL_addition(
            kwargs_lens=[{"e1": 0.111, "e2": 0.0}],
            kwargs_lens_light=[{"sigma_min": 0.01, "sigma_width": 1.0}],
        )
        assert prior == 0

    def test_get_mge_n_comp(self):
        """Test _get_mge_n_comp() helper method."""
        config = deepcopy(self.config_1)

        # Default when no mge_config
        assert config._get_mge_n_comp(0) == 20

        # With mge_config using integer keys (hits `if` branch)
        config.settings["lens_light_option"] = {"mge_config": {0: {"n_comp": 15}}}
        assert config._get_mge_n_comp(0) == 15
        assert config._get_mge_n_comp(1) == 20  # Not configured, returns default

        # With mge_config using string keys (hits `elif` branch)
        config.settings["lens_light_option"] = {"mge_config": {"0": {"n_comp": 12}}}
        assert config._get_mge_n_comp(0) == 12
        assert config._get_mge_n_comp(1) == 20  # Not configured, returns default

    def test_get_kwargs_likelihood_mge(self):
        """Test `get_kwargs_likelihood` disables `check_positive_flux` for MGE."""
        # Non-MGE config: check_positive_flux is True
        assert self.config_1.get_kwargs_likelihood()["check_positive_flux"] is True

        # MGE_SET: check_positive_flux is False
        config_mge = deepcopy(self.config_1)
        config_mge.settings["model"]["lens_light"] = ["MGE_SET"]
        assert config_mge.get_kwargs_likelihood()["check_positive_flux"] is False

        # MGE_SET_ELLIPSE: check_positive_flux is False
        config_mge_e = deepcopy(self.config_1)
        config_mge_e.settings["model"]["lens_light"] = ["MGE_SET_ELLIPSE"]
        assert config_mge_e.get_kwargs_likelihood()["check_positive_flux"] is False
