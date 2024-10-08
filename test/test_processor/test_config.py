# -*- coding: utf-8 -*-
"""Tests for config module."""
import pytest
from copy import deepcopy
import numpy as np
from pathlib import Path

from dolphin.processor.config import Config
from dolphin.processor.config import ModelConfig

_ROOT_DIR = Path(__file__).resolve().parents[2]


class TestConfig(object):
    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def test_load(self):
        """Test the `load` method in `class ModelConfig`.

        :return:
        :rtype:
        """
        test_setting_file = (
            _ROOT_DIR / "io_directory_example" / "settings" / "lens_system1_config.yml"
        )
        config = Config()
        config.load(str(test_setting_file.resolve()))


class TestModelConfig(object):
    """"""

    def setup_class(self):
        self.test_setting_file = (
            _ROOT_DIR / "io_directory_example" / "settings" / "lens_system1_config.yml"
        )
        self.config_1 = ModelConfig(str(self.test_setting_file.resolve()))

        self.test_setting_file2 = (
            _ROOT_DIR / "io_directory_example" / "settings" / "_test_config.yml"
        )
        self.config_2 = ModelConfig(str(self.test_setting_file2.resolve()))
        self.test_setting_file3 = (
            _ROOT_DIR / "io_directory_example" / "settings" / "lens_system3_config.yml"
        )
        self.config_3 = ModelConfig(str(self.test_setting_file3.resolve()))
        self.test_setting_file4 = (
            _ROOT_DIR / "io_directory_example" / "settings" / "lens_system4_config.yml"
        )
        self.config_4 = ModelConfig(str(self.test_setting_file4.resolve()))

    @classmethod
    def teardown_class(cls):
        pass

    def test_load_settings_from_file(self):
        test_config = ModelConfig()
        test_config.load_settings_from_file(str(self.test_setting_file.resolve()))

        assert test_config.settings is not None

    def test_pixel_size(self):
        """Test the `pixel_size` property.

        :return:
        :rtype:
        """
        assert self.config_1.pixel_size == [0.04]
        assert self.config_3.pixel_size == [0.04, 0.08]

        config = deepcopy(self.config_3)
        assert config.pixel_size == [0.04, 0.08]

        config.settings["pixel_size"] = 0.04
        assert config.pixel_size == [0.04, 0.04]

    def test_deflector_center_ra(self):
        """Test the `deflector_center_ra` property.

        :return:
        :rtype:
        """
        assert self.config_1.deflector_center_ra == 0.04
        assert self.config_2.deflector_center_ra == 0.0

    def test_deflector_center_dec(self):
        """Test the `deflector_center_ra` property.

        :return:
        :rtype:
        """
        assert self.config_1.deflector_center_dec == -0.04
        assert self.config_2.deflector_center_dec == 0.0

    def test_deflector_centroid_bound(self):
        """Test the `deflector_centroid_bound` property.

        :return:
        :rtype:
        """
        assert self.config_1.deflector_centroid_bound == 0.5
        assert self.config_2.deflector_centroid_bound == 0.2

    def test_band_number(self):
        """Test the `test_band_number` property.

        :return:
        :rtype:
        """
        assert self.config_1.band_number == 1

        with pytest.raises(ValueError):
            self.config_2.band_number

        self.config_2.settings["band"] = []

        with pytest.raises(ValueError):
            self.config_2.band_number

    def test_get_kwargs_model(self):
        """Test `get_kwargs_model` method.

        :return:
        :rtype:
        """
        kwargs_model = {
            "lens_model_list": ["EPL", "SHEAR_GAMMA_PSI"],
            "source_light_model_list": ["SERSIC_ELLIPSE"],
            "lens_light_model_list": ["SERSIC_ELLIPSE"],
            "point_source_model_list": [],
            "index_lens_light_model_list": [[0]],
            "index_source_light_model_list": [[0]],
        }

        assert kwargs_model == self.config_1.get_kwargs_model()

        self.config_2.settings["band"] = ["F390W"]
        kwargs_model2 = self.config_2.get_kwargs_model()

        assert kwargs_model2["key1"] == "value1"
        assert kwargs_model2["key2"] == "value2"

        self.config_2.settings["band"] = []

        kwargs_model_4 = self.config_4.get_kwargs_model()
        assert kwargs_model_4["lens_model_list"] == ["EPL", "SHEAR_GAMMA_PSI"]

    def test_get_kwargs_constraints(self):
        """Test `get_kwargs_constraints` method.

        :return:
        :rtype:
        """

        kwargs_constraints = {
            "joint_source_with_source": [],
            "joint_lens_light_with_lens_light": [],
            "joint_source_with_point_source": [],
            "joint_lens_with_light": [[0, 0, ["center_x", "center_y"]]],
            "joint_lens_with_lens": [],
        }

        kwargs_constraints_2 = {
            "joint_source_with_source": [[0, 1, ["center_x", "center_y"]]],
            "joint_lens_light_with_lens_light": [
                [0, 1, ["center_x", "center_y"]],
                [0, 2, ["center_x", "center_y"]],
                [0, 3, ["center_x", "center_y"]],
            ],
            "joint_source_with_point_source": [],
            "joint_lens_with_light": [],
            "joint_lens_with_lens": [],
        }

        assert kwargs_constraints == self.config_1.get_kwargs_constraints()
        self.config_2.settings["band"] = ["F390W"]
        kwargs_constraints = self.config_2.get_kwargs_constraints()

        assert kwargs_constraints["joint_source_with_source"] == [
            [0, 1, ["center_x", "center_y"]]
        ]
        assert kwargs_constraints["joint_source_with_point_source"] == [[0, 0], [0, 1]]
        self.config_2.settings["band"] = []

        assert kwargs_constraints_2 == self.config_3.get_kwargs_constraints()

    def test_get_kwargs_likelihood(self):
        """Test `get_kwargs_likelihood` method.

        :return:
        :rtype:
        """
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
        """Test `custom_logL_addition` method.

        :return:
        :rtype:
        """
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
            kwargs_source=[{"beta": 0.1}, {"beta": 0.1}],
        )
        assert round(prior, 2) == round(-2 * np.log(0.1), 2)

        # Settings set to False  (phi_L = 20 deg, q_L = 0.9)
        config2 = deepcopy(self.config_1)
        config2.settings["lens_option"][
            "constrain_position_angle_from_lens_light"
        ] = False
        config2.settings["lens_option"]["limit_mass_eccentricity_from_light"] = False
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
        config3.settings["lens_option"]["imit_mass_eccentricity_from_light"] = 0.2
        config3.settings["lens_option"]["constrain_position_angle_from_lens_light"] = 5
        prior = config3.custom_logL_addition(
            kwargs_lens=[{"e1": 0.111, "e2": 0.0}],
            kwargs_lens_light=[{"e1": 0.0403, "e2": 0.0338}],
        )
        assert round(prior, -3) == -((20 - 5) ** 2) / (1e-3)

        # Raise error when settings are not bool, int or float
        config4a = deepcopy(self.config_1)
        config4a.settings["lens_option"][
            "constrain_position_angle_from_lens_light"
        ] = "Test"
        with pytest.raises(TypeError):
            config4a.custom_logL_addition(
                kwargs_lens=[{"e1": 0.111, "e2": 0.0}],
                kwargs_lens_light=[{"e1": 0.166, "e2": 0.060}],
            )

        config4b = deepcopy(self.config_1)
        config4b.settings["lens_option"]["limit_mass_eccentricity_from_light"] = "Test"
        with pytest.raises(TypeError):
            config4b.custom_logL_addition(
                kwargs_lens=[{"e1": 0.111, "e2": 0.0}],
                kwargs_lens_light=[{"e1": 0.166, "e2": 0.060}],
            )

    def test_get_masks(self):
        """Test `get_masks` method.

        :return:
        :rtype:
        """
        masks = self.config_1.get_masks()

        assert len(masks) == self.config_1.band_number

        for n in range(self.config_1.band_number):
            assert masks[n].shape == (
                self.config_1.settings["mask"]["size"][n],
                self.config_1.settings["mask"]["size"][n],
            )

        masks2 = self.config_2.get_masks()
        assert masks2 == [[[0.0, 0.0], [0.0, 0.0]]]

        self.config_2.settings["mask"]["provided"] = None
        self.config_2.settings["band"] = ["F390W"]

        masks2 = self.config_2.get_masks()
        assert len(masks2) == self.config_2.band_number

        for n in range(self.config_2.band_number):
            assert masks2[n].shape == (
                self.config_2.settings["mask"]["size"][n],
                self.config_2.settings["mask"]["size"][n],
            )

        self.config_2.settings["mask"] = None
        assert self.config_2.get_masks() is None

        masks2 = self.config_2.get_masks()

        masks3 = self.config_3.get_masks()
        # Test custom mask (Alternating Pixel Mask)
        assert masks3[0][0, 0:6].tolist() == [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
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
        assert len(masks_elliptical) == config_elliptial_mask.band_number
        for n in range(config_elliptial_mask.band_number):
            assert masks_elliptical[n].shape == (
                config_elliptial_mask.settings["mask"]["size"][n],
                config_elliptial_mask.settings["mask"]["size"][n],
            )

    def test_get_kwargs_psf_iteration(self):
        """Test `get_psf_iteration` method.

        :return:
        :rtype:
        """
        assert self.config_1.get_kwargs_psf_iteration() == {}

        kwargs_psf_iteration = self.config_2.get_kwargs_psf_iteration()

        assert kwargs_psf_iteration == {
            "stacking_method": "median",
            "keep_psf_error_map": True,
            "psf_symmetry": 4,
            "block_center_neighbour": 0.0,
            "num_iter": 20,
            "psf_iter_factor": 0.5,
        }

    def test_get_kwargs_params(self):
        """Test `get_kwargs_params` method.

        :return:
        :rtype:
        """
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
        """Test `get_kwargs_numerics` method.

        :return:
        :rtype:
        """
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

        self.config_2.settings["band"] = ["F390W"]
        assert test_numerics == self.config_2.get_kwargs_numerics()

        config = deepcopy(self.config_1)
        config.settings["kwargs_numerics"]["supersampling_factor"] = None
        kwargs_numerics = config.get_kwargs_numerics()
        for kwargs_numerics_band in kwargs_numerics:
            assert kwargs_numerics_band["supersampling_factor"] == 3

    def test_get_point_source_params(self):
        """Test `get_point_source_params` method.

        :return:
        :rtype:
        """
        ps_params = self.config_1.get_point_source_params()
        assert ps_params == [[]] * 5

        ps_params = self.config_2.get_point_source_params()
        assert np.all(ps_params[0][0]["ra_image"] == [1.0, 0.0, 1.0, 0.0])
        assert np.all(ps_params[0][0]["dec_image"] == [0.0, 1.0, 0.0, -1.0])

    def test_get_lens_model_list(self):
        """Test `get_lens_model_list` method.

        :return:
        :rtype:
        """
        assert self.config_2.get_lens_model_list() == []

    def test_get_source_light_model_list(self):
        """Test `get_source_light_model_list` method.

        :return:
        :rtype:
        """
        config = deepcopy(self.config_2)
        del config.settings["model"]["source_light"]
        assert config.get_source_light_model_list() == []

        config2 = deepcopy(self.config_3)
        assert config2.get_source_light_model_list() == ["SHAPELETS", "SHAPELETS"]

    def test_get_lens_light_model_list(self):
        """Test `get_lens_light_model_list` method.

        :return:
        :rtype:
        """
        config = deepcopy(self.config_2)
        del config.settings["model"]["lens_light"]
        assert config.get_lens_light_model_list() == []

        config2 = deepcopy(self.config_3)
        assert config2.get_lens_light_model_list() == [
            "SERSIC_ELLIPSE",
            "SERSIC_ELLIPSE",
            "SERSIC_ELLIPSE",
            "SERSIC_ELLIPSE",
        ]

    def test_get_point_source_model_list(self):
        """Test `get_point_source_model_list` method.

        :return:
        :rtype:
        """
        config = deepcopy(self.config_2)
        del config.settings["model"]["point_source"]
        assert config.get_point_source_model_list() == []

    def test_get_lens_model_params(self):
        """Test `get_lens_model_params` method.

        :return:
        :rtype:
        """
        self.config_2.settings["model"]["lens"] = ["INVALID"]
        with pytest.raises(ValueError):
            self.config_2.get_lens_model_params()

        self.config_2.settings["model"]["lens"] = ["SPEP"]
        self.config_2.get_lens_model_params()

        params = self.config_4.get_lens_model_params()
        assert self.config_4.settings["model"]["lens"][0] == "SIE"
        assert params[2][0] == {"gamma": 2.0}

    def test_get_lens_light_model_params(self):
        """Test `get_lens_light_model_params` method.

        :return:
        :rtype:
        """
        config = deepcopy(self.config_2)
        config.settings["model"]["lens_light"] = ["INVALID"]
        with pytest.raises(ValueError):
            config.get_lens_light_model_params()

    def test_get_source_light_model_params(self):
        """Test `get_source_light_model_params` method.

        :return:
        :rtype:
        """
        config = deepcopy(self.config_1)
        config.settings["model"]["source_light"] = ["INVALID"]
        with pytest.raises(ValueError):
            config.get_source_light_model_params()

        config2 = deepcopy(self.config_3)
        config2.get_source_light_model_params()
        assert config2.settings["source_light_option"]["n_max"] == [2, 4]

    def test_fill_in_fixed_from_settings(self):
        """Test `fill_in_fixed_from_settings` method.

        :return:
        :rtype:
        """
        fixed = [{}]
        fixed = self.config_1.fill_in_fixed_from_settings("lens_light", fixed)
        assert fixed == [{"n_sersic": 4.0}]

        fixed2 = [{}, {}, {}, {}]
        fixed2 = self.config_3.fill_in_fixed_from_settings("lens_light", fixed2)
        assert fixed2 == [{"n_sersic": 4.0}, {}, {"n_sersic": 4.0}, {}]

    def test_get_psf_supersampling_factor(self):
        """Test `get_psf_supersampling_factor` method.

        :return:
        :rtype:
        """
        assert self.config_1.get_psf_supersampled_factor() == 1

        self.config_1.settings["psf_supersampled_factor"] = 3
        assert self.config_1.get_psf_supersampled_factor() == 3

    def test_get_index_lens_light_model_list(self):
        """Test `get_index_lens_light_model_list` method.

        :return:
        :rtype:
        """
        assert self.config_1.get_index_lens_light_model_list() == [[0]]
        assert self.config_3.get_index_lens_light_model_list() == [[0, 1], [2, 3]]
        config = deepcopy(self.config_2)
        del config.settings["model"]["lens_light"]
        assert config.get_index_lens_light_model_list() == []

        config2 = deepcopy(self.config_2)
        config2.settings["band"] = ["F390W", "F555W"]
        with pytest.raises(ValueError):
            config2.get_index_lens_light_model_list()

    def test_get_index_source_light_model_list(self):
        """Test `get_index_source_light_model_list` method.

        :return:
        :rtype:
        """
        assert self.config_1.get_index_source_light_model_list() == [[0]]
        assert self.config_3.get_index_source_light_model_list() == [[0], [1]]

        config = deepcopy(self.config_2)
        del config.settings["model"]["source_light"]
        assert config.get_index_source_light_model_list() == []

        config2 = deepcopy(self.config_2)
        config2.settings["band"] = ["F390W", "F555W"]
        with pytest.raises(ValueError):
            config2.get_index_source_light_model_list()
