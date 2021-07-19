# -*- coding: utf-8 -*-
"""
Tests for config module.
"""
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
        """
        Test the `load` method in `class ModelConfig`.
        :return:
        :rtype:
        """
        test_setting_file = _ROOT_DIR / 'io_directory_example' \
            / 'settings' / 'lens_system1_config.yml'
        config = Config()
        config.load(str(test_setting_file.resolve()))


class TestModelConfig(object):
    """

    """
    def setup_class(self):
        self.test_setting_file = _ROOT_DIR / 'io_directory_example' \
                            / 'settings' / 'lens_system1_config.yml'
        self.config = ModelConfig(str(self.test_setting_file.resolve()))

        self.test_setting_file2 = _ROOT_DIR / 'io_directory_example' \
            / 'settings' / '_test_config.yml'
        self.config2 = ModelConfig(str(self.test_setting_file2.resolve()))
        self.test_setting_file3 = _ROOT_DIR / 'io_directory_example' \
            / 'settings' / 'lens_system3_config.yml'
        self.config3 = ModelConfig(str(self.test_setting_file3.resolve()))

    @classmethod
    def teardown_class(cls):
        pass

    def test_load_settings_from_file(self):
        test_config = ModelConfig()
        test_config.load_settings_from_file(str(
            self.test_setting_file.resolve()))

        assert test_config.settings is not None

    def test_pixel_size(self):
        """
        Test the `pixel_size` property.
        :return:
        :rtype:
        """
        assert self.config.pixel_size == [0.04]
        assert self.config3.pixel_size == [0.04, 0.08]

        config = deepcopy(self.config3)
        assert config.pixel_size == [0.04, 0.08]

        config.settings['pixel_size'] = 0.04
        assert config.pixel_size == [0.04, 0.04]

    def test_deflector_center_ra(self):
        """
        Test the `deflector_center_ra` property.
        :return:
        :rtype:
        """
        assert self.config.deflector_center_ra == 0.04
        assert self.config2.deflector_center_ra == 0.

    def test_deflector_center_dec(self):
        """
        Test the `deflector_center_ra` property.
        :return:
        :rtype:
        """
        assert self.config.deflector_center_dec == -0.04
        assert self.config2.deflector_center_dec == 0.

    def test_deflector_centroid_bound(self):
        """
        Test the `deflector_centroid_bound` property.
        :return:
        :rtype:
        """
        assert self.config.deflector_centroid_bound == 0.5
        assert self.config2.deflector_centroid_bound == 0.2

    def test_band_number(self):
        """
        Test the `test_band_number` property.
        :return:
        :rtype:
        """
        assert self.config.band_number == 1

        with pytest.raises(ValueError):
            self.config2.band_number

        self.config2.settings['band'] = []

        with pytest.raises(ValueError):
            self.config2.band_number

    def test_get_kwargs_model(self):
        """
        Test `get_kwargs_model` method.
        :return:
        :rtype:
        """
        kwargs_model = {
            'lens_model_list': ['SPEP', 'SHEAR_GAMMA_PSI'],
            'source_light_model_list': ['SERSIC_ELLIPSE'],
            'lens_light_model_list': ['SERSIC_ELLIPSE'],
            'point_source_model_list': [],
            'index_lens_light_model_list': [[0]],
            'index_source_light_model_list': [[0]],
        }

        assert kwargs_model == self.config.get_kwargs_model()

        self.config2.settings['band'] = ['F390W']
        kwargs_model2 = self.config2.get_kwargs_model()

        assert kwargs_model2['key1'] == 'value1'
        assert kwargs_model2['key2'] == 'value2'

        self.config2.settings['band'] = []

    def test_get_kwargs_constraints(self):
        """
        Test `get_kwargs_constraints` method.
        :return:
        :rtype:
        """

        kwargs_constraints = {
            'joint_source_with_source': [],
            'joint_lens_light_with_lens_light': [],
            'joint_source_with_point_source': [],
            'joint_lens_with_light': [[0, 0, ['center_x', 'center_y']]],
            'joint_lens_with_lens': []
        }

        kwargs_constraints_2 = {
            'joint_source_with_source': [[0, 1, ['center_x', 'center_y']]],
            'joint_lens_light_with_lens_light':
                                        [[0, 1, ['center_x', 'center_y']],
                                         [0, 2, ['center_x', 'center_y']],
                                         [0, 3, ['center_x', 'center_y']]],
            'joint_source_with_point_source': [],
            'joint_lens_with_light':  [],
            'joint_lens_with_lens': []
        }

        assert kwargs_constraints == self.config.get_kwargs_constraints()
        self.config2.settings['band'] = ['F390W']
        kwargs_constraints = self.config2.get_kwargs_constraints()

        assert kwargs_constraints['joint_source_with_source'] == [[0, 1,
                                                                   ['center_x',
                                                                    'center_y']]
                                                                  ]
        assert kwargs_constraints['joint_source_with_point_source'] == [
            [0, 0],
            [0, 1]
        ]
        self.config2.settings['band'] = []

        assert kwargs_constraints_2 == self.config3.get_kwargs_constraints()

    def test_get_kwargs_likelihood(self):
        """
        Test `get_kwargs_likelihood` method.
        :return:
        :rtype:
        """
        test_likelihood = {
            'force_no_add_image': False,
            'source_marg': False,
            # 'point_source_likelihood': True,
            # 'position_uncertainty': 0.00004,
            # 'check_solver': False,
            # 'solver_tolerance': 0.001,
            'check_positive_flux': True,
            'check_bounds': True,
            'bands_compute': [True],
            # 'image_likelihood_mask_list': self.config.get_masks()
        }
        kwargs_likelihood = self.config.get_kwargs_likelihood()
        kwargs_likelihood.pop('image_likelihood_mask_list')
        assert kwargs_likelihood == test_likelihood

    def test_get_masks(self):
        """
        Test `get_masks` method.
        :return:
        :rtype:
        """
        masks = self.config.get_masks()

        assert len(masks) == self.config.band_number

        for n in range(self.config.band_number):
            assert masks[n].shape == (self.config.settings['mask']['size'][n],
                                      self.config.settings['mask']['size'][n]
                                      )

        masks2 = self.config2.get_masks()
        assert masks2 == [[[0., 0.], [0., 0.]]]

        self.config2.settings['mask']['provided'] = None
        self.config2.settings['band'] = ['F390W']

        masks2 = self.config2.get_masks()
        assert len(masks2) == self.config2.band_number

        for n in range(self.config2.band_number):
            assert masks2[n].shape == (self.config2.settings['mask']['size'][n],
                                       self.config2.settings['mask']['size'][n]
                                       )

        self.config2.settings['mask'] = None
        assert self.config2.get_masks() is None

    def test_get_kwargs_psf_iteration(self):
        """
        Test `get_psf_iteration` method.
        :return:
        :rtype:
        """
        assert self.config.get_kwargs_psf_iteration() == {}

        kwargs_psf_iteration = self.config2.get_kwargs_psf_iteration()

        assert kwargs_psf_iteration == {
            'stacking_method': 'median',
            'keep_psf_error_map': True,
            'psf_symmetry': 4,
            'block_center_neighbour': 0.,
            'num_iter': 20,
            'psf_iter_factor': 0.5,
        }

    def test_get_kwargs_params(self):
        """
        Test `get_kwargs_params` method.
        :return:
        :rtype:
        """
        for key in ['lens_model', 'source_model', 'lens_light_model',
                    'point_source_model']:
            kwargs_params = self.config.get_kwargs_params()

            assert key in kwargs_params

            assert len(kwargs_params[key]) == 5

    def test_get_kwargs_numerics(self):
        """
        Test `get_kwargs_numerics` method.
        :return:
        :rtype:
        """
        test_numerics = [{
            'supersampling_factor': 3,
            'supersampling_convolution': False,
            'supersampling_kernel_size': 3,
            'flux_evaluate_indexes': None,
            'point_source_supersampling_factor': 1,
            'compute_mode': 'regular',
        }]

        assert test_numerics == self.config.get_kwargs_numerics()

        self.config2.settings['band'] = ['F390W']
        assert test_numerics == self.config2.get_kwargs_numerics()

        config = deepcopy(self.config)
        config.settings['kwargs_numerics']['supersampling_factor'] = None
        kwargs_numerics = config.get_kwargs_numerics()
        for kwargs_numerics_band in kwargs_numerics:
            assert kwargs_numerics_band['supersampling_factor'] == 3

    def test_get_point_source_params(self):
        """
        Test `get_point_source_params` method.
        :return:
        :rtype:
        """
        ps_params = self.config.get_point_source_params()
        assert ps_params == [[]]*5

        ps_params = self.config2.get_point_source_params()
        assert np.all(ps_params[0][0]['ra_image'] == [1., 0., 1., 0.])
        assert np.all(ps_params[0][0]['dec_image'] == [0., 1., 0., -1.])

    def test_get_lens_model_list(self):
        """
        Test `get_lens_model_list` method.
        :return:
        :rtype:
        """
        assert self.config2.get_lens_model_list() == []

    def test_get_source_light_model_list(self):
        """
        Test `get_source_light_model_list` method.
        :return:
        :rtype:
        """
        config = deepcopy(self.config2)
        del config.settings['model']['source_light']
        assert config.get_source_light_model_list() == []

        config2 = deepcopy(self.config3)
        assert config2.get_source_light_model_list() == \
               ['SHAPELETS', 'SHAPELETS']

    def test_get_lens_light_model_list(self):
        """
        Test `get_lens_light_model_list` method.
        :return:
        :rtype:
        """
        config = deepcopy(self.config2)
        del config.settings['model']['lens_light']
        assert config.get_lens_light_model_list() == []

        config2 = deepcopy(self.config3)
        assert config2.get_lens_light_model_list() == \
               ['SERSIC_ELLIPSE', 'SERSIC_ELLIPSE',
                'SERSIC_ELLIPSE', 'SERSIC_ELLIPSE']

    def test_get_point_source_model_list(self):
        """
        Test `get_point_source_model_list` method.
        :return:
        :rtype:
        """
        config = deepcopy(self.config2)
        del config.settings['model']['point_source']
        assert config.get_point_source_model_list() == []

    def test_get_lens_model_params(self):
        """
        Test `get_lens_model_params` method.
        :return:
        :rtype:
        """
        self.config2.settings['model']['lens'] = ['INVALID']
        with pytest.raises(ValueError):
            self.config2.get_lens_model_params()

        self.config2.settings['model']['lens'] = ['SPEP']
        self.config2.get_lens_model_params()

    def test_get_lens_light_model_params(self):
        """
        Test `get_lens_light_model_params` method.
        :return:
        :rtype:
        """
        config = deepcopy(self.config2)
        config.settings['model']['lens_light'] = ['INVALID']
        with pytest.raises(ValueError):
            config.get_lens_light_model_params()

    def test_get_source_light_model_params(self):
        """
        Test `get_source_light_model_params` method.
        :return:
        :rtype:
        """
        config = deepcopy(self.config)
        config.settings['model']['source_light'] = ['INVALID']
        with pytest.raises(ValueError):
            config.get_source_light_model_params()

        config2 = deepcopy(self.config3)
        config2.get_source_light_model_params()
        assert config2.settings['source_light_option']['n_max'] == [2, 4]

    def test_fill_in_fixed_from_settings(self):
        """
        Test `fill_in_fixed_from_settings` method.
        :return:
        :rtype:
        """
        fixed = [{}]
        fixed = self.config.fill_in_fixed_from_settings('lens_light',
                                                        fixed)
        assert fixed == [{'n_sersic': 4.}]

    def test_get_psf_supersampling_factor(self):
        """
        Test `get_psf_supersampling_factor` method.
        :return:
        :rtype:
        """
        assert self.config.get_psf_supersampled_factor() == 1

        self.config.settings['psf_supersampled_factor'] = 3
        assert self.config.get_psf_supersampled_factor() == 3

    def test_get_index_lens_light_model_list(self):
        """
        Test `get_index_lens_light_model_list` method.
        :return:
        :rtype:
        """
        assert self.config.get_index_lens_light_model_list() == [[0]]
        assert self.config3.get_index_lens_light_model_list() == [[0, 1],
                                                                  [2, 3]]
        config = deepcopy(self.config2)
        del config.settings['model']['lens_light']
        assert config.get_index_lens_light_model_list() == []

    def test_get_index_source_light_model_list(self):
        """
        Test `get_index_source_light_model_list` method.
        :return:
        :rtype:
        """
        assert self.config.get_index_source_light_model_list() == [[0]]
        assert self.config3.get_index_source_light_model_list() == [[0], [1]]
        config = deepcopy(self.config2)
        del config.settings['model']['lens_light']
        assert config.get_index_source_light_model_list() == []
