# -*- coding: utf-8 -*-
"""
Tests for config module.
"""

import pytest
from pathlib import Path

from dolphin.processor.config import *

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
        test_setting_file = _ROOT_DIR / 'test_working_directory' \
                            / 'settings' / 'test_system_config.yml'
        config = Config()
        config.load(str(test_setting_file.resolve()))


class TestModelConfig(object):
    """

    """
    def setup_class(self):
        test_setting_file = _ROOT_DIR / 'test_working_directory' \
                            / 'settings' / 'test_system_config.yml'
        self.config = ModelConfig(str(test_setting_file.resolve()))

    @classmethod
    def teardown_class(cls):
        pass

    def test_pixel_size(self):
        """
        Test the `pixel_size` property.
        :return:
        :rtype:
        """
        assert self.config.pixel_size == 0.04

    def test_deflector_center_ra(self):
        """
        Test the `deflector_center_ra` property.
        :return:
        :rtype:
        """
        assert self.config.deflector_center_ra == 0.04

    def test_deflector_center_dec(self):
        """
        Test the `deflector_center_ra` property.
        :return:
        :rtype:
        """
        assert self.config.deflector_center_dec == -0.04

    def test_deflector_centroid_bound(self):
        """
        Test the `deflector_centroid_bound` property.
        :return:
        :rtype:
        """
        assert self.config.deflector_centroid_bound == 0.5

    def test_band_number(self):
        """
        Test the `test_band_number` property.
        :return:
        :rtype:
        """
        assert self.config.band_number == 1

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
            'point_source_model_list': []
        }

        assert kwargs_model == self.config.get_kwargs_model()

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
            'joint_lens_with_light': [[0, 0, ['center_x', 'center_y']]]
        }

        assert kwargs_constraints == self.config.get_kwargs_constraints()

    def test_get_kwargs_likelihood(self):
        """
        Test `get_kwargs_likelihood` method.
        :return:
        :rtype:
        """
        test_likelihood = {
            'force_no_add_image': False,
            'source_marg': False,
            #'point_source_likelihood': True,
            #'position_uncertainty': 0.00004,
            #'check_solver': False,
            #'solver_tolerance': 0.001,
            'check_positive_flux': True,
            'check_bounds': True,
            'bands_compute': [True],
            #'image_likelihood_mask_list': self.config.get_masks()
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

    def test_get_psf_iteration(self):
        """
        Test `get_psf_iteration` method.
        :return:
        :rtype:
        """
        assert self.config.get_kwargs_psf_iteration() == {}

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

    def test_get_fitting_kwargs_list(self):
        """
        Test `get_fitting_kwargs_list` method.
        :return:
        :rtype:
        """
        fitting_kwargs_list = [[
            'MCMC',
            {
                'sampler_type': 'EMCEE',
                'n_burn': 2,
                'n_run': 2,
                'walkerRatio': 2
            }
        ]]

        self.config.settings['fitting']['sampling'] = True
        self.config.settings['fitting']['pso'] = False

        assert fitting_kwargs_list == self.config.get_fitting_kwargs_list()


if __name__ == '__main__':
    pytest.main()