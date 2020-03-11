# -*- coding: utf-8 -*-
"""
Tests for Recipe module.
"""

import pytest
from pathlib import Path
import numpy as np
from copy import deepcopy
from lenstronomy.Workflow.fitting_sequence import FittingSequence

from dolphin.processor.config import ModelConfig
from dolphin.processor.recipe import Recipe

_ROOT_DIR = Path(__file__).resolve().parents[2]


class TestRecipe(object):
    """
    Test the `Recipe` module.
    """
    def setup_class(self):
        self.test_setting_file = _ROOT_DIR / 'test_working_directory' \
                                 / 'settings' / 'test_system_config.yml'
        self.config = ModelConfig(str(self.test_setting_file.resolve()))
        self.recipe = Recipe(self.config)

    @classmethod
    def teardown_class(cls):
        pass

    def test_get_recipe(self):
        """
        Test `get_recipe` method.
        :return:
        :rtype:
        """
        fitting_kwargs_list = self.recipe.get_recipe()
        assert isinstance(fitting_kwargs_list, list)

    def test_get_power_law_model_index(self):
        """
        Test `get_power_law_model_index` method.
        :return:
        :rtype:
        """
        assert self.recipe._get_power_law_model_index() == 0

    def test_get_external_shear_model_index(self):
        """
        Test `_get_external_shear_model_index` method.
        :return:
        :rtype:
        """
        assert self.recipe._get_external_shear_model_index() == 1

    def test_get_shapelet_model_index(self):
        """
        Test `get_power_law_model_index` method.
        :return:
        :rtype:
        """
        self.recipe._config.settings['model']['source_light'] = [
                                                'SERSIC_ELLIPSE', 'SHAPELETS']
        assert self.recipe._get_shapelet_model_index() == 1

        self.recipe._config.settings['model']['source_light'] = [
                                                            'SERSIC_ELLIPSE']

    def test_get_default_recipe(self):
        """
        Test `get_default_recipe` method.
        :return:
        :rtype:
        """
        self.recipe.reconstruct_psf = True
        fitting_kwargs_list = self.recipe.get_default_recipe()
        assert isinstance(fitting_kwargs_list, list)

    def test_get_sampling_sequence(self):
        """
        Test `get_sampling_sequence` method.
        :return:
        :rtype:
        """
        self.recipe.do_sampling = True
        fitting_kwargs_list = self.recipe.get_sampling_sequence()
        assert isinstance(fitting_kwargs_list, list)

    def test_get_galaxy_galaxy_recipe(self):
        """
        Test `get_galaxy_galaxy_recipe` method.
        :return:
        :rtype:
        """
        image = np.random.normal(size=(120, 120))
        kwargs_data_joint = {
            'multi_band_list': [[{'image_data': image,
                                  'background_rms': 0.01,
                                  'exposure_time': np.ones_like(image),
                                  'ra_at_xy_0': 0.,
                                  'dec_at_xy_0': 0.,
                                  'transform_pix2angle': np.array([[-0.01, 0],
                                                                   [0, 0.01]])
                                  },
                                 {}, {}]],
            'multi_band_type': 'multi-linear'
        }
        fitting_kwargs_list = self.recipe.get_galaxy_galaxy_recipe(
                                                            kwargs_data_joint)
        assert isinstance(fitting_kwargs_list, list)

        # test the recipe by running it fully
        config = deepcopy(self.config)
        config.settings['model']['source_light'] = ['SHAPELETS']

        recipe = Recipe(config)

        fitting_sequence = FittingSequence(
            kwargs_data_joint,
            config.get_kwargs_model(),
            config.get_kwargs_constraints(),
            config.get_kwargs_likelihood(),
            config.get_kwargs_params(),
        )

        fitting_kwargs_list = recipe.get_recipe(
            kwargs_data_joint=kwargs_data_joint,
            recipe_name='galaxy-galaxy')

        fit_output = fitting_sequence.fit_sequence(fitting_kwargs_list)

    def test_get_arc_mask(self):
        """
        Test `get_arc_mask` method.
        :return:
        :rtype:
        """
        image = np.random.normal(size=(100, 100))

        mask = self.recipe.get_arc_mask(image)
        assert mask.shape == (100, 100)

    def test_fix_params(self):
        """
        Test `fix_params` method.
        :return:
        :rtype:
        """
        test = self.recipe.fix_params('lens', [0])
        assert set(test[1]['lens_add_fixed'][0][1]) == {'theta_E',
                                                        'center_x',
                                                        'center_y',  'e1',
                                                        'gamma', 'e2'}

        test = self.recipe.fix_params('lens', [1])
        assert set(test[1]['lens_add_fixed'][0][1]) == {'gamma_ext',
                                                        'psi_ext'}

        test = self.recipe.fix_params('lens_light', [0])
        assert set(test[1]['lens_light_add_fixed'][0][1]) == {'e1', 'center_x',
                                                              'center_y',
                                                              'R_sersic',
                                                              'e2'}

        test = self.recipe.fix_params('source', [0])
        assert set(test[1]['source_add_fixed'][0][1]) == {'R_sersic',
                                                          'n_sersic',
                                                          'center_x',
                                                          'center_y',
                                                          'e1', 'e2'}

    def test_unfix_params(self):
        """
        Test `unfix_params` method.
        :return:
        :rtype:
        """
        test = self.recipe.unfix_params('lens', [0])
        assert set(test[1]['lens_remove_fixed'][0][1]) == {'theta_E',
                                                           'center_x',
                                                           'center_y',  'e1', 'gamma',
                                                           'e2'}

        test = self.recipe.unfix_params('lens', [1])
        assert set(test[1]['lens_remove_fixed'][0][1]) == {'gamma_ext',
                                                           'psi_ext'}

        test = self.recipe.unfix_params('lens_light', [0])
        assert set(test[1]['lens_light_remove_fixed'][0][1]) == {'e1',
                                                                 'center_x',
                                                                 'center_y',
                                                                 'R_sersic', 'e2'}

        test = self.recipe.unfix_params('source', [0])
        assert set(test[1]['source_remove_fixed'][0][1]) == {'R_sersic',
                                                             'n_sersic',
                                                             'center_x', 'center_y',
                                                             'e1', 'e2'}