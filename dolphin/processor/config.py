# -*- coding: utf-8 -*-
"""
This module loads settings from a configuration file.
"""
__author__ = 'ajshajib'

import yaml
import numpy as np
from copy import deepcopy

from lenstronomy.Data.coord_transforms import Coordinates
import lenstronomy.Util.util as util
import lenstronomy.Util.mask_util as mask_util


class Config(object):
    """
    This class contains the methods to load an read YAML configuration
    files. This is a parent class for other classes that needs to load
    a configuration file. If the file type of the configuration files
    changes, then only this class needs to be modified.
    """

    def __init__(self):
        pass

    @classmethod
    def load(cls, file):
        """
        Load configuration from `file`.

        :return:
        :rtype:
        """
        with open(file, 'r') as f:
            settings = yaml.load(f, yaml.FullLoader)

        return settings


class ModelConfig(Config):
    """
    This class contains the methods to load and interact with modeling
    settings for a particular system.
    """

    def __init__(self, file=None, settings=None):
        """
        Initiate a Model Config object. If the file path is given, `settings`
        will be loaded from it. Otherwise, the `settings` can be
        loaded/reloaded later with the `load_settings_from_file` method.

        :param file: path to a settings file, file will
        :type file: `str`
        :param settings: a dictionary containing settings. If both `file`
            and `settings` are provided, `file` will be prioritized.
        :type settings: `dict`
        """
        super(ModelConfig, self).__init__()

        self.settings = settings
        if file is not None:
            self.load_settings_from_file(file)

    def load_settings_from_file(self, file):
        """
        Load the settings.

        :param file: path to a settings file
        :type file: `str`
        :return:
        :rtype:
        """
        self.settings = self.load(file)

    @property
    def pixel_size(self):
        """
        The pixel size.

        :return:
        :rtype:
        """
        if type(self.settings['pixel_size']) == float:
            return [self.settings['pixel_size']]
        else:
            return self.settings['pixel_size']

    @property
    def maximun_pixel_size(self):
        """
        The maximun pixel size.

        :return:
        :rtype:
        """
        if type(self.settings['pixel_size']) == float:
            return [self.settings['pixel_size']]
        else:
            return [max(self.settings['pixel_size'])]

    @property
    def deflector_center_ra(self):
        """
        The RA offset for the deflector's center from the zero-point
        in the coordinate system of the data. Default is 0.

        :return:
        :rtype:
        """
        if 'lens_option' in self.settings and 'centroid_init' in \
                self.settings['lens_option']:
            return float(self.settings['lens_option'][
                             'centroid_init'][0])
        else:
            return 0.

    @property
    def deflector_center_dec(self):
        """
        The dec offset for the deflector's center from the zero-point
        in the coordinate system of the data. Default is 0.

        :return:
        :rtype:
        """
        if 'lens_option' in self.settings and 'centroid_init' in \
                self.settings['lens_option']:
            return float(self.settings['lens_option'][
                             'centroid_init'][1])
        else:
            return 0.

    @property
    def deflector_centroid_bound(self):
        """
        Half of the box width to constrain the deflector's centroid.
        Default is 0.5 arcsec.

        :return:
        :rtype:
        """
        if 'lens_option' in self.settings:
            if 'centroid_bound' in self.settings['lens_option']:
                bound = self.settings['lens_option']['centroid_bound']
                if bound is not None:
                    return bound

        return 0.5

    @property
    def band_number(self):
        """
        The number of bands.

        :return:
        :rtype:
        """
        try:
            num = len(self.settings['band'])
        except (KeyError, TypeError, NameError):
            raise ValueError('Name of band(s) not properly specified!')
        else:
            if num < 1:
                raise ValueError("Number of bands less than 1!")
            else:
                return num

    def get_kwargs_model(self):
        """
        Create `kwargs_model`.

        :return:
        :rtype:
        """

        kwargs_model = {
            'lens_model_list': self.get_lens_model_list(),
            'source_light_model_list': self.get_source_light_model_list(),
            'lens_light_model_list': self.get_lens_light_model_list(),
            'point_source_model_list': self.get_point_source_model_list(),
            'index_lens_light_model_list':
                self.get_index_lens_light_model_list(),
            'index_source_light_model_list':
                self.get_index_source_light_model_list(),
        }

        if 'kwargs_model' in self.settings and self.settings['kwargs_model'] \
                is not None:
            for key, value in self.settings['kwargs_model'].items():
                kwargs_model[key] = value

        return kwargs_model

    def get_kwargs_constraints(self):

        """
        Create `kwargs_constraints`.

        :return:
        :rtype:
        """
        joint_source_with_source = []
        num_source_profiles = len(self.get_source_light_model_list())

        if num_source_profiles > 1:
            for n in range(1, num_source_profiles):
                joint_source_with_source.append([
                    0, n, ['center_x', 'center_y']
                ])

        joint_lens_light_with_lens_light = []
        num_lens_light_profiles = len(self.get_lens_light_model_list())
        if num_lens_light_profiles > 1:
            for n in range(1, num_lens_light_profiles):
                joint_lens_light_with_lens_light.append([
                    0, n, ['center_x', 'center_y']
                ])

        joint_source_with_point_source = []
        if len(self.get_point_source_model_list()) > 0 and \
                num_source_profiles > 0:
            for n in range(num_source_profiles):
                joint_source_with_point_source.append([
                    0, n
                ])

        kwargs_constraints = {
            'joint_source_with_source': joint_source_with_source,
            'joint_lens_light_with_lens_light':
                joint_lens_light_with_lens_light,
            'joint_source_with_point_source': joint_source_with_point_source,
            'joint_lens_with_light': [],
            'joint_lens_with_lens': []
        }

        if 'kwargs_constraints' in self.settings and self.settings[
                                            'kwargs_constraints'] is not None:
            for key, value in self.settings['kwargs_constraints'].items():
                kwargs_constraints[key] = value

        return kwargs_constraints

    def get_kwargs_likelihood(self):
        """
        Create `kwargs_likelihood`.

        :return:
        :rtype:
        """
        kwargs_likelihood = {
            'force_no_add_image': False,
            'source_marg': False,
            # 'point_source_likelihood': True,
            # 'position_uncertainty': 0.00004,
            # 'check_solver': False,
            # 'solver_tolerance': 0.001,
            'check_positive_flux': True,
            'check_bounds': True,
            'bands_compute': [True] * self.band_number,
            'image_likelihood_mask_list': self.get_masks()
        }

        return kwargs_likelihood

    def get_masks(self):
        """
        Create masks.

        :return:
        :rtype:
        """
        if 'mask' in self.settings:
            if self.settings['mask'] is not None:
                if 'provided' in self.settings['mask'] \
                        and self.settings['mask']['provided'] is not None:
                    return self.settings['mask']['provided']
                else:
                    masks = []
                    mask_options = deepcopy(self.settings['mask'])

                    for n in range(self.band_number):
                        ra_at_xy_0 = mask_options['ra_at_xy_0'][n]
                        dec_at_xy_0 = mask_options['dec_at_xy_0'][n]
                        transform_pix2angle = np.array(
                            mask_options['transform_matrix'][n]
                        )
                        num_pixel = mask_options['size'][n]
                        radius = mask_options['radius'][n]
                        offset = mask_options['centroid_offset'][n]

                        coords = Coordinates(transform_pix2angle,
                                             ra_at_xy_0, dec_at_xy_0)

                        x_coords, y_coords = coords.coordinate_grid(num_pixel,
                                                                    num_pixel)

                        mask_outer = mask_util.mask_center_2d(
                            self.deflector_center_ra + offset[0],
                            self.deflector_center_dec + offset[1],
                            radius,
                            util.image2array(x_coords),
                            util.image2array(y_coords)
                        )

                        extra_masked_regions = []
                        try:
                            self.settings['mask']['extra_regions']
                        except (NameError, KeyError):
                            pass
                        else:
                            if self.settings['mask']['extra_regions'] is \
                                    not None:
                                for reg in self.settings['mask'][
                                                        'extra_regions'][n]:
                                    extra_masked_regions.append(
                                        mask_util.mask_center_2d(
                                            self.deflector_center_ra + reg[0],
                                            self.deflector_center_dec + reg[1],
                                            reg[2],
                                            util.image2array(x_coords),
                                            util.image2array(y_coords)
                                        )
                                    )

                        mask = 1. - mask_outer

                        for extra_region in extra_masked_regions:
                            mask *= extra_region

                        # sanity check
                        mask[mask >= 1.] = 1.
                        mask[mask <= 0.] = 0.

                        masks.append(util.array2image(mask))

                return masks

        return None

    def get_kwargs_psf_iteration(self):
        """
        Create `kwargs_psf_iteration`.

        :return:
        :rtype:
        """
        if 'psf_iteration' in self.settings['fitting'] \
                and self.settings['fitting']['psf_iteration']:
            kwargs_psf_iteration = {
                'stacking_method': 'median',
                'keep_psf_error_map': True,
                'psf_symmetry': 4,
                'block_center_neighbour': 0.,
                'num_iter': 50,
                'psf_iter_factor': 0.5
            }

            if 'psf_iteration_settings' in self.settings['fitting']:
                for key in ['stacking_method', 'keep_psf_error_map',
                            'psf_symmetry', 'block_center_neighbour',
                            'num_iter', 'psf_iter_factor']:
                    if key in self.settings['fitting'][
                                                    'psf_iteration_settings']:
                        kwargs_psf_iteration[key] = self.settings['fitting'][
                            'psf_iteration_settings'][key]

            return kwargs_psf_iteration
        else:
            return {}

    def get_kwargs_numerics(self):
        """
        Create `kwargs_numerics`.

        :return:
        :rtype:
        """
        try:
            self.settings['kwargs_numerics']['supersampling_factor']
        except (KeyError, NameError, TypeError):
            supersampling_factor = [3] * self.band_number
        else:
            supersampling_factor = deepcopy(self.settings['kwargs_numerics'][
                                                'supersampling_factor'])

            if supersampling_factor is None:
                supersampling_factor = [3] * self.band_number

        kwargs_numerics = []
        for n in range(self.band_number):
            kwargs_numerics.append({
                'supersampling_factor': supersampling_factor[n],
                'supersampling_convolution': False,
                'supersampling_kernel_size': 3,
                'flux_evaluate_indexes': None,
                'point_source_supersampling_factor': 1,
                'compute_mode': 'regular',
            })

        return kwargs_numerics

    def get_lens_model_list(self):
        """
        Return `lens_model_list`.

        :return:
        :rtype:
        """
        if 'lens' in self.settings['model']:
            return self.settings['model']['lens']
        else:
            return []

    def get_source_light_model_list(self):
        """
        Return `source_model_list`.

        :return:
        :rtype:
        """
        if 'source_light' in self.settings['model']:
            combined_source_light_model_list = []
            for i in range(self.band_number):
                combined_source_light_model_list.extend(
                           self.settings['model']['source_light'])
            return combined_source_light_model_list
        else:
            return []

    def get_lens_light_model_list(self):
        """
        Return `lens_light_model_list`.

        :return:
        :rtype:
        """
        if 'lens_light' in self.settings['model']:
            combined_lens_light_model_list = []
            for i in range(self.band_number):
                combined_lens_light_model_list.extend(self.settings[
                                                        'model']['lens_light'])
            return combined_lens_light_model_list

        else:
            return []

    def get_point_source_model_list(self):
        """
        Return `ps_model_list`.

        :return:
        :rtype:
        """
        if 'point_source' in self.settings['model'] and \
                self.settings['model']['point_source'] is not None:
            return self.settings['model']['point_source']
        else:
            return []

    def get_index_lens_light_model_list(self):
        """
        Create list with of index for the different lens light profile
         (for multiple filters)
        """
        if 'lens_light' in self.settings['model']:
            index_lens_light_model_list = []
            index_num = 0
            for i in range(self.band_number):
                single_index_list = []
                for j in range(len(self.settings['model']['lens_light'])):
                    single_index_list.append(index_num)
                    index_num += 1
                index_lens_light_model_list.append(single_index_list)
            return index_lens_light_model_list
        else:
            return []

    def get_index_source_light_model_list(self):
        """
        Create list with of index for the different source light profiles
         (for multiple filters)

        """
        if 'lens_light' in self.settings['model']:
            index_source_light_model_list = []
            index_num = 0
            for i in range(self.band_number):
                single_index_list = []
                for j in range(len(self.settings['model']['source_light'])):
                    single_index_list.append(index_num)
                    index_num += 1
                index_source_light_model_list.append(single_index_list)
            return index_source_light_model_list

        else:
            return []

    def get_lens_model_params(self):
        """
        Create `lens_params`.

        :return:
        :rtype:
        """
        lens_model_list = self.get_lens_model_list()

        fixed = []
        init = []
        sigma = []
        lower = []
        upper = []

        for i, model in enumerate(lens_model_list):
            if model in ['SPEP', 'PEMD']:
                fixed.append({})
                init.append({
                    'center_x': self.deflector_center_ra,
                    'center_y': self.deflector_center_dec,
                    'e1': 0., 'e2': 0.,
                    'gamma': 2., 'theta_E': 1.
                })

                sigma.append({
                    'theta_E': .1, 'e1': 0.1, 'e2': 0.1,
                    'gamma': 0.02, 'center_x': 0.1,
                    'center_y': 0.1
                })

                lower.append({
                    'theta_E': 0.3, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.,
                    'center_x': self.deflector_center_ra
                                    - self.deflector_centroid_bound,
                    'center_y': self.deflector_center_dec
                                    - self.deflector_centroid_bound
                })

                upper.append({
                    'theta_E': 3., 'e1': 0.5, 'e2': 0.5, 'gamma': 3.,
                    'center_x': self.deflector_center_ra
                    + self.deflector_centroid_bound,
                    'center_y': self.deflector_center_dec
                    + self.deflector_centroid_bound
                })

            elif model == 'SHEAR_GAMMA_PSI':
                fixed.append({'ra_0': 0, 'dec_0': 0})
                init.append({'gamma_ext': 0.001, 'psi_ext': 0.0})
                sigma.append({'gamma_ext': 0.001, 'psi_ext': np.pi / 90.})
                lower.append({'gamma_ext': 0.0, 'psi_ext': -np.pi})
                upper.append({'gamma_ext': 0.5, 'psi_ext': np.pi})
            else:
                raise ValueError('{} not implemented as a lens '
                                 'model!'.format(model))

        fixed = self.fill_in_fixed_from_settings('lens', fixed)

        params = [init, sigma, fixed, lower, upper]
        return params

    def get_lens_light_model_params(self):
        """
        Create `lens_light_params`.

        :return:
        :rtype:
        """
        lens_light_model_list = self.get_lens_light_model_list()

        fixed = []
        init = []
        sigma = []
        lower = []
        upper = []

        for n in range(self.band_number):
            for i, model in enumerate(lens_light_model_list):
                if model == 'SERSIC_ELLIPSE':
                    fixed.append({})
                    init.append({
                        'amp': 1., 'R_sersic': .2,
                        'center_x': self.deflector_center_ra,
                        'center_y': self.deflector_center_dec,
                        'e1': 0, 'e2': 0, 'n_sersic': 4.0
                    })
                    sigma.append({
                        'center_x': self.pixel_size[n] / 10.,
                        'center_y': self.pixel_size[n] / 10.,
                        'R_sersic': 0.05, 'n_sersic': 0.5,
                        'e1': 0.1, 'e2': 0.1
                    })

                    lower.append({
                        'e1': -0.5, 'e2': -0.5,
                        'n_sersic': .5, 'R_sersic': 0.1,
                        'center_x': self.deflector_center_ra
                                    - self.deflector_centroid_bound,
                        'center_y': self.deflector_center_dec
                                    - self.deflector_centroid_bound
                    })

                    upper.append({
                        'e1': 0.5, 'e2': 0.5,
                        'n_sersic': 8., 'R_sersic': 5.,
                        'center_x': self.deflector_center_ra
                                        + self.deflector_centroid_bound,
                        'center_y': self.deflector_center_dec
                                        + self.deflector_centroid_bound
                    })
                else:
                    raise ValueError('{} not implemented as a lens light'
                                     'model!'.format(model))

        fixed = self.fill_in_fixed_from_settings('lens_light', fixed)

        params = [init, sigma, fixed, lower, upper]
        return params

    def get_source_light_model_params(self):
        """
        Create `source_params`.

        :return:
        :rtype:
        """
        source_light_model_list = self.get_source_light_model_list()

        fixed = []
        init = []
        sigma = []
        lower = []
        upper = []

        for n in range(self.band_number-1):
            self.settings['source_light_option']['n_max'].extend(
                             self.settings['source_light_option']['n_max'])

        for n in range(self.band_number):
            for i, model in enumerate(source_light_model_list):
                if model == 'SERSIC_ELLIPSE':
                    fixed.append({})

                    init.append({
                        'amp': 1., 'R_sersic': 0.2, 'n_sersic': 1.,
                        'center_x': 0.,
                        'center_y': 0.,
                        'e1': 0., 'e2': 0.
                    })

                    sigma.append({
                        'center_x': 0.5,
                        'center_y': 0.5,
                        'R_sersic': 0.01, 'n_sersic': 0.5,
                        'e1': 0.05, 'e2': 0.05
                    })

                    lower.append({
                        'R_sersic': 0.04, 'n_sersic': .5,
                        'center_y': -2., 'center_x': -2.,
                        'e1': -0.5, 'e2': -0.5
                    })

                    upper.append({
                        'R_sersic': .5, 'n_sersic': 8.,
                        'center_y': 2., 'center_x': 2.,
                        'e1': 0.5, 'e2': 0.5
                    })
                elif model == 'SHAPELETS':
                    fixed.append(
                        {'n_max': self.settings['source_light_option'][
                                                                'n_max'][n]})
                    init.append({'center_x': 0., 'center_y': 0., 'beta': 0.15,
                                 'n_max': self.settings['source_light_option'][
                                                                'n_max'][n]})
                    sigma.append({'center_x': 0.5, 'center_y': 0.5,
                                  'beta': 0.015 / 10., 'n_max': 2})
                    lower.append({'center_x': -1.2, 'center_y': -1.2,
                                  'beta': 0.02, 'n_max': -1})
                    upper.append({'center_x': 1.2, 'center_y': 1.2,
                                  'beta': 0.25, 'n_max': 55})
                else:
                    raise ValueError('{} not implemented as a source light'
                                     'model!'.format(model))

        fixed = self.fill_in_fixed_from_settings('source_light', fixed)

        params = [init, sigma, fixed, lower, upper]
        return params

    def get_point_source_params(self):
        """
        Create `ps_params`.

        :return:
        :rtype:
        """
        point_source_model_list = self.get_point_source_model_list()

        fixed = []
        init = []
        sigma = []
        lower = []
        upper = []

        if len(point_source_model_list) > 0:
            fixed.append({})

            init.append({
                'ra_image': np.array(self.settings['point_source_option'][
                                                                'ra_init']),
                'dec_image': np.array(self.settings['point_source_option'][
                                                                'dec_init']),
            })

            num_point_sources = len(init[0]['ra_image'])
            sigma.append({
                'ra_image': self.maximun_pixel_size[0] *
                            np.ones(num_point_sources),
                'dec_image': self.maximun_pixel_size[0] *
                             np.ones(num_point_sources),
            })

            lower.append({
                'ra_image': init[0]['ra_image'] - self.settings[
                                            'point_source_option']['bound'],
                'dec_image': init[0]['dec_image'] - self.settings[
                                            'point_source_option']['bound'],
            })

            upper.append({
                'ra_image': init[0]['ra_image'] + self.settings[
                                            'point_source_option']['bound'],
                'dec_image': init[0]['dec_image'] + self.settings[
                                            'point_source_option']['bound'],
            })

        params = [init, sigma, fixed, lower, upper]
        return params

    def fill_in_fixed_from_settings(self, component, fixed_list):
        """
        Fill in fixed values from settings for lens, source light and lens
        light.

        :param component: name of component, 'lens', 'lens_light', or
            'source_light'
        :type component: `str`
        :param fixed_list: list of fixed params
        :type fixed_list: `list`
        :return:
        :rtype:
        """
        assert component in ['lens', 'lens_light', 'source_light']
        option_str = component + '_option'

        try:
            self.settings[option_str]['fix']
        except(NameError, KeyError):
            pass
        else:
            if self.settings[option_str]['fix'] is not None:
                for index, param_dict in self.settings[option_str][
                                                                'fix'].items():
                    for key, value in param_dict.items():
                        for n in range(self.band_number):
                            n_lens = len(self.settings['model']['lens_light'])
                            fixed_list[int(index)+n_lens*n][key] = value

        return fixed_list

    def get_kwargs_params(self):
        """
        Create `kwargs_params`.

        :return:
        :rtype:
        """
        kwargs_params = {
            'lens_model': self.get_lens_model_params(),
            'source_model': self.get_source_light_model_params(),
            'lens_light_model': self.get_lens_light_model_params(),
            'point_source_model': self.get_point_source_params(),
            # 'cosmography': []
        }

        return kwargs_params

    def get_psf_supersampled_factor(self):
        """
        Retrieve PSF supersampling factor if specified in the config file.
        :return: PSF supersampling factor
        :rtype: `float`
        """
        try:
            self.settings['psf_supersampled_factor']
        except (NameError, KeyError):
            return 1
        else:
            return self.settings['psf_supersampled_factor']
