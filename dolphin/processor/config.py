# -*- coding: utf-8 -*-
"""This module loads settings from a configuration file."""
__author__ = "ajshajib"

import yaml
import numpy as np
from copy import deepcopy

from lenstronomy.Util.param_util import ellipticity2phi_q
import lenstronomy.Util.util as util
import lenstronomy.Util.mask_util as mask_util
import os

from .data import ImageData
from .files import FileSystem


class Config(object):
    """This class contains the methods to load an read YAML configuration files.

    This is a parent class for other classes that needs to load a configuration file. If
    the file type of the configuration files changes, then only this class needs to be
    modified.
    """

    def __init__(self):
        pass

    @classmethod
    def load_config_from_yaml(cls, file):
        """Load configuration from `file`.

        :return:
        :rtype:
        """
        with open(file, "r") as f:
            settings = yaml.load(f, yaml.FullLoader)

        return settings


class ModelConfig(Config):
    """This class contains the methods to load and interact with modeling settings for a
    particular system."""

    def __init__(
        self, lens_name=None, file_system=None, io_directory=None, settings=None
    ):
        """Initiate a Model Config object. If the file path is given, `settings` will be
        loaded from it. Otherwise, the `settings` can be loaded/reloaded later with the
        `load_settings_from_file` method.

        :param lens_name: name of the lens system
        :type lens_name: `str`
        :param file_system: a FileSystem object
        :type file_system: `FileSystem`
        :param io_directory: path to the input-output directory
        :type io_directory: `str`
        :param settings: a dictionary containing settings. If both `file`
            and `settings` are provided, `file` will be prioritized.
        :type settings: `dict`
        """
        super(ModelConfig, self).__init__()

        if file_system is not None:
            self._file_system = file_system
        elif io_directory is not None:
            self._file_system = FileSystem(io_directory)
        else:
            raise ValueError("Either `file_system` or `io_directory` must be provided!")

        self._lens_name = lens_name

        if settings is not None:
            self.settings = settings
        else:
            self._config_file_path = self.file_system.get_config_file_path(lens_name)

            self._settings_dir = os.path.dirname(self.config_file_path)

            if self._config_file_path is not None:
                self.settings = self.load_config_from_yaml(self._config_file_path)

    @property
    def pixel_size(self):
        """The pixel size.

        :return:
        :rtype:
        """
        if isinstance(self.settings["pixel_size"], float):
            return [self.settings["pixel_size"]] * self.band_number
        else:
            return self.settings["pixel_size"]

    @property
    def deflector_center_ra(self):
        """The RA offset for the deflector's center from the zero-point in the
        coordinate system of the data. Default is 0.

        :return:
        :rtype:
        """
        if (
            "lens_option" in self.settings
            and "centroid_init" in self.settings["lens_option"]
        ):
            return float(self.settings["lens_option"]["centroid_init"][0])
        else:
            return 0.0

    @property
    def deflector_center_dec(self):
        """The dec offset for the deflector's center from the zero-point in the
        coordinate system of the data. Default is 0.

        :return:
        :rtype:
        """
        if (
            "lens_option" in self.settings
            and "centroid_init" in self.settings["lens_option"]
        ):
            return float(self.settings["lens_option"]["centroid_init"][1])
        else:
            return 0.0

    @property
    def deflector_centroid_bound(self):
        """Half of the box width to constrain the deflector's centroid. Default is 0.5
        arcsec.

        :return:
        :rtype:
        """
        if "lens_option" in self.settings:
            if "centroid_bound" in self.settings["lens_option"]:
                bound = self.settings["lens_option"]["centroid_bound"]
                if bound is not None:
                    return bound

        return 0.5

    @property
    def band_number(self):
        """The number of bands.

        :return:
        :rtype:
        """
        try:
            num = len(self.settings["band"])
        except (KeyError, TypeError, NameError):
            raise ValueError("Name of band(s) not properly specified!")
        else:
            if num < 1:
                raise ValueError("Number of bands less than 1!")
            else:
                return num

    def get_kwargs_model(self):
        """Create `kwargs_model`.

        :return:
        :rtype:
        """
        lens_model_list = deepcopy(self.get_lens_model_list())
        if "SIE" in lens_model_list:
            index = lens_model_list.index("SIE")
            lens_model_list[index] = "EPL"

        kwargs_model = {
            "lens_model_list": lens_model_list,
            "source_light_model_list": self.get_source_light_model_list(),
            "lens_light_model_list": self.get_lens_light_model_list(),
            "point_source_model_list": self.get_point_source_model_list(),
            "index_lens_light_model_list": self.get_index_lens_light_model_list(),
            "index_source_light_model_list": self.get_index_source_light_model_list(),
        }

        if (
            "kwargs_model" in self.settings
            and self.settings["kwargs_model"] is not None
        ):
            for key, value in self.settings["kwargs_model"].items():
                kwargs_model[key] = value

        return kwargs_model

    def get_kwargs_constraints(self):
        """Create `kwargs_constraints`.

        :return:
        :rtype:
        """
        joint_source_with_source = []
        num_source_profiles = len(self.get_source_light_model_list())

        if num_source_profiles > 1:
            for n in range(1, num_source_profiles):
                joint_source_with_source.append([0, n, ["center_x", "center_y"]])

        joint_lens_light_with_lens_light = []
        num_lens_light_profiles = len(self.get_lens_light_model_list())
        if num_lens_light_profiles > 1:
            for n in range(1, num_lens_light_profiles):
                joint_lens_light_with_lens_light.append(
                    [0, n, ["center_x", "center_y"]]
                )

        joint_source_with_point_source = []
        if len(self.get_point_source_model_list()) > 0 and num_source_profiles > 0:
            for n in range(num_source_profiles):
                joint_source_with_point_source.append([0, n])

        kwargs_constraints = {
            "joint_source_with_source": joint_source_with_source,
            "joint_lens_light_with_lens_light": joint_lens_light_with_lens_light,
            "joint_source_with_point_source": joint_source_with_point_source,
            "joint_lens_with_light": [],
            "joint_lens_with_lens": [],
        }

        if len(self.get_point_source_model_list()) > 0:
            num_image = len(self.settings["point_source_option"]["ra_init"])
            kwargs_constraints["num_point_source_list"] = [num_image]
            kwargs_constraints["solver_type"] = (
                "PROFILE_SHEAR" if num_image > 2 else "CENTER"
            )

        if (
            "kwargs_constraints" in self.settings
            and self.settings["kwargs_constraints"] is not None
        ):
            for key, value in self.settings["kwargs_constraints"].items():
                kwargs_constraints[key] = value

        return kwargs_constraints

    def get_kwargs_likelihood(self):
        """Create `kwargs_likelihood`.

        :return:
        :rtype:
        """
        kwargs_likelihood = {
            "force_no_add_image": False,
            "source_marg": False,
            # 'point_source_likelihood': True,
            # 'position_uncertainty': 0.00004,
            # 'check_solver': False,
            # 'solver_tolerance': 0.001,
            "check_positive_flux": True,
            "check_bounds": True,
            "bands_compute": [True] * self.band_number,
            "image_likelihood_mask_list": self.get_masks(),
            "prior_lens": [],
            "prior_lens_light": [],
            "prior_ps": [],
            "prior_source": [],
        }

        if (
            "lens_option" in self.settings
            and "gaussian_prior" in self.settings["lens_option"]
        ):
            for index, param_dict in self.settings["lens_option"][
                "gaussian_prior"
            ].items():
                for i in param_dict:
                    prior_param = [index]
                    prior_param.extend(i)
                    kwargs_likelihood["prior_lens"].append(prior_param)

        if (
            "lens_light_option" in self.settings
            and "gaussian_prior" in self.settings["lens_light_option"]
        ):
            for index, param_dict in self.settings["lens_light_option"][
                "gaussian_prior"
            ].items():
                for i in param_dict:
                    prior_param = [index]
                    prior_param.extend(i)
                    kwargs_likelihood["prior_lens_light"].append(prior_param)

        if (
            "source_light_option" in self.settings
            and "gaussian_prior" in self.settings["source_light_option"]
        ):
            for index, param_dict in self.settings["source_light_option"][
                "gaussian_prior"
            ].items():
                for i in param_dict:
                    prior_param = [index]
                    prior_param.extend(i)
                    kwargs_likelihood["prior_source"].append(prior_param)

        if (
            "point_source_option" in self.settings
            and "gaussian_prior" in self.settings["point_source_option"]
        ):
            for index, param_dict in self.settings["point_source_option"][
                "gaussian_prior"
            ].items():
                for i in param_dict:
                    prior_param = [index]
                    prior_param.extend(i)
                    kwargs_likelihood["prior_ps"].append(prior_param)

        use_custom_logL_addition = False

        if "lens_option" in self.settings:
            if (
                "constrain_position_angle_from_lens_light"
                in self.settings["lens_option"]
            ):
                use_custom_logL_addition = True
            if "limit_mass_eccentricity_from_light" in self.settings["lens_option"]:
                use_custom_logL_addition = True

        if "source_light_option" in self.settings:
            if (
                "shapelet_scale_logarithmic_prior"
                in self.settings["source_light_option"]
            ):
                use_custom_logL_addition = True

        if use_custom_logL_addition:
            kwargs_likelihood["custom_logL_addition"] = self.custom_logL_addition

        return kwargs_likelihood

    def custom_logL_addition(
        self,
        kwargs_lens=None,
        kwargs_source=None,
        kwargs_lens_light=None,
        kwargs_ps=None,
        kwargs_special=None,
        kwargs_extinction=None,
        kwargs_tracer_source=None,
    ):
        """Provide additional likelihood terms to be sent to `lenstronomy`.

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

        # Allign pa_light and pa_mass for the lensing galaxy, where pa is the
        # orientation angle of the profile
        if (
            "lens_option" in self.settings
            and "constrain_position_angle_from_lens_light"
            in self.settings["lens_option"]
        ):
            setting_input = self.settings["lens_option"][
                "constrain_position_angle_from_lens_light"
            ]

            if isinstance(setting_input, (bool)) and setting_input:
                max_delta = 15
            elif isinstance(setting_input, (bool)) and not setting_input:
                max_delta = np.nan
            elif isinstance(setting_input, (int, float)):
                max_delta = setting_input
            else:
                raise (
                    TypeError(
                        "constrain_position_angle_from_lens_light \
                                  should be float, int or bool"
                    )
                )

            if not np.isnan(max_delta):
                pa_mass = (
                    ellipticity2phi_q(kwargs_lens[0]["e1"], kwargs_lens[0]["e2"])[0]
                    * 180
                    / np.pi
                )
                pa_light = (
                    ellipticity2phi_q(
                        kwargs_lens_light[0]["e1"], kwargs_lens_light[0]["e2"]
                    )[0]
                    * 180
                    / np.pi
                )

                diff = min(abs(pa_light - pa_mass), 180 - abs(pa_light - pa_mass))
                if diff > np.abs(max_delta):
                    prior += -((diff - np.abs(max_delta)) ** 2) / (1e-3)

        # Ensure q_mass is smaller than q_light for the lensing galaxy, where
        # q is the ratio between the minor axis to the major axis of a profile
        if (
            "lens_option" in self.settings
            and "limit_mass_eccentricity_from_light" in self.settings["lens_option"]
        ):
            setting_input2 = self.settings["lens_option"][
                "limit_mass_eccentricity_from_light"
            ]

            if isinstance(setting_input2, (bool)) and setting_input2:
                max_diff = 0.0
            elif isinstance(setting_input2, (bool)) and not setting_input2:
                max_diff = np.nan
            elif isinstance(setting_input2, (int, float)):
                max_diff = setting_input2
            else:
                raise (
                    TypeError(
                        "limit_mass_eccentricity_from_light \
                                  should be float, int or bool"
                    )
                )
            q_mass = ellipticity2phi_q(kwargs_lens[0]["e1"], kwargs_lens[0]["e2"])[1]
            q_light = ellipticity2phi_q(
                kwargs_lens_light[0]["e1"], kwargs_lens_light[0]["e2"]
            )[1]
            if not np.isnan(max_diff):
                q_mass = ellipticity2phi_q(kwargs_lens[0]["e1"], kwargs_lens[0]["e2"])[
                    1
                ]
                q_light = ellipticity2phi_q(
                    kwargs_lens_light[0]["e1"], kwargs_lens_light[0]["e2"]
                )[1]
                diff = q_light - q_mass
                if diff > max_diff:
                    prior += -((diff - max_diff) ** 2) / (1e-4)

        # Provide logarithmic_prior on the source light profile beta param
        if (
            "source_light_option" in self.settings
            and "shapelet_scale_logarithmic_prior"
            in self.settings["source_light_option"]
        ):
            setting_input3 = self.settings["source_light_option"][
                "shapelet_scale_logarithmic_prior"
            ]
            if setting_input3:
                for i, model in enumerate(self.settings["model"]["source_light"]):
                    if model == "SHAPELETS":
                        beta = kwargs_source[i]["beta"]
                        prior += -np.log(beta)

        return prior

    @staticmethod
    def load_mask(mask_file_path):
        """Load mask from file.

        :param mask_file_path: path to the mask file
        :type mask_file_path: `str`
        :return: mask
        :rtype: `numpy.ndarray`
        """

    def get_masks(self):
        """Create masks.

        :return:
        :rtype:
        """
        if "mask" in self.settings:
            if self.settings["mask"] is not None:
                if (
                    "provided" in self.settings["mask"]
                    and self.settings["mask"]["provided"]
                ):
                    masks = []
                    for n in range(self.band_number):
                        masks.append(
                            self._file_system.load_mask(
                                self.lens_name, self.settings["band"][n]
                            )
                        )
                else:
                    masks = []
                    mask_options = deepcopy(self.settings["mask"])

                    for n in range(self.band_number):
                        band_name = self.settings["band"][n]

                        # go to ../data/system_name from settings_dir
                        image_file_path = self._file_system.get_image_file_path(
                            self.lens_name, band_name
                        )
                        # os.path.join(
                        #     self._settings_dir,
                        #     f"../data/{self.settings['lens_name']}/image_{self.settings['lens_name']}_{self.settings['band'][n]}.h5",
                        # )

                        image_data = ImageData(image_file_path)
                        coordinate_system = image_data.get_image_coordinate_system()
                        num_pixel = image_data.get_image_size()

                        # ra_at_xy_0 = mask_options["ra_at_xy_0"][n]
                        # dec_at_xy_0 = mask_options["dec_at_xy_0"][n]
                        # transform_pix2angle = np.array(
                        #     mask_options["transform_matrix"][n]
                        # )
                        # num_pixel = mask_options["size"][n]
                        # coords = Coordinates(
                        #     transform_pix2angle, ra_at_xy_0, dec_at_xy_0
                        # )

                        offset = mask_options["centroid_offset"][n]

                        x_coords, y_coords = coordinate_system.coordinate_grid(
                            num_pixel, num_pixel
                        )

                        if "radius" in mask_options:
                            radius = mask_options["radius"][n]
                            mask = mask_util.mask_azimuthal(
                                util.image2array(x_coords),
                                util.image2array(y_coords),
                                self.deflector_center_ra + offset[0],
                                self.deflector_center_dec + offset[1],
                                radius,
                            )
                        elif (
                            "a" in mask_options
                            and "b" in mask_options
                            and "angle" in mask_options
                        ):
                            a = mask_options["a"][n]
                            b = mask_options["b"][n]
                            angle = mask_options["angle"][n]
                            mask = mask_util.mask_ellipse(
                                util.image2array(x_coords),
                                util.image2array(y_coords),
                                self.deflector_center_ra + offset[0],
                                self.deflector_center_dec + offset[1],
                                a,
                                b,
                                angle,
                            )
                        else:
                            raise ValueError("Mask shape not properly specified!")

                        extra_masked_regions = []
                        try:
                            self.settings["mask"]["extra_regions"]
                        except (NameError, KeyError):
                            pass
                        else:
                            if self.settings["mask"]["extra_regions"] is not None:
                                for reg in self.settings["mask"]["extra_regions"][n]:
                                    extra_masked_regions.append(
                                        1
                                        - mask_util.mask_azimuthal(
                                            util.image2array(x_coords),
                                            util.image2array(y_coords),
                                            self.deflector_center_ra + reg[0],
                                            self.deflector_center_dec + reg[1],
                                            reg[2],
                                        )
                                    )

                        for extra_region in extra_masked_regions:
                            mask *= extra_region
                        # Mask Edge Pixels
                        try:
                            self.settings["mask"]["mask_edge_pixels"]
                        except (NameError, KeyError):
                            pass
                        else:
                            border_length = self.settings["mask"]["mask_edge_pixels"][n]
                            if border_length > 0:
                                edge_mask = 0 * np.ones(
                                    (num_pixel, num_pixel), dtype=int
                                )

                                edge_mask[
                                    border_length:-border_length,
                                    border_length:-border_length,
                                ] = 1
                                edge_mask = (edge_mask.flatten()).tolist()
                            elif border_length == 0:
                                edge_mask = 1 * np.ones(
                                    (num_pixel, num_pixel), dtype=int
                                )
                                edge_mask = (edge_mask.flatten()).tolist()

                            mask *= edge_mask
                        # Add custom Mask
                        try:
                            self.settings["mask"]["custom_mask"]
                        except (NameError, KeyError):
                            pass
                        else:
                            if self.settings["mask"]["custom_mask"][n] is not None:
                                provided_mask = self.settings["mask"]["custom_mask"][n]
                                provided_mask = np.array(provided_mask)
                                # make sure that mask consist of only 0 and 1
                                provided_mask[provided_mask > 0.0] = 1.0
                                provided_mask[provided_mask <= 0.0] = 0.0
                                mask *= provided_mask

                        # sanity check
                        mask[mask >= 1.0] = 1.0
                        mask[mask <= 0.0] = 0.0

                        masks.append(util.array2image(mask))

                return masks

        return None

    def get_kwargs_psf_iteration(self):
        """Create `kwargs_psf_iteration`.

        :return:
        :rtype:
        """
        if (
            "psf_iteration" in self.settings["fitting"]
            and self.settings["fitting"]["psf_iteration"]
        ):
            kwargs_psf_iteration = {
                "stacking_method": "median",
                "keep_psf_error_map": True,
                "psf_symmetry": 4,
                "block_center_neighbour": 0.0,
                "num_iter": 50,
                "psf_iter_factor": 0.5,
            }

            if "psf_iteration_settings" in self.settings["fitting"]:
                for key in [
                    "stacking_method",
                    "keep_psf_error_map",
                    "psf_symmetry",
                    "block_center_neighbour",
                    "num_iter",
                    "psf_iter_factor",
                ]:
                    if key in self.settings["fitting"]["psf_iteration_settings"]:
                        kwargs_psf_iteration[key] = self.settings["fitting"][
                            "psf_iteration_settings"
                        ][key]

            return kwargs_psf_iteration
        else:
            return {}

    def get_kwargs_numerics(self):
        """Create `kwargs_numerics`.

        :return:
        :rtype:
        """
        try:
            self.settings["kwargs_numerics"]["supersampling_factor"]
        except (KeyError, NameError, TypeError):
            supersampling_factor = [3] * self.band_number
        else:
            supersampling_factor = deepcopy(
                self.settings["kwargs_numerics"]["supersampling_factor"]
            )

            if supersampling_factor is None:
                supersampling_factor = [3] * self.band_number

        kwargs_numerics = []
        for n in range(self.band_number):
            kwargs_numerics.append(
                {
                    "supersampling_factor": supersampling_factor[n],
                    "supersampling_convolution": False,
                    "supersampling_kernel_size": 3,
                    "flux_evaluate_indexes": None,
                    "point_source_supersampling_factor": 1,
                    "compute_mode": "regular",
                }
            )

        return kwargs_numerics

    def get_lens_model_list(self):
        """Return `lens_model_list`.

        :return:
        :rtype:
        """
        if "lens" in self.settings["model"]:
            return self.settings["model"]["lens"]
        else:
            return []

    def get_source_light_model_list(self):
        """Return `source_model_list`.

        :return:
        :rtype:
        """
        if "source_light" in self.settings["model"]:
            return self.settings["model"]["source_light"]
        else:
            return []

    def get_lens_light_model_list(self):
        """Return `lens_light_model_list`.

        :return:
        :rtype:
        """
        if "lens_light" in self.settings["model"]:
            return self.settings["model"]["lens_light"]
        else:
            return []

    def get_point_source_model_list(self):
        """Return `ps_model_list`.

        :return:
        :rtype:
        """
        if (
            "point_source" in self.settings["model"]
            and self.settings["model"]["point_source"] is not None
        ):
            return self.settings["model"]["point_source"]
        else:
            return []

    def get_index_lens_light_model_list(self):
        """Create list with of index for the different lens light profile (for multiple
        filters)"""
        if "lens_light" in self.settings["model"]:
            if self.band_number == 1:
                index_list = [[]]
                for k, model in enumerate(self.settings["model"]["lens_light"]):
                    index_list[0].append(k)
                return index_list
            else:
                if "lens_light_band_indices" in self.settings["model"]:
                    index_list = [[] for _ in range(self.band_number)]
                    for i, model in enumerate(
                        self.settings["model"]["lens_light_band_indices"]
                    ):
                        index_list[model].append(i)
                    for k in index_list:
                        assert k != [], "One of the bands have no lens light"
                    return index_list
                else:
                    raise ValueError(
                        'Missing "lens_light_band_indices" in the settings file!'
                    )
        else:
            return []

    def get_index_source_light_model_list(self):
        """Create list with of index for the different source light profiles (for
        multiple filters)"""
        if "source_light" in self.settings["model"]:
            if self.band_number == 1:
                index_list = [[]]
                for k, model in enumerate(self.settings["model"]["source_light"]):
                    index_list[0].append(k)
                return index_list
            else:
                if "source_light_band_indices" in self.settings["model"]:
                    index_list = [[] for _ in range(self.band_number)]
                    for i, model in enumerate(
                        self.settings["model"]["source_light_band_indices"]
                    ):
                        index_list[model].append(i)
                    for k in index_list:
                        assert k != [], "One of the bands have no source light"
                    return index_list
                else:
                    raise ValueError(
                        'Missing "source_light_band_indices" ' "in the settings file!"
                    )
        else:
            return []

    def get_lens_model_params(self):
        """Create `lens_params`.

        :return:
        :rtype:
        """
        lens_model_list = self.get_lens_model_list()

        fixed = []
        init = []
        sigma = []
        lower = []
        upper = []

        for model in lens_model_list:
            if model in ["SPEP", "PEMD", "EPL", "SIE"]:
                if model == "SIE":
                    fixed.append({"gamma": 2.0})
                else:
                    fixed.append({})
                init.append(
                    {
                        "center_x": self.deflector_center_ra,
                        "center_y": self.deflector_center_dec,
                        "e1": 0.0,
                        "e2": 0.0,
                        "gamma": 2.0,
                        "theta_E": 1.0,
                    }
                )

                sigma.append(
                    {
                        "theta_E": 0.1,
                        "e1": 0.01,
                        "e2": 0.01,
                        "gamma": 0.02,
                        "center_x": 0.1,
                        "center_y": 0.1,
                    }
                )

                lower.append(
                    {
                        "theta_E": 0.3,
                        "e1": -0.5,
                        "e2": -0.5,
                        "gamma": 1.3,
                        "center_x": self.deflector_center_ra
                        - self.deflector_centroid_bound,
                        "center_y": self.deflector_center_dec
                        - self.deflector_centroid_bound,
                    }
                )

                upper.append(
                    {
                        "theta_E": 3.0,
                        "e1": 0.5,
                        "e2": 0.5,
                        "gamma": 2.8,
                        "center_x": self.deflector_center_ra
                        + self.deflector_centroid_bound,
                        "center_y": self.deflector_center_dec
                        + self.deflector_centroid_bound,
                    }
                )

            elif model == "SHEAR_GAMMA_PSI":
                fixed.append({"ra_0": 0, "dec_0": 0})
                init.append({"gamma_ext": 0.05, "psi_ext": 0.0})
                sigma.append({"gamma_ext": 0.05, "psi_ext": np.pi / 90.0})
                lower.append({"gamma_ext": 0.0, "psi_ext": -np.pi})
                upper.append({"gamma_ext": 0.5, "psi_ext": np.pi})
            else:
                raise ValueError("{} not implemented as a lens " "model!".format(model))

        fixed = self.fill_in_fixed_from_settings("lens", fixed)

        params = [init, sigma, fixed, lower, upper]
        return params

    def get_lens_light_model_params(self):
        """Create `lens_light_params`.

        :return:
        :rtype:
        """
        lens_light_model_list = self.get_lens_light_model_list()

        fixed = []
        init = []
        sigma = []
        lower = []
        upper = []

        for i, model in enumerate(lens_light_model_list):
            if model == "SERSIC_ELLIPSE":
                fixed.append({})
                init.append(
                    {
                        "amp": 1.0,
                        "R_sersic": 0.2,
                        "center_x": self.deflector_center_ra,
                        "center_y": self.deflector_center_dec,
                        "e1": 0,
                        "e2": 0,
                        "n_sersic": 4.0,
                    }
                )
                sigma.append(
                    {
                        "center_x": np.max(self.pixel_size) / 10.0,
                        "center_y": np.max(self.pixel_size) / 10.0,
                        "R_sersic": 0.05,
                        "n_sersic": 0.5,
                        "e1": 0.01,
                        "e2": 0.01,
                    }
                )

                lower.append(
                    {
                        "e1": -0.5,
                        "e2": -0.5,
                        "n_sersic": 0.5,
                        "R_sersic": 0.1,
                        "center_x": self.deflector_center_ra
                        - self.deflector_centroid_bound,
                        "center_y": self.deflector_center_dec
                        - self.deflector_centroid_bound,
                    }
                )

                upper.append(
                    {
                        "e1": 0.5,
                        "e2": 0.5,
                        "n_sersic": 8.0,
                        "R_sersic": 5.0,
                        "center_x": self.deflector_center_ra
                        + self.deflector_centroid_bound,
                        "center_y": self.deflector_center_dec
                        + self.deflector_centroid_bound,
                    }
                )
            else:
                raise ValueError(
                    "{} not implemented as a lens light" "model!".format(model)
                )

        fixed = self.fill_in_fixed_from_settings("lens_light", fixed)

        params = [init, sigma, fixed, lower, upper]
        return params

    def get_source_light_model_params(self):
        """Create `source_params`.

        :return:
        :rtype:
        """
        source_light_model_list = self.get_source_light_model_list()

        fixed = []
        init = []
        sigma = []
        lower = []
        upper = []

        shapelets_index = 0
        for i, model in enumerate(source_light_model_list):
            if model == "SERSIC_ELLIPSE":
                fixed.append({})

                init.append(
                    {
                        "amp": 1.0,
                        "R_sersic": 0.2,
                        "n_sersic": 1.0,
                        "center_x": 0.0,
                        "center_y": 0.0,
                        "e1": 0.0,
                        "e2": 0.0,
                    }
                )

                sigma.append(
                    {
                        "center_x": 0.5,
                        "center_y": 0.5,
                        "R_sersic": 0.01,
                        "n_sersic": 0.5,
                        "e1": 0.05,
                        "e2": 0.05,
                    }
                )

                lower.append(
                    {
                        "R_sersic": 0.04,
                        "n_sersic": 0.5,
                        "center_y": -2.0,
                        "center_x": -2.0,
                        "e1": -0.5,
                        "e2": -0.5,
                    }
                )

                upper.append(
                    {
                        "R_sersic": 0.5,
                        "n_sersic": 8.0,
                        "center_y": 2.0,
                        "center_x": 2.0,
                        "e1": 0.5,
                        "e2": 0.5,
                    }
                )
            elif model == "SHAPELETS":
                fixed.append(
                    {
                        "n_max": self.settings["source_light_option"]["n_max"][
                            shapelets_index
                        ]
                    }
                )
                init.append(
                    {
                        "center_x": 0.0,
                        "center_y": 0.0,
                        "beta": 0.10,
                        "n_max": self.settings["source_light_option"]["n_max"][
                            shapelets_index
                        ],
                    }
                )
                sigma.append(
                    {"center_x": 0.5, "center_y": 0.5, "beta": 0.010 / 10.0, "n_max": 2}
                )
                lower.append(
                    {"center_x": -1.2, "center_y": -1.2, "beta": 0.02, "n_max": -1}
                )
                upper.append(
                    {"center_x": 1.2, "center_y": 1.2, "beta": 0.20, "n_max": 55}
                )
                shapelets_index += 1
            else:
                raise ValueError(
                    "{} not implemented as a source light" "model!".format(model)
                )

        fixed = self.fill_in_fixed_from_settings("source_light", fixed)

        params = [init, sigma, fixed, lower, upper]
        return params

    def get_point_source_params(self):
        """Create `ps_params`.

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

            init.append(
                {
                    "ra_image": np.array(
                        self.settings["point_source_option"]["ra_init"]
                    ),
                    "dec_image": np.array(
                        self.settings["point_source_option"]["dec_init"]
                    ),
                }
            )

            num_point_sources = len(init[0]["ra_image"])
            sigma.append(
                {
                    "ra_image": np.max(self.pixel_size) * np.ones(num_point_sources),
                    "dec_image": np.max(self.pixel_size) * np.ones(num_point_sources),
                }
            )

            lower.append(
                {
                    "ra_image": init[0]["ra_image"]
                    - self.settings["point_source_option"]["bound"],
                    "dec_image": init[0]["dec_image"]
                    - self.settings["point_source_option"]["bound"],
                }
            )

            upper.append(
                {
                    "ra_image": init[0]["ra_image"]
                    + self.settings["point_source_option"]["bound"],
                    "dec_image": init[0]["dec_image"]
                    + self.settings["point_source_option"]["bound"],
                }
            )

        params = [init, sigma, fixed, lower, upper]
        return params

    def fill_in_fixed_from_settings(self, component, fixed_list):
        """Fill in fixed values from settings for lens, source light and lens light.

        :param component: name of component, 'lens', 'lens_light', or
            'source_light'
        :type component: `str`
        :param fixed_list: list of fixed params
        :type fixed_list: `list`
        :return:
        :rtype:
        """
        assert component in ["lens", "lens_light", "source_light"]
        option_str = component + "_option"

        try:
            self.settings[option_str]["fix"]
        except (NameError, KeyError):
            pass
        else:
            if self.settings[option_str]["fix"] is not None:
                for index, param_dict in self.settings[option_str]["fix"].items():
                    for key, value in param_dict.items():
                        fixed_list[int(index)][key] = value
        return fixed_list

    def get_kwargs_params(self):
        """Create `kwargs_params`.

        :return:
        :rtype:
        """
        kwargs_params = {
            "lens_model": self.get_lens_model_params(),
            "source_model": self.get_source_light_model_params(),
            "lens_light_model": self.get_lens_light_model_params(),
            "point_source_model": self.get_point_source_params(),
            # 'cosmography': []
        }

        return kwargs_params

    def get_psf_supersampled_factor(self):
        """Retrieve PSF supersampling factor if specified in the config file.

        :return: PSF supersampling factor
        :rtype: `float`
        """
        try:
            self.settings["psf_supersampled_factor"]
        except (NameError, KeyError):
            return 1
        else:
            return self.settings["psf_supersampled_factor"]
