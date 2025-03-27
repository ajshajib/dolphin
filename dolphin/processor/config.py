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

    def __init__(self, lens_name, file_system=None, io_directory=None, settings=None):
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

        self._lens_name = lens_name

        if settings is not None:
            self.settings = settings
        else:
            self._config_file_path = self._file_system.get_config_file_path(lens_name)

            self._settings_dir = os.path.dirname(self._config_file_path)
            self.settings = self.load_config_from_yaml(self._config_file_path)

        assert self.settings["lens_name"] == self._lens_name

    @property
    def lens_name(self):
        """The name of the lens system.

        :return:
        :rtype:
        """
        return self._lens_name

    @property
    def pixel_size(self):
        """The pixel size.

        :return:
        :rtype:
        """
        # if isinstance(self.settings["pixel_size"], float):
        #     return [self.settings["pixel_size"]] * self.band_number
        # else:
        #     return self.settings["pixel_size"]
        if "pixel_size" not in self.settings:
            pixel_size = []

            for band in self.settings["band"]:
                image_data = self.get_image_data(band)
                pixel_size.append(image_data.get_image_pixel_scale())

            self.settings["pixel_size"] = pixel_size

        return self.settings["pixel_size"]

    def get_image_data(self, band):
        """Get image data.

        :param band: name of the band
        :type band: `str`
        :return: image data
        :rtype: `ImageData`
        """
        return ImageData(self._file_system.get_image_file_path(self.lens_name, band))

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
    def number_of_bands(self):
        """The number of bands.

        :return:
        :rtype:
        """
        return len(self.settings["band"])

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
        joint_source_with_source, num_source_profiles = (
            self.get_joint_source_with_source()
        )

        joint_lens_light_with_lens_light = self.get_joint_lens_light_with_lens_light()

        joint_source_with_point_source = self.get_joint_source_with_point_source(
            num_source_profiles
        )

        joint_lens_with_light = self.get_joint_lens_with_light()

        kwargs_constraints = {
            "joint_source_with_source": joint_source_with_source,
            "joint_lens_light_with_lens_light": joint_lens_light_with_lens_light,
            "joint_source_with_point_source": joint_source_with_point_source,
            "joint_lens_with_light": joint_lens_with_light,
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

    def get_joint_lens_with_light(self):
        """Create `joint_lens_with_light`."""
        _, lens_light_satellite_flags = self.get_lens_light_model_list_with_flags()
        _, lens_satellite_flags = self.get_lens_model_list_with_flags()

        joint_lens_with_light = []

        if (np.array(lens_light_satellite_flags) > -1).any():
            for i, flag in enumerate(lens_light_satellite_flags):
                if flag > -1:
                    lens_sat_index = lens_satellite_flags.index(flag)
                    joint_lens_with_light.append(
                        [i, lens_sat_index, ["center_x", "center_y"]]
                    )

        return joint_lens_with_light

    def get_joint_source_with_point_source(self, num_source_profiles):
        """Create `joint_source_with_point_source`."""
        joint_source_with_point_source = []
        if len(self.get_point_source_model_list()) > 0 and num_source_profiles > 0:
            for n in range(num_source_profiles):
                joint_source_with_point_source.append([0, n])
        return joint_source_with_point_source

    def get_joint_lens_light_with_lens_light(self):
        """Create `joint_lens_light_with_lens_light`."""
        joint_lens_light_with_lens_light = []
        lens_light_model_list = self.get_lens_light_model_list()

        num_lens_light_profile_central = len(self.settings["model"]["lens_light"])

        if num_lens_light_profile_central > 1:
            for n in range(1, num_lens_light_profile_central * self.number_of_bands):
                joint_lens_light_with_lens_light.append(
                    [0, n, ["center_x", "center_y"]]
                )

        # Join Sersic ellipticities in multiband fitting
        if self.number_of_bands > 1:
            for i in range(num_lens_light_profile_central):
                model = lens_light_model_list[i]
                if "SERSIC" in model:
                    join_list = ["n_sersic"]
                    if "ELLIPSE" in model:
                        join_list += ["e1", "e2"]
                    joint_lens_light_with_lens_light.append(
                        [
                            i,
                            i + num_lens_light_profile_central,
                            join_list,
                        ]
                    )

        if self.number_of_bands > 1 and self.num_satellites > 0:
            for i in range(self.num_satellites):
                model = lens_light_model_list[i + num_lens_light_profile_central]

                join_list = ["center_x", "center_y", "n_sersic"]
                if "ELLIPSE" in model:
                    join_list += ["e1", "e2"]
                joint_lens_light_with_lens_light.append(
                    [
                        i + num_lens_light_profile_central * self.number_of_bands,
                        i
                        + num_lens_light_profile_central * self.number_of_bands
                        + self.num_satellites,
                        join_list,
                    ]
                )

        return joint_lens_light_with_lens_light

    def get_joint_source_with_source(self):
        """Create `joint_source_with_source`."""
        joint_source_with_source = []
        num_source_profiles = len(self.get_source_light_model_list())

        if num_source_profiles > 1:
            for n in range(1, num_source_profiles):
                joint_source_with_source.append([0, n, ["center_x", "center_y"]])

        # Join Sersic ellipticities in multiband fitting
        num_bands = self.number_of_bands
        if num_bands > 1:
            num_source_profile_single_band = int(num_source_profiles / num_bands)

            for i in range(num_source_profile_single_band):
                model = self.get_source_light_model_list()[i]
                if "SERSIC" in model:
                    join_list = ["n_sersic"]
                    if "ELLIPSE" in model:
                        join_list += ["e1", "e2"]
                    joint_source_with_source.append(
                        [
                            i,
                            i + num_source_profile_single_band,
                            join_list,
                        ]
                    )

        return joint_source_with_source, num_source_profiles

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
            "bands_compute": [True] * self.number_of_bands,
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
            if any(
                key in self.settings["lens_option"]
                for key in ["limit_mass_pa_from_light", "limit_mass_q_from_light"]
            ):
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

        # Limit the difference between pa_light and pa_mass for the deflector, where pa is the
        # position angle of the major axis
        if (
            "lens_option" in self.settings
            and "limit_mass_pa_from_light" in self.settings["lens_option"]
        ):
            max_mass_pa_difference = self.settings["lens_option"][
                "limit_mass_pa_from_light"
            ]

            if not isinstance(max_mass_pa_difference, (int, float)):
                raise ValueError(
                    "The value for limit_mass_pa_from_light should be a number!"
                )

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
            if diff > np.abs(max_mass_pa_difference):
                prior += -((diff - np.abs(max_mass_pa_difference)) ** 2) / 1e-3

        # Limit the difference between q_light and q_mass for the deflector, where q is the axis
        # ratio of the elliptical profile
        if (
            "lens_option" in self.settings
            and "limit_mass_q_from_light" in self.settings["lens_option"]
        ):
            max_mass_q_difference = self.settings["lens_option"][
                "limit_mass_q_from_light"
            ]

            if not isinstance(max_mass_q_difference, (int, float)):
                raise ValueError(
                    "The value for limit_mass_q_from_light should be a number!"
                )

            q_mass = ellipticity2phi_q(kwargs_lens[0]["e1"], kwargs_lens[0]["e2"])[1]
            q_light = ellipticity2phi_q(
                kwargs_lens_light[0]["e1"], kwargs_lens_light[0]["e2"]
            )[1]

            q_mass = ellipticity2phi_q(kwargs_lens[0]["e1"], kwargs_lens[0]["e2"])[1]
            q_light = ellipticity2phi_q(
                kwargs_lens_light[0]["e1"], kwargs_lens_light[0]["e2"]
            )[1]
            diff = q_light - q_mass
            if diff > max_mass_q_difference:
                prior += -((diff - max_mass_q_difference) ** 2) / 1e-4

        # Provide logarithmic_prior on the source light profile beta param
        if (
            "source_light_option" in self.settings
            and "shapelet_scale_logarithmic_prior"
            in self.settings["source_light_option"]
        ):
            if self.settings["source_light_option"]["shapelet_scale_logarithmic_prior"]:
                for i, model in enumerate(self.get_source_light_model_list()):
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
        if "mask" in self.settings and self.settings["mask"] is not None:
            if (
                "provided" in self.settings["mask"]
                and self.settings["mask"]["provided"]
            ):
                masks = []
                for n in range(self.number_of_bands):
                    masks.append(
                        self._file_system.load_mask(
                            self.settings["lens_name"], self.settings["band"][n]
                        )
                    )
            else:
                masks = []
                mask_options = deepcopy(self.settings["mask"])

                for n in range(self.number_of_bands):
                    band_name = self.settings["band"][n]

                    image_data = self.get_image_data(band_name)
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
                    if "mask_edge_pixels" in self.settings["mask"]:
                        border_length = self.settings["mask"]["mask_edge_pixels"][n]
                        if border_length > 0:
                            edge_mask = 0 * np.ones((num_pixel, num_pixel), dtype=int)

                            edge_mask[
                                border_length:-border_length,
                                border_length:-border_length,
                            ] = 1
                            edge_mask = (edge_mask.flatten()).tolist()
                        elif border_length == 0:
                            edge_mask = 1 * np.ones((num_pixel, num_pixel), dtype=int)
                            edge_mask = (edge_mask.flatten()).tolist()

                        mask *= edge_mask

                    # sanity check
                    mask[mask >= 1.0] = 1.0
                    mask[mask <= 0.0] = 0.0

                    masks.append(util.array2image(mask))

            return masks
        else:
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
                "keep_psf_variance_map": True,
                "psf_symmetry": 4,
                "block_center_neighbour": 0.0,
                "num_iter": 50,
                "psf_iter_factor": 0.5,
            }

            if "psf_iteration_settings" in self.settings["fitting"]:
                for key in self.settings["fitting"]["psf_iteration_settings"].keys():
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
            supersampling_factor = [3] * self.number_of_bands
        else:
            supersampling_factor = deepcopy(
                self.settings["kwargs_numerics"]["supersampling_factor"]
            )

            if supersampling_factor is None:
                supersampling_factor = [3] * self.number_of_bands

        kwargs_numerics = []
        for n in range(self.number_of_bands):
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

    @property
    def num_satellites(self):
        """Check if the system has satellites.

        :return:
        :rtype:
        """
        if "satellites" not in self.settings:
            return 0
        else:
            return len(self.settings["satellites"]["centroid_init"])

    def get_lens_model_list(self):
        """Return `lens_model_list`."""
        return self.get_lens_model_list_with_flags()[0]

    def get_lens_model_list_with_flags(self):
        """Return `lens_model_list` and `satellite_flags`.

        :return: lens_model_list, satellite_flags
        :rtype: list, list
        """
        lens_model_list = []
        satellite_flag = []

        if "lens" in self.settings["model"]:
            lens_model_list = [model for model in self.settings["model"]["lens"]]
            satellite_flag = [-1 for _ in range(len(lens_model_list))]

            if self.num_satellites > 0:
                if "is_elliptical" not in self.settings["satellites"]:
                    is_elliptical = [False] * self.num_satellites
                else:
                    is_elliptical = self.settings["satellites"]["is_elliptical"]
                for i, yes in enumerate(is_elliptical):
                    if yes:
                        lens_model_list.append("SIE")
                    else:
                        lens_model_list.append("SIS")
                    satellite_flag.append(i)

        return lens_model_list, satellite_flag

    def get_source_light_model_list(self):
        """Return `source_model_list`.

        :return:
        :rtype:
        """
        source_light_model_list = []

        if "source_light" in self.settings["model"]:
            # return self.settings["model"]["source_light"]
            for i in range(self.number_of_bands):
                source_light_model_list += self.settings["model"]["source_light"]

        return source_light_model_list

    def get_lens_light_model_list(self):
        """Return `lens_light_model_list`."""
        return self.get_lens_light_model_list_with_flags()[0]

    def get_lens_light_model_list_with_flags(self):
        """Return `lens_light_model_list` and `satellite_flags`.

        :return: lens_light_model_list, satellite_flags
        :rtype: list, list
        """
        lens_light_model_list = []
        satellite_flag = []

        if "lens_light" in self.settings["model"]:
            for i in range(self.number_of_bands):
                lens_light_model_list += [
                    model for model in self.settings["model"]["lens_light"]
                ]
                satellite_flag += [-1 for model in self.settings["model"]["lens_light"]]

            if self.num_satellites > 0:
                if "is_elliptical" not in self.settings["satellites"]:
                    is_elliptical = [False] * self.num_satellites
                else:
                    is_elliptical = self.settings["satellites"]["is_elliptical"]

                for i in range(self.number_of_bands):
                    for j, yes in enumerate(is_elliptical):
                        if yes:
                            lens_light_model_list.append("SERSIC_ELLIPSE")
                        else:
                            lens_light_model_list.append("SERSIC")
                        satellite_flag.append(j)

        return lens_light_model_list, satellite_flag

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

    def get_index_list(self, light_type="lens_light"):
        """Create list with of index for the different light profiles (for multiple
        filters)"""
        index_list = []

        if light_type in self.settings["model"]:
            index_list = [[] for _ in range(self.number_of_bands)]
            counter = 0
            n = len(self.settings["model"][light_type])

            for num_band in range(self.number_of_bands):
                for _ in range(n):
                    index_list[num_band].append(counter)
                    counter += 1

            if light_type == "lens_light":
                for i in range(self.num_satellites):
                    for num_band in range(self.number_of_bands):
                        index_list[num_band].append(counter)
                        counter += 1

        return index_list

    def get_index_lens_light_model_list(self):
        """Create list with of index for the different lens light profile (for multiple
        filters)"""
        index_list = self.get_index_list("lens_light")

        return index_list

    def get_index_source_light_model_list(self):
        """Create list with of index for the different source light profiles (for
        multiple filters)"""
        return self.get_index_list("source_light")

    def get_lens_model_params(self):
        """Create `lens_params`.

        :return:
        :rtype:
        """
        lens_model_list, satellite_flags = self.get_lens_model_list_with_flags()

        fixed = []
        init = []
        sigma = []
        lower = []
        upper = []

        for i, model in enumerate(lens_model_list):
            if satellite_flags[i] == -1:
                # central deflector
                bound = self.deflector_centroid_bound
                center_x = self.deflector_center_ra
                center_y = self.deflector_center_dec
                try:
                    theta_E_init = self.settings["guess_params"]["lens"][0]["theta_E"]
                except (NameError, KeyError):
                    theta_E_init = 1.0
            else:
                # satellite
                bound = self.settings["satellites"]["centroid_bound"]
                center_x = self.settings["satellites"]["centroid_init"][
                    satellite_flags[i]
                ][0]
                center_y = self.settings["satellites"]["centroid_init"][
                    satellite_flags[i]
                ][1]
                theta_E_init = 0.1

            if model in ["SPEP", "PEMD", "EPL", "SIE"]:
                if model == "SIE":
                    fixed.append({"gamma": 2.0})
                else:
                    fixed.append({})
                init.append(
                    {
                        "center_x": center_x,
                        "center_y": center_y,
                        "e1": 0.0,
                        "e2": 0.0,
                        "gamma": 2.0,
                        "theta_E": theta_E_init,
                    }
                )
                sigma.append(
                    {
                        "theta_E": 0.1,
                        "e1": 0.01,
                        "e2": 0.01,
                        "gamma": 0.02,
                        "center_x": bound / 3,
                        "center_y": bound / 3,
                    }
                )

                lower.append(
                    {
                        "theta_E": 0.3,
                        "e1": -0.5,
                        "e2": -0.5,
                        "gamma": 1.3,
                        "center_x": center_x - bound,
                        "center_y": center_y - bound,
                    }
                )

                upper.append(
                    {
                        "theta_E": 3.0,
                        "e1": 0.5,
                        "e2": 0.5,
                        "gamma": 2.8,
                        "center_x": center_x + bound,
                        "center_y": center_y + bound,
                    }
                )
            elif model == "SHEAR_GAMMA_PSI":
                fixed.append({"ra_0": 0, "dec_0": 0})
                init.append({"gamma_ext": 0.05, "psi_ext": 0.0})
                sigma.append({"gamma_ext": 0.05, "psi_ext": np.pi / 90.0})
                lower.append({"gamma_ext": 0.0, "psi_ext": -np.pi})
                upper.append({"gamma_ext": 0.5, "psi_ext": np.pi})
            elif model == "SIS":
                fixed.append({})
                init.append(
                    {
                        "center_x": center_x,
                        "center_y": center_y,
                        "theta_E": theta_E_init,
                    }
                )
                sigma.append(
                    {
                        "theta_E": 0.1,
                        "center_x": bound / 3,
                        "center_y": bound / 3,
                    }
                )
                lower.append(
                    {
                        "theta_E": 0.0,
                        "center_x": center_x - bound,
                        "center_y": center_y - bound,
                    }
                )
                upper.append(
                    {
                        "theta_E": 1.0,
                        "center_x": center_x + bound,
                        "center_y": center_y + bound,
                    }
                )
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
        lens_light_model_list, satellite_flags = (
            self.get_lens_light_model_list_with_flags()
        )

        fixed = []
        init = []
        sigma = []
        lower = []
        upper = []

        for i, model in enumerate(lens_light_model_list):
            if satellite_flags[i] == -1:
                bound = self.deflector_centroid_bound
                center_x = self.deflector_center_ra
                center_y = self.deflector_center_dec
            else:
                bound = self.settings["satellites"]["centroid_bound"]
                center_x = self.settings["satellites"]["centroid_init"][
                    satellite_flags[i]
                ][0]
                center_y = self.settings["satellites"]["centroid_init"][
                    satellite_flags[i]
                ][1]
            if "SERSIC" in model:
                if satellite_flags[i] == -1:
                    _fixed = {}
                else:
                    _fixed = {"n_sersic": 4.0}
                _init = {
                    "amp": 1.0,
                    "R_sersic": 0.2,
                    "center_x": center_x,
                    "center_y": center_y,
                    "n_sersic": 4.0,
                }
                _sigma = {
                    "center_x": np.max(self.pixel_size) / 10.0,
                    "center_y": np.max(self.pixel_size) / 10.0,
                    "R_sersic": 0.05,
                    "n_sersic": 0.5,
                }
                _lower = {
                    "n_sersic": 0.5,
                    "R_sersic": 0.1,
                    "center_x": center_x - bound,
                    "center_y": center_y - bound,
                }
                _upper = {
                    "n_sersic": 8.0,
                    "R_sersic": 5.0,
                    "center_x": center_x + bound,
                    "center_y": center_y + bound,
                }
                if "_ELLIPSE" in model:
                    _init.update({"e1": 0.0, "e2": 0.0})
                    _sigma.update({"e1": 0.05, "e2": 0.05})
                    _lower.update({"e1": -0.5, "e2": -0.5})
                    _upper.update({"e1": 0.5, "e2": 0.5})

                fixed.append(_fixed)
                init.append(_init)
                sigma.append(_sigma)
                lower.append(_lower)
                upper.append(_upper)
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
                # If n_max is given as a single integer, convert it to a list
                if isinstance(self.settings["source_light_option"]["n_max"], int):
                    self.settings["source_light_option"]["n_max"] = [
                        self.settings["source_light_option"]["n_max"]
                        for _ in range(self.number_of_bands)
                    ]

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
                        # Propagting the fixed values in light profile to all bands
                        if component in ["lens_light", "source_light"]:
                            for n in range(self.number_of_bands):
                                num_profiles = len(self.settings["model"][component])
                                fixed_list[int(index) + n * num_profiles][key] = value
                        elif (
                            component == "lens"
                        ):  # for mass model that doesn't need duplication for multiple bands
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
