# -*- coding: utf-8 -*-
"""This module creates `fitting_kwargs_list` for `FittingSequence.fit_sequence()` with
pre-defined recipes."""
__author__ = "ajshajib"

from copy import deepcopy
import numpy as np
from scipy import ndimage


class Recipe(object):
    """This class contains methods to create fitting recipes.

    It builds an optimization workflow (currently using particle-swarm optimization) to
    first find a good enough lens model within the total parameter space. Then, the
    sampling can be done starting from the neighborhood of this point.
    """

    def __init__(self, config, thread_count=1):
        """Initiate the class from the given settings for a lens system.

        :param config: `ModelConfig` instance
        :type config: `class`
        :param thread_count: number of threads if `multiprocess` is used
        :type thread_count: `int`
        """
        self._config = config
        try:
            config.settings["fitting"]["pso"]
        except (NameError, KeyError):
            self.do_pso = False
        else:
            self.do_pso = deepcopy(config.settings["fitting"]["pso"])

            self._pso_num_particle = self._config.settings["fitting"]["pso_settings"][
                "num_particle"
            ]
            self._pso_num_iteration = self._config.settings["fitting"]["pso_settings"][
                "num_iteration"
            ]

            if self.do_pso is None:
                self.do_pso = False

        try:
            config.settings["fitting"]["psf_iteration"]
        except (NameError, KeyError):
            self.reconstruct_psf = False
        else:
            self.reconstruct_psf = deepcopy(config.settings["fitting"]["psf_iteration"])

            if self.reconstruct_psf is None:
                self.reconstruct_psf = False

        try:
            config.settings["fitting"]["sampling"]
        except (NameError, KeyError):
            self.do_sampling = False
        else:
            self.do_sampling = deepcopy(config.settings["fitting"]["sampling"])

            if self.do_sampling is None:
                self.do_sampling = False

        self._thread_count = thread_count

        self.guess_params = {}
        for component in ["lens", "source", "lens_light", "ps"]:
            try:
                self.guess_params[component] = deepcopy(
                    config.settings["guess_params"][component]
                )
            except (NameError, KeyError):
                self.guess_params[component] = None

    def get_recipe(self, kwargs_data_joint=None, recipe_name="galaxy-quasar"):
        """Get `fitting_kwargs_list` according to the requested `recipe`.

        :param config: `ModelConfig` instance
        :type config:
        :param kwargs_data_joint: `kwargs_data_joint` dictionary
        :type kwargs_data_joint:
        :param recipe_name: recipe name, 'galaxy-quasar' or 'galaxy-galaxy'
        :type recipe_name: `str`
        :return: fitting kwargs list
        :rtype: `list`
        """
        fitting_kwargs_list = []

        if recipe_name == "galaxy-quasar":
            try:
                # if this key exists then return it
                self._config.settings["fitting_kwargs_list"]
            except (KeyError, NameError):
                fitting_kwargs_list += self.get_galaxy_quasar_recipe()
            else:
                if self._config.settings["fitting_kwargs_list"] is not None:
                    fitting_kwargs_list += self._config.settings["fitting_kwargs_list"]
                else:
                    fitting_kwargs_list += self.get_galaxy_quasar_recipe()
        elif recipe_name == "galaxy-galaxy":
            if kwargs_data_joint is None:
                raise ValueError(
                    "kwargs_data_joint is necessary to use "
                    "galaxy-galaxy optimization recipe!"
                )
            fitting_kwargs_list += self.get_galaxy_galaxy_recipe(kwargs_data_joint)
        elif recipe_name == "skip":
            pass
        else:
            raise ValueError("Recipe name '{}' not recognized!!".format(recipe_name))

        fitting_kwargs_list += self.get_sampling_sequence()

        return fitting_kwargs_list

    def _get_power_law_model_index(self):
        """Get the index of the power-law model, if included in the lens model list.

        :return: index or `None`
        :rtype: `int`
        """
        lens_model_list = self._config.get_lens_model_list()

        if "SPEMD" in lens_model_list:
            index = lens_model_list.index("SPEMD")
        elif "PEMD" in lens_model_list:
            index = lens_model_list.index("PEMD")
        elif "SPEP" in lens_model_list:
            index = lens_model_list.index("SPEP")
        elif "EPL" in lens_model_list:
            index = lens_model_list.index("EPL")
        else:
            index = None

        return index

    def _get_external_shear_model_index(self):
        """Get the index of the external shear model, if included in the lens model
        list.

        :return: index or `None`
        :rtype: `int`
        """
        lens_model_list = self._config.get_lens_model_list()
        if "SHEAR_GAMMA_PSI" in lens_model_list or "SHEAR" in lens_model_list:
            if "SHEAR_GAMMA_PSI" in lens_model_list:
                index = lens_model_list.index("SHEAR_GAMMA_PSI")
            else:
                index = lens_model_list.index("SHEAR")
        else:
            index = None

        return index

    def _get_shapelet_model_index(self):
        """Get the index of the shapelets model, if included in the source model list.

        :return: index or `None`
        :rtype: `int`
        """
        source_model_list = self._config.get_source_light_model_list()
        if "SHAPELETS" in source_model_list:
            index = source_model_list.index("SHAPELETS")
        else:
            index = None

        return index

    def get_galaxy_quasar_recipe(self):
        """Get the default pre-sampling optimization routine.

        :return: fitting kwargs list
        :rtype: `list`
        """
        fitting_kwargs_list = []

        if self.do_pso:
            pso_range_multipliers = [1.0, 0.1, 0.1]

            pl_model_index = self._get_power_law_model_index()

            for epoch in range(2):
                if epoch == 0 and pl_model_index is not None:
                    fitting_kwargs_list.append(
                        [
                            "update_settings",
                            {"lens_add_fixed": [[pl_model_index, ["gamma"]]]},
                        ]
                    )
                elif pl_model_index is not None:
                    fitting_kwargs_list.append(
                        [
                            "update_settings",
                            {"lens_remove_fixed": [[pl_model_index, ["gamma"]]]},
                        ]
                    )

                for multiplier in pso_range_multipliers:
                    # if multiplier in [10., 1.]:
                    #     fitting_kwargs_list.append([
                    #         'update_settings',
                    #         {'lens_add_fixed': [[index, ['gamma']]]}
                    #
                    #     ])
                    # elif multiplier == .1:
                    #     fitting_kwargs_list.append([
                    #         'update_settings',
                    #         {'lens_remove_fixed': [[index, ['gamma']]]}
                    #     ])

                    fitting_kwargs_list.append(
                        [
                            "PSO",
                            {
                                "sigma_scale": multiplier,
                                "n_particles": self._pso_num_particle,
                                "n_iterations": self._pso_num_iteration,
                                "threadCount": self._thread_count,
                            },
                        ]
                    )

                    if self.reconstruct_psf:
                        fitting_kwargs_list.append(
                            ["psf_iteration", self._config.get_kwargs_psf_iteration()]
                        )
        return fitting_kwargs_list

    def get_sampling_sequence(self):
        """Get the sampling sequence. Currently only MCMC with emcee is supported.

        :return:
        :rtype:
        """
        fitting_kwargs_list = []

        if self.do_sampling:
            supported_samplers = [
                "emcee",
                # "zeus",
                # "dynesty",
                # "dyPolyChord",
                # "MultiNest",
                # "nested_sampling",
                # "Nautilus",
            ]
            if self._config.settings["fitting"]["sampler"] not in supported_samplers:
                raise ValueError(
                    "Sampler '{}' not supported! ".format(
                        self._config.settings["fitting"]["sampler"]
                    )
                    + "Supported ones are: {}".format(supported_samplers)
                )

            sampling_kwargs = self._config.settings["fitting"]["sampler_settings"]
            if self._config.settings["fitting"]["sampler"] in ["emcee"]:
                if "threadCount" not in sampling_kwargs:
                    sampling_kwargs["threadCount"] = self._thread_count

                try:
                    self._config.settings["fitting"]["sampler_settings"]["init_samples"]
                except (NameError, KeyError):
                    pass
                else:
                    if (
                        self._config.settings["fitting"]["sampler_settings"][
                            "init_samples"
                        ]
                        is not None
                    ):
                        sampling_kwargs["init_samples"] = np.array(
                            self._config.settings["fitting"]["sampler_settings"][
                                "init_samples"
                            ]
                        )

                        sampling_kwargs["re_use_samples"] = True

            fitting_kwargs_list.append(
                [self._config.settings["fitting"]["sampler"], sampling_kwargs]
            )

        return fitting_kwargs_list

    def get_galaxy_galaxy_recipe(self, kwargs_data_joint, epochs=2):
        """Get the pre-sampling optimization routine for a galaxy-galaxy lens. PSF
        iteration is not added.

        :param kwargs_data_joint:
        :type kwargs_data_joint:
        :param epochs: number of times to repeat the fitting sequence
        :type epochs: `int`
        :return:
        :rtype:
        """
        fitting_kwargs_list = []

        if self.do_pso:
            arc_masks = []
            masks = self._config.get_masks()
            for i, band_item in enumerate(kwargs_data_joint["multi_band_list"]):
                mask = masks[i] if masks is not None else None
                image = band_item[0]["image_data"]
                arc_masks.append(self.get_arc_mask(image, mask=mask))

            pl_model_index = self._get_power_law_model_index()
            external_shear_model_index = self._get_external_shear_model_index()
            shapelets_index = self._get_shapelet_model_index()

            temp_constraints = self._config.get_kwargs_constraints()
            for epoch in range(epochs):
                # first fix everything else except for lens light and use arc
                # mask to fit the lens light only. Join the centroids of lens
                # and lens light
                fitting_kwargs_list += [
                    self.fix_params("lens"),
                    self.fix_params("source"),
                    [
                        "update_settings",
                        {
                            "kwargs_likelihood": {
                                "image_likelihood_mask_list": arc_masks
                            }
                        },
                    ],
                    [
                        "update_settings",
                        {
                            "kwargs_constraints": {
                                "joint_lens_with_light": [
                                    [0, 0, ["center_x", "center_y"]]
                                ]
                            }
                        },
                    ],
                    [
                        "PSO",
                        {
                            "sigma_scale": 1.0,
                            "n_particles": self._pso_num_particle,
                            "n_iterations": self._pso_num_iteration,
                            "threadCount": self._thread_count,
                        },
                    ],
                ]

                # unfix the source except for beta, keep lens fixed, fix lens
                # light, use regular mask
                fitting_kwargs_list += [self.unfix_params("source")]

                # fix the shapelets beta parameter
                if shapelets_index is not None:
                    fitting_kwargs_list += [
                        [
                            "update_settings",
                            {"source_add_fixed": [[shapelets_index, ["beta"], [0.1]]]},
                        ]
                    ]

                fitting_kwargs_list += [
                    # self.unfix_params('lens'),
                    self.fix_params("lens_light"),
                    [
                        "update_settings",
                        {"kwargs_likelihood": {"image_likelihood_mask_list": masks}},
                    ],
                ]

                # set lens parameter values to guess values, if provided
                if self.guess_params["lens"] is not None:
                    param_list = []
                    for index, params in self.guess_params["lens"].items():
                        param_list.append(
                            [index, list(params.keys()), list(params.values())]
                        )

                    fitting_kwargs_list += [
                        ["update_settings", {"lens_add_fixed": param_list}]
                    ]

                # optimize for the source only
                fitting_kwargs_list += [
                    # self.fix_params('lens'),
                    [
                        "PSO",
                        {
                            "sigma_scale": 1.0,
                            "n_particles": self._pso_num_particle,
                            "n_iterations": self._pso_num_iteration,
                            "threadCount": self._thread_count,
                        },
                    ],
                ]

                # unfix the central deflector parameters, keep beta fixed
                fitting_kwargs_list += [
                    self.unfix_params("lens"),
                    self.fix_params("lens", external_shear_model_index),
                ]

                # optimize for lens and source together, fix power-law gamma to
                # 2, as all the lens parameters are unfixed
                if pl_model_index is not None:
                    fitting_kwargs_list += [
                        [
                            "update_settings",
                            {"lens_add_fixed": [[pl_model_index, ["gamma"], [2.0]]]},
                        ]
                    ]

                fitting_kwargs_list += [
                    [
                        "PSO",
                        {
                            "sigma_scale": 1.0,
                            "n_particles": self._pso_num_particle,
                            "n_iterations": self._pso_num_iteration,
                            "threadCount": self._thread_count,
                        },
                    ],
                ]

                # unfix the shapelets beta parameter
                if shapelets_index is not None:
                    fitting_kwargs_list += [
                        [
                            "update_settings",
                            {"source_remove_fixed": [[shapelets_index, ["beta"]]]},
                        ]
                    ]

                # finally optimize with all of lens, lens light and source free
                fitting_kwargs_list += [
                    [
                        "PSO",
                        {
                            "sigma_scale": 1.0,
                            "n_particles": self._pso_num_particle,
                            "n_iterations": self._pso_num_iteration,
                            "threadCount": self._thread_count,
                        },
                    ],
                    self.unfix_params("lens_light"),
                    [
                        "PSO",
                        {
                            "sigma_scale": 1.0,
                            "n_particles": self._pso_num_particle,
                            "n_iterations": self._pso_num_iteration,
                            "threadCount": self._thread_count,
                        },
                    ],
                ]

                # finally, relax shear parameters for MCMC later
                # disjoin lens and lens light centroids
                fitting_kwargs_list += [
                    self.unfix_params("lens"),
                    ["update_settings", {"kwargs_constraints": temp_constraints}],
                ]

            # fitting_kwargs_list += self.get_default_recipe()

        return fitting_kwargs_list

    def get_arc_mask(self, image, clear_center=0.4, mask=None):
        """Create a mask for lensed galaxy arcs from the image of the lens. The lens
        galaxy is required to be close to the center (within a few pixels) of the image.

        :param image: image of the lensing system
        :type image: `ndarray`
        :param clear_center: radius of the central region to **not** mask
        :type clear_center: `float`
        :param mask: a mask to multiply with the arc mask. If the central
            region is masked out in `mask`, then a circle with radius
            `clear_center` will be unmasked.
        :type mask: `ndarray`
        :return: mask for the lensed galaxy arcs
        :rtype: `ndarray`
        """
        # take x- and y- gradient of the image
        x_diff = np.diff(image, axis=1)[1:, :]
        y_diff = np.diff(image, axis=0)[:, 1:]

        w = len(x_diff) - 1
        x, y = np.meshgrid(
            np.linspace(-w / 2, w / 2, int(w + 1)),
            np.linspace(-w / 2, w / 2, int(w + 1)),
        )
        r = np.sqrt(x * x + y * y)

        # compute the radial gradient of the image
        softening = 1e-10
        radial_gradient = -(x_diff * x / (r + softening) + y_diff * y / (r + softening))

        # convert radial_gradient to binary map (+ve to 0 and -ve to 1).
        # where the arc starts when going radially outward, the gradient
        # will be +ve, so this operation marks the inner edge of the arcs
        radial_gradient[np.isnan(radial_gradient)] = 0
        radial_gradient[radial_gradient > 0] = 1
        radial_gradient[radial_gradient <= 0] = 0
        radial_gradient = 1 - radial_gradient

        # unmark any marked pixels from the central region
        radial_gradient[r < int(clear_center / np.max(self._config.pixel_size))] = 0

        # remove connected regions with area less than 5 pixels to remove
        # masked regions created by noise
        structure = np.ones((3, 3))
        filtered_map = deepcopy(radial_gradient)
        id_regions, num_ids = ndimage.label(filtered_map, structure=structure)
        id_sizes = np.array(
            ndimage.sum(radial_gradient, id_regions, range(num_ids + 1))
        )
        area_mask = id_sizes < 5
        filtered_map[area_mask[id_regions]] = 0

        # dilate the binary marked-pixel map
        dilated = np.zeros_like(filtered_map)

        # create structural elements for dilating the image radially outward
        # in each of the four quadrants separately
        a2 = np.tril(np.ones((8, 8)))
        np.fill_diagonal(a2, 0)
        a2 = np.flip(a2, axis=1)  # 1's at lower than anti-diagonal
        a4 = np.flip(a2)  # 1's at upper than anti-diagonal
        a3 = np.rot90(a2)  # 1's at upper than the diagonal
        a1 = np.flip(a3)  # 1's at lower than the diagonal

        dilated[:50, :50] = ndimage.binary_dilation(filtered_map[:50, :50], a4)
        dilated[:50, 50:] = ndimage.binary_dilation(filtered_map[:50, 50:], a3)
        dilated[50:, :50] = ndimage.binary_dilation(filtered_map[50:, :50], a1)
        dilated[50:, 50:] = ndimage.binary_dilation(filtered_map[50:, 50:], a2)

        # increase the size by 1 along both axes to match the image size
        # the mask is the negative of the marked pixel-map
        arc_mask = 1 - np.pad(dilated, ((0, 1), (0, 1)), "minimum")

        # check for bad values
        arc_mask[arc_mask > 0] = 1
        arc_mask[arc_mask <= 0] = 0

        if mask is not None:
            arc_mask *= mask

            w = len(arc_mask) - 1
            x, y = np.meshgrid(
                np.linspace(-w / 2, w / 2, int(w + 1)),
                np.linspace(-w / 2, w / 2, int(w + 1)),
            )
            r = np.sqrt(x * x + y * y)

            arc_mask[r < int(clear_center / np.max(self._config.pixel_size))] = 1

        return arc_mask

    def fix_params(self, model_component, index=None):
        """Fix all the params in `name` that are not fixed by settings.

        :param model_component: name of params type, e.g., 'lens_model'
        :type model_component: `str`
        :param index: profile indices, if `None` all will be fixed
        :type index: `list`
        :return: formatted fit-sequence code to go into `fitting_kwargs_list`
        :rtype: `list`
        """
        if model_component == "lens":
            kwargs_params = self._config.get_lens_model_params()
        # elif model_component == 'point_source':
        #    kwargs_params = self.get_point_source_params()
        elif model_component == "lens_light":
            kwargs_params = self._config.get_lens_light_model_params()
        elif model_component == "source":
            kwargs_params = self._config.get_source_light_model_params()
        else:
            raise ValueError(
                "{} not recognized! Must be lens or "
                "lens_light or source.".format(model_component)
            )

        lower_list = kwargs_params[3]
        fixed_list = kwargs_params[2]

        if index is None:
            index = [i for i, _ in enumerate(lower_list)]

        if not isinstance(index, list):
            index = [index]

        param_list_with_index = []

        for i, (sigma, fixed) in enumerate(zip(lower_list, fixed_list)):
            if i in index:
                param_list = []
                for key, value in sigma.items():
                    if key not in fixed:
                        param_list.append(key)

                param_list_with_index.append([i, param_list])

        key = "{}_add_fixed".format(model_component)

        return ["update_settings", {key: param_list_with_index}]

    def unfix_params(self, model_component, index=None):
        """Unfix all the params in `name` that are not fixed from settings.

        :param model_component: name of params type, e.g., 'lens_model'
        :type model_component: `str`
        :param index: profile indices, if `None` all will be unfixed
        :type index: `list`
        :return: formatted fit-sequence code to go into `fitting_kwargs_list`
        :rtype: `list`
        """
        code = self.fix_params(model_component, index=index)

        old_key = "{}_add_fixed".format(model_component)
        key = "{}_remove_fixed".format(model_component)

        code[1][key] = deepcopy(code[1][old_key])
        del code[1][old_key]

        return code
