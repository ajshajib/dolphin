import numpy as np
import scipy.stats as stats
import os
import copy
import yaml
from tqdm import trange

import lenstronomy.Util.util as util
import lenstronomy.Util.param_util as param_util
from lenstronomy.SimulationAPI.sim_api import SimAPI
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver


class TrainingData(object):
    """
    Contains all the methods to simulate the training set.
    """

    def __init__(
        self,
        filter,
        pixel_size,
        num_pixel,
        source_type="galaxy",
        max_num_satellite=2,
        verbose=False,
    ):
        """
        Setup the training set simulation settings.

        :param filters: list of filters used for simulation, must be defined in filter_settings.yaml
        :type filters: `list`
        :param pixel_sizes: list of pixel sizes for each filter
        :type pixel_sizes: `list`
        :param num_pixels: number of pixels for each filter
        :type num_pixels: `list`
        :param max_num_satellite: maximum number of satellites in the lens
        :type max_satellite_num: `int`
        :param verbose: if `True`, print the satellite numbers
        :type verbose: `bool`
        """
        self.filter = filter
        self.pixel_size = pixel_size
        self.num_pixel = num_pixel
        self._with_point_source = source_type == "quasar"
        self.source_type = source_type
        self.max_num_satellite = max_num_satellite
        self.verbose = verbose

        self._shapelet_coeffs = np.load(
            "resources/source_galaxy_shapelet_coefficients_nmax50.npz"
        )["arr_0"]

        self.filter_settings = self.get_filter_settings()

    def get_filter_settings(self):
        """
        Get the filter settings from the filter_settings.yaml file.
        """
        with open("filter_settings.yaml", "r") as file:
            filter_spec = yaml.load(file, Loader=yaml.FullLoader)

            filter_settings = filter_spec[self.filter]
            filter_settings["psf_type"] = "PIXEL"
            filter_settings["kernel_point_source"] = self.get_psf(
                filter_settings["psf_file"]
            )

            del filter_settings["psf_file"]

        return filter_settings

    def get_psf(self, filename):
        """
        Load the PSF from the file.

        :param filename: name of the file containing the PSF
        :type filename: `str`
        :return: PSF
        :rtype: `numpy.ndarray`
        """
        filename = os.path.join(os.path.dirname(__file__), filename)
        return np.load(filename)

    def create_dataset(
        self,
        num_system,
        max_satellite_num=2,
        no_lens_light_fracion=0.5,
        random_seed=None,
    ):
        """ """
        if random_seed is not None:
            np.random.seed(random_seed)

        dataset = np.zeros(shape=(num_system, self.num_pixel, self.num_pixel))
        masks = np.zeros(shape=(num_system, self.num_pixel, self.num_pixel))

        for i in trange(num_system):
            relative_probability = np.ones(max_satellite_num + 1) / (
                np.arange(max_satellite_num + 1) + 1
            )
            num_satellites = np.random.choice(
                np.arange(max_satellite_num + 1),
                p=relative_probability / np.sum(relative_probability),
            )
            if self.verbose:
                print("Number of satellites:", num_satellites)

            theta_E_multiplier = np.linspace(0.1, 0.4, 100)
            probability_theta_E_sat = 1 / theta_E_multiplier
            probability_theta_E_sat = probability_theta_E_sat / np.sum(
                probability_theta_E_sat
            )

            ## Randomized settings for the lens, these distributions are simple choices
            # More realistic distributions can be used depending on the necesity
            phi_epl = np.random.uniform(0, np.pi)
            q_epl = np.random.uniform(0.7, 0.95)
            theta_E = np.random.uniform(0.8, 1.1)
            gamma_epl = np.random.uniform(1.9, 2.1)
            phi_shear = np.random.uniform(0, np.pi)
            gamma_shear = np.random.uniform(0, 0.1)
            x_lens = np.random.uniform(-0.1, 0.1)
            y_lens = np.random.uniform(-0.1, 0.1)

            ## Randomized settings for the source galaxy
            e1_source = np.random.uniform(-0.2, 0.2)
            e2_source = np.random.uniform(-0.2, 0.2)
            source_r_sersic = np.random.uniform(0.1, 0.2)
            n_sersic_source = 1  # np.random.uniform(0.8, 2.5)
            r = np.random.uniform(0.05, 0.35) * theta_E
            phi = np.random.uniform(-np.pi, np.pi)
            x_source = r * np.cos(phi)
            y_source = r * np.sin(phi)
            mag_ps = np.random.uniform(19.5, 23) + np.random.uniform(-1.5, 0.1)

            ## Randomized settings for the lens light
            phi = np.random.uniform(0, np.pi)
            q = np.random.uniform(0.6, 0.95)
            e1, e2 = param_util.phi_q2_ellipticity(phi=phi, q=q)
            n_sersic = np.random.uniform(0.8, 5)
            # truncated normal distribution for r_sersic
            mu = 1.5  # mean
            sigma = 0.3  # standard deviation
            a, b = 1.0, 1.8  # truncation limits
            trunc_norm = stats.truncnorm(
                (a - mu) / sigma, (b - mu) / sigma, loc=mu, scale=sigma
            )
            r_sersic = trunc_norm.rvs()

            mag_lens = np.random.uniform(17, 19.5)  # np.random.uniform(16,20.5)

            mag_source = min(
                24 + np.random.normal(0, 1), mag_lens + np.random.normal(4.5, 0.2)
            )  # np.random.normal(3.8, 1.95)
            # mag_sat = mag + np.random.normal(2.5, 0.5)

            lens_model_list = ["EPL", "SHEAR"]
            e1_epl, e2_epl = param_util.phi_q2_ellipticity(phi=phi_epl, q=q_epl)
            e1_shear, e2_shear = param_util.shear_polar2cartesian(
                phi=phi_shear, gamma=gamma_shear
            )
            kwargs_epl = {
                "theta_E": theta_E,
                "e1": e1_epl,
                "e2": e2_epl,
                "center_x": x_lens,
                "center_y": y_lens,
                "gamma": gamma_epl,
            }
            kwargs_shear = {"gamma1": e1_shear, "gamma2": e2_shear}
            kwargs_lens = [kwargs_epl, kwargs_shear]

            lens_light_model_list = ["SERSIC_ELLIPSE"]
            kwargs_lens_light_mag = [
                {
                    "magnitude": mag_lens,
                    "R_sersic": r_sersic,
                    "n_sersic": n_sersic,
                    "e1": e1,
                    "e2": e2,
                    "center_x": x_lens,
                    "center_y": y_lens,
                }
            ]

            if num_satellites != 0:
                r_eff = 1.2 * r_sersic  # 1.2 is a hyperparameter

            x_sats, y_sats, R_sats = [], [], []
            for j in range(num_satellites):
                lens_model_list.append("EPL")

                theta_E_sat = theta_E * np.random.choice(
                    theta_E_multiplier, p=probability_theta_E_sat
                )
                # R_sat = np.random.uniform(0.1, 0.5)
                R_sat = r_sersic * theta_E_sat / theta_E
                e1_sat = np.random.uniform(-0.1, 0.1)
                e2_sat = np.random.uniform(-0.1, 0.1)

                r = r_eff + np.random.normal(0, 0.2)  # check the 13 lenses
                phi = np.random.uniform(0, 2 * np.pi)

                x_sat, y_sat = param_util.polar2cart(r, phi, [x_lens, y_lens])
                n_sat = np.random.uniform(3, 5)

                mag_sat = mag_lens + 2.5 * np.log10((theta_E / theta_E_sat) ** 2)
                # np.random.normal(
                #     2.5, 0.2
                # )  # np.random.uniform(20, 22.5)

                x_sats.append(x_sat)
                y_sats.append(y_sat)
                R_sats.append(R_sat)

                kwargs_sat_epl = {
                    "theta_E": theta_E_sat,
                    "gamma": 2,
                    "e1": e1_sat,
                    "e2": e2_sat,
                    "center_x": x_sat,
                    "center_y": y_sat,
                }
                kwargs_lens.append(kwargs_sat_epl)

                lens_light_model_list.append("SERSIC_ELLIPSE")
                kwargs_sat_light = {
                    "magnitude": mag_sat,
                    "R_sersic": R_sat,
                    "n_sersic": n_sat,
                    "e1": e1_sat,
                    "e2": e2_sat,
                    "center_x": x_sat,
                    "center_y": y_sat,
                }
                kwargs_lens_light_mag.append(kwargs_sat_light)

            kwargs_model = {
                "lens_model_list": lens_model_list,
                "lens_light_model_list": lens_light_model_list,
                "source_light_model_list": ["SHAPELETS"],
                "point_source_model_list": ["SOURCE_POSITION"],
            }

            sim_api = SimAPI(
                numpix=self.num_pixel,
                kwargs_single_band=self.filter_settings,
                kwargs_model=kwargs_model,
            )

            kwargs_model_copy = copy.deepcopy(kwargs_model)

            kwargs_model_copy["source_light_model_list"] = ["SERSIC_ELLIPSE"]
            sim_api_smooth_source = SimAPI(
                numpix=self.num_pixel,
                kwargs_single_band=self.filter_settings,
                kwargs_model=kwargs_model_copy,
            )

            kwargs_numerics = {"point_source_supersampling_factor": 1}
            im_sim = sim_api.image_model_class(kwargs_numerics)

            lens_model_class = LensModel(lens_model_list=lens_model_list)
            lensEquationSolver = LensEquationSolver(lens_model_class)
            x_image, y_image = lensEquationSolver.findBrightImage(
                x_source,
                y_source,
                kwargs_lens,
                numImages=4,
                min_distance=self.filter_settings["pixel_scale"],
                search_window=self.num_pixel * self.filter_settings["pixel_scale"],
            )

            # point source
            kwargs_ps_mag = [
                {
                    "magnitude": mag_ps,
                    "ra_source": x_source,
                    "dec_source": y_source,
                }
            ]

            # source light
            kwargs_source_smooth_mag = [
                {
                    "magnitude": mag_source,
                    "R_sersic": source_r_sersic,
                    "n_sersic": n_sersic_source,
                    "e1": e1_source,
                    "e2": e2_source,
                    "center_x": 0,
                    "center_y": 0,
                }
            ]

            kwargs_source = [
                {
                    "n_max": 20,
                    "beta": source_r_sersic,
                    "amp": self._shapelet_coeffs[
                        np.random.randint(len(self._shapelet_coeffs))
                    ],
                    "center_x": 0,
                    "center_y": 0,
                }
            ]

            kwargs_lens_light, kwargs_source_smooth, kwargs_ps = (
                sim_api_smooth_source.magnitude2amplitude(
                    kwargs_lens_light_mag, kwargs_source_smooth_mag, kwargs_ps_mag
                )
            )

            smooth_light_model = LightModel(["SERSIC_ELLIPSE"])
            shapelet_light_model = LightModel(["SHAPELETS"])

            num_source_pixels = 150
            x, y = util.make_grid(num_source_pixels, 0.02)
            x -= np.mean(x)
            y -= np.mean(y)

            smooth_source_flux = smooth_light_model.surface_brightness(
                x, y, kwargs_source_smooth
            )
            smooth_flux_total = np.sum(smooth_source_flux)

            shapelet_flux = shapelet_light_model.surface_brightness(x, y, kwargs_source)
            shapelet_flux[shapelet_flux < 0] = 1e-6
            shapelet_flux_total = np.sum(shapelet_flux)

            kwargs_source[0]["amp"] *= smooth_flux_total / shapelet_flux_total
            kwargs_source[0]["center_x"] = x_source
            kwargs_source[0]["center_y"] = y_source

            if num_satellites != 0:
                kwargs_sat_light = copy.deepcopy(kwargs_lens_light)

                for j in range(num_satellites):
                    kwargs_lens_light[-1 - j]["amp"] = 0
                for j in range(len(kwargs_lens_light) - num_satellites):
                    kwargs_sat_light[j]["amp"] = 0

            if np.random.uniform() > no_lens_light_fracion:
                lens_light_image = lens_light_image = im_sim.image(
                    kwargs_lens,
                    kwargs_source,
                    kwargs_lens_light,
                    kwargs_ps,
                    source_add=False,
                    point_source_add=False,
                )
            else:
                lens_light_image = 0

            source_light_image = im_sim.image(
                kwargs_lens,
                kwargs_source,
                kwargs_lens_light,
                kwargs_ps,
                lens_light_add=False,
                point_source_add=False,
            )
            source_light_image[source_light_image < 0] = 1e-6

            if self.source_type == "quasar":
                ps_image = im_sim.image(
                    kwargs_lens,
                    kwargs_source,
                    kwargs_lens_light,
                    kwargs_ps,
                    lens_light_add=False,
                    source_add=False,
                )
            else:
                ps_image = 0

            if num_satellites != 0:
                sat_light_image = im_sim.image(
                    kwargs_lens,
                    kwargs_source,
                    kwargs_sat_light,
                    kwargs_ps,
                    source_add=False,
                    point_source_add=False,
                )
            else:
                sat_light_image = 0

            image = lens_light_image + source_light_image + ps_image + sat_light_image

            mask = np.zeros_like(image)
            xs, ys = np.meshgrid(np.arange(self.num_pixel), np.arange(self.num_pixel))

            ########## source galaxy mask ##########
            #  mask[(source_light_images[1] > 3*sim.background_noise) &
            #      (source_light_images[1] > 0.5*lens_light_images[1])] = 2

            source_mask = np.zeros_like(mask).astype(bool)
            source_mask[
                (source_light_image > 3 * sim_api.background_noise)
                #  & (source_light_images[0] > lens_light_images[0])
            ] = True

            source_min = max(
                0.8 * np.max(source_light_image), np.mean(sim_api.background_noise)
            )
            source_mask[source_light_image > source_min] = True

            center_x_pixel = int(x_lens / sim_api.pixel_scale) + self.num_pixel // 2
            center_y_pixel = int(y_lens / sim_api.pixel_scale) + self.num_pixel // 2

            r_min = theta_E
            for p in range(len(x_image)):
                r_image = np.sqrt(
                    (x_image[p] - x_lens) ** 2 + (y_image[p] - y_lens) ** 2
                )

                if r_image < r_min:
                    r_min = r_image

            rs = np.sqrt((xs - center_x_pixel) ** 2 + (ys - center_y_pixel) ** 2)

            source_mask[rs < 0.7 * r_min / sim_api.pixel_scale] = False

            mask[source_mask] = 2

            ########## lens light mask ##########
            if np.sum(lens_light_image) > 0:
                galaxy_x_pixel = int(x_lens / sim_api.pixel_scale) + self.num_pixel // 2
                galaxy_y_pixel = int(y_lens / sim_api.pixel_scale) + self.num_pixel // 2

                mask[
                    (galaxy_x_pixel - xs) ** 2 + (galaxy_y_pixel - ys) ** 2 < 10**2
                ] = 1

            # mask[lens_light_image > 5 * sim_api.background_noise] = 1

            # mask[np.array(image) < 5 * sim_api.background_noise] = 0

            ########### point source mask ##########
            if self.source_type == "quasar":
                # mask[
                #     (ps_image > 10 * sim_api.background_noise)
                #     & (ps_image > lens_light_image)
                # ] = 4

                for j in range(len(x_image)):
                    x_pixel = (
                        int(x_image[j] / sim_api.pixel_scale) + self.num_pixel // 2
                    )
                    y_pixel = (
                        int(y_image[j] / sim_api.pixel_scale) + self.num_pixel // 2
                    )

                    # mask[
                    #     (ps_image > 10 * sim_api.background_noise)
                    #     & (ps_image > lens_light_image)
                    #     & (ps_image > source_light_image)
                    # ] = 3

                    mask[(xs - x_pixel) ** 2 + (ys - y_pixel) ** 2 < 3.5**2] = 4
            else:
                ps_image = 0

            ########## satellite masks ##########
            if num_satellites != 0:
                # mask[
                #     (sat_light_image > 5 * sim_api.background_noise)
                #     & (sat_light_image > lens_light_image)
                #     #  & (sat_light_images[0] > source_light_images[0])
                # ] = 3

                for j in range(num_satellites):
                    x_pixel = int(x_sats[j] / sim_api.pixel_scale) + self.num_pixel // 2
                    y_pixel = int(y_sats[j] / sim_api.pixel_scale) + self.num_pixel // 2

                    sat_mask = np.zeros_like(mask).astype(bool)
                    sat_mask[(sat_light_image > 5 * sim_api.background_noise)] = True

                    sat_mask[
                        (xs - x_pixel) ** 2 + (ys - y_pixel) ** 2
                        > (0.5 * R_sats[j] / sim_api.pixel_scale) ** 2
                    ] = False

                    sat_mask[(xs - x_pixel) ** 2 + (ys - y_pixel) ** 2 < 3.5**2] = True

                    mask[sat_mask] = 3
                    # mask[(xs - x_pixel) ** 2 + (ys - y_pixel) ** 2 < 3.5**2] = 3

            # add noise
            image += sim_api.noise_for_model(model=image)

            dataset[i, :, :] = image
            masks[i, :, :] = mask

        return dataset, masks
