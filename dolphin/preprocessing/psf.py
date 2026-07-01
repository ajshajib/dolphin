# -*- coding: utf-8 -*-
"""This class contains helper functions to create a PSF using either PSFr or STARRED."""

__author__ = "brady-ryan"

import os

import matplotlib.pyplot as plt
import numpy as np
import subprocess
from astroObjectAnalyser.DataAnalysis.analysis import Analysis
from astroObjectAnalyser.astro_object_superclass import StrongLensSystem
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

from mpl_toolkits.axes_grid1 import make_axes_locatable
from ..processor.files import FileSystem

from psfr import psfr

from starred.psf.psf import PSF as STARRED_PSF
from starred.psf.loss import Loss
from starred.optim.optimization import Optimizer
from starred.psf.parameters import ParametersPSF
from starred.utils.noise_utils import propagate_noise
from starred.plots import plot_function as pltf


class PSF:
    """This class contains helper functions to create a PSF estimate from observational
    data."""

    def __init__(self, io_directory, lens_name, data_band, instrument):
        """Initiate the class from the following inputs:

        :param io_directory: path to the input/output directory. Should not end with slash.
        :type io_directory: `str`
        :param lens_name: name of the system to create a PSF for
        :type lens_name: `str`
        :param data_band: data band of desired PSF
        :type data_band: `str`
        :param instrument: instrument which took the data
        :type intrument: `str`
        """
        self.file_system = FileSystem(io_directory)
        self.lens_name = lens_name
        self.data_band = data_band

        self.instrument = instrument
        supported_instruments = ["HST", "JWST"]
        if instrument not in supported_instruments:
            raise ValueError(
                f"{instrument} is not supported! Options are: {supported_instruments}"
            )

        self.data_dir = self.file_system.get_data_directory()
        self.full_image_path = f"{self.data_dir}/{self.lens_name}"
        self.image_file_name = (
            f"{self.full_image_path}/full_image_{self.lens_name}_{self.data_band}.fits"
        )
        self.weight_file_name = f"{self.full_image_path}/weight_image_{self.lens_name}_{self.data_band}.fits"

    def make_image_catalog(self):
        """Run Source Extractor to obtain the full SExtractor catalog.

        :return: full SExtractor catalog
        :rtype: `BinTableHDU`
        """
        subprocess.run(
            f"sex {self.image_file_name} -c default.sex "
            f"-CATALOG_NAME {self.lens_name}_{self.data_band}.cat "
            "-PARAMETERS_NAME default.param "
            "-FILTER_NAME default.conv "
            "-STARNNW_NAME default.nnw",
            shell=True,
        )

        catalog_str = f"{self.lens_name}_{self.data_band}.cat"
        catalog = fits.open(catalog_str)[2]

        # move catalog from working directory to respective data directory
        output_dir = f"{self.full_image_path}/preprocessing/{self.data_band}"
        os.makedirs(output_dir, exist_ok=True)
        catalog_destination = os.path.join(output_dir, catalog_str)
        os.replace(catalog_str, catalog_destination)

        return catalog

    def get_kwargs_cut(self):
        """Estimate the cuts in the different SExtractor quantities for masking, and get
        the SExtractor catalog.

        :return: a tuple containing a dictionary of SExtractor quantities and
          the catalog of objects obtained by SExtractor
        :rtype: `tuple` (`dict`, `BinTableHDU`)
        """
        catalog = self.make_image_catalog()
        mag = np.array(catalog.data["MAG_BEST"], dtype=float)
        size = np.array(catalog.data["FLUX_RADIUS"], dtype=float)
        # ellipticity = catalog.data['ELLIPTICITY']

        kwargs_cut = {}
        mag_max = min(np.max(mag), 34)
        mag_min = np.min(mag)
        delta_mag = mag_max - mag_min
        kwargs_cut["MagMaxThresh"] = mag_max - 0.7 * delta_mag
        kwargs_cut["MagMinThresh"] = mag_min  # + 0.01*delta_mag

        mask = mag < mag_max - 0.5 * delta_mag
        kwargs_cut["SizeMinThresh"] = max(0, np.min(size[mask]))
        kwargs_cut["SizeMaxThresh"] = max(0, np.min(size[mask]) + 4)
        kwargs_cut["EllipticityThresh"] = 0.1
        kwargs_cut["ClassStarMax"] = 1.0
        kwargs_cut["ClassStarMin"] = 0.5

        return kwargs_cut, catalog

    def get_psf_candidates(
        self,
        catalog,
        kwargs_cut,
        radius_pix=100,
        exclude_flags=True,
        exclude_specific=None,
        save=False,
    ):
        """Obtain PSF candidates based upon the catalog and cutting dictionary. In
        addition to cutouts of the candidate objects being created, weight cutouts and
        noise map cutouts are made. To save the cutouts, toggle `save = True`. Can be
        run again excluding specific objects to narrow down the initial candidates.

        :param catalog: fits table SExtractor catalog as determined by
          :meth:`~get_kwargs_cut`
        :type catalog: `table`
        :param radius_pix: (optional) Exclusion radius around RA/DEC of target object in pixels.
          This is so that the lens/quasar images are not included in the PSF making.
        :type radius_pix: `int`
        :param exclude_flags: (optional) remove all objects with non-zero SExtractor flags
        :type exclude_flags: `bool`
        :param exclude_specific: (optional) list of SExtractor object numbers to exclude
        :type exclude_specific: `list`
        :param save: (optional) boolean dictating whether or not to save the star cutouts
          and corresponding noise maps
        :type plot: `bool`

        :return: tuple containing the cutout, weight map, and noise map for each star
        :rtype: `tuple` (`np.ndarray`, `np.ndarray`, `np.ndarray`)
        """

        mag = np.array(catalog.data["MAG_BEST"], dtype=float)
        size = np.array(catalog.data["FLUX_RADIUS"], dtype=float)
        ellipticity = catalog.data["ELLIPTICITY"]
        classStar = catalog.data["CLASS_STAR"]
        SizeMaxThresh = kwargs_cut["SizeMaxThresh"]
        SizeMinThresh = kwargs_cut["SizeMinThresh"]
        EllipticityThresh = kwargs_cut["EllipticityThresh"]
        MagMaxThresh = kwargs_cut["MagMaxThresh"]
        MagMinThresh = kwargs_cut["MagMinThresh"]
        ClassStarMax = kwargs_cut["ClassStarMax"]
        ClassStarMin = kwargs_cut["ClassStarMin"]

        mask = (
            (size < SizeMaxThresh)
            & (ellipticity < EllipticityThresh)
            & (size > SizeMinThresh)
            & (mag < MagMaxThresh)
            & (mag > MagMinThresh)
            & (classStar < ClassStarMax)
            & (classStar > ClassStarMin)
        )

        with fits.open(self.image_file_name) as hdul:
            header = hdul[0].header

            if self.instrument == "JWST":
                ra_targ = header["TARG_RA"] * u.deg
                dec_targ = header["TARG_DEC"] * u.deg
                wcs = WCS(hdul["SCI"])
            else:
                ra_targ = header["RA_TARG"] * u.deg
                dec_targ = header["DEC_TARG"] * u.deg
                wcs = WCS(header)

        coord = SkyCoord(ra=ra_targ, dec=dec_targ, unit="deg")

        x_targ, y_targ = wcs.world_to_pixel(coord)
        x = catalog.data["X_IMAGE"]
        y = catalog.data["Y_IMAGE"]
        dist = np.sqrt((x - x_targ) ** 2 + (y - y_targ) ** 2)

        mask &= dist > radius_pix
        mask_mask = mask[mask].copy()

        flags = catalog.data["FLAGS"][mask]
        obj_num = catalog.data["NUMBER"][mask]

        if exclude_flags:
            mask_mask &= flags == 0
            print("Objects with non-zero flags excluded.")

        if exclude_specific is not None:
            mask_mask &= ~np.isin(obj_num, exclude_specific)
            print(f"Excluded {len(exclude_specific)} specified objects.")

        mask[mask] = mask_mask
        print(f"Found {len(mag[mask > 0])} candidate objects!")

        analysis = Analysis()
        if self.instrument == "HST":
            system = StrongLensSystem(self.lens_name)

            system.add_image_data_init(
                self.data_band,
                self.image_file_name,
                local_wht_filename=self.weight_file_name,
                cutout_scale=100,
                ra=ra_targ,
                dec=dec_targ,
                ra_cutout_cent=None,
                dec_cutout_cent=None,
            )

            mean_bkd, sigma_bkd = system.get_background(self.data_band)
            img_full = system.get_full_image(self.data_band)  # get the full image
            star_exposures = analysis.get_objects_image(
                img_full - mean_bkd, catalog, mask, 50
            )  # get star cutouts, background subtracted

            wht_full = system.get_full_exposure(
                self.data_band
            )  # get the full exposure map
            wht_full[wht_full <= 0] = 10 ** (-10)
            star_weights = analysis.get_objects_image(
                wht_full, catalog, mask, 50
            )  # get weight cutouts for stars

            noise_maps = []
            for i, star_img in enumerate(star_exposures):
                noise_map = np.abs(star_img) / star_weights[i] + sigma_bkd**2
                noise_maps.append(np.sqrt(noise_map))
        elif self.instrument == "JWST":
            with fits.open(self.image_file_name) as hdul:
                sci = hdul["SCI"].data
                wht = hdul["WHT"].data
                variance = (
                    hdul["VAR_POISSON"].data
                    + hdul["VAR_RNOISE"].data
                    + hdul["VAR_FLAT"].data
                    + 1.0 / wht
                )
                err = np.sqrt(variance)

            star_exposures = analysis.get_objects_image(
                sci,
                catalog,
                mask,
                50,
            )

            star_weights = analysis.get_objects_image(
                wht,
                catalog,
                mask,
                50,
            )

            noise_maps = analysis.get_objects_image(
                err,
                catalog,
                mask,
                50,
            )
        else:
            print(f"{self.instrument} not yet supported!")

        # sort data by object magnitude
        star_magnitudes = np.array(catalog.data["MAG_BEST"][mask])
        mag_sort_idx = np.argsort(star_magnitudes)
        star_exposures = [star_exposures[i] for i in mag_sort_idx]
        star_weights = [star_weights[i] for i in mag_sort_idx]
        noise_maps = [noise_maps[i] for i in mag_sort_idx]

        self.plot_psf_candidates(
            mask=mask,
            star_exposures=star_exposures,
            star_weights=star_weights,
            noise_maps=noise_maps,
            catalog=catalog,
        )

        if save:
            self.file_system.save_star_cutouts(
                psf_class=self,
                star_exposures=star_exposures,
                star_weights=star_weights,
                noise_maps=noise_maps,
            )

        return star_exposures, star_weights, noise_maps

    def make_psf_psfr(
        self,
        oversampling=1,
        error_map_list=None,
        saturation_limit=None,
        num_iteration=20,
        n_recenter=5,
        kwargs_one_step=None,
        verbose=False,
        psf_initial_guess=None,
        kwargs_psf_stacking=None,
        centroid_optimizer="Nelder-Mead",
        cut_threshold=1.0e-20,
        save=False,
    ):
        """Create a PSF using the `PSFr` methodology.

        :param oversampling: (optional) higher-resolution PSF reconstruction and return
        :type oversampling: `int`
        :param error_map_list: (optional) Variance in the uncorrelated uncertainties in the data for individual pixels.
          If not set, assumes equal variances for all pixels.
        :type error_map_list: `list of np.ndarray`
        :param saturation_limit: (optional) float or list of floats of length of star_list
          pixel values above this threshold will not be considered in the reconstruction.
        :type saturation_limit: `float` or `list of floats` of length of star_list
        :param num_iteration: (optional)  number of iterative corrections applied on the PSF based on previous guess
        :type num_iteration: `int`
        :param n_recenter: (optional) every n_recenter iterations of the updated PSF, a re-centering of
          the centroids are performed with the updated PSF guess
        :type n_recenter: `int`
        :param kwargs_one_step: (optional) keyword arguments to be passed to one_step_psf_estimate() method
        :type kwargs_one_step: `dict`
        :param verbose: (optional) If True, provides plots of updated PSF during the iterative process
        type verose: `bool`
        :param psf_initial_guess: (optional) Initial guess PSF on oversampled scale. If not provided, estimates
          an initial guess with the stacked stars.
        :type psf_initial_guess: `None` or `2d numpy array with square odd axis`
        :param kwargs_psf_stacking: (optional)
          stacking_option: option of stacking, 'mean',  'median' or 'median_weight'.
          symmetry: integer, imposed symmetry of PSF estimate
        :type kwargs_psf_stacking: `list of keyword arguments`
        :param centroid_optimizer: (optional) Option for the optimizing algorithm used to find the center of each PSF in data.
          Options are 'Nelder-Mead' or 'PSO'. Default is 'Nelder-Mead'
        :type centroid_optimizer: `str`
        :param cut_threshold: (optional) signal threshold in which pixels under this value will not be
          saved in the final PSF
        :type cut_threshold: `float`
        :param save: (optional) whether or not to save the output PSF and variance map in the expected
          `Dolphin` format
        :type save: `bool`

        :return: a tuple containing the PSF array and PSF variance map array
        :rtype: `tuple` (`np.ndarray`, `np.ndarray`)
        """

        star_data_list, mask_data_list, _, _ = self.load_psf_candidate_attributes()

        psf_returns = psfr.stack_psf(
            star_list=star_data_list,
            oversampling=oversampling,
            mask_list=mask_data_list,
            error_map_list=error_map_list,
            saturation_limit=saturation_limit,
            num_iteration=num_iteration,
            n_recenter=n_recenter,
            kwargs_one_step=kwargs_one_step,
            verbose=verbose,
            psf_initial_guess=psf_initial_guess,
            kwargs_psf_stacking=kwargs_psf_stacking,
            centroid_optimizer=centroid_optimizer,
        )

        psf_guess = psf_returns[0]
        center_list = np.array(psf_returns[1])

        # Process the center list for the error map
        new_center_list = []
        for i, _ in enumerate(center_list):
            new_center_list.append([center_list[i][0], center_list[i][1]])
        new_center_list = np.array(new_center_list)
        error_map = psfr.psf_error_map(
            star_list=star_data_list,
            psf_kernel=psf_guess,
            center_list=new_center_list,
            mask_list=mask_data_list,
            oversampling=oversampling,
        )

        final_psf_mask = psf_guess > cut_threshold
        final_psf = np.where(final_psf_mask, psf_guess, 0)
        if oversampling > 1:
            # Downsample mask to variance map resolution
            native_nx = error_map.shape[0]
            native_ny = error_map.shape[1]

            error_map_mask = final_psf_mask.reshape(
                native_nx, oversampling, native_ny, oversampling
            ).any(axis=(1, 3))

            final_error_map = np.where(error_map_mask, error_map, 0)
        else:
            final_error_map = np.where(final_psf_mask, error_map, 0)

        self.plot_psf_and_variance_map(
            method="PSFr",
            psf_guess=psf_guess,
            variance_map=error_map**2,
            psf_cut=final_psf,
            variance_map_cut=final_error_map**2,
        )

        if save:
            self.file_system.save_psf_and_variance_map(
                psf_class=self, psf_guess=final_psf, variance_map=final_error_map**2
            )

        return final_psf, final_error_map**2

    def make_psf_starred(
        self,
        max_iterations=1500,
        subsampling_factor=1,
        convolution_method="scipy",
        include_moffat=True,
        elliptical_moffat=False,
        regularization_terms="l1_starlet",
        regularization_strength_scales=0,
        regularization_strength_hf=0,
        cut_threshold=1.0e-20,
        save=False,
    ):
        """Create a PSF using the `STARRED` methodology.

        :param max_iterations: (optional) maximum number of iterations to use in the final minimization
        :type max_iterations: `int`
        :param subsampling_factor: (optional) higher-resolution PSF reconstruction and return
        :type subsampling_factor: `int`
        :param convolution_method: (optional) method to use to calculate the convolution, choose between 'fft', 'scipy', and 'lax. Recommended if jax>=0.4.9 - 'scipy'
        :type convolution_method: `str`
        :param include_moffat: (optional) True for the PSF to be expressed as the sum of a Moffat and a grid of pixels. False to not include the Moffat. Default: True
        :type include_moffat: bool
        :param elliptical_moffat: (optional) Allow elliptical Moffat.
        :type elliptical_moffat: bool
        :param regularization_terms: (optional) information about the regularization terms
        :type regularization_terms: `str`
        :param regularization_strength_scales: (optional) Lagrange parameter that weights intermediate scales in the transformed domain.
        :type regularization_strength_scales: `float`
        :param regularization_strength_hf: (optional) Lagrange parameter weighting the highest frequency scale
        :type regularization_strength_hf: `float`
        :param cut_threshold: (optional) signal threshold in which pixels under this value will not be
          saved in the final PSF
        :type cut_threshold: `float`
        :param save: (optional) whether or not to save the output PSF and variance map in the expected
          `Dolphin` format
        :type save: `bool`

        :return: a tuple containing the PSF array and PSF variance map array
        :rtype: `tuple` (`np.ndarray`, `np.ndarray`)
        """

        star_data_list, mask_data_list, _, noise_maps = (
            self.load_psf_candidate_attributes()
        )
        variance = noise_maps**2

        model = STARRED_PSF(
            image_size=star_data_list[0].shape[1],
            number_of_sources=len(star_data_list),
            upsampling_factor=subsampling_factor,
            convolution_method=convolution_method,
            include_moffat=include_moffat,
            elliptical_moffat=elliptical_moffat,
        )

        kwargs_init, kwargs_fixed, kwargs_up, kwargs_down = model.smart_guess(
            data=star_data_list, fixed_background=True
        )

        parameters = ParametersPSF(
            kwargs_init=kwargs_init,
            kwargs_fixed=kwargs_fixed,
            kwargs_up=kwargs_up,
            kwargs_down=kwargs_down,
        )

        loss = Loss(
            data=star_data_list,
            psf_class=model,
            param_class=parameters,
            sigma_2=variance,
            N=len(star_data_list),
            regularization_terms=regularization_terms,
            regularization_strength_scales=0,
            regularization_strength_hf=0,
            masks=mask_data_list,
        )

        optim = Optimizer(loss_class=loss, param_class=parameters, method="Newton-CG")
        optimizer_options = {"maxiter": 1000, "restart_from_init": True}

        best_fit, _, extra_fields, _ = optim.minimize(**optimizer_options)
        kwargs_partial = parameters.args2kwargs(best_fit)

        # compute noise level in starlet space and propagate Poisson noise
        W = propagate_noise(
            model=model,
            noise_maps=noise_maps,
            kwargs=kwargs_partial,
            masks=mask_data_list,
            wavelet_type_list=["starlet"],
            method="MC",
            num_samples=500,
            seed=1,
            likelihood_type="chi2",
            verbose=False,
            upsampling_factor=subsampling_factor,
            scaling_noise_ref=None,
        )[0]

        # run the full model on a regularized grid
        kwargs_moffat_fixed = {"C": kwargs_partial["kwargs_moffat"]["C"]}
        kwargs_fixed = {
            "kwargs_moffat": kwargs_moffat_fixed,
            "kwargs_gaussian": {},
            "kwargs_background": {},
            "kwargs_distortion": kwargs_partial["kwargs_distortion"],
        }
        parameters = ParametersPSF(
            kwargs_init=kwargs_partial,
            kwargs_fixed=kwargs_fixed,
            kwargs_up=kwargs_up,
            kwargs_down=kwargs_down,
        )

        loss = Loss(
            data=star_data_list,
            psf_class=model,
            param_class=parameters,
            sigma_2=variance,
            N=len(star_data_list),
            regularization_terms=regularization_terms,
            regularization_strength_scales=regularization_strength_scales,
            regularization_strength_hf=regularization_strength_hf,
            regularization_strength_positivity=0,
            W=W,
            regularize_full_psf=False,
            masks=mask_data_list,
        )

        optim = Optimizer(loss_class=loss, param_class=parameters, method="adabelief")

        kwargs_optim = {
            "max_iterations": max_iterations,
            "min_iterations": None,
            "init_learning_rate": 1.0e-2,
            "schedule_learning_rate": True,
            "restart_from_init": False,
            "stop_at_loss_increase": False,
            "progress_bar": True,
            "return_param_history": True,
        }

        best_fit, _, extra_fields, _ = optim.minimize(**kwargs_optim)
        kwargs_final = parameters.args2kwargs(best_fit)
        psf_guess = model.get_full_psf(**kwargs_final)

        error_map = model.get_psf_error_map(
            kwargs=kwargs_final,
            data=star_data_list,
            sigma_2=variance,
            masks=mask_data_list,
            error_method="std_residuals",
            high_res=True,
        )

        final_psf_mask = psf_guess > cut_threshold
        final_psf = np.where(final_psf_mask, psf_guess, 0)
        # STARRED returns a downsampled PSF guess, so no need to reshape the error map
        final_error_map = np.where(final_psf_mask, error_map, 0)

        kwargs_starred = {
            "extra_fields": {"loss_history": extra_fields["loss_history"]},
            "model": model,
            "data": star_data_list,
            "sigma_2": variance,
            "kwargs_final": kwargs_final,
            "masks": mask_data_list,
        }

        self.plot_psf_and_variance_map(
            method="STARRED",
            psf_guess=psf_guess,
            variance_map=error_map**2,
            psf_cut=final_psf,
            variance_map_cut=final_error_map**2,
            kwargs_starred=kwargs_starred,
        )

        if save:
            self.file_system.save_psf_and_variance_map(
                psf_class=self, psf_guess=final_psf, variance_map=final_error_map**2
            )

        return final_psf, final_error_map**2

    def plot_psf_candidates(
        self, mask, star_exposures, star_weights, noise_maps, catalog
    ):
        """Plot some diagnostics on the PSF candidate stars.

        :param mask: mask corresponding to candidate stars from the catalog
        :type mask: `bool`
        :param star_exposures: candidate star cutouts
        :type star_exposures: `np.ndarray`
        :param star_weights: candidate star weight cutouts
        :type star_weights: `np.ndarray`
        :param noise_maps: candidate star noise maps
        :type noise_maps: `np.ndarray`
        :param catalog: fits table SExtractor catalog as determined by
        :meth:`~get_kwargs_cut`
        :type catalog: `Table`

        :return: figures of candidate star Flux Radius vs. Magnitude, cutouts,
            weight maps, error maps, and locations in the full science image
        :rtype: 4 `fig`
        """

        mag = np.array(catalog.data["MAG_BEST"], dtype=float)
        size = np.array(catalog.data["FLUX_RADIUS"], dtype=float)

        with fits.open(self.image_file_name) as hdul:
            header = hdul[0].header
            wcs = WCS(header)

        plt.plot(mag[mask == 0], size[mask == 0], "og")
        plt.plot(mag[mask > 0], size[mask > 0], "or")
        plt.xlim([0, np.max(mag)])
        plt.ylim([0, np.max(size)])
        plt.xlabel("Magnitude")
        plt.ylabel("Flux Radius")
        plt.show()

        # sort by object magnitude
        star_magnitudes = np.array(catalog.data["MAG_BEST"][mask], dtype=float)
        mag_sort_idx = np.argsort(star_magnitudes)
        star_magnitudes = star_magnitudes[mag_sort_idx]
        star_ids = np.array(catalog.data["NUMBER"][mask])[mag_sort_idx]

        num_stars = len(mask.nonzero()[0])
        ncols = 4
        nrows = (num_stars + ncols - 1) // ncols  # calculate number of rows needed

        # PLOT STAR CUTOUTS
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3 * nrows))
        ax = ax.flatten()
        for i in range(num_stars):
            ax[i].imshow(np.log10(star_exposures[i]), cmap="viridis")
            ax[i].set_title(f"Star {i}: ID {star_ids[i]} Mag {star_magnitudes[i]:.2f}")
            ax[i].axis("off")
        fig.suptitle("STAR CUTOUTS", fontsize=15)

        # hide any remaining unused subplots
        for j in range(num_stars, nrows * ncols):
            ax[j].axis("off")

        plt.tight_layout()
        plt.show()

        # PLOT WEIGHT MAPS
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3 * nrows))
        ax = ax.flatten()
        for i in range(num_stars):
            ax[i].imshow(np.log10(star_weights[i]), cmap="viridis")
            ax[i].set_title(f"Star {i}: ID {star_ids[i]} Mag {star_magnitudes[i]:.2f}")
            ax[i].axis("off")
        fig.suptitle("WEIGHT CUTOUTS", fontsize=15)

        # hide any remaining unused subplots
        for j in range(num_stars, nrows * ncols):
            ax[j].axis("off")

        plt.tight_layout()
        plt.show()

        # PLOT NOISE MAPS
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3 * nrows))
        ax = ax.flatten()
        for i in range(num_stars):
            ax[i].imshow(np.log10(noise_maps[i]), cmap="viridis")
            ax[i].set_title(f"Star {i}: ID {star_ids[i]} Mag {star_magnitudes[i]:.2f}")
            ax[i].axis("off")
        fig.suptitle(r"$\sigma$", fontsize=15)

        # hide any remaining unused subplots
        for j in range(num_stars, nrows * ncols):
            ax[j].axis("off")

        plt.tight_layout()
        plt.show()

        # PLOT COUNTS VS VARIANCE
        fig, ax = plt.subplots(figsize=(8, 6))

        for i in range(num_stars):
            counts = star_exposures[i].flatten()

            # variance = sigma^2
            variance = noise_maps[i] ** 2
            variance = variance.flatten()

            # remove bad pixels
            mask_good = (
                np.isfinite(counts)
                & np.isfinite(variance)
                & (counts > 0)
                & (variance > 0)
            )

            ax.scatter(
                counts[mask_good], variance[mask_good], alpha=0.2, label=f"Star {i}"
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Counts")
        ax.set_ylabel(r"$\sigma^2$")
        ax.set_title("Counts vs Variance of Stars")
        ax.legend()

        plt.show()

        star_coords_list = [
            (int(i), int(j))
            for i, j in zip(
                catalog.data[mask]["X_IMAGE"], catalog.data[mask]["Y_IMAGE"]
            )
        ]

        # Turn the pixel coordinates in pixels to WCS coordinates
        star_ang = [wcs.all_pix2world(i[0], i[1], 0) for i in star_coords_list]
        star_coords = {}
        star_coords = np.round(  # Convert WCS coordinates to a pixel center
            [wcs.all_world2pix(i[0], i[1], 0) for i in star_ang]
        ).astype(int)

        with fits.open(self.image_file_name) as hdul:
            if self.instrument == "JWST":
                data_full = hdul[1].data
            else:
                data_full = hdul[0].data

        _, ax = plt.subplots(figsize=(10, 10))
        plt_data = np.log10(data_full + 0.1)

        im = ax.matshow(plt_data, origin="lower", vmin=-1, vmax=2.0)
        ax.autoscale(False)
        divider = make_axes_locatable(ax)

        # put the points on the image
        for ix, i in enumerate(star_coords):
            plt.scatter(i[0], i[1], 10)
            plt.text(i[0] + 50, i[1] + 50, f"{ix}", color="white")

        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.show()

    def plot_saved_psf_candidates(self):
        """Plot the saved star cutouts, weight cutouts, and noise map cutouts with their
        correspdoning masks to probe appropriate mask configurations.

        :return: figures of candidate star cutouts, weight maps, and error maps
          with masks applied
        :rtype: `fig`
        """

        star_exposures, mask_data_list, star_weights, noise_maps = (
            self.load_psf_candidate_attributes()
        )

        num_stars = len(star_exposures)
        ncols = 4
        nrows = (num_stars + ncols - 1) // ncols  # calculate number of rows needed

        # PLOT STAR CUTOUTS
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3 * nrows))
        ax = ax.flatten()
        for i in range(num_stars):
            image = np.log10(star_exposures[i])
            mask = mask_data_list[i]

            ax[i].imshow(image, cmap="viridis", origin="lower")

            # show masked pixels in red
            ax[i].imshow(
                np.ma.masked_where(mask, ~mask),  # only display masked pixels
                cmap="Reds",
                alpha=1,
            )

            ax[i].set_title(f"Star {i}")
            ax[i].axis("off")
        fig.suptitle("STAR CUTOUTS", fontsize=15)

        # hide any remaining unused subplots
        for j in range(num_stars, nrows * ncols):
            ax[j].axis("off")

        plt.tight_layout()
        plt.show()

        # PLOT WEIGHT MAPS
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3 * nrows))
        ax = ax.flatten()
        for i in range(num_stars):
            mask = mask_data_list[i]
            ax[i].imshow(np.log10(star_weights[i]), cmap="viridis")
            # show masked pixels in red
            ax[i].imshow(
                np.ma.masked_where(mask, ~mask),  # only display masked pixels
                cmap="Reds",
                alpha=1,
            )
            ax[i].set_title(f"Star {i}")
            ax[i].axis("off")
        fig.suptitle("WEIGHT CUTOUTS", fontsize=15)

        # hide any remaining unused subplots
        for j in range(num_stars, nrows * ncols):
            ax[j].axis("off")

        plt.tight_layout()
        plt.show()

        # PLOT NOISE MAPS
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3 * nrows))
        ax = ax.flatten()
        for i in range(num_stars):
            mask = mask_data_list[i]
            ax[i].imshow(np.log10(noise_maps[i]), cmap="viridis")
            # show masked pixels in red
            ax[i].imshow(
                np.ma.masked_where(mask, ~mask),  # only display masked pixels
                cmap="Reds",
                alpha=1,
            )
            ax[i].set_title(f"Star {i}")
            ax[i].axis("off")
        fig.suptitle(r"$\sigma$", fontsize=15)

        # hide any remaining unused subplots
        for j in range(num_stars, nrows * ncols):
            ax[j].axis("off")

        plt.tight_layout()
        plt.show()

        # PLOT COUNTS VS VARIANCE
        fig, ax = plt.subplots(figsize=(8, 6))

        for i in range(num_stars):
            counts = star_exposures[i].flatten()

            # variance = sigma^2
            variance = noise_maps[i] ** 2
            variance = variance.flatten()

            # remove bad pixels
            mask_good = (
                np.isfinite(counts)
                & np.isfinite(variance)
                & (counts > 0)
                & (variance > 0)
            )

            ax.scatter(
                counts[mask_good], variance[mask_good], alpha=0.2, label=f"Star {i}"
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Counts")
        ax.set_ylabel(r"$\sigma^2$")
        ax.set_title("Counts vs Variance of Stars")
        ax.legend()

        plt.show()

    @staticmethod
    def plot_psf_and_variance_map(
        method,
        psf_guess,
        variance_map,
        psf_cut=None,
        variance_map_cut=None,
        kwargs_starred=None,
    ):
        """Plot the PSF and variance map determined by the respective fitting method.

        :param method: fitting method used to option the PSF. Options are "PSFr" and "STARRED"
        :type method: `str`
        :param psf_guess: initial PSF as determined by either :meth:`~dolphin.preprocessing.PSF.make_psf_psfr`
        or `~dolphin.preprocessing.PSF.make_psf_starred`
        :type psf_guess: `array`
        :param variance_map: initial PSF variance map as determined by either :meth:`~dolphin.preprocessing.PSF.make_psf_psfr`
        or `~dolphin.preprocessing.PSF.make_psf_starred`
        :type variance_map: `array`
        :param psf_cut: (optional) cut PSF
        :type psf_cut: `array`
        :param variance_map_cut: (optional) cut PSF variance map
        :type variance_map_cut: `array`
        :param kwargs_starred: (optional) STARRED arguments corresponding to their helper functions
        :type kwargs_starred: `dict`

        :return: plot of the PSF guess from the respective fitting method, alongisde the error
            map and cut PSF/variance map if applicable
        :rtype: `fig`
        """

        if method == "PSFr":
            _, ax = plt.subplots(1, 2)
            ax[0].imshow(np.log10(psf_guess), origin="lower", cmap="viridis")
            ax[0].set_title(r"$\log_{10}$(PSF)")
            ax[1].imshow(np.log10(variance_map), origin="lower")
            ax[1].set_title(r"$\log_{10}(\sigma^2$)")
            plt.tight_layout()
            plt.show()

            if psf_cut is not None and variance_map_cut is not None:
                _, ax = plt.subplots(1, 2)
                ax[0].imshow(np.log10(psf_cut))
                ax[0].set_title(r"$\log_{10}$(PSF) CUT")
                cut_fraction = 100 * (1 - np.sum(psf_cut) / np.sum(psf_guess))
                ax[0].set_xlabel(f"Cut Fraction: {abs(cut_fraction):.2f}%")
                ax[1].imshow(np.log10(variance_map_cut))
                ax[1].set_xlabel(f"Cut Fraction: {abs(cut_fraction):.2f}%")
                ax[1].set_title(r"$\log_{10}(\sigma^2$) CUT")
                plt.tight_layout()
                plt.show()
        elif method == "STARRED":
            _ = pltf.plot_loss(kwargs_starred["extra_fields"]["loss_history"])
            plt.show()

            _, ax = plt.subplots(1, 2)
            ax[0].imshow(np.log10(psf_guess), origin="lower", cmap="viridis")
            ax[0].set_title(r"$\log_{10}$(PSF)")
            ax[1].imshow(np.log10(variance_map), origin="lower")
            ax[1].set_title(r"$\log_{10}(\sigma^2$)")
            plt.tight_layout()
            plt.show()

            if psf_cut is not None and variance_map_cut is not None:
                _, ax = plt.subplots(1, 2)
                ax[0].imshow(np.log10(psf_cut))
                ax[0].set_title(r"$\log_{10}$(PSF) CUT")
                cut_fraction = 100 * (1 - np.sum(psf_cut) / np.sum(psf_guess))
                ax[0].set_xlabel(f"Cut Fraction: {abs(cut_fraction):.2f}%")
                ax[1].imshow(np.log10(variance_map_cut))
                ax[1].set_xlabel(f"Cut Fraction: {abs(cut_fraction):.2f}%")
                ax[1].set_title(r"$\log_{10}(\sigma^2$) CUT")
                plt.tight_layout()
                plt.show()

    def load_saved_psf(self, plot=True):
        """Load the saved PSF and variance map generated by
        :class:`~dolphin.preprocessing.psf.PSF`.

        :param plot: whether or not to plot the saved PSF and variance map
        :type plot: `bool`

        :return: a tuple containing the saved PSF and variance map
        :rtype: `tuple` (`array`, `array`)
        """

        psf_data, variance_map = self.file_system.load_saved_psf(self)

        if plot:
            if variance_map is None:
                _, ax = plt.subplots(1)
                ax[0].imshow(np.log10(psf_data), origin="lower", cmap="viridis")
                ax[0].set_title(r"$\log_{10}$(PSF)")
            else:
                _, ax = plt.subplots(1, 2)
                ax[0].imshow(np.log10(psf_data), origin="lower", cmap="viridis")
                ax[0].set_title(r"$\log_{10}$(PSF)")
                ax[1].imshow(np.log10(variance_map), origin="lower")
                ax[1].set_title(r"$\log_{10}(\sigma^2$)")

            plt.tight_layout()
            plt.show()

        return psf_data, variance_map

    def load_catalog_table(self):
        """Get the SExtractor catalog if already made.

        :return: fits table SExtractor catalog as determined by
        :meth:`~dolphin.preprocessing.psf.PSF.get_kwargs_cut`
        :rtype: `table`
        """

        catalog = self.file_system.load_catalog_table(self)
        return catalog

    def load_psf_candidate_attributes(self):
        """Reload the saved star cutouts, corresponding masks, weight maps, and noise
        maps needed by :class:`~dolphin.preprocessing.psf.PSF`.

        return: A tuple containing the saved star cutouts, matched masks, weight maps, and saved noise maps.
        :rtype: `tuple` (`np.ndarray`, `np.ndarray`, `np.ndarray`, `np.ndarray`)
        """

        star_data_list, mask_data_list, weight_maps, noise_maps = (
            self.file_system.load_psf_candidate_attributes(self)
        )
        return star_data_list, mask_data_list, weight_maps, noise_maps
