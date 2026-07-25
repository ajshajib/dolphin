# -*- coding: utf-8 -*-
"""This class contains helper functions to create a PSF using the STARRED and PSFr
methodologies."""

__author__ = "brady-ryan"

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import NDData
from astropy.table import Table

from photutils.psf import extract_stars
from photutils.detection import find_peaks

from dolphin.processor.files import FileSystem
from dolphin.preprocessing import preprocessing_util

from psfr import psfr

from starred.psf.psf import PSF as STARRED_PSF
from starred.psf.loss import Loss
from starred.optim.optimization import Optimizer
from starred.psf.parameters import ParametersPSF
from starred.utils.noise_utils import propagate_noise
from starred.plots import plot_function as pltf


class PSF:
    """This class contains helper functions to create a PSF using the STARRED and PSFr
    methodologies."""

    def __init__(
        self,
        io_directory,
        lens_name,
        data_band,
        instrument,
        full_image_file,
        weight_image_file=None,
    ):
        """Initiate the class from the following inputs:

        :param io_directory: path to the input/output directory. Should not end with slash.
        :type io_directory: `str`
        :param lens_name: name of the system to create a PSF for
        :type lens_name: `str`
        :param data_band: data band of desired PSF
        :type data_band: `str`
        :param instrument: instrument which took the data. Current options are
          "HST" and "JWST"
        :type instrument: `str`
        :param full_image_file: path to the full science image FITS file
        :type full_image_file: `str`
        :param weight_image_file: (optional) if analyzing HST data,
          the path to the full weight image FITS file
        :type weight_image_file: `str`
        """
        self.io_directory = io_directory
        self.file_system = FileSystem(io_directory)
        self.lens_name = lens_name
        self.data_band = data_band

        self.instrument = instrument
        supported_instruments = ["HST", "JWST"]
        if instrument not in supported_instruments:
            raise ValueError(
                f"{instrument} is not supported! Options are: {supported_instruments}"
            )

        self.image_file_name = full_image_file
        self.weight_file_name = weight_image_file

    def get_psf_candidates(
        self,
        threshold_over_background=1000,
        cutout_size=51,
        exclude_specific=None,
        include_specific=None,
        save=False,
    ):
        """Obtain PSF candidates using `photutils`. In addition to cutouts of the
        candidate objects being created, weight cutouts and noise map cutouts are made.
        To save the cutouts, toggle `save = True`. Can be run again excluding or
        including specific objects to narrow down the initial candidates.

        :param threshold_over_background: (optional) threshold over the background for which
          candidate objects will be indentified
        :type threshold_over_background: `int`
        :param cutout_size: (optional) size (in pixels) of one side of the square cutout
        :type cutout_size: `int`
        :param exclude_specific: (optional) list of specific star numbers to exclude
        :type exclude_specific: `list` of `int`
        :param include_specific: (optional) list of specific star numbers to include. Will take
          priority over `exclude_specific`.
        :type include_specific: `list` of `int`
        :param save: (optional) boolean dictating whether or not to save the star cutouts
          and corresponding weight maps and noise maps
        :type plot: `bool`
        :return: tuple containing the cutout, weight, and noise map data for each star
        :rtype: `tuple` (`np.ndarray`, `np.ndarray`, `np.ndarray`)
        """
        mean_bkd, sigma_bkd = preprocessing_util.get_background(self.image_file_name)
        if self.instrument == "HST":
            with fits.open(self.image_file_name) as hdul:
                sci = hdul[0].data
            image_reduced = sci - mean_bkd

            with fits.open(self.weight_file_name) as hdul:
                wht = hdul[0].data
            wht[wht <= 0] = 10 ** (-10)
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
            image_reduced = sci - mean_bkd
            err = np.sqrt(variance)

        peaks_table = find_peaks(
            image_reduced,
            threshold=threshold_over_background * sigma_bkd,
        )
        peaks_table.sort("peak_value", reverse=True)
        half_size = (cutout_size - 1) / 2
        x = peaks_table["x_peak"]
        y = peaks_table["y_peak"]

        # mask out sources near the edges
        mask = (
            (x > half_size)
            & (x < (image_reduced.shape[1] - 1 - half_size))
            & (y > half_size)
            & (y < (image_reduced.shape[0] - 1 - half_size))
        )

        stars_table = Table()
        stars_table["x"] = x[mask]
        stars_table["y"] = y[mask]
        stars_table["x_peak"] = peaks_table["x_peak"][mask]
        stars_table["y_peak"] = peaks_table["y_peak"][mask]

        data_nddata_obj = NDData(data=image_reduced)
        weight_nddata_obj = NDData(data=wht)
        if self.instrument == "HST":
            noise_nddata_obj = NDData(
                data=np.sqrt(np.abs(image_reduced / wht) + sigma_bkd**2)
            )
        else:
            noise_nddata_obj = NDData(data=err)

        keep = np.ones(len(stars_table), dtype=bool)
        # remove specific star numbers
        if exclude_specific is not None:
            keep = ~np.isin(np.arange(len(stars_table)), exclude_specific)

        # keep specific star numbers if there are too many candidates
        if include_specific is not None:
            keep = np.zeros(len(stars_table), dtype=bool)
            keep = np.isin(np.arange(len(stars_table)), include_specific)

        stars_table = stars_table[keep]

        star_cutouts = extract_stars(data_nddata_obj, stars_table, size=cutout_size)
        weight_cutouts = extract_stars(weight_nddata_obj, stars_table, size=cutout_size)
        noise_cutouts = extract_stars(noise_nddata_obj, stars_table, size=cutout_size)

        print(f"Found {len(star_cutouts)} candidate objects!")

        self.plot_psf_candidates(
            star_exposures=star_cutouts,
            star_weights=weight_cutouts,
            noise_maps=noise_cutouts,
            stars_table=stars_table,
        )

        if save:
            self.file_system.save_star_cutouts(
                lens_name=self.lens_name,
                data_band=self.data_band,
                star_exposures=star_cutouts,
                star_weights=weight_cutouts,
                noise_maps=noise_cutouts,
            )

        star_data_list = star_cutouts.data
        weight_data_list = weight_cutouts.data
        noise_data_list = noise_cutouts.data

        return star_data_list, weight_data_list, noise_data_list

    def make_candidate_mask(
        self, star_data_list, weight_data_list, noise_map_list, star_num, kwargs_mask
    ):
        """Create a mask for a PSF candidate object.

        :param star_data_list: list of arrays corresponding to the cutout star data
          as returned by :meth:`~dolphin.preprocessing.psf.get_psf_candidates`
        :type star_data_list: `list` of `np.ndarray`
        :param weight_data_list: list of arrays corresponding to the cutout weight data
          as returned by :meth:`~dolphin.preprocessing.psf.get_psf_candidates`
        :type weight_data_list: `list` of `np.ndarray`
        :param noise_map_list: list of arrays corresponding to the cutout noise map data
            as returned by :meth:`~dolphin.preprocessing.psf.get_psf_candidates`
        :type noise_map_list: `list` of `np.ndarray`
        :param star_num: the number of the saved star cutout to apply the mask to
        :type star_num: `int`
        :param kwargs_mask: list of dictionaries corresponding to masking keywork arguments. Supported types,
            with all required keywords, are as follows: [{"type": "circle", "center": `tuple` (`int`, `int`),
            "radius": `int`}, {"type": "square", "center": `tuple` (`int`, `int`), "size": `int`},
            {"type": "ellipse", "center": `tuple` (`int`, `int`), "a": `int`, "b": `int`}]. To invert the boolean
            logic of a specific mask index, one must place "invert": True in that dictionary.
        :type kwargs_mask: `list` of `dict`
        :return: a boolean mask corresponding to the specified configuration
        :rtype: `bool`
        """
        star_exposure = star_data_list[star_num]
        weight_map = weight_data_list[star_num]
        noise_map = noise_map_list[star_num]

        mask = preprocessing_util.build_mask(star_exposure.shape, kwargs_mask)

        fig, ax = plt.subplots(1, 3, figsize=(10, 6))
        im_star = ax[0].imshow(np.log10(star_exposure * mask))
        ax[0].set_title("Candidate Cutout")
        fig.colorbar(im_star, ax=ax[0], fraction=0.05)

        im_weight = ax[1].imshow(np.log10(weight_map * mask))
        ax[1].set_title("Weight Map")
        fig.colorbar(im_weight, ax=ax[1], fraction=0.05)

        im_noise = ax[2].imshow(np.log10(noise_map * mask))
        ax[2].set_title(r"$\sigma$")
        fig.colorbar(im_noise, ax=ax[2], fraction=0.05)

        plt.tight_layout()

        return mask

    def make_psf_psfr(
        self,
        star_data_list,
        noise_map_list,
        mask_list=None,
        oversampling=1,
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

        :param star_data_list: list of arrays corresponding to the cutout star data
          as returned by :meth:`~dolphin.preprocessing.psf.get_psf_candidates`
        :type star_data_list: `list` of `np.ndarray`
        :param noise_map_list: list of arrays corresponding to the cutout noise map data
            as returned by :meth:`~dolphin.preprocessing.psf.get_psf_candidates`
        :type noise_map_list: `list` of `np.ndarray`
        :param mask_list: (optional) list of boolean arrays corresponding to pixels to be masked
          for individual stars across the candidate cutouts. If not provided, all pixels will
          be assumed `True`.
        :type mask_list: `list` of `np.ndarray` of `bool`
        :param oversampling: (optional) higher-resolution PSF reconstruction and return
        :type oversampling: `int`
        :param saturation_limit: (optional) float or list of floats of length of star_list
          pixel values above this threshold will not be considered in the reconstruction.
        :type saturation_limit: `float` or `list of floats` of length of star_data_list
        :param num_iteration: (optional)  number of iterative corrections applied on the PSF based on previous guess
        :type num_iteration: `int`
        :param n_recenter: (optional) every n_recenter iterations of the updated PSF, a re-centering of
          the centroids are performed with the updated PSF guess
        :type n_recenter: `int`
        :param kwargs_one_step: keyword arguments to be passed to one_step_psf_estimate() method
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
          `dolphin` format
        :type save: `bool`
        :return: a tuple containing the PSF array and PSF variance map array
        :rtype: `tuple` (`np.ndarray`, `np.ndarray`)
        """
        if mask_list is None:
            mask_list = []
            num_stars = len(star_data_list)
            star_shape = star_data_list[0].shape
            for _ in range(num_stars):
                mask_list.append(np.ones(star_shape))

        variance = [noise_map**2 for noise_map in noise_map_list]

        psf_returns = psfr.stack_psf(
            star_list=star_data_list,
            oversampling=oversampling,
            mask_list=mask_list,
            error_map_list=variance,
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

        # process the center list for the error map
        new_center_list = []
        for i, _ in enumerate(center_list):
            new_center_list.append([center_list[i][0], center_list[i][1]])
        new_center_list = np.array(new_center_list)
        variance_map = psfr.psf_error_map(
            star_list=star_data_list,
            error_map_list=variance,
            psf_kernel=psf_guess,
            center_list=new_center_list,
            mask_list=mask_list,
            oversampling=oversampling,
        )

        final_psf_mask = psf_guess > cut_threshold
        final_psf = np.where(final_psf_mask, psf_guess, 0)
        if oversampling > 1:
            # downsample mask to variance map resolution
            native_nx = variance_map.shape[0]
            native_ny = variance_map.shape[1]

            error_map_mask = final_psf_mask.reshape(
                native_nx, oversampling, native_ny, oversampling
            ).any(axis=(1, 3))

            final_variance_map = np.where(error_map_mask, variance_map, 0)
        else:
            final_variance_map = np.where(final_psf_mask, variance_map, 0)

        self.plot_psf_and_variance_map(
            method="PSFr",
            psf_guess=psf_guess,
            variance_map=final_variance_map,
            psf_cut=final_psf,
            variance_map_cut=final_variance_map,
        )

        if save:
            self.file_system.save_psf_and_variance_map(
                lens_name=self.lens_name,
                data_band=self.data_band,
                psf_guess=final_psf,
                variance_map=final_variance_map,
            )

        return final_psf, final_variance_map

    def make_psf_starred(
        self,
        star_data_list,
        noise_map_list,
        mask_list=None,
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

        :param star_data_list: list of arrays corresponding to the cutout star data
          as returned by :meth:`~dolphin.preprocessing.psf.get_psf_candidates`
        :type star_data_list: `list` of `np.ndarray`
        :param noise_map_list: list of arrays corresponding to the cutout noise map data
            as returned by :meth:`~dolphin.preprocessing.psf.get_psf_candidates`
        :type noise_map_list: `list` of `np.ndarray`
        :param mask_list: (optional) list of boolean arrays corresponding to pixels to be masked
          for individual stars across the candidate cutouts. If not provided, all pixels will
          be assumed `True`.
        :type mask_list: `list` of `np.ndarray` of `bool`
        :param max_iterations: (optional) maximum number of iterations to use in the final minimization
        :type max_iterations: `int`
        :param subsampling_factor: (optional) higher-resolution PSF reconstruction and return
        :type subsampling_factor: `int`
        :param convolution_method: (optional) method to use to calculate the convolution,
          choose between 'fft', 'scipy', and 'lax`. Recommended if jax>=0.4.9 - 'scipy'
        :type convolution_method: `str`
        :param include_moffat: (optional) True for the PSF to be expressed as the sum of a
          Moffat and a grid of pixels. False to not include the Moffat. Default: True
        :type include_moffat: bool
        :param elliptical_moffat: (optional) Allow elliptical Moffat.
        :type elliptical_moffat: bool
        :param regularization_terms: (optional) information about the regularization terms
        :type regularization_terms: `str`
        :param regularization_strength_scales: (optional) Lagrange parameter that weights
          intermediate scales in the transformed domain.
        :type regularization_strength_scales: `float`
        :param regularization_strength_hf: (optional) Lagrange parameter weighting the highest frequency scale
        :type regularization_strength_hf: `float`
        :param cut_threshold: (optional) signal threshold in which pixels under this value will not be
          saved in the final PSF
        :type cut_threshold: `float`
        :param save: (optional) whether or not to save the output PSF and variance map in the expected
          `dolphin` format
        :type save: `bool`
        :return: a tuple containing the PSF array and PSF variance map array
        :rtype: `tuple` (`np.ndarray`, `np.ndarray`)
        """
        star_data_list = np.asarray(star_data_list)
        noise_map_list = np.asarray(noise_map_list)
        variance = noise_map_list**2

        if mask_list is None:
            mask_list = []
            num_stars = len(star_data_list)
            star_shape = star_data_list[0].shape
            for _ in range(num_stars):
                mask_list.append(np.ones(star_shape))
        mask_list = np.asarray(mask_list)

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
            masks=mask_list,
        )

        optimizer = Optimizer(
            loss_class=loss, param_class=parameters, method="Newton-CG"
        )
        optimizer_options = {"maxiter": 1000, "restart_from_init": True}

        best_fit, _, extra_fields, _ = optimizer.minimize(**optimizer_options)
        kwargs_partial = parameters.args2kwargs(best_fit)

        # compute noise level in starlet space and propagate Poisson noise
        W = propagate_noise(
            model=model,
            noise_maps=noise_map_list,
            kwargs=kwargs_partial,
            masks=mask_list,
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
            masks=mask_list,
        )

        optimizer = Optimizer(
            loss_class=loss, param_class=parameters, method="adabelief"
        )

        kwargs_optimizer = {
            "max_iterations": max_iterations,
            "min_iterations": None,
            "init_learning_rate": 1.0e-2,
            "schedule_learning_rate": True,
            "restart_from_init": False,
            "stop_at_loss_increase": False,
            "progress_bar": True,
            "return_param_history": True,
        }

        best_fit, _, extra_fields, _ = optimizer.minimize(**kwargs_optim)
        kwargs_final = parameters.args2kwargs(best_fit)
        psf_guess = model.get_full_psf(**kwargs_final)

        error_map = model.get_psf_error_map(
            kwargs=kwargs_final,
            data=star_data_list,
            sigma_2=variance,
            masks=mask_list,
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
            "masks": mask_list,
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
                lens_name=self.lens_name,
                data_band=self.data_band,
                psf_guess=final_psf,
                variance_map=final_error_map**2,
            )

        return final_psf, final_error_map**2

    def plot_psf_candidates(
        self, star_exposures, star_weights, noise_maps, stars_table
    ):
        """Plot some diagnostics on the PSF candidate stars.

        :param star_exposures: candidate star cutouts
        :type star_exposures: `EPSFStar`
        :param star_weights: candidate star weight cutouts
        :type star_weights: `EPSFStar`
        :param noise_maps: candidate star noise maps
        :type noise_maps: `EPSFStar`
        :param stars_table: table of cutout objects and their coordinates,
          as determined by `photutils.find_peaks`
        :type stars_table: `Table`
        :return: figures of candidate star cutouts, weight maps, error maps,
          variance vs. counts, and locations in the full science image
        :rtype: 5 `fig`
        """
        x_peaks = stars_table["x_peak"]
        y_peaks = stars_table["y_peak"]

        num_stars = len(star_exposures)
        ncols = 4
        nrows = (num_stars + ncols - 1) // ncols  # calculate number of rows needed

        # plot star cutouts
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3 * nrows))
        ax = ax.flatten()
        for i in range(num_stars):
            im_star = ax[i].imshow(np.log10(star_exposures[i].data), cmap="viridis")
            ax[i].set_title(f"Star {i}")
            ax[i].axis("off")
            fig.colorbar(im_star, ax=ax[i], fraction=0.05)
        fig.suptitle("STAR CUTOUTS", fontsize=15)

        # hide any remaining unused subplots
        for j in range(num_stars, nrows * ncols):
            ax[j].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.985])
        plt.show()

        # plot weight maps
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3 * nrows))
        ax = ax.flatten()
        for i in range(num_stars):
            im_weight = ax[i].imshow(np.log10(star_weights[i].data), cmap="viridis")
            ax[i].set_title(f"Star {i}")
            ax[i].axis("off")
            fig.colorbar(im_weight, ax=ax[i], fraction=0.05)
        fig.suptitle("WEIGHT CUTOUTS", fontsize=15)

        # hide any remaining unused subplots
        for j in range(num_stars, nrows * ncols):
            ax[j].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.985])
        plt.show()

        # plot noise maps
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3 * nrows))
        ax = ax.flatten()
        for i in range(num_stars):
            im_noise = ax[i].imshow(np.log10(noise_maps[i].data), cmap="viridis")
            ax[i].set_title(f"Star {i}")
            ax[i].axis("off")
            fig.colorbar(im_noise, ax=ax[i], fraction=0.05)
        fig.suptitle(r"$\sigma$", fontsize=15)

        # hide any remaining unused subplots
        for j in range(num_stars, nrows * ncols):
            ax[j].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.985])
        plt.show()

        # plot variance vs. counts
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = []

        for i in range(num_stars):
            counts = star_exposures[i].data

            # variance = sigma^2
            variance = noise_maps[i].data ** 2

            sc = ax.scatter(counts, variance, alpha=0.8)
            colors.append(sc.get_facecolor()[0])  # save the scatter color

        # create color-coded text in columns of 10
        n_per_col = 10
        x0 = 0.02
        dx = 0.18
        y0 = 0.98
        dy = 0.04

        for i in range(num_stars):
            col = i // n_per_col
            row = i % n_per_col

            ax.text(
                x0 + col * dx,
                y0 - row * dy,
                f"Star {i}",
                transform=ax.transAxes,
                fontsize=10,
                va="top",
                ha="left",
                color=colors[i],
                bbox=dict(facecolor="white", alpha=1.0, edgecolor="none", pad=0.2),
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Counts")
        ax.set_ylabel(r"$\sigma^2$")
        ax.set_title("Variance vs. Counts of Stars")

        plt.show()

        star_coords_list = [(int(i), int(j)) for i, j in zip(x_peaks, y_peaks)]

        data_full, header = fits.getdata(self.image_file_name, header=True)
        wcs = WCS(header)
        # turn the pixel coordinates in pixels to WCS coordinates
        star_ang = [wcs.all_pix2world(i[0], i[1], 0) for i in star_coords_list]
        star_coords = {}
        star_coords = np.round(  # Convert WCS coordinates to a pixel center
            [wcs.all_world2pix(i[0], i[1], 0) for i in star_ang]
        ).astype(int)

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
        return fig

    def plot_saved_psf_candidates(self):
        """Plot the saved star cutouts, weight cutouts, and noise map cutouts with their
        correspdoning masks. Also, make a plot of variance vs. counts.

        :return: figures of candidate star cutouts, weight maps, and error maps
          with masks applied, plot of variance vs. counts of stars.
        :rtype: `fig`
        """

        star_exposures, star_weights, noise_maps = self.load_psf_candidate_attributes()

        num_stars = len(star_exposures)
        ncols = 4
        nrows = (num_stars + ncols - 1) // ncols  # calculate number of rows needed

        # plot star cutouts
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3 * nrows))
        ax = ax.flatten()
        for i in range(num_stars):
            image = np.log10(star_exposures[i])
            im_star = ax[i].imshow(image, cmap="viridis", origin="lower")
            ax[i].set_title(f"Star {i}")
            ax[i].axis("off")
            fig.colorbar(im_star, ax=ax[i], fraction=0.05)
        fig.suptitle("STAR CUTOUTS", fontsize=15)

        # hide any remaining unused subplots
        for j in range(num_stars, nrows * ncols):
            ax[j].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.985])
        plt.show()

        # plot weight cutouts
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3 * nrows))
        ax = ax.flatten()
        for i in range(num_stars):
            im_weight = ax[i].imshow(np.log10(star_weights[i]), cmap="viridis")
            ax[i].set_title(f"Star {i}")
            ax[i].axis("off")
            fig.colorbar(im_weight, ax=ax[i], fraction=0.05)
        fig.suptitle("WEIGHT CUTOUTS", fontsize=15)

        # hide any remaining unused subplots
        for j in range(num_stars, nrows * ncols):
            ax[j].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.985])
        plt.show()

        # plot noise maps
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3 * nrows))
        ax = ax.flatten()
        for i in range(num_stars):
            im_noise = ax[i].imshow(np.log10(noise_maps[i]), cmap="viridis")
            ax[i].set_title(f"Star {i}")
            ax[i].axis("off")
            fig.colorbar(im_noise, ax=ax[i], fraction=0.05)
        fig.suptitle(r"$\sigma$", fontsize=15)

        # hide any remaining unused subplots
        for j in range(num_stars, nrows * ncols):
            ax[j].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.985])
        plt.show()

        # plot variance vs. counts
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = []

        for i in range(num_stars):
            counts = star_exposures[i].flatten()

            # variance = sigma^2
            variance = noise_maps[i] ** 2
            variance = variance.flatten()

            sc = ax.scatter(counts, variance, alpha=0.8)
            colors.append(sc.get_facecolor()[0])  # save the scatter color

        # create color-coded text in columns of 10
        n_per_col = 10
        x0 = 0.02
        dx = 0.18
        y0 = 0.98
        dy = 0.04

        for i in range(num_stars):
            col = i // n_per_col
            row = i % n_per_col

            ax.text(
                x0 + col * dx,
                y0 - row * dy,
                f"Star {i}",
                transform=ax.transAxes,
                fontsize=10,
                va="top",
                ha="left",
                color=colors[i],
                bbox=dict(facecolor="white", alpha=1.0, edgecolor="none", pad=0.2),
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Counts")
        ax.set_ylabel(r"$\sigma^2$")
        ax.set_title("Variance vs. Counts of Stars")

        plt.show()
        return fig

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
        or :meth:`~dolphin.preprocessing.PSF.make_psf_starred`
        :type psf_guess: `array`
        :param variance_map: initial PSF variance map as determined by either :meth:`~dolphin.preprocessing.PSF.make_psf_psfr`
        or :meth:`~dolphin.preprocessing.PSF.make_psf_starred`
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
            fig, ax = plt.subplots(1, 2)
            im_psf = ax[0].imshow(np.log10(psf_guess), origin="lower", cmap="viridis")
            ax[0].set_title(r"$\log_{10}$(PSF)")
            fig.colorbar(im_psf, ax=ax[0], fraction=0.05)

            im_variance = ax[1].imshow(np.log10(variance_map), origin="lower")
            ax[1].set_title(r"$\log_{10}(\sigma^2$)")
            fig.colorbar(im_variance, ax=ax[1], fraction=0.05)

            plt.tight_layout()
            plt.show()

            if psf_cut is not None and variance_map_cut is not None:
                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(np.log10(psf_cut))
                ax[0].set_title(r"$\log_{10}$(PSF) CUT")
                cut_fraction = 100 * (1 - np.sum(psf_cut) / np.sum(psf_guess))
                ax[0].set_xlabel(f"Cut Fraction: {abs(cut_fraction):.2f}%")
                fig.colorbar(im_psf, ax=ax[0], fraction=0.05)

                ax[1].imshow(np.log10(variance_map_cut))
                ax[1].set_xlabel(f"Cut Fraction: {abs(cut_fraction):.2f}%")
                ax[1].set_title(r"$\log_{10}(\sigma^2$) CUT")
                fig.colorbar(im_variance, ax=ax[1], fraction=0.05)

                plt.tight_layout()
                plt.show()
        elif method == "STARRED":
            if kwargs_starred is not None:
                _ = pltf.plot_loss(kwargs_starred["extra_fields"]["loss_history"])
                plt.show()

            fig, ax = plt.subplots(1, 2)
            im_psf = ax[0].imshow(np.log10(psf_guess), origin="lower", cmap="viridis")
            ax[0].set_title(r"$\log_{10}$(PSF)")
            fig.colorbar(im_psf, ax=ax[0], fraction=0.05)

            im_variance = ax[1].imshow(np.log10(variance_map), origin="lower")
            ax[1].set_title(r"$\log_{10}(\sigma^2$)")
            fig.colorbar(im_variance, ax=ax[1], fraction=0.05)

            plt.tight_layout()
            plt.show()

            if psf_cut is not None and variance_map_cut is not None:
                _, ax = plt.subplots(1, 2)
                ax[0].imshow(np.log10(psf_cut))
                ax[0].set_title(r"$\log_{10}$(PSF) CUT")
                cut_fraction = 100 * (1 - np.sum(psf_cut) / np.sum(psf_guess))
                ax[0].set_xlabel(f"Cut Fraction: {abs(cut_fraction):.2f}%")
                fig.colorbar(im_psf, ax=ax[0], fraction=0.05)

                ax[1].imshow(np.log10(variance_map_cut))
                ax[1].set_xlabel(f"Cut Fraction: {abs(cut_fraction):.2f}%")
                ax[1].set_title(r"$\log_{10}(\sigma^2$) CUT")
                fig.colorbar(im_variance, ax=ax[1], fraction=0.05)

                plt.tight_layout()
                plt.show()

        return fig

    def load_saved_psf(self, plot=True):
        """Load the saved PSF and variance map generated by
        :class:`~dolphin.preprocessing.psf.PSF`.

        :param plot: whether or not to plot the saved PSF and variance map
        :type plot: `bool`
        :return: a tuple containing the saved PSF and variance map
        :rtype: `tuple` (`array`, `array`)
        """

        psf_data, variance_map = self.file_system.load_saved_psf(
            self.lens_name, self.data_band
        )

        if plot:
            fig, ax = plt.subplots(1, 2)
            im_psf = ax[0].imshow(np.log10(psf_data), origin="lower", cmap="viridis")
            ax[0].set_title(r"$\log_{10}$(PSF)")
            fig.colorbar(im_psf, ax=ax[0], fraction=0.05)

            im_variance = ax[1].imshow(np.log10(variance_map), origin="lower")
            ax[1].set_title(r"$\log_{10}(\sigma^2$)")
            fig.colorbar(im_variance, ax=ax[1], fraction=0.05)

            plt.tight_layout()
            plt.show()

        return psf_data, variance_map

    def load_psf_candidate_attributes(self):
        """Reload the saved star cutouts, corresponding masks, weight maps, and noise
        maps needed by :class:`~dolphin.preprocessing.psf.PSF`.

        :return: A tuple containing the saved star cutouts, matched masks, weight maps, and saved noise maps.
        :rtype: `tuple` (`np.ndarray`, `np.ndarray`, `np.ndarray`, `np.ndarray`)
        """

        star_data_list, weight_maps, noise_maps = (
            self.file_system.load_psf_candidate_attributes(
                self.lens_name, self.data_band
            )
        )
        return star_data_list, weight_maps, noise_maps

    def clean_psf_workspace(self):
        """Remove the saved PSF candidate cutouts, weight maps, and noise maps from the
        :class:`~dolphin.preprocessing.psf.PSF` workspace.

        :return: None
        """
        self.file_system.clean_psf_workspace(self.lens_name, self.data_band)
