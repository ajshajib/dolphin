"""This module contains the methods to perform various cosmological analyses."""

__author__ = "brady-ryan"

import warnings

import numpy as np
from lenstronomy.Analysis.td_cosmography import TDCosmography

from ..analysis.output import Output
from ..analysis.photometry import Photometry
from ..processor.config import ModelConfig
from ..processor.files import FileSystem


class Cosmology:
    """This module contains the methods to perform various cosmological analyses."""

    def __init__(
        self,
        io_directory,
        lens_name,
        model_id,
        z_lens,
        z_source,
        cosmology_model,
        walker_ratio,
        burn_in=0,
        n_samples=None,
        rng_seed=42,
        lens_model_kinematics_bool=None,
        light_model_kinematics_bool=None,
        kwargs_seeing=None,
        kwargs_aperture=None,
        anisotropy_model=None,
        **kwargs_kin_api,
    ):
        """Initialize the class with the following inputs:

        :param io_directory: path to the input/output directory. It should not end with a slash.
        :type io_directory: `str`
        :param lens_name: name of the system to analyze
        :type lens_name: `str`
        :param model_id: model ID of the lens system being analyzed
        :type model_id: `str`
        :param z_lens: redshift of the main deflector
        :type z_lens: `float`
        :param z_source: redshift of the source
        :type z_source: `float`
        :param cosmology_model: model `astropy.cosmology` instance
        :type cosmology_model: `astropy.cosmology` instance
        :param walker_ratio: number of walkers per parameter in MCMC
        :type walker_ratio: `int`
        :param burn_in: (optional) number of burn-in steps to discard
        :type burn_in: `int`
        :param n_samples: (optional) number of samples to randomly select from the flattened chain.
          If `None`, all samples after `burn_in` will be used.
        :type n_samples: `int`
        :param rng_seed: (optional) seed for the random number generator to ensure reproducibility
        :type rng_seed: `int`
        :param lens_model_kinematics_bool: (optional) bool list, corresponding to lens models being included into the
         kinematics modeling
        :type lens_model_kinematics_bool: `list` of `bool`
        :param light_model_kinematics_bool: (optional) bool list, corresponding to lens light models being included
         into the kinematics modeling
        :type light_model_kinematics_bool: `list` of `bool`
        :param kwargs_seeing: (optional) seeing conditions (see observation class in Galkin)
        :type kwargs_seeing: `dict`
        :param kwargs_aperture: (optional) aperture keyword arguments (see aperture class in Galkin)
        :type kwargs_aperture: `dict`
        :param anisotropy_model: (optional) string, anisotropy model type
        :type anisotropy_model: `str`
        :param kwargs_kin_api: additional keyword arguments for KinematicsAPI class instance
        :type kwargs_kin_api: `dict`
        """
        self.io_directory = io_directory
        self.output = Output(io_directory)
        self.lens_name = lens_name
        self.model_id = model_id
        self.z_lens = z_lens
        self.z_source = z_source
        self.cosmology = cosmology_model
        self.cosmology_model_class = type(self.cosmology)
        self.cosmology_model_base_params = dict(self.cosmology.parameters)
        self.walker_ratio = walker_ratio
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(self.rng_seed)

        self.output.load_output(
            self.lens_name,
            self.model_id,
            verbose=False,
        )

        self.param = self.output.get_param_class(
            lens_name=self.lens_name,
            model_id=self.model_id,
        )

        self.config = ModelConfig(
            lens_name=self.lens_name,
            io_directory=self.output.io_directory,
            settings=self.output.model_settings,
        )

        self.kwargs_model = self.config.get_kwargs_model()
        # disable multi-plane if in kwargs_model, as
        # velocity dispersion calculator cannot handle it
        if "multi_plane" in self.kwargs_model:
            self.kwargs_model["multi_plane"] = False

        # set sample number to length of posterior after burn-in
        # if not specified
        if self.n_samples is None:
            # get MCMC chain
            chain = self.output.get_reshaped_emcee_chain(
                self.lens_name,
                self.model_id,
                self.walker_ratio,
                self.burn_in,
            )

            # flatten walkers and steps
            flat_chain = chain.reshape(-1, chain.shape[-1])
            self.n_samples = flat_chain.shape[0]

        self.file_system = FileSystem(self.output.io_directory)
        self.TDCosmography = TDCosmography(
            self.z_lens,
            self.z_source,
            self.kwargs_model,
            cosmo_fiducial=self.cosmology,
            lens_model_kinematics_bool=lens_model_kinematics_bool,
            light_model_kinematics_bool=light_model_kinematics_bool,
            kwargs_seeing=kwargs_seeing,
            kwargs_aperture=kwargs_aperture,
            anisotropy_model=anisotropy_model,
            **kwargs_kin_api,
        )

    def fermat_potential(
        self,
        original_ps_position=False,
    ):
        """Sample the lens model posterior and compute the Fermat potential differences
        relative to the first image.

        :param original_ps_position: (optional) boolean (only applies when first point source model
          is of type 'LENSED_POSITION'), uses the image positions in the model
          parameters and does not re-compute images (which might be differently
          ordered) in case of the lens equation solver
        :type original_ps_position: `bool`

        :return: Array of Fermat potential differences relative
          to the first image in units of arcsec^2
        :rtype: `numpy.ndarray`
        """
        # get MCMC chain
        chain = self.output.get_reshaped_emcee_chain(
            self.lens_name,
            self.model_id,
            self.walker_ratio,
            self.burn_in,
        )

        # flatten walkers and steps
        flat_chain = chain.reshape(-1, chain.shape[-1])

        # randomly select indices from the converged chain
        random_indices = self.rng.choice(
            flat_chain.shape[0],
            size=self.n_samples,
            replace=False,
        )

        # select random samples
        flat_chain = flat_chain[random_indices]

        dphi_list = []
        for sample in flat_chain:
            # convert sampled parameters back to lenstronomy kwargs
            kwargs_out = self.param.args2kwargs(sample)
            kwargs_lens = kwargs_out["kwargs_lens"]
            kwargs_ps = kwargs_out["kwargs_ps"]

            # number of lensed images
            num_images = len(kwargs_ps[0]["ra_image"])

            # calculate Fermat potentials
            fermat_potential = self.TDCosmography.fermat_potential(
                kwargs_lens=kwargs_lens,
                kwargs_ps=kwargs_ps,
                original_ps_position=original_ps_position,
            )

            # differences relative to image 1
            delta_fermat_12 = fermat_potential[1] - fermat_potential[0]

            if num_images == 4:
                delta_fermat_13 = fermat_potential[2] - fermat_potential[0]

                delta_fermat_14 = fermat_potential[3] - fermat_potential[0]

                dphi_list.append(
                    [
                        delta_fermat_12,
                        delta_fermat_13,
                        delta_fermat_14,
                    ]
                )
            else:
                dphi_list.append([delta_fermat_12])

        dphi_array = np.asarray(dphi_list)

        return dphi_array

    def velocity_dispersion(
        self,
        beta_draws,
        kinematics_band,
        kappa_ext_samples=None,
        band_config=None,
    ):
        r"""Compute the velocity dispersion (in km/s) for Galkin/Jampy configurations.

        :param beta_draws: stellar anisotropy samples, must be the same in number as `n_samples`
        :type beta_draws: `numpy.array`
        :param kinematics_band: data band in which the stellar kinematics are modeled
        :type kinematics_band: `str`
        :param kappa_ext_samples: (optional) array containing external convergence samples
        :type kappa_ext_samples: `numpy.array`
        :param band_config: (optional) dictionary describing the filters and profile indices to
          utilize in the linear inversion to retrieve `r_eff` (see `Photometry` class documentation).
          For multi-component lens light models, this must be supplied if there are no saved
          outputs from the `Photometry` module's linear inversion routine.
        :type band_config: `dict` of `dict`

        :return: Model-predicted velocity dispersion in units of km/s
        :rtype: `numpy.array`
        """

        if kappa_ext_samples is None:
            kappa_ext_samples = np.zeros(shape=self.n_samples)

        # get MCMC chain
        chain = self.output.get_reshaped_emcee_chain(
            self.lens_name,
            self.model_id,
            self.walker_ratio,
            self.burn_in,
        )

        # flatten walkers and steps
        flat_chain = chain.reshape(-1, chain.shape[-1])

        random_indices = self.rng.choice(
            flat_chain.shape[0],
            size=self.n_samples,
            replace=False,
        )

        # select random samples
        flat_chain = flat_chain[random_indices]

        # if multi-component lens light model, need to calculate r_eff for
        # each sample in the chain
        if len(self.kwargs_model["index_lens_light_model_list"][0]) > 1:
            photometry = Photometry(
                io_directory=self.io_directory,
                lens_name=self.lens_name,
                model_id=self.model_id,
            )
            try:
                morphology_results = photometry.load_morphology_chain()
            except FileNotFoundError:
                _, morphology_results = photometry.do_linear_inversion(
                    band_config=band_config,
                    walker_ratio=self.walker_ratio,
                    burn_in=self.burn_in,
                    do_morphology=True,
                    n_samples=self.n_samples,
                    rng_seed=self.rng_seed,
                )
            r_eff_chain = morphology_results[kinematics_band]["r_eff"]
        else:
            # if single-component lens light model, use the r_eff from the reshaped chain
            bands = np.array(self.config.settings["band"])
            band_index = int(np.where(bands == kinematics_band)[0][0])
            index_r_eff = self.output.params_sampled.index(
                f"R_sersic_lens_light{band_index}"
            )
            r_eff_chain = flat_chain[:, index_r_eff]

        sigma_v_list = []
        for i, sample in enumerate(flat_chain):
            kwargs_aniso_i = {"beta": beta_draws[i]}

            kwargs_out = self.param.args2kwargs(sample)
            theta_E = kwargs_out["kwargs_lens"][0]["theta_E"]
            gamma = kwargs_out["kwargs_lens"][0]["gamma"]
            kwargs_lens = kwargs_out["kwargs_lens"]
            kwargs_lens_light = kwargs_out["kwargs_lens_light"]
            r_eff = r_eff_chain[i]

            # unphysical r_effs will be given no weight
            if r_eff <= 0:
                sigma_v_list.append(np.array([np.nan]))
                continue

            sigma_v = self.TDCosmography.velocity_dispersion(
                kwargs_lens=kwargs_lens,
                kwargs_lens_light=kwargs_lens_light,
                kwargs_anisotropy=kwargs_aniso_i,
                r_eff=r_eff,
                theta_E=theta_E,
                gamma=gamma,
                kappa_ext=kappa_ext_samples[i],
            )
            sigma_v_list.append(sigma_v)

        # reshape to 1D
        sigma_v_array = np.array(sigma_v_list).reshape(len(sigma_v_list), -1)[:, 0]
        if np.isnan(sigma_v_array).any():
            num_nans = np.sum(np.isnan(sigma_v_array))
            warnings.warn(f"Warning: {num_nans} indices of r_eff are unphysical!")
        return sigma_v_array

    def ddt_from_time_delay(
        self,
        fermat_model_samples,
        dt_measured,
        dt_measured_uncertainty,
    ):
        """Compute the time-delay distance posterior from the modeled Fermat potential
        and measured time delays. The function draws measured time delays from a
        Gaussian distribution with mean and standard deviation given by the measured
        values and their uncertainties, respectively.

        :param fermat_model_samples: posterior samples of the relative Fermat
            potential in units of arcsec^2, as returned by
            :meth:`~dolphin.analysis.Cosmology.fermat_potential`.
            For a quad, the columns correspond to the different image pairs.
        :type fermat_model_samples: `numpy.ndarray`
        :param dt_measured: measured relative time delays [days]. For a quad,
            provide one value for each image pair.
        :type dt_measured: `list` of `float`
        :param dt_measured_uncertainty: 1-sigma uncertainties on the measured
            relative time delays [days]. For a quad, provide one uncertainty
            for each image pair.
        :type dt_measured_uncertainty: `list` of `float`

        :return: Time-delay distance model posterior samples for each image pair in units of Mpc.
        :rtype: `numpy.ndarray`
        """

        fermat_model_samples = np.asarray(fermat_model_samples)
        dt_measured = np.asarray(dt_measured)

        # draw measured time delays for each image pair
        dt_draws = self.rng.normal(
            loc=dt_measured,
            scale=dt_measured_uncertainty,
            size=(len(fermat_model_samples), len(dt_measured)),
        )

        # calculate D_dt for each image pair
        ddt_model = np.empty_like(fermat_model_samples, dtype=float)
        for i in range(len(dt_measured)):
            ddt_model[:, i] = self.TDCosmography.ddt_from_time_delay(
                d_fermat_model=fermat_model_samples[:, i],
                dt_measured=dt_draws[:, i],
                kappa_s=0,
                kappa_ds=0,
                kappa_d=0,
            )

        return ddt_model

    def calculate_sampled_h0(
        self,
        ddt_samples,
        kappa_ext_samples=None,
        omega_m_min=0.05,
        omega_m_max=0.5,
    ):
        """Compute :math:`\rm{H_0}` posterior samples from :math:`\rm D_{\\Delta t}`
        posterior samples in the cosmology of `self.cosmology` with :math:`\\Omega_{\rm
        m}` varied uniformly and external convergence corrections. For multiple image
        pairs, :math:`\rm{H_0}` is calculated independently for each pair. As this
        methodology assumes a flat prior in :math:`\rm D_{\\Delta t}`, an additional
        weight term is computed to determine the proprtional prior in :math:`\rm{H_0}`
        (see, e.g., Eq. 20 in TDCOSMO XXVII).

        :param ddt_samples: time-delay distance samples (in Mpc)
        :type ddt_samples: `numpy.ndarray`
        :param kappa_ext_samples: (optional) array containing external convergence samples
        :type kappa_ext_samples: `numpy.array`
        :param omega_m_min: (optional) minimum value of :math:`\\Omega_{\rm m}` to sample from
        :type omega_m_min: float
        :param omega_m_max: (optional) maximum value of :math:`\\Omega_{\rm m}` to sample from
        :type omega_m_max: float

        :return: :math:`\rm{H_0}` posterior samples (in km/s/Mpc), sampled :math:`\\Omega_{\rm m}` values, flat :math:`\rm{H_0}` weighting
        :rtype: `tuple (numpy.ndarray, numpy.array, numpy.ndarray)`
        """

        ddt_samples = np.asarray(ddt_samples)
        if ddt_samples.ndim == 1:
            ddt_samples = ddt_samples[:, None]

        if kappa_ext_samples is None:
            kappa_ext_samples = np.zeros(shape=self.n_samples)

        # one :math:`\Omega_{\rm m}` draw per MCMC realization
        omega_m_samples = self.rng.uniform(
            omega_m_min,
            omega_m_max,
            size=self.n_samples,
        )

        H0_samples = np.empty_like(ddt_samples)
        flat_H0_weights = np.empty_like(ddt_samples)
        for i in range(self.n_samples):
            params = {
                **self.cosmology_model_base_params,
                "Om0": float(omega_m_samples[i]),
            }
            cos_temp = self.cosmology_model_class(name="temp", **params)
            H0_reference = float(cos_temp.H0.value)

            dd = cos_temp.angular_diameter_distance(self.z_lens).value
            ds = cos_temp.angular_diameter_distance(self.z_source).value
            dds = cos_temp.angular_diameter_distance_z1z2(
                self.z_lens, self.z_source
            ).value

            ddt_cosmology = (1 + self.z_lens) * dd * ds / dds

            # same cosmology for every image pair
            A = H0_reference * ddt_cosmology
            ddt_samples_corrected = ddt_samples[i, :] / (1 - kappa_ext_samples[i])
            H0_samples[i, :] = A / ddt_samples_corrected

            flat_weight = A * (1 - kappa_ext_samples[i]) / ddt_samples[i, :] ** 2
            flat_H0_weights[i, :] = flat_weight

        return H0_samples, omega_m_samples, flat_H0_weights

    def compute_logarithmic_kinematic_likelihood(
        self,
        sigma_v_observed,
        sigma_v_observed_error,
        sigma_v_model,
        omega_m_min=0.05,
        omega_m_max=0.5,
    ):
        """Evaluate the log-probability of the modeled velocity dispersion
        under a normal distribution centered on `sigma_v_observed` with standard
        deviation `sigma_v_observed_error`. The function also applies cosmology weights
        to account for the cosmological dependence of the kinematics. The predicted
        velocity dispersion scales with the distance ratio :math:`R = D_{\rm s} / D_{\rm ds}`,
        so each realization is weighted by:

            :math:`g_i = \\sqrt(R(\\Omega_\rm{m},i) / R_{\rm model})`,

        :param sigma_v_observed: observed velocity dispersion (in km/s) of the
          main deflector
        :type sigma_v_observed: `float`
        :param sigma_v_observed_error: 1-sigma error (in km/s) on the velocity
          dispersion of the main deflector
        :type sigma_v_observed_error: `float`
        :param sigma_v_model: modeled velocity dispersion samples (in km/s), as computed
          by :meth:`~dolphin.analysis.Cosmology.velocity_dispersion`
        :type sigma_v_model: `numpy.array`
        :param omega_m_min: (optional) minimum value of :math:`\\Omega_{\rm m}` to sample from
        :type omega_m_min: float
        :param omega_m_max: (optional) maximum value of :math:`\\Omega_{\rm m}` to sample from
        :type omega_m_max: float

        :return: logarithmic kinematic likelihood term
        :rtype: `numpy.array`
        """

        # fiducial kinematic weighting
        ds_fiducial = self.cosmology.angular_diameter_distance(self.z_source).value
        dds_fiducial = self.cosmology.angular_diameter_distance_z1z2(
            self.z_lens, self.z_source
        ).value
        R_fiducial = ds_fiducial / dds_fiducial

        # one :math:`\Omega_{\rm m}` draw per MCMC realization
        omega_m_samples = self.rng.uniform(
            omega_m_min,
            omega_m_max,
            size=self.n_samples,
        )

        logarithmic_likelihood = []
        for i in range(self.n_samples):
            params = {
                **self.cosmology_model_base_params,
                "Om0": float(omega_m_samples[i]),
            }
            cos_temp = self.cosmology_model_class(name="temp", **params)
            ds = cos_temp.angular_diameter_distance(self.z_source).value
            dds = cos_temp.angular_diameter_distance_z1z2(
                self.z_lens, self.z_source
            ).value
            R = ds / dds
            g = np.sqrt(R / R_fiducial)

            sigma_v_predicted = g * sigma_v_model[i]
            logarithmic_likelihood_single = (
                -0.5
                * ((sigma_v_observed - sigma_v_predicted) / sigma_v_observed_error) ** 2
            )

            # unphysical r_effs given no weight
            logarithmic_likelihood_single = np.where(
                np.isfinite(logarithmic_likelihood_single),
                logarithmic_likelihood_single,
                -np.inf,
            )

            logarithmic_likelihood.append(logarithmic_likelihood_single)

        return logarithmic_likelihood

    def compute_bic_weight(
        self,
        bic_all,
        n_grid=300,
        n_sigma=5,
    ):
        """Compute the BIC weight of the current model (see TDCOSMO IX, Eq. 31 on BIC
        weighting). The model's BIC is treated as a Gaussian random variable with
        intrinsic scatter determined from `bic_all`, and the piecewise function:

        .. math::

            f(x) =
            \\begin{cases}
                1, & x < \\mathrm{BIC}_{\\min}, \\\\
                \\exp\\left[-\\frac{1}{2}(x - \\mathrm{BIC}_{\\min})\\right],
                & x \\geq \\mathrm{BIC}_{\\min}.
            \\end{cases}

        is convolved over this Gaussian.

        :param bic_all: BIC values of all models considered for this lens
            (including the current one)
        :type bic_all: `list` of `float`
        :param n_grid: (optional) number of grid points for the integration
        :type n_grid: `int`
        :param n_sigma: (optional) integration range in units of sigma
        :type n_sigma: `float`

        :return: BIC weight of the current model
        :rtype: `float`
        """
        bic_all = np.asarray(bic_all)
        bic_min = bic_all.min()

        # BIC of the current model
        bic_model = self.output._bic

        # compute standard deviation of all BICs
        bic_std = np.std(bic_all, ddof=1)

        x_vals = np.linspace(
            bic_model - n_sigma * bic_std,
            bic_model + n_sigma * bic_std,
            n_grid,
        )
        gauss = np.exp(-0.5 * ((x_vals - bic_model) / bic_std) ** 2) / (
            np.sqrt(2 * np.pi) * bic_std
        )

        # piecewise f(x)
        f_x = np.ones_like(x_vals)
        mask = x_vals >= bic_min
        f_x[mask] = np.exp(-0.5 * (x_vals[mask] - bic_min))

        bic_weight = np.trapezoid(f_x * gauss, x_vals)
        return bic_weight
