"""This class contains the methods to perform various cosmological analyses."""

__author__ = "brady-ryan"

import numpy as np
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.Analysis.td_cosmography import TDCosmography

from ..analysis.output import Output
from ..analysis.photometry import Photometry
from ..processor.config import ModelConfig
from ..processor.files import FileSystem


class Cosmology:
    """This class contains the methods to perform various cosmological analyses."""

    def __init__(
        self,
        io_directory,
        lens_name,
        model_id,
        z_lens,
        z_source,
        cosmo_fiducial,
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

        :param io_directory: path to the input/output directory. Should not end with slash.
        :type io_directory: `str`
        :param lens_name: name of the system to analyze
        :type lens_name: `str`
        :param model_id: model ID of the lens system being analyzed
        :type model_id: `str`
        :param z_lens: redshift of the main deflector
        :type z_lens: `float`
        :param z_source: redshift of the source
        :type z_source: `float`
        :param cosmo_fiducial: fiducial `astropy.cosmology` instance
        :type cosmo_fiducial: `astropy.cosmology` instance
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
        self.cosmology = cosmo_fiducial
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
            z_lens,
            z_source,
            self.kwargs_model,
            cosmo_fiducial=cosmo_fiducial,
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
          to the first image
        :rtype: `np.ndarray`
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

    def velocity_dispersion_dimension_less(
        self,
        beta_draws,
        kinematics_band,
        band_config=None,
    ):
        """Sigma^2 = Dd/Dds * c^2 * J(kwargs_lens, kwargs_light, anisotropy) (Equation
        4.11 in Birrer et al. 2016 or Equation 6 in Birrer et al. 2019) J() is a
        dimensionless and cosmological independent quantity only depending on angular
        units. This function returns J given the lens and light parameters and the
        anisotropy choice without an external mass sheet correction.


        :param beta_draws: stellar anisotropy samples for computation of J, must be the same
          in number as `n_samples`
        :type beta_draws: `np.array`
        :type kwargs_anisotropy: `np.array`
        :param kinematics_band: data band in which the stellar kinematics are modeled
        :type kinematics_band: `str`
        :param band_config: (optional) dictionary describing the filters and profile indices to
          utilize in the linear inversion to retrieve `r_eff` (see `Photometry` class documentation)

        :return: Dimensionless velocity dispersion samples.
        :rtype: `np.ndarray`
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

            # r_eff corresponding to this MCMC sample
            r_eff = r_eff_chain[i]

            # convert sampled parameters back to lenstronomy kwargs
            kwargs_out = self.param.args2kwargs(sample)
            kwargs_lens = kwargs_out["kwargs_lens"]
            kwargs_lens_light = kwargs_out["kwargs_lens_light"]

            sigma_v = self.TDCosmography.velocity_dispersion_dimension_less(
                kwargs_lens=kwargs_lens,
                kwargs_lens_light=kwargs_lens_light,
                kwargs_anisotropy=kwargs_aniso_i,
                r_eff=r_eff,
                theta_E=theta_E,
                gamma=gamma,
            )
            sigma_v_list.append(sigma_v)

        # reshape to 1D
        sigma_v_array = np.array(sigma_v_list).reshape(len(sigma_v_list), -1)[:, 0]
        return sigma_v_array

    def ddt_from_time_delay(
        self,
        fermat_model_samples,
        dt_measured,
        dt_measured_uncertainty,
    ):
        """Compute time-delay distance posterior samples from the modeled Fermat
        potential and measured time delays. The function draws measured time delays from
        a Gaussian distribution with mean and standard deviation given by the measured
        values and their uncertainties, respectively.

        :param fermat_model_samples: posterior samples of the relative Fermat
            potential in units of arcsec^2, as returned by
            :meth:`~dolphin.analysis.cosmology.fermat_potential`.
            For a quad, the columns correspond to the different image pairs.
        :type fermat_model_samples: `np.ndarray`
        :param dt_measured: measured relative time delays [days]. For a quad,
            provide one value for each image pair.
        :type dt_measured: `list` of `float`
        :param dt_measured_uncertainty: 1-sigma uncertainties on the measured
            relative time delays [days]. For a quad, provide one uncertainty
            for each image pair.
        :type dt_measured_uncertainty: `list` of `float`

        :return: D_dt model posterior samples for each image pair
        :rtype: `np.ndarray`
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
        Om_min=0.05,
        Om_max=0.5,
    ):
        """Compute H0 posterior samples from D_dt posterior samples. For multiple image
        pairs, H0 is calculated independently for each pair.

        :param ddt_samples: time-delay distance samples
        :type ddt_samples: `np.ndarray`
        :param Om_min: (optional) minimum value of Omega_m to sample from
        :type Om_min: float
        :param Om_max: (optional) maximum value of Omega_m to sample from
        :type Om_max: float

        :return: H0 posterior samples, sampled Omega_m values, and cosmology weights
        :rtype: tuple (np.ndarray, np.ndarray, np.ndarray)
        """

        ddt_samples = np.asarray(ddt_samples)
        if ddt_samples.ndim == 1:
            ddt_samples = ddt_samples[:, None]

        # fiducial kinematic weighting
        Ds_fiducial = self.cosmology.angular_diameter_distance(self.z_source).value
        Dds_fiducial = self.cosmology.angular_diameter_distance_z1z2(
            self.z_lens, self.z_source
        ).value
        R_fiducial = Ds_fiducial / Dds_fiducial

        # one Omega_m draw per MCMC realization
        Om_samples = self.rng.uniform(
            Om_min,
            Om_max,
            size=self.n_samples,
        )

        H0_samples = np.empty_like(ddt_samples)
        g_samples = np.empty(self.n_samples)
        H0_fiducial = float(self.cosmology.H0.value)
        for i in range(self.n_samples):
            cos_temp = FlatLambdaCDM(
                H0=H0_fiducial,
                Om0=float(Om_samples[i]),
                Tcmb0=2.725,
            )

            Dd = cos_temp.angular_diameter_distance(self.z_lens).value
            Ds = cos_temp.angular_diameter_distance(self.z_source).value
            Dds = cos_temp.angular_diameter_distance_z1z2(
                self.z_lens, self.z_source
            ).value

            Ddt = (1 + self.z_lens) * Dd * Ds / Dds

            # same cosmology for every image pair
            H0_samples[i, :] = H0_fiducial * Ddt / ddt_samples[i, :]

            R = Ds / Dds
            g_samples[i] = np.sqrt(R / R_fiducial)

        return H0_samples, Om_samples, g_samples

    def compute_logarithmic_kinematic_likelihood(
        self,
        sigma_v_observed,
        sigma_v_observed_error,
        J_dimensionless_samples,
        kappa_ext_samples,
        g_samples,
        Om_min=0.05,
        Om_max=0.5,
    ):
        """Evaluates the log-probability of the modeled velocity dispersion, derived
        from `J_dimensionless_samples`, under a normal distribution centered on
        `sigma_v_observed` with standard deviation `sigma_v_observed_error`.

        :param sigma_v_observed: observed velocity dispersion (in km/s) of the
          main deflector
        :type sigma_v_observed: `float`
        :param sigma_v_observed_error: 1-sigma error (in km/s) on the velocity
          dispersion of the main deflector
        :type sigma_v_observed_error: `float`
        :param J_dimensionless_samples: modeled dimensionless velocity dispersion `J` samples, as computed
          by :meth:`~dolphin.analysis.cosmology.velocity_dispersion_dimension_less`
        :type J_dimensionless_samples: `np.array`
        :param kappa_ext_samples: array containing external convergence samples
        :type kappa_ext_samples: `array`
        :param g_samples: cosmological weighting as determined by
          :meth:`~dolphin.analysis.cosmology.calculate_sampled_h0`
        :type g_samples: `np.ndarray`
        :param Om_min: (optional) minimum value of Omega_m to sample from
        :type Om_min: float
        :param Om_max: (optional) maximum value of Omega_m to sample from
        :type Om_max: float

        :return: logarithmic kinematic likelihood term
        :rtype: `np.array`
        """

        # one Omega_m draw per MCMC realization
        Om_samples = self.rng.uniform(
            Om_min,
            Om_max,
            size=self.n_samples,
        )

        # compute predicted velocity dispersion based upon dimensionless value,
        # external convergence, and g-term weighting. Then, compute the
        # logarithmic kinematic likelihood term
        H0_fiducial = float(self.cosmology.H0.value)
        logarithmic_likelihood = []
        for i in range(self.n_samples):
            cos_temp = FlatLambdaCDM(
                H0=H0_fiducial,
                Om0=float(Om_samples[i]),
                Tcmb0=2.725,
            )
            D_d = cos_temp.angular_diameter_distance(self.z_lens).value
            D_ds = cos_temp.angular_diameter_distance_z1z2(
                self.z_lens, self.z_source
            ).value

            c_km_s = 299792.458

            sigma_v = c_km_s * np.sqrt(D_d / D_ds * J_dimensionless_samples[i])

            sigma_v_predicted = (
                np.sqrt(1 - kappa_ext_samples[i]) * g_samples[i] * sigma_v
            )
            logarithmic_likelihood_single = (
                -0.5
                * ((sigma_v_observed - sigma_v_predicted) / sigma_v_observed_error) ** 2
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

            f(x) = 1                      for x <  BIC_min

            f(x) = exp(BIC_min - x)       for x >= BIC_min

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
