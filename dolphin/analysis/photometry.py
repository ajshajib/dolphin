from lenstronomy.Sampling.parameters import Param
from lenstronomy.ImSim.image_linear_solve import ImageLinearFit
from lenstronomy.Data.psf import PSF
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Analysis.light_profile import LightProfileAnalysis
from lenstronomy.Util import param_util

from dolphin.processor.config import ModelConfig
import numpy as np
import h5py
import os

class Photometry:
    """This class performs a linear inversion on the model outputs to 
    obtain lens, image, and source fluxes/magnitudes, as well as morphological properties of the lens
    light."""

    def __init__(self, output, band_config, 
                 model_id, walker_ratio, burn_in=0):
        """Initiate the class from the following inputs:
        
        :param output: `Output` instance 
        :type output: `class`
        :param band_config: Dictionary describing the filters and profile indices to
        utilize in the inversion.
        :type band_config: `dict`
        :param model_id: Model ID of the lens system being analyzed
        :type model_id: `str`
        :param walker_ratio: number of walkers per parameter in MCMC
        :type walker_ratio: `int`
        :param burn_in: (optional) number of burn-in steps to compute the medians after
            convergence of the MCMC chain
        :type burn_in: `int`
        """

        self.output = output
        self.system_name = output._model_settings["lens_name"]
        self.band_config = band_config
        self.model_id = model_id
        self.walker_ratio = walker_ratio
        self.burn_in = burn_in

        self.filters = []
        for filter_name, values in self.band_config.items():
            self.filters.append(filter_name)

        band_list = self.output._model_settings["band"]
        self.band_map = {
            band: i for i, band in enumerate(band_list)
        }

        self.multi_band_list = output._multi_band_list_out

        config = ModelConfig(
            lens_name=self.system_name,
            io_directory=self.output.io_directory,
            settings=output.model_settings,
        )

        self.kwargs_model = config.get_kwargs_model()
        self.kwargs_likelihood = config.get_kwargs_likelihood()

        self.param = output.get_param_class(
            lens_name=self.system_name,
            model_id=self.model_id
        )

        self.band_models = self._build_band_models()

    def _build_band_models(self):
        """Build model components per band from results chain
        """
        band_models = {}

        for filt in self.filters:
            i_band = self.band_map[filt]

            kwargs_data = self.multi_band_list[i_band][0]
            kwargs_psf = self.multi_band_list[i_band][1]
            kwargs_numerics = self.multi_band_list[i_band][2]
            mask_list = self.kwargs_likelihood.get("image_likelihood_mask_list", None)

            if mask_list is None:
                likelihood_mask = None
            else:
                likelihood_mask = mask_list[i_band]
        
            band_models[filt] = {
                "data_class": ImageData(**kwargs_data),
                "psf_class": PSF(**kwargs_psf),
                "kwargs_numerics": kwargs_numerics,
                "likelihood_mask": likelihood_mask,
            }

        return band_models

    def _load_photometry(self, filt):

        if not hasattr(self, "_phot_cache"):
            self._phot_cache = {}

        if filt in self._phot_cache:
            return self._phot_cache[filt]

        filename = (
            f"{self.output.io_directory}/data/{self.system_name}/"
            f"image_{self.system_name}_{filt}.h5"
        )

        if not os.path.exists(filename):
            raise FileNotFoundError(filename)

        with h5py.File(filename, "r") as f:

            # JWST calibration
            if "PIXAR_SR" in f:

                calib = {
                    "instrument": "JWST",
                    "pixar_sr": f["PIXAR_SR"][()],
                }

            # HST calibration
            elif all(key in f for key in ["PHOTFLAM", "PHOTZPT", "PHOTPLAM"]):

                calib = {
                    "instrument": "HST",
                    "photflam": f["PHOTFLAM"][()],
                    "photzpt": f["PHOTZPT"][()],
                    "photplam": f["PHOTPLAM"][()],
                }

        self._phot_cache[filt] = calib

        print(f"Instrument identified as: {calib['instrument']}")

        return calib
    
    def _get_abmag(self, flux, filt):
        """
        Convert integrated flux to AB magnitude
        """

        calib = self._load_photometry(filt)

        flux = np.asarray(flux)

        if calib["instrument"] == "JWST":
    
            pixar_sr = calib["pixar_sr"]

            # MJy/sr to Jy
            flux_jy = flux * pixar_sr * 1e6

            abmag = -2.5 * np.log10(flux_jy / 3631.0)

            return abmag

        elif calib["instrument"] == "HST":

            photflam = calib["photflam"]
            photzpt = calib["photzpt"]
            photplam = calib["photplam"]

            flux_cgs = flux * photflam

            stmag = -2.5 * np.log10(flux_cgs) + photzpt

            abmag = (
                stmag
                - 5.0 * np.log10(photplam)
                + 2.5 * np.log10(299792458e10) 
                - 27.5
            )

            return abmag

    def evaluate_band(
        self,
        filt,
        kwargs_lens_all,
        kwargs_lens_light_all,
        kwargs_source_all,
        kwargs_ps_all,
        grid_spacing=0.02,
        grid_num=200,
    ):
        """Perform linear inversion on a band
        """

        band = self.band_models[filt]
        band_config = self.band_config.get(filt)

        lens_light_indices = band_config.get("lens_light_indices")
        source_indices = band_config.get("source_indices")
        exclude_lens_light_indices = band_config.get("exclude_lens_light_indices", [])
        has_point_source = (
            kwargs_ps_all is not None and len(kwargs_ps_all) > 0
        )

        if lens_light_indices is None:
            raise ValueError(f"lens_light_indices must be provided for filter {filt}")

        lens_light_model_list = [
            self.kwargs_model["lens_light_model_list"][k]
            for k in lens_light_indices
        ]

        lens_light_kwargs = [
            kwargs_lens_light_all[k]
            for k in lens_light_indices
        ]

        if source_indices is None:
            raise ValueError(f"source_indices must be provided for filter {filt}")
            
        source_model_list = [
            self.kwargs_model["source_light_model_list"][k]
            for k in source_indices
        ]

        source_kwargs = [
            kwargs_source_all[k]
            for k in source_indices
        ]

        lightModel = LightModel(lens_light_model_list)
        sourceModel = LightModel(source_model_list)
        lensModel = LensModel(self.kwargs_model["lens_model_list"])

        if has_point_source:
            pointSource = PointSource(self.kwargs_model["point_source_model_list"])
        else:
            pointSource = None
            
        imageLinearFit = ImageLinearFit(
            data_class=band["data_class"],
            psf_class=band["psf_class"],
            lens_model_class=lensModel,
            source_model_class=sourceModel,
            lens_light_model_class=lightModel,
            point_source_class=pointSource,
            kwargs_numerics=band["kwargs_numerics"],
            likelihood_mask=band["likelihood_mask"],
            psf_error_map_bool_list=[True],
        )

        imageLinearFit.image_linear_solve(
            kwargs_lens=kwargs_lens_all,
            kwargs_source=source_kwargs,
            kwargs_lens_light=lens_light_kwargs,
            kwargs_ps=kwargs_ps_all if has_point_source else None,
        )

        flux_images = []

        if has_point_source:
            flux_images = []
            for ps_model in kwargs_ps_all:
                if "point_amp" in ps_model:
                    flux_images.extend(ps_model["point_amp"])
            flux_images = np.array(flux_images)
        else:
            flux_images = np.array([])  # no images

        imageModel = imageLinearFit

        flux_lens = 0
        for k in range(len(lens_light_model_list)):
            if k not in exclude_lens_light_indices:
                flux_lens += np.sum(
                    imageModel.lens_surface_brightness(
                        lens_light_kwargs, k=k
                    )
                )

        flux_host_lensed = np.sum(
            imageModel.source_surface_brightness(
                source_kwargs, kwargs_lens_all, de_lensed=False
            )
        )

        flux_host_intrinsic = np.sum(
            imageModel.source_surface_brightness(
             source_kwargs, kwargs_lens_all, de_lensed=True
            )
        )

        light_analysis = LightProfileAnalysis(lightModel)

        cx = lens_light_kwargs[0]["center_x"]
        cy = lens_light_kwargs[0]["center_y"]

        model_bool_list = [
            k not in exclude_lens_light_indices
            for k in range(len(lens_light_model_list))
        ]

        e1, e2 = light_analysis.ellipticity(
            lens_light_kwargs,
            center_x=cx,
            center_y=cy,
            grid_spacing=grid_spacing,
            grid_num=grid_num,
            model_bool_list=model_bool_list,
            iterative=True
        )

        r_eff = light_analysis.half_light_radius(
            lens_light_kwargs,
            center_x=cx,
            center_y=cy,
            grid_spacing=grid_spacing,
            grid_num=grid_num,
            model_bool_list=model_bool_list,
        )

        phi, q = param_util.ellipticity2phi_q(e1, e2)
        phi = (phi * 180 / np.pi) % 180

        return {
            "fluxes": {
                "images": flux_images,
                "lens": flux_lens,
                "host_lensed": flux_host_lensed,
                "host_intrinsic": flux_host_intrinsic,
            },
            "morphology": {
                "phi": phi,
                "q": q,
                "r_eff": r_eff
            }
        }

    def get_flux_and_morphology(self):
        """Perform the linear inversion on all bands

        :return: Flux chains, Morphology Chains
        """
        results = []
        morph_results = {f: {"phi": [], "q": [], "r_eff": []} for f in self.filters}
        self.n_images = None  # will infer from first sample

        chain = self.output.get_reshaped_emcee_chain(
            self.system_name,
            self.model_id,
            self.walker_ratio,
            self.burn_in
        )

        flat_chain = chain.reshape(-1, chain.shape[-1])

        for sample in flat_chain:

            kwargs_out = self.param.args2kwargs(sample)

            kwargs_lens = kwargs_out["kwargs_lens"]
            kwargs_lens_light = kwargs_out["kwargs_lens_light"]
            kwargs_source = kwargs_out["kwargs_source"]
            kwargs_ps = kwargs_out["kwargs_ps"]

            sample_fluxes = []

            for filt in self.filters:
                res = self.evaluate_band(
                    filt,
                    kwargs_lens,
                    kwargs_lens_light,
                    kwargs_source,
                    kwargs_ps
             )

                flux_dict = res["fluxes"]

                if self.n_images is None:
                    self.n_images = len(flux_dict["images"])

                sample_fluxes.extend(flux_dict["images"])
                sample_fluxes.append(flux_dict["lens"])
                sample_fluxes.append(flux_dict["host_lensed"])
                sample_fluxes.append(flux_dict["host_intrinsic"])

                morph = res["morphology"]
                morph_results[filt]["phi"].append(morph["phi"])
                morph_results[filt]["q"].append(morph["q"])
                morph_results[filt]["r_eff"].append(morph["r_eff"])

            results.append(sample_fluxes)

        return np.array(results), morph_results

    def get_ab_magnitude(self, flux_chain):
        """Get AB Magnitude from the flux chains.

        :param flux_chain: flux chain
        :type flux_chain: array

        :return mag_chain: AB magnitude chain
        :rtype mag_chain: array
        """

        n_flux_per_filt = self.n_images + 3
        mag_chain = np.zeros_like(flux_chain)

        for i, filt in enumerate(self.filters):

            start = i * n_flux_per_filt
            end = start + n_flux_per_filt

            flux_block = flux_chain[:, start:end]

            mag_block = self._get_abmag(flux_block, filt)

            mag_chain[:, start:end] = mag_block

        return mag_chain
    
    def save_to_hdf5(self, flux_chain, mag_chain=None, morph_chain=None):
        """Save linear inversion outputs to HDF5 for later analysis.
        
        :param flux_chain: Flux chain as computed from `get_flux_and_morphology`
        :type flux_chain: array
        :param mag_chain: (Optional) AB magnitude chain as computed from `get_ab_magnitude`
        :type mag_chain: array
        :param morph_chain: (Optional) Morphology chain as computed from `get_flux_and_morphology`
        :tpye morph_chain: dict
        """

        n_flux_per_filt = self.n_images + 3

        flux_labels = (
            [f"Image{i+1}" for i in range(self.n_images)]
            + ["Lens", "Host_lensed", "Host_intrinsic"]
        )

        filename = (
            f"{self.output.io_directory}/outputs/"
            f"photometry_{self.system_name}_{self.model_id}.h5"
        )

        with h5py.File(filename, "w") as f:

            f.attrs["system_name"] = self.system_name
            f.attrs["filters"] = self.filters

            for i, filt in enumerate(self.filters):

                grp = f.create_group(filt)

                start = i * n_flux_per_filt
                end = start + n_flux_per_filt

                flux_block = flux_chain[:, start:end]

                if mag_chain is not None:
                    mag_block = mag_chain[:, start:end]

                for j, label in enumerate(flux_labels):

                    subgrp = grp.create_group(label)

                    subgrp.create_dataset(
                        "flux",
                        data=flux_block[:, j]
                    )

                    if mag_chain is not None:
                        subgrp.create_dataset(
                            "magnitude",
                            data=mag_block[:, j]
                        )

            if morph_chain is not None:

                morph_grp = f.create_group("lens_light_morphology")

                for filt in self.filters:

                    filt_grp = morph_grp.create_group(filt)

                    filt_grp.create_dataset(
                        "phi",
                        data=np.array(morph_chain[filt]["phi"])
                    )

                    filt_grp.create_dataset(
                        "q",
                        data=np.array(morph_chain[filt]["q"])
                    )

                    filt_grp.create_dataset(
                        "r_eff",
                        data=np.array(morph_chain[filt]["r_eff"])
                    )
            
    def get_flux_chain(self):
        """
        Load flux chain

        :return flux_chain: flux chain
        :rtype flux_chain: np.ndarray
        """

        filename = f"{self.output.io_directory}/outputs/photometry_{self.system_name}_{self.model_id}.h5"
        with h5py.File(filename, "r") as f:
            
            filters = list(f.attrs["filters"])
            chains = []

            for filt in filters:
                grp = f[filt]

                labels = list(grp.keys())

                # ensure consistent ordering
                image_labels = sorted([l for l in labels if "Image" in l])
                other_labels = ["Lens", "Host_lensed", "Host_intrinsic"]

                ordered_labels = image_labels + other_labels

                block = np.vstack([
                    grp[label]["flux"][:] for label in ordered_labels
                ]).T

                chains.append(block)

        flux_chain = np.hstack(chains)

        return flux_chain

    def get_magnitude_chain(self):
        """
        Load magnitude chain

        :return mag_chain: AB magnitude chain
        :type mag_chain: np.ndarray
        """

        filename = f"{self.output.io_directory}/outputs/photometry_{self.system_name}_{self.model_id}.h5"
        with h5py.File(filename, "r") as f:
            
            filters = list(f.attrs["filters"])
            chains = []

            for filt in filters:
                grp = f[filt]

                labels = list(grp.keys())

                image_labels = sorted([l for l in labels if "Image" in l])
                other_labels = ["Lens", "Host_lensed", "Host_intrinsic"]

                ordered_labels = image_labels + other_labels

                block = np.vstack([
                    grp[label]["magnitude"][:] for label in ordered_labels
                ]).T

                chains.append(block)

        mag_chain = np.hstack(chains)

        return mag_chain

    def get_morphology_chain(self):
        """
        Load morphology chains.

        :return morph_chain: morphology chain: {filter: {"phi": array, "q": array, "r_eff": array}}
        :rtype: dict
        """

        filename = f"{self.output.io_directory}/outputs/photometry_{self.system_name}_{self.model_id}.h5"
        with h5py.File(filename, "r") as f:

            if "lens_light_morphology" not in f:
                return None

            filters = list(f.attrs["filters"])
            morph = f["lens_light_morphology"]

            morph_chain = {}

            for filt in filters:
                morph_chain[filt] = {
                    "phi": morph[filt]["phi"][:],
                    "q": morph[filt]["q"][:],
                    "r_eff": morph[filt]["r_eff"][:],
                }

        return morph_chain