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


class Photometry:

    def __init__(
        self,
        output,
        band_config,
        model_id,
        walker_ratio,
        burn_in=0,
        aperture_radius=None,
        aperture_length=None,
        do_morphology=False,
    ):
        """This class performs a linear inversion on the model outputs to obtain lens,
        image, and source fluxes/magnitudes, as well as morphological properties of the
        lens light. Initiate the class from the following inputs:

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
        :param aperture_radius: (optional) Radius, in arcseconds, for a circular aperture centered around the
            lens light centroid in which the inversion is evaluated. If `None`, uses the domain of the model grid.
        :type aperature_radius: `float`
        :param aperture_length: (optional) Length, in arcseconds, for a square aperture centered around the
           lens light centroid in which the inversion is evaluated. If `None`, uses the domain of the model grid.
        :type aperature_length: `float`
        :param do_morphology: (optional) If `True`, solves for the morphological properties of multi-component
            lens light profiles after the linear inversion.
        :type do_morphology: `bool`
        """

        self.output = output
        self.system_name = output._model_settings["lens_name"]
        self.band_config = band_config
        self.model_id = model_id
        self.walker_ratio = walker_ratio
        self.burn_in = burn_in
        self.aperature_radius = aperture_radius
        self.aperature_length = aperture_length
        self.do_morphology = do_morphology

        self.filters = []
        for filter_name, values in self.band_config.items():
            self.filters.append(filter_name)

        band_list = self.output._model_settings["band"]
        self.band_map = {band: i for i, band in enumerate(band_list)}

        self.multi_band_list = output._multi_band_list_out

        config = ModelConfig(
            lens_name=self.system_name,
            io_directory=self.output.io_directory,
            settings=output.model_settings,
        )

        self.kwargs_model = config.get_kwargs_model()
        self.kwargs_likelihood = config.get_kwargs_likelihood()

        self.param = output.get_param_class(
            lens_name=self.system_name, model_id=self.model_id
        )

        self.band_models = self._build_band_models()

    def _build_band_models(self):
        """Build model components per band from Dolphin model configuration."""
        band_models = {}

        for data_band in self.filters:
            i_band = self.band_map[data_band]

            kwargs_data = self.multi_band_list[i_band][0]
            kwargs_psf = self.multi_band_list[i_band][1]
            kwargs_numerics = self.multi_band_list[i_band][2]
            mask_list = self.kwargs_likelihood.get("image_likelihood_mask_list", None)

            if mask_list is None:
                likelihood_mask = None
            else:
                likelihood_mask = mask_list[i_band]

            band_models[data_band] = {
                "data_class": ImageData(**kwargs_data),
                "psf_class": PSF(**kwargs_psf),
                "kwargs_numerics": kwargs_numerics,
                "likelihood_mask": likelihood_mask,
            }

        return band_models

    def _aperture_mask(self, data_class, center_x, center_y, radius=None, length=None):
        """Generate aperture mask centered around the lens light centroid."""

        x_grid, y_grid = data_class.pixel_coordinates

        if radius is not None:

            r = np.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)

            return r <= radius

        elif length is not None:

            return (np.abs(x_grid - center_x) <= length / 2) & (
                np.abs(y_grid - center_y) <= length / 2
            )

        else:
            return np.ones_like(x_grid, dtype=bool)

    def _load_photometry(self, data_band, magnitude_config):
        """Load photometric calibration for a band."""

        calib = magnitude_config[data_band]

        # infer instrument type from keys
        if "pixar_sr" in calib:

            calib_out = {
                "instrument": "JWST",
                "pixar_sr": calib["pixar_sr"],
            }

        elif all(key in calib for key in ["photflam", "photplam"]):

            calib_out = {
                "instrument": "HST",
                "photflam": calib["photflam"],
                "photplam": calib["photplam"],
            }

        else:
            raise ValueError(
                f"Could not determine instrument type for band '{data_band}'"
            )

        print(f"Instrument identified as: {calib_out['instrument']}")

        return calib_out

    def _get_abmag(self, flux, data_band, magnitude_config):
        """Convert flux to AB magnitude."""

        calib = self._load_photometry(data_band, magnitude_config)

        flux = np.asarray(flux)

        if calib["instrument"] == "JWST":

            pixar_sr = calib["pixar_sr"]

            # MJy/sr to Jy
            flux_jy = flux * pixar_sr * 1e6

            abmag = -2.5 * np.log10(flux_jy / 3631.0)

            return abmag

        elif calib["instrument"] == "HST":

            photflam = calib["photflam"]
            photplam = calib["photplam"]
            photzpt = -21.10

            flux_cgs = flux * photflam

            stmag = -2.5 * np.log10(flux_cgs) + photzpt

            abmag = (
                stmag - 5.0 * np.log10(photplam) + 2.5 * np.log10(299792458e10) - 27.5
            )

            return abmag

    def _do_linear_inversion_single_band(
        self,
        data_band,
        kwargs_lens_all,
        kwargs_lens_light_all,
        kwargs_source_all,
        kwargs_ps_all,
        grid_spacing=0.02,
        grid_num=200,
    ):
        """Perform linear inversion on a single band."""

        band = self.band_models[data_band]
        band_config = self.band_config.get(data_band)

        lens_light_indices = band_config.get("lens_light_indices")
        source_indices = band_config.get("source_indices")
        exclude_lens_light_indices = band_config.get("exclude_lens_light_indices", [])
        has_point_source = kwargs_ps_all is not None and len(kwargs_ps_all) > 0

        if lens_light_indices is None:
            raise ValueError(
                f"lens_light_indices must be provided for filter {data_band}"
            )

        lens_light_model_list = [
            self.kwargs_model["lens_light_model_list"][k] for k in lens_light_indices
        ]

        lens_light_kwargs = [kwargs_lens_light_all[k] for k in lens_light_indices]

        if source_indices is None:
            raise ValueError(f"source_indices must be provided for filter {data_band}")

        source_model_list = [
            self.kwargs_model["source_light_model_list"][k] for k in source_indices
        ]

        source_kwargs = [kwargs_source_all[k] for k in source_indices]

        light_model = LightModel(lens_light_model_list)
        source_model = LightModel(source_model_list)
        lens_model = LensModel(self.kwargs_model["lens_model_list"])

        if has_point_source:
            point_source = PointSource(self.kwargs_model["point_source_model_list"])
        else:
            point_source = None

        image_linear_fit = ImageLinearFit(
            data_class=band["data_class"],
            psf_class=band["psf_class"],
            lens_model_class=lens_model,
            source_model_class=source_model,
            lens_light_model_class=light_model,
            point_source_class=point_source,
            kwargs_numerics=band["kwargs_numerics"],
            likelihood_mask=band["likelihood_mask"],
            psf_error_map_bool_list=[True],
        )

        image_linear_fit.image_linear_solve(
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

        aperture_mask = self._aperture_mask(
            data_class=band["data_class"],
            center_x=lens_light_kwargs[0]["center_x"],
            center_y=lens_light_kwargs[0]["center_y"],
            radius=self.aperature_radius,
            length=self.aperature_length,
        )

        flux_lens = 0
        flux_source_lensed = 0
        flux_source_instrinsic = 0
        for k in range(len(lens_light_model_list)):
            if k not in exclude_lens_light_indices:
                lens_surface_brightness = image_linear_fit.lens_surface_brightness(
                    lens_light_kwargs,
                    k=k,
                )

                flux_lens += np.sum(lens_surface_brightness[aperture_mask])

        source_lensed_surface_brightness = image_linear_fit.source_surface_brightness(
            source_kwargs, kwargs_lens_all, de_lensed=False
        )

        flux_source_lensed += np.sum(source_lensed_surface_brightness[aperture_mask])

        source_intrinsic_surface_brightness = (
            image_linear_fit.source_surface_brightness(
                source_kwargs, kwargs_lens_all, de_lensed=True
            )
        )

        flux_source_instrinsic += np.sum(
            source_intrinsic_surface_brightness[aperture_mask]
        )

        phi = None
        q = None
        r_eff = None

        if self.do_morphology:
            light_analysis = LightProfileAnalysis(light_model)

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
                iterative=True,
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
                "source_lensed": flux_source_lensed,
                "source_intrinsic": flux_source_instrinsic,
            },
            "morphology": {"phi": phi, "q": q, "r_eff": r_eff},
        }

    def do_linear_inversion(self):
        """Perform the linear inversion on all bands provided in `band_config`.

        :return flux_results: Array corresponding to the flux results from the linear
            inversion for each model component.
        :rtype flux_results: np.ndarray
        :return morphology_results: Dictionary corresponding to the fitted lens light
            parameters of mulit-component models.
        :rtype morphology_results: dict
        """
        flux_results = []
        morphology_results = {
            f: {"phi": [], "q": [], "r_eff": []} for f in self.filters
        }
        self.n_images = None  # will infer from first sample

        chain = self.output.get_reshaped_emcee_chain(
            self.system_name, self.model_id, self.walker_ratio, self.burn_in
        )

        flat_chain = chain.reshape(-1, chain.shape[-1])

        for sample in flat_chain:

            kwargs_out = self.param.args2kwargs(sample)

            kwargs_lens = kwargs_out["kwargs_lens"]
            kwargs_lens_light = kwargs_out["kwargs_lens_light"]
            kwargs_source = kwargs_out["kwargs_source"]
            kwargs_ps = kwargs_out["kwargs_ps"]

            sample_fluxes = []

            for data_band in self.filters:
                result = self._do_linear_inversion_single_band(
                    data_band, kwargs_lens, kwargs_lens_light, kwargs_source, kwargs_ps
                )

                flux_dict = result["fluxes"]

                if self.n_images is None:
                    self.n_images = len(flux_dict["images"])

                sample_fluxes.extend(flux_dict["images"])
                sample_fluxes.append(flux_dict["lens"])
                sample_fluxes.append(flux_dict["source_lensed"])
                sample_fluxes.append(flux_dict["source_intrinsic"])

                morphology = result["morphology"]
                morphology_results[data_band]["phi"].append(morphology["phi"])
                morphology_results[data_band]["q"].append(morphology["q"])
                morphology_results[data_band]["r_eff"].append(morphology["r_eff"])

            flux_results.append(sample_fluxes)

        return np.array(flux_results), morphology_results

    def calculate_ab_magnitude(self, flux_chain, magnitude_config):
        """Helper functions to calculate the AB magnitude from the flux chains. Currently supported instruments and
           needed calibration parameters are:

           1) JWST:
                - `pixar_sr`: for a given data band, the average pixel area in units of steradians
           2) HST:
                - `photflam`: mean flux density (in erg cm-2 sec-1 Angstrom-1) that produces 1 count per second in the HST observing mode
                - `photplam`: HST data band pivot wavelength

        :param flux_chain: flux chain computed from `do_linear_inversion`
        :type flux_chain: np.ndarray
        :param magnitude_config: dictionary corresponding to the filter and keyword values for magnitude conversions
        :type magnitude_config: dict
        :return magnitude_chain: AB magnitude chain
        :rtype magnitude_chain: np.ndarray
        """

        n_flux_per_filt = self.n_images + 3
        magnitude_chain = np.zeros_like(flux_chain)

        for i, data_band in enumerate(self.filters):

            start = i * n_flux_per_filt
            end = start + n_flux_per_filt

            flux_block = flux_chain[:, start:end]

            mag_block = self._get_abmag(flux_block, data_band, magnitude_config)

            magnitude_chain[:, start:end] = mag_block

        return magnitude_chain

    def save_to_hdf5(self, flux_chain, magnitude_chain=None, morphology_chain=None):
        """Save linear inversion outputs to HDF5 for later analysis.

        :param flux_chain: Flux chain as computed from `do_linear_inversion`
        :type flux_chain: np.ndarray
        :param magnitude_chain: (Optional) AB magnitude chain as computed from `calculate_ab_magnitude`
        :type magnitude_chain: np.ndarray
        :param morphology_chain: (Optional) Morphology chain as computed from `do_linear_inversion`
        :tpye morphology_chain: dict
        """

        n_flux_per_filt = self.n_images + 3

        flux_labels = [f"Image{i+1}" for i in range(self.n_images)] + [
            "Lens",
            "Host_lensed",
            "Host_intrinsic",
        ]

        filename = (
            f"{self.output.io_directory}/outputs/"
            f"photometry_{self.system_name}_{self.model_id}.h5"
        )

        with h5py.File(filename, "w") as f:

            f.attrs["system_name"] = self.system_name
            f.attrs["filters"] = self.filters

            for i, data_band in enumerate(self.filters):

                group = f.create_group(data_band)

                start = i * n_flux_per_filt
                end = start + n_flux_per_filt

                flux_block = flux_chain[:, start:end]

                if magnitude_chain is not None:
                    mag_block = magnitude_chain[:, start:end]

                for j, label in enumerate(flux_labels):

                    subgrp = group.create_group(label)

                    subgrp.create_dataset("flux", data=flux_block[:, j])

                    if magnitude_chain is not None:
                        subgrp.create_dataset("magnitude", data=mag_block[:, j])

            if self.do_morphology:

                morphology_group = f.create_group("lens_light_morphology")

                for data_band in self.filters:

                    filter_group = morphology_group.create_group(data_band)

                    filter_group.create_dataset(
                        "phi", data=np.array(morphology_chain[data_band]["phi"])
                    )

                    filter_group.create_dataset(
                        "q", data=np.array(morphology_chain[data_band]["q"])
                    )

                    filter_group.create_dataset(
                        "r_eff", data=np.array(morphology_chain[data_band]["r_eff"])
                    )

    def load_flux_chain(self):
        """Load flux chain.

        :return flux_chain: flux chain
        :rtype flux_chain: np.ndarray
        """

        filename = f"{self.output.io_directory}/outputs/photometry_{self.system_name}_{self.model_id}.h5"
        with h5py.File(filename, "r") as f:

            filters = list(f.attrs["filters"])
            chains = []

            for data_band in filters:
                group = f[data_band]

                labels = list(group.keys())

                # ensure consistent ordering
                image_labels = sorted([label for label in labels if "Image" in label])
                other_labels = ["Lens", "Host_lensed", "Host_intrinsic"]

                ordered_labels = image_labels + other_labels

                block = np.vstack(
                    [group[label]["flux"][:] for label in ordered_labels]
                ).T

                chains.append(block)

        flux_chain = np.hstack(chains)

        return flux_chain

    def load_magnitude_chain(self):
        """Load magnitude chain.

        :return magnitude_chain: AB magnitude chain
        :type magnitude_chain: np.ndarray
        """

        filename = f"{self.output.io_directory}/outputs/photometry_{self.system_name}_{self.model_id}.h5"
        with h5py.File(filename, "r") as f:

            filters = list(f.attrs["filters"])
            chains = []

            for data_band in filters:
                group = f[data_band]

                labels = list(group.keys())

                image_labels = sorted([label for label in labels if "Image" in label])
                other_labels = ["Lens", "Host_lensed", "Host_intrinsic"]

                ordered_labels = image_labels + other_labels

                block = np.vstack(
                    [group[label]["magnitude"][:] for label in ordered_labels]
                ).T

                chains.append(block)

        magnitude_chain = np.hstack(chains)

        return magnitude_chain

    def load_morphology_chain(self):
        """Load morphology chains.

        :return morphology_chain: morphology chain: {filter: {"phi": array, "q": array,
            "r_eff": array}}
        :rtype: dict
        """

        filename = f"{self.output.io_directory}/outputs/photometry_{self.system_name}_{self.model_id}.h5"
        with h5py.File(filename, "r") as f:

            if "lens_light_morphology" not in f:
                return None

            filters = list(f.attrs["filters"])
            morphology = f["lens_light_morphology"]

            morphology_chain = {}

            for data_band in filters:
                morphology_chain[data_band] = {
                    "phi": morphology[data_band]["phi"][:],
                    "q": morphology[data_band]["q"][:],
                    "r_eff": morphology[data_band]["r_eff"][:],
                }

        return morphology_chain
