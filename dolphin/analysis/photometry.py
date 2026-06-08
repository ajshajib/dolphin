from lenstronomy.ImSim.image_linear_solve import ImageLinearFit
from lenstronomy.Data.psf import PSF
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Analysis.light_profile import LightProfileAnalysis
from lenstronomy.Util import param_util

from dolphin.processor.config import ModelConfig
from dolphin.processor.files import FileSystem
import numpy as np


class Photometry:
    """This class performs a linear inversion on the model outputs to obtain lens,
    image, and source fluxes/magnitudes, as well as morphological properties of the lens
    light. Initiate the class from the following inputs:

    :param output: `Output` instance
    :type output: `class`
    :param lens_name: name of the system to analyze
    :type lens_name: `str`
    :param model_id: Model ID of the lens system being analyzed
    :type model_id: `str`
    :param band_config: Dictionary describing the filters and profile indices to
        utilize in the inversion.
    :type band_config: `dict`
    :param walker_ratio: number of walkers per parameter in MCMC
    :type walker_ratio: `int`
    :param burn_in: (optional) number of burn-in steps to compute the medians after
            convergence of the MCMC chain
    :type burn_in: `int`
    :param aperture_type: (optional) type of aperture in which the inversion is to be calculated within.
          Options are: "circle" and "square." If not specified, the inversion will be computed over
          the full image grid.
    :type aperture_type: `bool`
    :param aperture_size: (optional) angular size (in arcsec) over which the aperture mask is applied. For
        type "circle," this corresponds to the radius of the aperture. For type "square," this corresponds
        to the length of one side of the aperture.
    :param do_morphology: (optional) If `True`, solves for the morphological properties of multi-component
            lens light profiles after the linear inversion.
    :type do_morphology: `bool`
    """

    def __init__(
        self,
        output,
        lens_name,
        model_id,
        band_config,
        walker_ratio,
        burn_in=0,
        aperture_type=None,
        aperture_size=None,
        do_morphology=False,
    ):

        self.output = output
        self.lens_name = lens_name
        self.model_id = model_id
        output.load_output(f"{self.lens_name}", f"{self.model_id}", verbose=False)

        self.band_config = band_config
        self.walker_ratio = walker_ratio
        self.burn_in = burn_in
        self.aperture_type = aperture_type
        self.aperature_size = aperture_size
        self.do_morphology = do_morphology

        self.filters = []
        for filter_name, values in self.band_config.items():
            self.filters.append(filter_name)

        band_list = self.output._model_settings["band"]
        self.band_map = {band: i for i, band in enumerate(band_list)}

        self.multi_band_list = output._multi_band_list_out

        config = ModelConfig(
            lens_name=self.lens_name,
            io_directory=self.output.io_directory,
            settings=output.model_settings,
        )

        self.kwargs_model = config.get_kwargs_model()
        self.kwargs_likelihood = config.get_kwargs_likelihood()

        self.param = output.get_param_class(
            lens_name=self.lens_name, model_id=self.model_id
        )

        self.band_models = self._build_band_models()
        self.file_system = FileSystem(self.output.io_directory)

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

    def _aperture_mask(
        self, data_class, center_x, center_y, aperture_type=None, aperture_size=None
    ):
        """Generate an aperture mask centered around the lens light centroid.

        :param data_class: instance of `ImageData` class
        :type data_class: `class`
        :param center_x: center x-axis pixel (in arcesc) for the aperture
        :type center_x: `float`
        :param center_y: center y-axis pixel (in arcsec) for the aperture
        :type center_y: `float`
        :param aperture_type: type of aperture in which the inversion is to be calculated within.
          Options are: "circle" and "square." If not specified, the inversion will be computed over
          the full image grid.
        :type aperture_type: `bool`
        :param aperture_size: angular size (in arcsec) over which the aperture mask is applied. For
        type "circle," this corresponds to the radius of the aperture. For type "square," this corresponds
        to the length of one side of the aperture.
        """

        x_grid, y_grid = data_class.pixel_coordinates

        if aperture_type == "circle":

            r = np.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)

            return r <= aperture_size

        elif aperture_type == "square":

            return (np.abs(x_grid - center_x) <= aperture_size / 2) & (
                np.abs(y_grid - center_y) <= aperture_size / 2
            )

        else:
            return np.ones_like(x_grid, dtype=bool)

    def _do_linear_inversion_single_band(
        self,
        data_band,
        kwargs_lens_all,
        kwargs_lens_light_all,
        kwargs_source_all,
        kwargs_special_all,
        kwargs_ps_all=None,
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
            kwargs_special=kwargs_special_all,
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
            aperture_type=self.aperture_type,
            aperture_size=self.aperature_size,
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

        :return flux_results: Dictionary corresponding to the flux results from the
            linear inversion for each model component per data band.
        :type flux_results: dict
        :return morphology_results: Dictionary corresponding to the fitted lens light
            parameters of mulit-component models per data band.
        :type morphology_results: dict
        """
        self.n_images = None  # will infer from first sample
        flux_results = {}

        for f in self.filters:
            flux_results[f] = {
                "lens": [],
                "source_lensed": [],
                "source_intrinsic": [],
            }
        morphology_results = {
            f: {"phi": [], "q": [], "r_eff": []} for f in self.filters
        }

        chain = self.output.get_reshaped_emcee_chain(
            self.lens_name, self.model_id, self.walker_ratio, self.burn_in
        )

        flat_chain = chain.reshape(-1, chain.shape[-1])

        for sample in flat_chain:

            kwargs_out = self.param.args2kwargs(sample)

            kwargs_lens = kwargs_out["kwargs_lens"]
            kwargs_lens_light = kwargs_out["kwargs_lens_light"]
            kwargs_source = kwargs_out["kwargs_source"]
            kwargs_ps = kwargs_out["kwargs_ps"]
            kwargs_special = kwargs_out["kwargs_special"]

            for data_band in self.filters:
                result = self._do_linear_inversion_single_band(
                    data_band=data_band,
                    kwargs_lens_all=kwargs_lens,
                    kwargs_lens_light_all=kwargs_lens_light,
                    kwargs_source_all=kwargs_source,
                    kwargs_special_all=kwargs_special,
                    kwargs_ps_all=kwargs_ps,
                )

                flux_dict = result["fluxes"]

                if self.n_images is None:
                    self.n_images = len(flux_dict["images"])
                    for f in self.filters:
                        for i in range(self.n_images):
                            flux_results[f][f"image{i+1}"] = []

                for i, flux in enumerate(flux_dict["images"]):
                    flux_results[data_band][f"image{i+1}"].append(flux)

                flux_results[data_band]["lens"].append(flux_dict["lens"])

                flux_results[data_band]["source_lensed"].append(
                    flux_dict["source_lensed"]
                )

                flux_results[data_band]["source_intrinsic"].append(
                    flux_dict["source_intrinsic"]
                )

                morphology = result["morphology"]
                morphology_results[data_band]["phi"].append(morphology["phi"])
                morphology_results[data_band]["q"].append(morphology["q"])
                morphology_results[data_band]["r_eff"].append(morphology["r_eff"])

        for band in self.filters:
            for key in flux_results[band]:
                flux_results[band][key] = np.array(flux_results[band][key])
        return flux_results, morphology_results

    def calculate_ab_magnitude(self, flux_chain, calibration_parameters):
        """Helper function to calculate the AB magnitude from the flux chains computed
        by `do_linear_inversion`.

        1) For JWST data, the required calibration parameter is:
            - `pixar_sr`: JWST average pixel area in units of steradians
        2) For HST data, the required calibartion parameters are:
            - `photflam`: mean flux density (in erg cm-2 sec-1 Angstrom-1) that produces 1 count per second in the HST observing mode
            - `photzpt`: STMAG HST data band zero point
            - `photplam`: HST data band pivot wavelength

        :param flux_chain: flux chain computed from `do_linear_inversion`
        :type flux_chain: np.ndarray
        :param calibration_parameters: dictionary corresponding to the filters, instruments, and keyword values for magnitude conversions. For example, if doing
          JWST and HST data conversions:

          calibration_parameters = {

          "F115W":
            {"instrument": JWST, "pixar_sr": ####},

            "F814W": {"instrument": HST, "photplam": ##, "photzpt": ##, "photflam": ##}

            }

        :type calibration_parameters: dict
        :return magnitude_results: Dictionary corresponding to the converted AB magnitudes for each
          model component per band.
        :type magnitude_results: dict
        """

        magnitude_results = {}

        for data_band in self.filters:

            calib = calibration_parameters[data_band]

            if "instrument" not in calib:
                raise ValueError(f"Instrument not specified for data band {data_band}!")

            instrument = calib["instrument"]

            if instrument not in ["JWST", "HST"]:
                raise ValueError(f"{instrument} not yet supported!")

            magnitude_results[data_band] = {}

            for component, flux in flux_chain[data_band].items():

                if instrument == "JWST":
                    pixar_sr = calib["pixar_sr"]

                    flux_jy = flux * pixar_sr * 1e6
                    magnitude_results[data_band][component] = -2.5 * np.log10(
                        flux_jy / 3631.0
                    )
                elif instrument == "HST":
                    photflam = calib["photflam"]
                    photzpt = calib["photzpt"]
                    photplam = calib["photplam"]

                    flux_cgs = flux * photflam
                    stmag = -2.5 * np.log10(flux_cgs) + photzpt
                    magnitude_results[data_band][component] = (
                        stmag
                        - 5.0 * np.log10(photplam)
                        + 2.5 * np.log10(299792458e10)
                        - 27.5
                    )

        return magnitude_results

    def save_to_hdf5(self, flux_chain, magnitude_chain=None, morphology_chain=None):
        """Save linear inversion outputs to HDF5 for later analysis.

        :param flux_chain: Flux chain as computed from `do_linear_inversion`
        :type flux_chain: np.ndarray
        :param magnitude_chain: (Optional) AB magnitude chain as computed from `calculate_ab_magnitude`
        :type magnitude_chain: np.ndarray
        :param morphology_chain: (Optional) Morphology chain as computed from `do_linear_inversion`
        :type morphology_chain: dict
        """

        self.file_system.save_photometry_to_hdf5(
            self, flux_chain, magnitude_chain, morphology_chain
        )

    def load_flux_chain(self):
        """Load flux chain as computed by `do_linear_inversion`.

        :return flux_chain: flux dictionary: {filter: {"image1": array, "image2": array,
            "lens": array, ...}}
        :type flux_chain: dict
        """

        return self.file_system.load_flux_chain(self)

    def load_magnitude_chain(self):
        """Load magnitude chain as computed by the respective helper function.

        :return magnitude_chain: AB magnitude dictionary: {filter: {"image1": array, "image2": array,
            "lens": array, ...}}
        :type magnitude_chain: dict
        """

        return self.file_system.load_magnitude_chain(self)

    def load_morphology_chain(self):
        """Load morphology dictionary as computed by `do_linear_inversion`.

        :return morphology_chain: morphology chain: {filter: {"phi": array, "q": array,
            "r_eff": array}}
        :type: dict
        """

        return self.file_system.load_morphology_chain(self)
