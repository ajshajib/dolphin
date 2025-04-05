# -*- coding: utf-8 -*-

# This module creates a configuration file from the output of the visual recognition model.


import numpy as np
import yaml

from .ai import AI


class Modeler(AI):
    """This class creates a configuration file from the output of the visual recognition
    model."""

    def __init__(self, io_directory_path, source_type):
        """Initialize the Configure object.

        :param io_directory_path: path to the input/output directory
        :type io_directory_path: `str`
        :param source_type: source type
        :type source_type: `str`
        """
        super(Modeler, self).__init__(io_directory_path)
        self._source_type = source_type

    def create_configuration_for_all_lenses(self, band_name, **kwargs):
        """Create configuration files for all lenses.

        :param band_name: band name
        :type band_name: `str`
        :param kwargs: additional keyword arguments to be passed to `get_configuration`
        :type kwargs: `dict`
        """
        lens_list = self.file_system.get_lens_list()

        for lens_name in lens_list:
            self.create_config_for_single_lens(lens_name, band_name, **kwargs)

        print(f"Done creating configuration files for {len(lens_list)} lenses.")

    def create_config_for_single_lens(self, lens_name, band_name, **kwargs):
        """Create configuration file for a single lens.

        :param lens_name: lens name
        :type lens_name: `str`
        :param band_name: band name
        :type band_name: `str`
        :param kwargs: additional keyword arguments to be passed to `get_configuration`
        :type kwargs: `dict`
        :return: configuration
        :rtype: `dict`
        """
        configuration = self.get_configuration(
            lens_name,
            band_name,
            **kwargs,
        )
        self.save_configuration(configuration, lens_name)

        return configuration

    def load_semantic_segmentation(self, lens_name, band_name):
        """Load semantic segmentation output from the visual recognition model.

        :param lens_name: lens name
        :type lens_name: `str`
        :param band_name: band name
        :type band_name: `str`
        """
        return self.file_system.load_semantic_segmentation(lens_name, band_name)

    def save_configuration(self, config, lens_name):
        """Save the configuration to a YAML file.

        :param config: configuration
        :type config: `dict`
        :param lens_name: lens name
        :type lens_name: `str`
        """
        config_file_path = self.file_system.get_config_file_path(lens_name)

        with open(config_file_path, "w") as config_file:
            yaml.dump(config, config_file, default_flow_style=False, sort_keys=False)

    def get_configuration(
        self,
        lens_name,
        band_name,
        pso_settings={"num_particle": 50, "num_iteration": 50},
        psf_iteration_settings=None,
        sampler_name="emcee",
        sampler_settings=None,
        supersampling_factor=[3],
        max_satellite_number=2,
        minimum_satellite_area=15,
        satellite_bound=0.25,
        clear_center=0.2,
    ):
        """Get configuration from the semantic segmentation output. This method
        currently works only for the single-band case.

        :param lens_name: lens name
        :type lens_name: `str`
        :param band_name: band name
        :type band_name: `str`
        :param pso_settings: PSO settings
        :type pso_settings: `dict`
        :param psf_iteration_settings: PSF iteration settings. If `None`, PSF
            iteration will not be performed. If settings are provided, PSF
            iteration will be performed. Example settings:
            {
                "stacking_method": "median",
                "num_iter": 20,
                "psf_iter_factor": 0.5,
                "keep_psf_variance_map": True,
                "psf_symmetry": 4,
            }
        :type psf_iteration_settings: `dict`
        :param sampler_name: sampler name, default is "emcee"
        :type sampler_name: `str`
        :param sampler_settings: sampler settings. If `None`, sampling will not be
            performed. If settings are provided, sampling will be performed.
            Example settings:
            {
                "n_burn": 0,
                "n_run": 100,
                "walkerRatio": 2,
            }
        :type sampler_settings: `dict`
        :param supersampling_factor: supersampling factor
        :type supersampling_factor: `List[int]`
        :param max_satellite_number: maximum number of satellites
        :type max_satellite_number: `int`
        :param minimum_satellite_area: minimum satellite area
        :type minimum_satellite_area: `int`
        :param satellite_bound: satellite bound
        :type satellite_bound: `float`
        :param clear_center: radius (arcsecond) to clear the center from any detected satellite or quasar
        :type clear_center: `float`
        :return: configuration
        :rtype: `dict`
        """
        # Get image data and coordinate system
        image_data = self.get_image_data(lens_name, band_name)
        coordinate_system = image_data.get_image_coordinate_system()
        semantic_segmentation = self.load_semantic_segmentation(lens_name, band_name)

        # Initialize configuration dictionary
        config = {}
        config["lens_name"] = lens_name
        config["band"] = [band_name]

        # Define model components
        config["model"] = {
            "lens": ["EPL", "SHEAR_GAMMA_PSI"],
            "lens_light": ["SERSIC_ELLIPSE"],
            "source_light": ["SERSIC_ELLIPSE"],
            "point_source": (
                ["LENSED_POSITION"] if self._source_type == "quasar" else []
            ),
        }

        # Set lens options
        galaxy_center_x, galaxy_center_y = self.get_lens_galaxy_center_init(
            semantic_segmentation, coordinate_system
        )
        config["lens_option"] = {
            "centroid_init": [galaxy_center_x.item(), galaxy_center_y.item()],
            "centroid_bound": 0.2,
        }
        config["lens_light_option"] = {"fix": {0: {"n_sersic": 4.0}}}
        config["source_light_option"] = {"n_max": [6]}

        # Set point source options
        if self._source_type == "quasar":
            point_source_init = self.get_quasar_image_position(
                semantic_segmentation, coordinate_system, clear_center=clear_center
            )
            config["point_source_option"] = {
                "ra_init": point_source_init[0].tolist(),
                "dec_init": point_source_init[1].tolist(),
                "bound": 0.2,
            }

        # Set satellite options
        satellite_positions = self.get_satellite_positions(
            semantic_segmentation,
            coordinate_system,
            clear_center=clear_center,
            minimum_pixel_area=minimum_satellite_area,
        )

        config["satellites"] = {
            "centroid_init": [a for a in satellite_positions[:max_satellite_number]],
            "centroid_bound": satellite_bound,
        }

        # Set guess params
        theta_E_init = self.get_theta_E_init(semantic_segmentation, coordinate_system)
        config["guess_params"] = {"lens": {0: {"theta_E": float(theta_E_init)}}}

        # Set numeric options
        config["numeric_option"] = {"supersampling_factor": supersampling_factor}

        # Set fitting options
        config["fitting"] = {
            "pso": pso_settings is not None,
            "pso_settings": pso_settings,
        }

        config["fitting"]["psf_iteration"] = (
            True
            if psf_iteration_settings is not None and self._source_type == "quasar"
            else False
        )
        config["fitting"]["psf_iteration_settings"] = psf_iteration_settings

        if config["fitting"]["psf_iteration"]:
            # Calculate distances between quasar images
            distances = []
            for i in range(len(point_source_init[0])):
                for j in range(i + 1, len(point_source_init[0])):
                    distances.append(
                        np.sqrt(
                            (point_source_init[0][i] - point_source_init[0][j]) ** 2
                            + (point_source_init[1][i] - point_source_init[1][j]) ** 2
                        )
                    )

            # Set PSF iteration settings
            # Second minimum distance is used to set the block center neighbour and error map radius
            if (
                "block_center_neighbour"
                not in config["fitting"]["psf_iteration_settings"]
            ):
                config["fitting"]["psf_iteration_settings"][
                    "block_center_neighbour"
                ] = float(np.sort(distances)[1] / 2.0)
            if "error_map_radius" not in config["fitting"]["psf_iteration_settings"]:
                config["fitting"]["psf_iteration_settings"]["error_map_radius"] = float(
                    np.sort(distances)[1] / 2.0
                )

        # Set sampling options
        config["fitting"]["sampling"] = sampler_settings is not None
        config["fitting"]["sampler"] = sampler_name
        config["fitting"]["sampler_settings"] = sampler_settings

        # Set mask options
        if self._source_type == "quasar":
            config["mask"] = {}
            config["mask"]["provided"] = True
            self.file_system.save_mask(
                lens_name,
                band_name,
                self.get_mask_from_semantic_segmentation(
                    semantic_segmentation, coordinate_system
                ),
            )

        return config

    def get_mask_from_semantic_segmentation(
        self, semantic_segmentation, coordinate_system
    ):
        """Get mask from the semantic segmentation output.

        :param semantic_segmentation: semantic segmentation output
        :type semantic_segmentation: `numpy.ndarray`
        :param coordinate_system: coordinate system
        :type coordinate_system: `Coordinates`
        :param image_positions: image positions
        :type image_positions: `List[np.ndarray]`
        :return: mask
        :rtype: `numpy.ndarray`
        """
        theta_E_init = self.get_theta_E_init(semantic_segmentation, coordinate_system)
        galaxy_center_x, galaxy_center_y = self.get_lens_galaxy_center_init(
            semantic_segmentation, coordinate_system
        )

        mask = np.zeros(semantic_segmentation.shape)

        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if (i - galaxy_center_x) ** 2 + (j - galaxy_center_y) ** 2 < (
                    2 * theta_E_init
                ) ** 2:
                    mask[i, j] = 1

        return mask

    def get_theta_E_init(self, semantic_segmentation, coordinate_system):
        """Get the initial guess for the Einstein radius.

        :param galaxy_center: galaxy center
        :type galaxy_center: `List[float]`
        :param image_positions: image positions
        :type image_positions: `List[List[float]]`
        :return: initial guess for the Einstein radius
        :rtype: `float`
        """
        galaxy_center = self.get_lens_galaxy_center_init(
            semantic_segmentation, coordinate_system
        )

        if self._source_type == "quasar":
            image_positions = self.get_quasar_image_position(
                semantic_segmentation, coordinate_system
            )
            x_0, y_0 = galaxy_center
            x_i, y_i = image_positions
            theta_E_init = np.mean(np.sqrt((x_i - x_0) ** 2 + (y_i - y_0) ** 2))
        elif self._source_type == "galaxy":
            # Get all x and y indices of the pixels where semantic segmentation is 2
            arc_x, arc_y = np.where(semantic_segmentation == 2)

            arc_ra, arc_dec = coordinate_system.map_pix2coord(arc_y, arc_x)
            rs = np.sqrt(
                (arc_ra - galaxy_center[0]) ** 2 + (arc_dec - galaxy_center[1]) ** 2
            )

            theta_E_init = np.mean(rs)

        return theta_E_init

    @classmethod
    def get_lens_galaxy_center_init(cls, semantic_segmentation, coordinate_system):
        """Get the initial guess for the lens galaxy center.

        :param semantic_segmentation: semantic segmentation output
        :type semantic_segmentation: `numpy.ndarray`
        :param coordinate_system: coordinate system
        :type coordinate_system: `Coordinates`
        :return: initial guess for the lens galaxy center
        :rtype: `[float, float]`
        """
        galaxy_center_pixels = cls.list_region_centers(semantic_segmentation, 1)
        galaxy_center_ra, galaxy_center_dec = coordinate_system.map_pix2coord(
            galaxy_center_pixels[0][1], galaxy_center_pixels[0][0]
        )
        return [galaxy_center_ra, galaxy_center_dec]

    def get_quasar_image_position(
        self, semantic_segmentation, coordinate_system, clear_center=0.0
    ):
        """Identify quasar image positions from the semantic segmentation output.

        :param semantic_segmentation: semantic segmentation output
        :type semantic_segmentation: `numpy.ndarray`
        :param coordinate_system: coordinate system
        :type coordinate_system: `Coordinates`
        :param clear_center: radius (arcsecond) to clear the center from any detected quasar
        :type clear_center: `float`
        :return: quasar image positions
        :rtype: `[np.ndarray, np.ndarray]`
        """
        if self._source_type != "quasar":
            raise NotImplementedError(
                "Quasar image position calculation is only implemented for quasar sources."
            )
        quasar_positions = self.list_region_centers(semantic_segmentation, 3)

        quasar_ra, quasar_dec = [], []

        galaxy_center = self.get_lens_galaxy_center_init(
            semantic_segmentation, coordinate_system
        )

        for position in quasar_positions:
            ra, dec = coordinate_system.map_pix2coord(position[1], position[0])

            distance = np.sqrt(
                (ra - galaxy_center[0]) ** 2 + (dec - galaxy_center[1]) ** 2
            )

            if distance > clear_center:
                quasar_ra.append(ra)
                quasar_dec.append(dec)

        return [np.array(quasar_ra), np.array(quasar_dec)]

    @classmethod
    def get_satellite_positions(
        cls,
        semantic_segmentation,
        coordinate_system,
        clear_center=0.0,
        minimum_pixel_area=1,
    ):
        """Identify satellite positions from the semantic segmentation output.

        :param semantic_segmentation: semantic segmentation output
        :type semantic_segmentation: `numpy.ndarray`
        :param coordinate_system: coordinate system
        :type coordinate_system: `Coordinates`
        :param clear_center: radius (arcsecond) to clear the center from any detected satellite
        :type clear_center: `float`
        :param minimum_pixel_area: minimum pixel area for a satellite
        :type minimum_pixel_area: `int`
        :return: satellite positions
        :rtype: `List[List[float]]`
        """
        satellite_positions = cls.list_region_centers(
            semantic_segmentation, 4, minimum_pixel_area=minimum_pixel_area
        )
        galaxy_center = cls.get_lens_galaxy_center_init(
            semantic_segmentation, coordinate_system
        )

        satellite_ra, satellite_dec = [], []

        for position in satellite_positions:
            ra, dec = coordinate_system.map_pix2coord(position[1], position[0])

            distance = np.sqrt(
                (ra - galaxy_center[0]) ** 2 + (dec - galaxy_center[1]) ** 2
            )
            if distance > clear_center:
                satellite_ra.append(float(ra))
                satellite_dec.append(float(dec))

        return [[ra, dec] for ra, dec in zip(satellite_ra, satellite_dec)]

    @classmethod
    def collect_connected_pixels(
        cls, x, y, pixels, visited, target_value, matrix, rows, cols
    ):
        """Collect all connected pixels in the matrix.

        :param x: x-coordinate
        :type x: `int`
        :param y: y-coordinate
        :type y: `int`
        :param pixels: list of pixels
        :type pixels: `List[Tuple[int, int]]`
        :param visited: matrix to track visited pixels
        :type visited: `List[List[bool]]`
        :param target_value: target value to match
        :type target_value: `int`
        :param matrix: input matrix
        :type matrix: `List[List[int]]`
        :param rows: number of rows in the matrix
        :type rows: `int`
        :param cols: number of columns in the matrix
        :type cols: `int`
        """
        # Check if the current pixel is out of bounds
        if x < 0 or x >= rows or y < 0 or y >= cols:
            return

        # Check if the current pixel is already visited or does not match the target value
        if visited[x][y] or matrix[x][y] != target_value:
            return

        # Mark the current pixel as visited and add it to the list of pixels
        visited[x][y] = True
        pixels.append((x, y))

        # Explore all 4 possible directions (up, down, left, right)
        cls.collect_connected_pixels(
            x + 1, y, pixels, visited, target_value, matrix, rows, cols
        )
        cls.collect_connected_pixels(
            x - 1, y, pixels, visited, target_value, matrix, rows, cols
        )
        cls.collect_connected_pixels(
            x, y + 1, pixels, visited, target_value, matrix, rows, cols
        )
        cls.collect_connected_pixels(
            x, y - 1, pixels, visited, target_value, matrix, rows, cols
        )

    @classmethod
    def list_region_centers(cls, matrix, target_value, minimum_pixel_area=1):
        """List the central pixel (x, y) for all regions in a matrix with the target
        value.

        :param matrix: input matrix
        :type matrix: `List[List[int]]`
        :param target_value: target value
        :type target_value: `int`
        :return: list of central pixels (x, y) for all regions
        :rtype: `List[Tuple[int, int]]`
        :param minimum_pixel_area: minimum size of the region
        :type minimum_pixel_area: `int`
        :return: list of central pixels (x, y) for all regions
        :rtype: `List[Tuple[int, int]]`
        """
        rows, cols = len(matrix), len(matrix[0])
        visited = [[False for _ in range(cols)] for _ in range(rows)]
        region_centers = []
        region_areas = []

        for i in range(rows):
            for j in range(cols):
                if matrix[i][j] == target_value and not visited[i][j]:
                    pixels = []
                    cls.collect_connected_pixels(
                        i, j, pixels, visited, target_value, matrix, rows, cols
                    )
                    if pixels and len(pixels) >= minimum_pixel_area:
                        center_x = sum(p[0] for p in pixels) // len(pixels)
                        center_y = sum(p[1] for p in pixels) // len(pixels)

                        region_centers.append((center_x, center_y))
                        region_areas.append(len(pixels))

        # sort by region_areas
        region_centers = [
            x for _, x in sorted(zip(region_areas, region_centers), reverse=True)
        ]

        return region_centers
