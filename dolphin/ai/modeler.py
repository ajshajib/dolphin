# -*- coding: utf-8 -*-

# This module creates a configuration file from the output of the visual recognition model.


import numpy as np
import yaml

from .ai import AI


class Modeler(AI):
    """This class creates a configuration file from the output of the visual recognition
    model."""

    def __init__(self, io_directory_path):
        """Initialize the Configure object."""
        super(Modeler, self).__init__(io_directory_path)

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
        type="quasar",
        pso_settings={"num_particle": 50, "num_iteration": 50},
        psf_iteration_settings={
            "stacking_method": "median",
            "num_iter": 20,
            "psf_iter_factor": 0.5,
            "keep_psf_error_map": True,
            "psf_symmetry": 4,
        },
        sampler_name="emcee",
        sampler_settings={
            "n_burn": 0,
            "n_run": 100,
            "walkerRatio": 2,
        },
        supersampling_factor=[2],
    ):
        """Get configuration from the semantic segmentation output. This method
        currently works only for the single-band case.

        :param lens_name: lens name
        :type lens_name: `str`
        :param band_name: band name
        :type band_name: `str`
        :param type: type of configuration
        :type type: `str`
        :param pso_settings: PSO settings
        :type pso_settings: `dict`
        :param psf_iteration_settings: PSF iteration settings
        :type psf_iteration_settings: `dict`
        :param sampler_settings: sampler settings
        :type sampler_settings: `dict`
        """
        image_data = self.get_image_data(lens_name, band_name)
        coordinate_system = image_data.get_image_coordinate_system()
        semantic_segmentation = self.load_semantic_segmentation(lens_name, band_name)

        config = {}
        config["lens_name"] = lens_name
        config["band"] = [band_name]

        # config["pixel_size"] = image_data.get_image_pixel_scale().item()

        config["model"] = {
            "lens": ["EPL", "SHEAR_GAMMA_PSI"],
            "lens_light": ["SERSIC_ELLIPSE"],
            "source_light": ["SERSIC_ELLIPSE"],
            "point_source": (["LENSED_POSITION"] if type == "quasar" else [""]),
        }

        galaxy_center_x, galaxy_center_y = self.get_lens_galaxy_center_init(
            semantic_segmentation, coordinate_system
        )
        config["lens_option"] = {
            "centroid_init": [galaxy_center_x.item(), galaxy_center_y.item()],
            "centroid_bound": 0.2,
        }
        config["lens_light_option"] = {"fix": {0: {"n_sersic": 4.0}}}
        config["source_light_option"] = {"n_max": [4]}

        point_source_init = self.get_quasar_image_position(
            semantic_segmentation, coordinate_system
        )
        config["point_source_option"] = {
            "ra_init": point_source_init[0].tolist(),
            "dec_init": point_source_init[1].tolist(),
            "bound": 0.2,
        }

        config["numeric_option"] = {"supersampling_factor": supersampling_factor}

        config["fitting"] = {
            "pso": psf_iteration_settings is not None,
            "pso_settings": pso_settings,
        }

        config["fitting"]["psf_iteration"] = (
            True if psf_iteration_settings is not None and type == "quasar" else False
        )
        config["fitting"]["psf_iteration_settings"] = psf_iteration_settings
        # second minimum distance between the four quasar images
        distances = []
        for i in range(len(point_source_init[0])):
            for j in range(i + 1, len(point_source_init[0])):
                distances.append(
                    np.sqrt(
                        (point_source_init[0][i] - point_source_init[0][j]) ** 2
                        + (point_source_init[1][i] - point_source_init[1][j]) ** 2
                    )
                )
        if "block_center_neighbour" not in config["fitting"]["psf_iteration_settings"]:
            config["fitting"]["psf_iteration_settings"]["block_center_neighbour"] = (
                float(np.sort(distances)[1] / 2.0)
            )
        if "error_map_radius" not in config["fitting"]["psf_iteration_settings"]:
            config["fitting"]["psf_iteration_settings"]["error_map_radius"] = float(
                np.sort(distances)[1] / 2.0
            )

        config["fitting"]["sampling"] = sampler_settings is not None
        config["fitting"]["sampler"] = sampler_name
        config["fitting"]["sampler_settings"] = sampler_settings

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

    @classmethod
    def get_mask_from_semantic_segmentation(
        cls, semantic_segmentation, coordinate_system
    ):
        """Get mask from the semantic segmentation output.

        :param semantic_segmentation: semantic segmentation output
        :type semantic_segmentation: `numpy.ndarray`
        :return: mask
        :rtype: `numpy.ndarray`
        """
        galaxy_center = cls.get_lens_galaxy_center_init(
            semantic_segmentation, coordinate_system
        )
        image_positions = cls.get_quasar_image_position(
            semantic_segmentation, coordinate_system
        )

        galaxy_center_x, galaxy_center_y = coordinate_system.map_coord2pix(
            galaxy_center[0], galaxy_center[1]
        )

        image_positions_x, image_positions_y = coordinate_system.map_coord2pix(
            image_positions[0], image_positions[1]
        )

        theta_E_init = cls.get_theta_E_init(
            [galaxy_center_x, galaxy_center_y], [image_positions_x, image_positions_y]
        )

        mask = np.zeros(semantic_segmentation.shape)

        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if (i - galaxy_center_x) ** 2 + (j - galaxy_center_y) ** 2 < (
                    2 * theta_E_init
                ) ** 2:
                    mask[i, j] = 1

        return mask

    @staticmethod
    def get_theta_E_init(galaxy_center, image_positions):
        """Get the initial guess for the Einstein radius.

        :param galaxy_center: galaxy center
        :type galaxy_center: `List[float]`
        :param image_positions: image positions
        :type image_positions: `List[List[float]]`
        :return: initial guess for the Einstein radius
        :rtype: `float`
        """
        x_0, y_0 = galaxy_center
        x_i, y_i = image_positions

        return np.mean(np.sqrt((x_i - x_0) ** 2 + (y_i - y_0) ** 2))

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

    @classmethod
    def get_quasar_image_position(cls, semantic_segmentation, coordinate_system):
        """Identify quasar image positions from the semantic segmentation output.

        :param semantic_segmentation: semantic segmentation output
        :type semantic_segmentation: `numpy.ndarray`
        :param coordinate_system: coordinate system
        :type coordinate_system: `Coordinates`
        :return: quasar image positions
        :rtype: `[np.ndarray, np.ndarray]`
        """
        quasar_positions = cls.list_region_centers(semantic_segmentation, 3)

        quasar_ra, quasar_dec = [], []

        for position in quasar_positions:
            ra, dec = coordinate_system.map_pix2coord(position[1], position[0])
            quasar_ra.append(ra)
            quasar_dec.append(dec)

        return [np.array(quasar_ra), np.array(quasar_dec)]

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
        """
        # if x < 0 or x >= rows or y < 0 or y >= cols:
        #     return
        if visited[x][y] or matrix[x][y] != target_value:
            return

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
    def list_region_centers(cls, matrix, target_value):
        """List the central pixel (x, y) for all regions in a matrix with the target
        value.

        :param matrix: input matrix
        :type matrix: `List[List[int]]`
        :param target_value: target value
        :type target_value: `int`
        :return: list of central pixels (x, y) for all regions
        :rtype: `List[Tuple[int, int]]`
        """
        rows, cols = len(matrix), len(matrix[0])
        visited = [[False for _ in range(cols)] for _ in range(rows)]
        region_centers = []

        for i in range(rows):
            for j in range(cols):
                if matrix[i][j] == target_value and not visited[i][j]:
                    pixels = []
                    cls.collect_connected_pixels(
                        i, j, pixels, visited, target_value, matrix, rows, cols
                    )
                    if pixels:
                        center_x = sum(p[0] for p in pixels) // len(pixels)
                        center_y = sum(p[1] for p in pixels) // len(pixels)
                        region_centers.append((center_x, center_y))

        return region_centers
