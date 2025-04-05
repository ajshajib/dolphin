from pathlib import Path
import numpy as np
import pytest

from dolphin.ai.modeler import Modeler

_ROOT_DIR = Path(__file__).resolve().parents[2]
_TEST_IO_DIR = _ROOT_DIR / "io_directory_example"


class TestModeler:
    def setup_class(self):
        self.qso_modeler = Modeler(_TEST_IO_DIR, source_type="quasar")
        self.galaxy_modeler = Modeler(_TEST_IO_DIR, source_type="galaxy")

    @classmethod
    def teardown_class(cls):
        pass

    def test_create_config_for_single_lens(self):
        """Test `create_config_for_single_lens` method.

        :return:
        :rtype:
        """
        lens_system = "lensed_quasar_2"
        config_file_path = _TEST_IO_DIR / "settings" / f"{lens_system}_config.yaml"

        if config_file_path.exists():
            config_file_path.unlink()

        self.qso_modeler.create_config_for_single_lens(
            lens_system,
            "F814W",
            psf_iteration_settings={
                "stacking_method": "median",
                "num_iter": 20,
                "psf_iter_factor": 0.5,
                "keep_psf_variance_map": True,
                "psf_symmetry": 4,
            },
        )

        assert config_file_path.exists()

        config_file_path.unlink()

    def test_create_configuration_for_all_lenses(self):
        """Test `create_configuration_for_all_lenses` method.

        :return:
        :rtype:
        """

        lens_systems = ["lensed_quasar_2"]
        for lens_system in lens_systems:
            config_file_path = _TEST_IO_DIR / "settings" / f"{lens_system}_config.yaml"

            if config_file_path.exists():
                config_file_path.unlink()

        self.qso_modeler.create_configuration_for_all_lenses("F814W")
        for lens_system in lens_systems:
            config_file_path = _TEST_IO_DIR / "settings" / f"{lens_system}_config.yaml"
            assert config_file_path.exists()

            config_file_path.unlink()

    def test_load_semantic_segmentation(self):
        """Test `load_semantic_segmentation` method.

        :return:
        :rtype:
        """
        segmentation = self.qso_modeler.load_semantic_segmentation(
            "lensed_quasar", "F814W"
        )

        assert segmentation is not None
        assert segmentation.shape == (120, 120)

    def test_save_configuration(self):
        """Test `save_configuration` method.

        :return:
        :rtype:
        """
        lens_system = "lensed_quasar_2"
        config_file_path = _TEST_IO_DIR / "settings" / f"{lens_system}_config.yaml"

        config = {"lens_system": lens_system, "band": "F814W"}
        if config_file_path.exists():
            config_file_path.unlink()
        self.qso_modeler.save_configuration(config, lens_system)

        assert config_file_path.exists()

    def test_get_configuration(self):
        """Test `get_configuration` method.

        :return:
        :rtype:
        """
        lens_system = "lensed_quasar"
        config = self.qso_modeler.get_configuration(lens_system, "F814W")

        keywords = [
            "lens_name",
            "band",
            "model",
            "lens_option",
            "lens_light_option",
            "source_light_option",
            "point_source_option",
            "numeric_option",
            "fitting",
            "mask",
        ]
        for keyword in keywords:
            assert keyword in config

    def test_get_mask_from_semantic_segmentation(self):
        """Test `get_mask_from_semantic_segmentation` method.

        :return:
        :rtype:
        """
        lens_name = "lensed_quasar"
        band_name = "F814W"
        image_data = self.qso_modeler.get_image_data(lens_name, band_name)
        coordinate_system = image_data.get_image_coordinate_system()
        semantic_segmentation = self.qso_modeler.load_semantic_segmentation(
            lens_name, band_name
        )
        mask = self.qso_modeler.get_mask_from_semantic_segmentation(
            semantic_segmentation=semantic_segmentation,
            coordinate_system=coordinate_system,
        )
        assert mask.shape == image_data.get_image().shape

    def test_get_theta_E_init(self):
        """Test `get_theta_E_init` method.

        :return:
        :rtype:
        """
        lens_name = "lensed_quasar"
        band_name = "F814W"
        image_data = self.qso_modeler.get_image_data(lens_name, band_name)
        coordinate_system = image_data.get_image_coordinate_system()

        semantic_segmentation = np.zeros_like(image_data.get_image())
        xs = np.arange(semantic_segmentation.shape[0])
        ys = np.arange(semantic_segmentation.shape[1])
        xx, yy = np.meshgrid(xs, ys)
        ras, decs = coordinate_system.map_pix2coord(xx.flatten(), yy.flatten())

        ras = ras.reshape(semantic_segmentation.shape)
        decs = decs.reshape(semantic_segmentation.shape)

        rs = np.sqrt(ras**2 + decs**2)

        semantic_segmentation[rs < 0.2] = 1  # deflector

        # quasar 1
        rs = np.sqrt((ras - 0.5) ** 2 + (decs - 0) ** 2)
        semantic_segmentation[rs < 0.1] = 3  # quasar 1

        # quasar 2
        rs = np.sqrt((ras + 0.5) ** 2 + (decs - 0) ** 2)
        semantic_segmentation[rs < 0.1] = 3  # quasar 2

        theta_E_init = self.qso_modeler.get_theta_E_init(
            semantic_segmentation, coordinate_system
        )

        assert round(theta_E_init, 2) == 0.5

        semantic_segmentation *= 0

        rs = np.sqrt(ras**2 + decs**2)
        semantic_segmentation[rs < 0.2] = 1  # deflector
        semantic_segmentation[(rs > 0.65) & (rs < 0.75)] = 2  # arc

        theta_E_init = self.galaxy_modeler.get_theta_E_init(
            semantic_segmentation, coordinate_system
        )

        assert round(theta_E_init, 2) == 0.7

    def test_get_lens_galaxy_center_init(self):
        """Test `get_lens_galaxy_center_init` method.

        :return:
        :rtype:
        """
        image_data = self.qso_modeler.get_image_data("lensed_quasar", "F814W")
        coordinate_system = image_data.get_image_coordinate_system()
        image_size = image_data.get_image_size()

        mask = np.zeros((image_size, image_size))
        center_pix = [40, 80]
        center_coords = coordinate_system.map_pix2coord(center_pix[0], center_pix[1])

        # make a circle with radius 10 around the center
        for i in range(image_size):
            for j in range(image_size):
                if (i - center_pix[0]) ** 2 + (j - center_pix[1]) ** 2 < 10**2:
                    mask[j, i] = 1

        lens_galaxy_center_init = self.qso_modeler.get_lens_galaxy_center_init(
            mask, coordinate_system
        )

        assert lens_galaxy_center_init == list(center_coords)

    def test_get_quasar_image_position(self):
        """Test `get_quasar_image_position` method.

        :return:
        :rtype:
        """
        image_data = self.qso_modeler.get_image_data("lensed_quasar", "F814W")
        coordinate_system = image_data.get_image_coordinate_system()
        image_size = image_data.get_image_size()

        mask = np.zeros((image_size, image_size))

        quasar_pixel_x, quasar_pixel_y = [30, 50], [80, 40]
        quasar_coords = coordinate_system.map_pix2coord(quasar_pixel_x, quasar_pixel_y)

        galaxy_pixel_x, galaxy_pixel_y = 80, 80

        for i in range(image_size):
            for j in range(image_size):
                for k in range(2):
                    if (i - quasar_pixel_x[k]) ** 2 + (
                        j - quasar_pixel_y[k]
                    ) ** 2 < 7**2:
                        mask[j, i] = 3

                if (i - galaxy_pixel_x) ** 2 + (j - galaxy_pixel_y) ** 2 < 7**2:
                    mask[j, i] = 1

        quasar_image_positions = self.qso_modeler.get_quasar_image_position(
            mask, coordinate_system
        )

        assert np.allclose(
            sorted(quasar_image_positions[0]), sorted(quasar_coords[0]), atol=1e-4
        )
        assert np.allclose(
            sorted(quasar_image_positions[1]), sorted(quasar_coords[1]), atol=1e-4
        )

        with pytest.raises(NotImplementedError):
            self.galaxy_modeler.get_quasar_image_position(mask, coordinate_system)

    def test_get_satellite_positions(self):
        """Test `get_satellite_positions` method."""
        image_data = self.qso_modeler.get_image_data("lensed_quasar", "F814W")
        coordinate_system = image_data.get_image_coordinate_system()
        image_size = image_data.get_image_size()

        mask = np.zeros((image_size, image_size))

        satellite_pixel_xs, satellite_pixel_ys = [30, 50, 10], [80, 40, 80]
        satellite_ras, satellite_decs = coordinate_system.map_pix2coord(
            satellite_pixel_xs, satellite_pixel_ys
        )

        galaxy_pixel_x, galaxy_pixel_y = 80, 80

        for i in range(image_size):
            for j in range(image_size):
                for k, (x, y) in enumerate(zip(satellite_pixel_xs, satellite_pixel_ys)):
                    if (i - x) ** 2 + (j - y) ** 2 < 7**2:
                        mask[j, i] = 4

                if (i - galaxy_pixel_x) ** 2 + (j - galaxy_pixel_y) ** 2 < 7**2:
                    mask[j, i] = 1

        satellite_positions = self.qso_modeler.get_satellite_positions(
            mask, coordinate_system
        )

        coords = [[ra, dec] for ra, dec in zip(satellite_ras, satellite_decs)]

        assert len(satellite_positions) == len(coords)

        # Sort coords by first elements first element
        coords = sorted(coords, key=lambda x: x[0])
        satellite_positions = sorted(satellite_positions, key=lambda x: x[0])

        for i in range(len(coords)):
            assert np.allclose(
                sorted(satellite_positions[i]), sorted(coords[i]), atol=1e-4
            )

    def test_collect_connected_pixels_boundary_check(self):
        """Test the boundary check in the `collect_connected_pixels` method."""
        # Define a small matrix
        matrix = [
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ]
        rows, cols = len(matrix), len(matrix[0])

        # Initialize visited matrix
        visited = [[False for _ in range(cols)] for _ in range(rows)]

        # Define parameters
        pixels = []
        target_value = 1

        # Test out-of-bounds indices
        self.galaxy_modeler.collect_connected_pixels(
            -1, 0, pixels, visited, target_value, matrix, rows, cols
        )
        self.galaxy_modeler.collect_connected_pixels(
            0, -1, pixels, visited, target_value, matrix, rows, cols
        )
        self.galaxy_modeler.collect_connected_pixels(
            rows, 0, pixels, visited, target_value, matrix, rows, cols
        )
        self.galaxy_modeler.collect_connected_pixels(
            0, cols, pixels, visited, target_value, matrix, rows, cols
        )

        # Ensure no pixels were added for out-of-bounds indices
        assert (
            len(pixels) == 0
        ), "Boundary check failed: Out-of-bounds indices should not add pixels."

        # Test valid indices
        self.galaxy_modeler.collect_connected_pixels(
            0, 0, pixels, visited, target_value, matrix, rows, cols
        )

        # Ensure valid pixels are added
        assert (
            len(pixels) > 0
        ), "Boundary check failed: Valid indices should add pixels."
