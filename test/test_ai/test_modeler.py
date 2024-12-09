from pathlib import Path
import numpy as np

from dolphin.ai.modeler import Modeler

_ROOT_DIR = Path(__file__).resolve().parents[2]
_TEST_IO_DIR = _ROOT_DIR / "io_directory_example"


class TestModeler:
    def setup_class(self):
        self.modeler = Modeler(_TEST_IO_DIR)

    @classmethod
    def teardown_class(cls):
        pass

    def test_create_config_for_single_lens(self):
        """Test `create_config_for_single_lens` method.

        :return:
        :rtype:
        """
        lens_system = "lensed_quasar_2"
        config_file_path = _TEST_IO_DIR / "settings" / f"{lens_system}_config.yml"

        if config_file_path.exists():
            config_file_path.unlink()

        self.modeler.create_config_for_single_lens(lens_system, "F814W")

        assert config_file_path.exists()

        config_file_path.unlink()

    def test_create_configuration_for_all_lenses(self):
        """Test `create_configuration_for_all_lenses` method.

        :return:
        :rtype:
        """
        
        lens_systems = ["lensed_quasar_2"]
        for lens_system in lens_systems:
            config_file_path = _TEST_IO_DIR / "settings" / f"{lens_system}_config.yml"
            
            if config_file_path.exists():
                config_file_path.unlink()

        self.modeler.create_configuration_for_all_lenses("F814W")
        for lens_system in lens_systems:
            config_file_path = _TEST_IO_DIR / "settings" / f"{lens_system}_config.yml"
            assert config_file_path.exists()

            config_file_path.unlink()

    def test_load_semantic_segmentation(self):
        """Test `load_semantic_segmentation` method.

        :return:
        :rtype:
        """
        segmentation = self.modeler.load_semantic_segmentation("lensed_quasar", "F814W")

        assert segmentation is not None
        assert segmentation.shape == (120, 120)

    def test_save_configuration(self):
        """Test `save_configuration` method.

        :return:
        :rtype:
        """
        lens_system = "lensed_quasar_2"
        config_file_path = _TEST_IO_DIR / "settings" / f"{lens_system}_config.yml"

        config = {"lens_system": lens_system, "band": "F814W"}
        if config_file_path.exists():
            config_file_path.unlink()
        self.modeler.save_configuration(config, lens_system)

        assert config_file_path.exists()

    def test_get_configuration(self):
        """Test `get_configuration` method.

        :return:
        :rtype:
        """
        lens_system = "lensed_quasar"
        config = self.modeler.get_configuration(lens_system, "F814W")

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
        image_data = self.modeler.get_image_data(lens_name, band_name)
        coordinate_system = image_data.get_image_coordinate_system()
        semantic_segmentation = self.modeler.load_semantic_segmentation(
            lens_name, band_name
        )
        mask = self.modeler.get_mask_from_semantic_segmentation(
            semantic_segmentation=semantic_segmentation,
            coordinate_system=coordinate_system,
        )
        assert mask.shape == image_data.get_image().shape

    def test_get_theta_E_init(self):
        """Test `get_theta_E_init` method.

        :return:
        :rtype:
        """
        center = [0, 0]
        image_positions = np.array([[1, -1, 0, 0], [0, 0, 1, -1]])
        theta_E_init = self.modeler.get_theta_E_init(center, image_positions)

        assert theta_E_init == 1

    def test_get_lens_galaxy_center_init(self):
        """Test `get_lens_galaxy_center_init` method.

        :return:
        :rtype:
        """
        image_data = self.modeler.get_image_data("lensed_quasar", "F814W")
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

        lens_galaxy_center_init = self.modeler.get_lens_galaxy_center_init(
            mask, coordinate_system
        )

        assert lens_galaxy_center_init == list(center_coords)

    def test_get_quasar_image_position(self):
        """Test `get_quasar_image_position` method.

        :return:
        :rtype:
        """
        image_data = self.modeler.get_image_data("lensed_quasar", "F814W")
        coordinate_system = image_data.get_image_coordinate_system()
        image_size = image_data.get_image_size()

        mask = np.zeros((image_size, image_size))

        quasar_pixel_x, quasar_pixel_y = [30, 50], [80, 40]
        quasar_coords = coordinate_system.map_pix2coord(quasar_pixel_x, quasar_pixel_y)

        for i in range(image_size):
            for j in range(image_size):
                if (i - quasar_pixel_x[0]) ** 2 + (j - quasar_pixel_y[0]) ** 2 < 7**2:
                    mask[j, i] = 3
                if (i - quasar_pixel_x[1]) ** 2 + (j - quasar_pixel_y[1]) ** 2 < 7**2:
                    mask[j, i] = 3

        quasar_image_positions = self.modeler.get_quasar_image_position(
            mask, coordinate_system
        )

        assert np.allclose(sorted(quasar_image_positions[0]), sorted(quasar_coords[0]))
        assert np.allclose(sorted(quasar_image_positions[1]), sorted(quasar_coords[1]))
