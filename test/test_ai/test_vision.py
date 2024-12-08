from pathlib import Path
import numpy as np
import pytest
from dolphin.ai.vision import Vision


_ROOT_DIR = Path(__file__).resolve().parents[2]
_TEST_IO_DIR = _ROOT_DIR / "io_directory_example"


class TestVision:
    @classmethod
    def setup_class(self):
        """Set up the Vision instance for testing."""
        self.vision = Vision(_TEST_IO_DIR)

    @classmethod
    def teardown_class(cls):
        """Clean up resources after tests are completed."""
        pass

    def test_init(self):
        """Test the __init__ method."""
        # Expected output
        with pytest.raises(ValueError):
            Vision(_TEST_IO_DIR, source_type="invalid")
        
    def test_resize_image(self):
        """Test the resize_image method to ensure it resizes images correctly."""
        # Create a sample image (e.g., 256x256)
        original_shape = (256, 256)
        sample_image = np.random.rand(*original_shape)

        # Call the resize_image function
        resized_image = self.vision.resize_image(sample_image)

        # Expected output shape
        expected_shape = (128, 128)

        # Assert the shape of the resized image
        assert (
            resized_image.shape == expected_shape
        ), f"Resized image shape: {resized_image.shape}, Expected shape: {expected_shape}"

        original_shape = (100, 100)
        sample_image = np.random.rand(*original_shape)
        resized_image = self.vision.resize_image(sample_image)
        expected_shape = (128, 128)
        assert (
            resized_image.shape == expected_shape
        ), f"Resized image shape: {resized_image.shape}, Expected shape: {expected_shape}"


    def create_segmentation_for_single_lens(self):
        """Test the create_segmentation_for_single_lens method."""
        # Call the create_segmentation_for_single_lens function
        segmentation = self.vision.create_segmentation_for_single_lens("lensed_quasar", "F814W")

        assert segmentation.shape == (120, 120)

    
