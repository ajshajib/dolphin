from pathlib import Path
import pytest
import numpy as np
from dolphin.ai.vision import Vision


_ROOT_DIR = Path(__file__).resolve().parents[2]
_TEST_IO_DIR = _ROOT_DIR / "io_directory_example"


class TestVision:
    @classmethod
    def setup_class(cls):
        """Set up the Vision instance for testing."""
        cls.vision = Vision(_TEST_IO_DIR)

    @classmethod
    def teardown_class(cls):
        """Clean up resources after tests are completed."""
        cls.vision = None

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
        ), f"Expected shape {expected_shape}, but got {resized_image.shape}"
