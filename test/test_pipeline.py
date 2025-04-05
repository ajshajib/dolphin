# -*- coding: utf-8 -*-
"""Tests for config module."""
from dolphin.processor import Processor
from dolphin.ai import Vision
from dolphin.ai import Modeler


class TestPipeline(object):
    """Test pipeline for the AI module."""

    def setup_method(self):
        """Setup method for the test class."""
        self.io_directory_path = "../io_directory_example/"

    def test_quasar_pipeline(self):
        """Test the entire pipeline."""
        lens_name = "ai_test"
        band = "F814W"

        vision = Vision(io_directory_path=self.io_directory_path, source_type="quasar")
        vision.create_segmentation_for_single_lens(lens_name, band)

        modeler = Modeler(
            io_directory_path=self.io_directory_path, source_type="quasar"
        )
        modeler.create_config_for_single_lens(
            lens_name,
            band,
            # psf_iteration_settings=None,
            pso_settings={"num_particle": 4, "num_iteration": 2},
            sampler_settings=None,
        )

        processor = Processor(self.io_directory_path)
        processor.swim(
            lens_name=lens_name,
            model_id="example",
            log=True,
            recipe_name="galaxy-quasar",
        )

    def test_galaxy_pipeline(self):
        """Test the entire pipeline."""
        lens_name = "lens_system2"
        band = "F390W"

        vision = Vision(io_directory_path=self.io_directory_path, source_type="galaxy")
        vision.create_segmentation_for_single_lens(lens_name, band)

        modeler = Modeler(
            io_directory_path=self.io_directory_path, source_type="galaxy"
        )
        modeler.create_config_for_single_lens(
            lens_name,
            band,
            psf_iteration_settings=None,
            pso_settings={"num_particle": 20, "num_iteration": 20},
            sampler_settings=None,
        )

        processor = Processor(self.io_directory_path)
        processor.swim(
            lens_name=lens_name,
            model_id="example",
            log=False,
            recipe_name="galaxy-galaxy",
        )
