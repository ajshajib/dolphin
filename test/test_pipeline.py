# -*- coding: utf-8 -*-
"""Tests for config module."""
import pytest
from dolphin.processor import Processor
from dolphin.analysis import Output
from dolphin.ai import Vision
from dolphin.ai import Modeler


class TestPipeline(object):
    """Test pipeline for the AI module."""

    def test_quasar_pipeline(self):
        """Test the entire pipeline."""
        io_directory_path = "io_directory_example/"
        lens_name = "ai_test"
        band = "F814W"

        vision = Vision(io_directory_path=io_directory_path, source_type="quasar")
        vision.create_segmentation_for_single_lens(lens_name, band)

        modeler = Modeler(io_directory_path=io_directory_path, source_type="quasar")
        modeler.create_config_for_single_lens(
            lens_name,
            band,
            # psf_iteration_settings=None,
            pso_settings={"num_particle": 4, "num_iteration": 2},
            sampler_settings=None,
        )

        processor = Processor(io_directory_path)
        processor.swim(
            lens_name=lens_name,
            model_id="example",
            log=True,
            recipe_name="galaxy-quasar",
        )

    def test_galaxy_pipeline(self):
        """Test the entire pipeline."""
        io_directory_path = "io_directory_example/"
        lens_name = "lens_system2"
        band = "F390W"

        vision = Vision(io_directory_path=io_directory_path, source_type="galaxy")
        vision.create_segmentation_for_single_lens(lens_name, band)

        modeler = Modeler(io_directory_path=io_directory_path, source_type="galaxy")
        modeler.create_config_for_single_lens(
            lens_name,
            band,
            psf_iteration_settings=None,
            pso_settings={"num_particle": 20, "num_iteration": 20},
            sampler_settings=None,
        )

        processor = Processor(io_directory_path)
        processor.swim(
            lens_name=lens_name,
            model_id="example",
            log=False,
            recipe_name="galaxy-galaxy",
        )
