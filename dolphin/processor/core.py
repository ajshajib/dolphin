# -*- coding: utf-8 -*-
"""This module handles the execution of modeling sequences for lens systems."""

__author__ = "ajshajib"

import sys
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from schwimmbad import choose_pool

from .files import FileSystem
from .config import ModelConfig
from .data import ImageData
from .data import PSFData
from .recipe import Recipe


class Processor(object):
    """This class contains methods to model a single lens system or a batch of systems
    using settings loaded from configuration files."""

    def __init__(self, io_directory):
        """Initialize the Processor with the base I/O directory.

        :param io_directory: path to the input/output directory. Should not end with a slash.
        :type io_directory: `str`
        """
        self.io_directory = io_directory
        self.file_system = FileSystem(io_directory)
        self.lens_list = self.file_system.get_lens_list()

    def swim(
        self,
        lens_name,
        model_id,
        log=True,
        mpi=False,
        recipe_name="galaxy-quasar",
        thread_count=1,
        use_jax=False,
    ):
        """Run lens modeling optimizations for a single lens system.

        :param lens_name: name of the lens system to model
        :type lens_name: `str`
        :param model_id: identifier for this specific model run
        :type model_id: `str`
        :param log: if `True`, standard output is logged to a file. Set to `False` in notebooks.
        :type log: `bool`
        :param mpi: enable MPI for parallel processing
        :type mpi: `bool`
        :param recipe_name: recipe for pre-sampling optimization. Supported: 'galaxy-quasar', 'galaxy-galaxy', 'skip'. 'skip' will skip pre-sampling optimization and directly sample the full model. See `Recipe` class for details.
        :type recipe_name: `str`
        :param thread_count: number of threads to use if multiprocess is enabled
        :type thread_count: `int`
        :param use_jax: if `True`, performs modeling through JAXtronomy instead of lenstronomy
        :type use_jax: `bool`
        :return: None
        :rtype: `None`
        """
        pool = choose_pool(mpi=mpi)

        if log and pool.is_master():
            log_file = open(
                self.file_system.get_log_file_path(lens_name, model_id), "wt"
            )
            sys.stdout = log_file

        config = self.get_lens_config(lens_name)
        recipe = Recipe(config, thread_count=thread_count)

        psf_supersampling_factor = config.get_psf_supersampled_factor()
        kwargs_data_joint = self.get_kwargs_data_joint(
            lens_name, psf_supersampled_factor=psf_supersampling_factor
        )

        if use_jax:
            from jaxtronomy.Workflow.fitting_sequence import (
                FittingSequence as FittingSequenceJAX,
            )

            FittingSequenceClass = FittingSequenceJAX
        else:
            FittingSequenceClass = FittingSequence

        fitting_sequence = FittingSequenceClass(
            kwargs_data_joint,
            config.get_kwargs_model(),
            config.get_kwargs_constraints(),
            config.get_kwargs_likelihood(use_jax=use_jax),
            config.get_kwargs_params(),
            mpi=mpi,
        )

        fitting_kwargs_list = recipe.get_recipe(
            kwargs_data_joint=kwargs_data_joint, recipe_name=recipe_name
        )
        print(f"Optimizing model for {lens_name} with recipe: {recipe_name}.")

        fit_output = fitting_sequence.fit_sequence(fitting_kwargs_list)
        kwargs_result = fitting_sequence.best_fit(bijective=False)
        multi_band_list_out = fitting_sequence.multi_band_list

        output = {
            "settings": config.settings,
            "kwargs_result": kwargs_result,
            "fit_output": fit_output,
            "multi_band_list_out": multi_band_list_out,
        }

        if pool.is_master():
            self.file_system.save_output(lens_name, model_id, output)

        if log and pool.is_master():
            log_file.close()

    def get_lens_config(self, lens_name):
        """Get the `ModelConfig` object populated with settings for a specific lens.

        :param lens_name: name of the lens system
        :type lens_name: `str`
        :return: instance of `ModelConfig` containing the lens configurations
        :rtype: `ModelConfig`
        """
        return ModelConfig(lens_name, file_system=self.file_system)

    def get_kwargs_data_joint(self, lens_name, psf_supersampled_factor=1):
        """Create a joint `kwargs_data` dictionary combining data and PSFs across
        filters.

        :param lens_name: name of the lens system
        :type lens_name: `str`
        :param psf_supersampled_factor: supersampling factor applied to the PSF
        :type psf_supersampled_factor: `int`
        :return: joint kwargs data mapping suitable for `lenstronomy`
        :rtype: `dict`
        """
        config = self.get_lens_config(lens_name)

        bands = config.settings["band"]

        kwargs_numerics = config.get_kwargs_numerics()

        multi_band_list = []

        for b, kwargs_num in zip(bands, kwargs_numerics):
            image_data = self.get_image_data(lens_name, b)
            psf_data = self.get_psf_data(lens_name, b)

            psf_data.kwargs_psf["point_source_supersampling_factor"] = (
                psf_supersampled_factor
            )

            multi_band_list.append(
                [image_data.kwargs_data, psf_data.kwargs_psf, kwargs_num]
            )

        kwargs_data_joint = {
            "multi_band_list": multi_band_list,
            "multi_band_type": "multi-linear",
        }

        return kwargs_data_joint

    def get_image_data(self, lens_name, band):
        """Get the `ImageData` instance for a given lens and observing band.

        :param lens_name: name of the lens system
        :type lens_name: `str`
        :param band: observing band or filter name
        :type band: `str`
        :return: loaded image data object
        :rtype: `ImageData`
        """
        return ImageData(self.file_system.get_image_file_path(lens_name, band))

    def get_psf_data(self, lens_name, band):
        """Get the `PSFData` instance for a given lens and observing band.

        :param lens_name: name of the lens system
        :type lens_name: `str`
        :param band: observing band or filter name
        :type band: `str`
        :return: loaded PSF data object
        :rtype: `PSFData`
        """
        return PSFData(self.file_system.get_psf_file_path(lens_name, band))
