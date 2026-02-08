# -*- coding: utf-8 -*-
"""This module loads settings from a configuration file."""
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
    """This class contains methods to model a single lens system or a bunch of systems
    from the config files."""

    def __init__(self, io_directory):
        """

        :param io_directory: path to the input/output directory. Should not
            end with slash.
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
    ):
        """Run models for a single lens.

        :param lens_name: lens name
        :type lens_name: `str`
        :param model_id: identifier for the model run
        :type model_id: `str`
        :param log: if `True`, all `print` statements will be logged. This should be `False` if running in a notebook.
        :type log: `bool`
        :param mpi: MPI option
        :type mpi: `bool`
        :param recipe_name: recipe for pre-sampling optimization, supported
            ones now: 'galaxy-quasar' and 'galaxy-galaxy'
        :type recipe_name: `str`
        :param sampler: 'EMCEE' or 'COSMOHAMMER', cosmohammer is kept for
            legacy
        :type sampler: `str`
        :param thread_count: number of threads if `multiprocess` is used
        :type thread_count: `int`
        :return:
        :rtype:
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

        fitting_sequence = FittingSequence(
            kwargs_data_joint,
            config.get_kwargs_model(),
            config.get_kwargs_constraints(),
            config.get_kwargs_likelihood(),
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
        """Get the `ModelConfig` object for a lens.

        :param lens_name: lens name
        :type lens_name: `str`
        :return: `ModelConfig` instance
        :rtype:
        """
        return ModelConfig(lens_name, file_system=self.file_system)

    def get_kwargs_data_joint(self, lens_name, psf_supersampled_factor=1):
        """Create `kwargs_data` for a lens and given filters.

        :param lens_name: lens name
        :type lens_name: `str`
        :param psf_supersampled_factor: Supersampled factor of given PSF.
        :rtype psf_supersampled_factor: `float`
        :return:
        :rtype:
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
        """Get the `ImageData` instance.

        :param lens_name: name of the lens
        :type lens_name: `str`
        :param band: image band/filter
        :type band: `str`
        :return: `ImageData` instance
        :rtype:
        """
        return ImageData(self.file_system.get_image_file_path(lens_name, band))

    def get_psf_data(self, lens_name, band):
        """Get the `PSFData` instance.

        :param lens_name: lens name
        :type lens_name: `str`
        :param band: image band/filter
        :type band: `str`
        :return: `PSFData` instance
        :rtype:
        """
        return PSFData(self.file_system.get_psf_file_path(lens_name, band))
