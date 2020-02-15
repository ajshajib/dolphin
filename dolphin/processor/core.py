# -*- coding: utf-8 -*-
"""
This module loads settings from a configuration file.
"""
import sys
import json
from lenstronomy.Workflow.fitting_sequence import FittingSequence

from .files import FileSystem
from .config import ModelConfig
from .data import ImageData
from .data import PSFData


class Processor(object):
    """
    This class contains methods to model a single lens system or a bunch of
    systems from the config files.
    """
    def __init__(self, working_directory):
        """

        :param working_directory: path to the working directory. Should not end
        with slash.
        :type working_directory: `str`
        """
        self.working_directory = working_directory
        self.file_system = FileSystem(working_directory)
        self.lens_list = self.file_system.get_lens_list()

    def swim(self, lens_name, model_id=''):
        """
        Run models for a single lens.
        :param lens_name: lens name
        :type lens_name: `str`
        :param model_id: identifier for the model run
        :type model_id: `str`
        :return:
        :rtype:
        """
        log_file = open(self.file_system.get_log_file_path(lens_name,
                                                             model_id),
                          'wt')
        sys.stdout = log_file

        config = ModelConfig(self.file_system.get_config_file_path(lens_name))

        fitting_sequence = FittingSequence(
            self.get_kwargs_data_joint(),
            config.get_kwargs_model(),
            config.get_kwargs_constraints(),
            config.get_kwargs_likelihood(),
            config.get_kwargs_params()
        )

        fit_output = fitting_sequence.fit_sequence(
                                            config.get_fitting_kwargs_list())
        kwargs_result = fitting_sequence.best_fit(bijective=False)


        log_file.close()

    def get_kwargs_data_joint(self, lens_name):
        """
        Create `kwargs_data` for a lens and given filters.
        :param lens_name:
        :type lens_name:
        :return:
        :rtype:
        """
        config = ModelConfig(self.file_system.get_config_file_path(lens_name))

        bands = config.settings['band']

        kwargs_numerics = config.get_kwargs_numerics()

        multi_band_list = []

        for b, kwargs_num in zip(bands, kwargs_numerics):
            image_data = ImageData(self.file_system.get_image_file_path(
                                                                    lens_name))
            psf_data = PSFData(self.file_system.get_psf_file_path(lens_name))

            multi_band_list.append([
                image_data.kwargs_data,
                psf_data.kwargs_psf,
                kwargs_num
            ])

        kwargs_data_joint = {
            'multi_band_list': multi_band_list,
            'multi_band_type': 'multi-linear'
        }

        return kwargs_data_joint

    def _save_output(self, lens_name, model_id, kwargs_result):
        """
        Save output from fitting sequence.
        :param kwargs_result:
        :type kwargs_result:
        :return:
        :rtype:
        """
        save_file = self.file_system.get_output_file_path(lens_name, model_id)
        with open(save_file) as f:
            json.dump(kwargs_result, f)

    def load_output(self, lens_name, model_id):
        """
        Load from saved output file.
        :param lens_name: lens name
        :type lens_name: `str`
        :param model_id: model identifier provided at run initiation
        :type model_id: `str`
        :return:
        :rtype:
        """
        save_file = self.file_system.get_output_file_path(lens_name, model_id)

        with open(save_file, "r") as f:
            output = json.load(f)

        return output