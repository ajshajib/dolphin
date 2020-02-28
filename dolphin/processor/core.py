# -*- coding: utf-8 -*-
"""
This module loads settings from a configuration file.
"""
import sys
import json
import numpy as np
from lenstronomy.Workflow.fitting_sequence import FittingSequence

from .files import FileSystem
from .config import ModelConfig
from .data import ImageData
from .data import PSFData

try:
    from mpi4py import MPI
    COMM_RANK = MPI.COMM_WORLD.Get_rank()
except:
    COMM_RANK = 0


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

    def swim(self, lens_name, model_id, log=True, mpi=False):
        """
        Run models for a single lens.
        :param lens_name: lens name
        :type lens_name: `str`
        :param model_id: identifier for the model run
        :type model_id: `str`
        :param log: if `True`, all `print` statements will be logged
        :type log: `bool`
        :param mpi: MPI option
        :type mpi: `bool`
        :return:
        :rtype:
        """
        if log and COMM_RANK == 0:
            log_file = open(self.file_system.get_log_file_path(lens_name,
                                                               model_id),
                            'wt')
            sys.stdout = log_file

        config = self.get_lens_config(lens_name)

        fitting_sequence = FittingSequence(
            self.get_kwargs_data_joint(lens_name),
            config.get_kwargs_model(),
            config.get_kwargs_constraints(),
            config.get_kwargs_likelihood(),
            config.get_kwargs_params(),
            mpi=mpi
        )

        fit_output = fitting_sequence.fit_sequence(
                                            config.get_fitting_kwargs_list())
        kwargs_result = fitting_sequence.best_fit(bijective=False)

        output = {
            'kwargs_result': kwargs_result,
            'fit_output': fit_output,
        }

        if COMM_RANK == 0:
            self._save_output(lens_name, model_id, output)

        if log and COMM_RANK == 0:
            log_file.close()

    def get_lens_config(self, lens_name):
        """
        Get the `ModelConfig` object for a lens.
        :param lens_name: lens name
        :type lens_name: `str`
        :return:
        :rtype:
        """
        return ModelConfig(self.file_system.get_config_file_path(lens_name))

    def get_kwargs_data_joint(self, lens_name):
        """
        Create `kwargs_data` for a lens and given filters.

        :param lens_name:
        :type lens_name:
        :return:
        :rtype:
        """
        config = self.get_lens_config(lens_name)

        bands = config.settings['band']

        kwargs_numerics = config.get_kwargs_numerics()

        multi_band_list = []

        for b, kwargs_num in zip(bands, kwargs_numerics):
            image_data = self.get_image_data(lens_name, b)
            psf_data = self.get_psf_data(lens_name, b)

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

    def get_image_data(self, lens_name, band):
        """
        Get the `ImageData` instance.

        :param lens_name: name of the lens
        :type lens_name: `str`
        :param band: name of band
        :type band: `str`
        :return: `ImageData` instance
        :rtype:
        """
        return ImageData(self.file_system.get_image_file_path(lens_name, band))

    def get_psf_data(self, lens_name, band):
        """
        Get the `PSFData` instance.

        :param lens_name:
        :type lens_name:
        :param band:
        :type band:
        :return:
        :rtype:
        """
        return PSFData(self.file_system.get_psf_file_path(lens_name, band))

    def _save_output(self, lens_name, model_id, output):
        """
        Save output from fitting sequence.

        :param kwargs_result:
        :type kwargs_result:
        :return:
        :rtype:
        """
        save_file = self.file_system.get_output_file_path(lens_name, model_id)
        with open(save_file, 'w') as f:
            json.dump(self.encode_numpy_arrays(output), f,
                      ensure_ascii=False, indent=4)

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

        with open(save_file, 'r') as f:
            output = json.load(f)

        return self.decode_numpy_arrays(output)

    @classmethod
    def encode_numpy_arrays(cls, obj):
        """
        Encode a list/dictionary containing numpy arrays through recursion
        for JSON serialization.

        :param obj: object
        :type obj:
        :return: object with ndarrays encoded in dictionaries
        :rtype:
        """
        if isinstance(obj, np.ndarray):
            return {
                '__ndarray__': obj.tolist(),
                'shape': obj.shape
            }
        elif isinstance(obj, list):
            encoded = []
            for element in obj:
                encoded.append(cls.encode_numpy_arrays(element))
            return encoded
        elif isinstance(obj, dict):
            encoded = {}
            for key, value in obj.items():
                encoded[key] = cls.encode_numpy_arrays(value)
            return encoded
        else:
            return obj

    @classmethod
    def decode_numpy_arrays(cls, obj):
        """
        Decode a list/dictionary containing encoded numpy arrays through
        recursion.

        :param obj: object with ndarrays encoded in dictionaries
        :type obj:
        :return: object
        :rtype:
        """
        if isinstance(obj, dict):
            if '__ndarray__' in obj:
                return np.asarray(obj['__ndarray__']).reshape(obj['shape'])
            else:
                decoded = {}
                for key, value in obj.items():
                    decoded[key] = cls.decode_numpy_arrays(value)
                return decoded
        elif isinstance(obj, list):
            decoded = []
            for element in obj:
                decoded.append(cls.decode_numpy_arrays(element))
            return decoded
        else:
            return obj
