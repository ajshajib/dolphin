# -*- coding: utf-8 -*-
"""
This module has a class for maintaining the file system.
"""

from pathlib import Path
import os


class FileSystem(object):
    """
    This class contains the method to handle the file system and directory
    addresses.
    """
    def __init__(self, working_directory):
        """
        Initiates a FileSystem object with `working_directory` as root.
        :param working_directory: path to working directory
        :type working_directory: str
        """
        self._root_path = Path(working_directory)
        self.root = working_directory

    @staticmethod
    def path2str(path):
        """
        Converts a pathlib Path into string.
        :param path: path to a file or directory
        :type path: `Path`
        :return: path to a file or directory
        :rtype: `str`
        """
        return str(path.resolve())

    def get_lens_list_file_path(self):
        """
        Get the address for the lens_list.txt file.
        :return:
        :rtype:
        """
        return self.path2str(self._root_path / 'lens_list.txt')

    def get_lens_list(self):
        """
        Get the list of lenses from lens_list.txt.
        :return:
        :rtype:
        """
        lens_list = []

        for line in open(self.get_lens_list_file_path(), 'r'):
            if not line.startswith('#'):
                lens_list.append(line.rstrip('\n').rstrip('\r'))

        return lens_list

    def get_config_file_path(self, lens_name):
        """
        Get the file path to the config file for `lens_name`.
        :param lens_name: lens name
        :type lens_name: `str`
        :return: path to the config file
        :rtype: `str`
        """
        return self.path2str(self._root_path / 'settings'
                             / '{}_config.yml'.format(lens_name))

    def get_logs_directory(self):
        """
        Get directory for logs folder. If the directory doesn't exist,
        a folder is created.
        :return:
        :rtype:
        """
        logs_dir = self.path2str(self._root_path / 'logs')

        if not os.path.isdir(logs_dir):
            os.mkdir(logs_dir)

        return logs_dir

    def get_settings_directory(self):
        """
        Get directory for settings folder. If the directory doesn't exist,
        a folder is created.
        :return:
        :rtype:
        """
        settings_dir = self.path2str(self._root_path / 'settings')

        if not os.path.isdir(settings_dir):
            os.mkdir(settings_dir)

        return settings_dir

    def get_outputs_directory(self):
        """
        Get directory for settings folder. If the directory doesn't exist,
        a folder is created.
        :return:
        :rtype:
        """
        outputs_dir = self.path2str(self._root_path / 'outputs')

        if not os.path.isdir(outputs_dir):
            os.mkdir(outputs_dir)

        return outputs_dir

    def get_data_directory(self):
        """
        Get directory for data folder. If the directory doesn't exist,
        a folder is created.
        :return:
        :rtype:
        """
        data_dir = self.path2str(self._root_path / 'data')

        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)

        return data_dir

    def get_image_file_path(self, lens_name, band):
        """
        Get the file path for the imaging data for `lens_name`.
        :param lens_name: lens name
        :type lens_name: `str`
        :param band: band name
        :type band: `str`
        :return: file path
        :rtype: `str`
        """
        return self.path2str(self._root_path / 'data'
                             / '{}'.format(lens_name)
                             / 'image_{}_{}.hdf5'.format(lens_name, band)
                             )

    def get_psf_file_path(self, lens_name, band):
        """
        Get the file path for the PSF data for `lens_name`.
        :param lens_name: lens name
        :type lens_name: `str`
        :param band: band name
        :type band: `str`
        :return: file path
        :rtype: `str`
        """
        return self.path2str(self._root_path / 'data'
                             / '{}'.format(lens_name)
                             / 'psf_{}_{}.hdf5'.format(lens_name, band)
                             )

    def get_log_file_path(self, lens_name, model_id):
        """
        Get the file path for the PSF data for `lens_name`.
        :param lens_name: lens name
        :type lens_name: `str`
        :param model_id: identifier for run model
        :type model_id: `str`
        :return: file path
        :rtype: `str`
        """
        return self.path2str(self._root_path / 'logs'
                             / 'log_{}_{}.txt'.format(lens_name, model_id)
                             )

    def get_output_file_path(self, lens_name, model_id):
        """
        Get the file path for the PSF data for `lens_name`.
        :param lens_name: lens name
        :type lens_name: `str`
        :param model_id: identifier for run model
        :type model_id: `str`
        :return: file path
        :rtype: `str`
        """
        return self.path2str(self._root_path / 'outputs'
                             / 'output_{}_{}.json'.format(lens_name, model_id)
                             )