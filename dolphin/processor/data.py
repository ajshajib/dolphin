# -*- coding: utf-8 -*-
"""
This module loads data and psfs from data files.
"""
__author__ = 'ajshajib'

import h5py
from copy import deepcopy


class Data(object):
    """
    This is a superclass to load datafiles.
    """
    def __init__(self):
        pass

    @staticmethod
    def load_from_file(file_path):
        """
        Load data from h5py file.

        :param file_path: path to a file
        :type file_path: `str`
        :return:
        :rtype:
        """
        f = h5py.File(file_path, 'r')

        data = {}
        for key in f:
            data[key] = f[key][()]

        f.close()

        return data


class ImageData(Data):
    """
    This class contains the image of a lens system.
    """
    def __init__(self, data_file_path):
        """

        :param data_file_path: path to a data file
        :type data_file_path: `str`
        """
        super(ImageData, self).__init__()

        self._data = self.load_from_file(data_file_path)

    @property
    def kwargs_data(self):
        """
        Get `kwargs_data` dictionary.

        :return: `kwargs_data`
        :rtype: `dict`
        """
        kwargs_data = deepcopy(self._data)

        return kwargs_data

    def get_image(self):
        """
        Get image `ndarray` from the saved in the class instance.

        :return: image
        :rtype: `ndarray`
        """
        return deepcopy(self._data['image_data'])


class PSFData(Data):
    """
    This class contains the PSF for a lens system.
    """
    def __init__(self, psf_file_path):
        """

        :param psf_file_path: path to a PSF data file
        :type psf_file_path: `str`
        """
        super(PSFData, self).__init__()

        self._data = self.load_from_file(psf_file_path)

    @property
    def kwargs_psf(self):
        """
        Get `kwargs_psf` dictionary.

        :return: `kwargs_psf`
        :rtype: `dict`
        """
        kwargs_psf = deepcopy(self._data)
        kwargs_psf['psf_type'] = 'PIXEL'
        kwargs_psf['kernel_point_source_init'] = deepcopy(
                                        kwargs_psf['kernel_point_source'])
        return kwargs_psf
