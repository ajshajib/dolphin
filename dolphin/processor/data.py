# -*- coding: utf-8 -*-
"""
This module loads data and psfs from data files.
"""

import numpy as np
import h5py
from copy import deepcopy


class Data(object):
    """
    This is a superclass to load datafiles.
    """
    def __init__(self):
        pass

    def load_from_file(self, file_dir):
        """
        Load data from h5py file.
        :param file_dir:
        :type file_dir:
        :return:
        :rtype:
        """
        f = h5py.File(file_dir, 'r')

        data = {}
        for key in f:
            data[key] = f[key][()]

        f.close()

        return data


class ImageData(Data):
    """
    This class contains the image of a lens system.
    """
    def __init__(self, data_file_dir):
        """

        """
        super(ImageData, self).__init__()

        self._data = self.load_from_file(data_file_dir)

    @property
    def kwargs_data(self):
        """
        Get `kwargs_data` dictionary.
        :return:
        :rtype:
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
    def __init__(self, psf_file_dir):
        """

        """
        super(PSFData, self).__init__()

        self._data = self.load_from_file(psf_file_dir)

    @property
    def kwargs_psf(self):
        """
        Get `kwargs_psf` dictionary.
        :return:
        :rtype:
        """
        kwargs_psf = deepcopy(self._data)
        kwargs_psf['psf_type'] = 'PIXEL'
        kwargs_psf['kernel_point_source_init'] = deepcopy(
                                        kwargs_psf['kernel_point_source'])
        return kwargs_psf
