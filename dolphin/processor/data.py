# -*- coding: utf-8 -*-
"""This module loads data and psfs from data files."""
__author__ = "ajshajib"

import h5py
import numpy as np
from copy import deepcopy
from lenstronomy.Data.coord_transforms import Coordinates


class Data(object):
    """This is a superclass to load datafiles."""

    def __init__(self):
        pass

    @staticmethod
    def load_from_file(file_path):
        """Load data from h5py file.

        :param file_path: path to a file
        :type file_path: `str`
        :return:
        :rtype:
        """
        with h5py.File(file_path, "r") as f:
            data = {}
            for key in f:
                data[key] = f[key][()]

        return data


class ImageData(Data):
    """This class contains the image of a lens system."""

    def __init__(self, data_file_path):
        """

        :param data_file_path: path to a data file
        :type data_file_path: `str`
        """
        super(ImageData, self).__init__()

        self._data = self.load_from_file(data_file_path)

    @property
    def kwargs_data(self):
        """Get `kwargs_data` dictionary.

        :return: `kwargs_data`
        :rtype: `dict`
        """
        kwargs_data = deepcopy(self._data)

        return kwargs_data

    def get_image(self):
        """Get image `ndarray` from the saved in the class instance.

        :return: image
        :rtype: `ndarray`
        """
        return deepcopy(self._data["image_data"])

    def get_image_coordinate_system(self):
        """Get the coordinate system of the image data.

        :return: coordinate system
        :rtype: `str`
        """
        ra_at_xy_0 = self.kwargs_data["ra_at_xy_0"]
        dec_at_xy_0 = self.kwargs_data["dec_at_xy_0"]
        transform_pix2angle = np.array(self.kwargs_data["transform_pix2angle"])

        coordinate_system = Coordinates(transform_pix2angle, ra_at_xy_0, dec_at_xy_0)

        return coordinate_system

    def get_image_size(self):
        """Get the number of pixels in the image.

        :return: number of pixels
        :rtype: `int`
        """
        return self.kwargs_data["image_data"].shape[0]

    def get_image_pixel_scale(self):
        """Get the pixel scale of the image.

        :return: pixel scale
        :rtype: `float`
        """
        transformation_matrix = np.array(self.kwargs_data["transform_pix2angle"])
        pixel_scale = np.sqrt(np.abs(np.linalg.det(transformation_matrix)))

        return pixel_scale


class PSFData(Data):
    """This class contains the PSF for a lens system."""

    def __init__(self, psf_file_path):
        """

        :param psf_file_path: path to a PSF data file
        :type psf_file_path: `str`
        """
        super(PSFData, self).__init__()

        self._data = self.load_from_file(psf_file_path)

    @property
    def kwargs_psf(self):
        """Get `kwargs_psf` dictionary.

        :return: `kwargs_psf`
        :rtype: `dict`
        """
        kwargs_psf = deepcopy(self._data)
        kwargs_psf["psf_type"] = "PIXEL"
        kwargs_psf["kernel_point_source_init"] = deepcopy(
            kwargs_psf["kernel_point_source"]
        )
        return kwargs_psf
