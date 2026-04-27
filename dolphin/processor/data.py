# -*- coding: utf-8 -*-
"""This module loads data and PSFs from HDF5 data files."""

__author__ = "ajshajib"

import h5py
import numpy as np
from copy import deepcopy
from lenstronomy.Data.coord_transforms import Coordinates


class Data(object):
    """This is a superclass to load data files securely and consistently."""

    def __init__(self):
        """Initialize the base Data object."""
        pass

    @staticmethod
    def load_from_file(file_path):
        """Load data dictionary from an HDF5 file.

        :param file_path: path to the HDF5 data file
        :type file_path: `str`
        :return: a dictionary containing the data loaded from the file
        :rtype: `dict`
        """
        with h5py.File(file_path, "r") as f:
            data = {}
            for key in f:
                data[key] = f[key][()]

        return data


class ImageData(Data):
    """This class manages the image data of a lens system."""

    def __init__(self, data_file_path):
        """Initialize the ImageData instance by loading from the given file path.

        :param data_file_path: path to the HDF5 data file containing image data
        :type data_file_path: `str`
        """
        super().__init__()

        self._data = self.load_from_file(data_file_path)

    @property
    def kwargs_data(self):
        """Get a deep copy of the `kwargs_data` dictionary.

        :return: a dictionary with the image configuration and data
        :rtype: `dict`
        """
        kwargs_data = deepcopy(self._data)

        return kwargs_data

    def get_image(self):
        """Get the image `ndarray` saved in the class instance.

        :return: a numpy array representing the image data
        :rtype: `ndarray`
        """
        return deepcopy(self._data["image_data"])

    def get_image_coordinate_system(self):
        """Get the coordinate system of the image data.

        :return: an instance representing the image coordinate system
        :rtype: `lenstronomy.Data.coord_transforms.Coordinates`
        """
        ra_at_xy_0 = self.kwargs_data["ra_at_xy_0"]
        dec_at_xy_0 = self.kwargs_data["dec_at_xy_0"]
        transform_pix2angle = np.array(self.kwargs_data["transform_pix2angle"])

        coordinate_system = Coordinates(transform_pix2angle, ra_at_xy_0, dec_at_xy_0)

        return coordinate_system

    def get_image_size(self):
        """Get the number of pixels along one axis in the image.

        :return: the dimension of the square image data
        :rtype: `int`
        """
        return self.kwargs_data["image_data"].shape[0]

    def get_image_pixel_scale(self):
        """Get the pixel scale of the image in arcseconds.

        :return: pixel scale width
        :rtype: `float`
        """
        return self.get_image_coordinate_system().pixel_width


class PSFData(Data):
    """This class manages the Point Spread Function (PSF) data for a lens system."""

    def __init__(self, psf_file_path):
        """Initialize the PSFData instance by loading from the given file path.

        :param psf_file_path: path to the HDF5 PSF data file
        :type psf_file_path: `str`
        """
        super().__init__()

        self._data = self.load_from_file(psf_file_path)

    @property
    def kwargs_psf(self):
        """Get a deep copy of the `kwargs_psf` dictionary with correct formatting.

        :return: a dictionary containing the PSF settings and kernel
        :rtype: `dict`
        """
        kwargs_psf = deepcopy(self._data)
        kwargs_psf["psf_type"] = "PIXEL"
        kwargs_psf["kernel_point_source_init"] = deepcopy(
            kwargs_psf["kernel_point_source"]
        )
        return kwargs_psf
