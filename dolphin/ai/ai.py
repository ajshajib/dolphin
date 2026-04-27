# -*- coding: utf-8 -*-
"""This module provides the base AI class for handling common AI-related operations."""

__author__ = "ajshajib"

from ..processor.data import ImageData
from ..processor.files import FileSystem


class AI(object):
    """This parent class holds common methods for AI-related operations and
    initialization."""

    def __init__(self, io_directory_path):
        """Initialize the AI class.

        :param io_directory_path: path to the input-output directory containing data
        :type io_directory_path: `str`
        """
        self.file_system = FileSystem(io_directory_path)

    def get_image_data(self, lens_name, band_name):
        """Get the image data for a specific lens and band.

        :param lens_name: name of the lens system
        :type lens_name: `str`
        :param band_name: name of the observing band (e.g., 'F814W')
        :type band_name: `str`
        :return: an `ImageData` instance containing the image data
        :rtype: `ImageData`
        """
        image_path = self.file_system.get_image_file_path(lens_name, band_name)
        image_data = ImageData(image_path)

        return image_data
