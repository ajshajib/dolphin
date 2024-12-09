# -*- coding: utf-8 -*-
from ..processor.data import ImageData
from ..processor.files import FileSystem


class AI(object):
    """This parent class holds common methods for AI classes."""

    def __init__(self, io_directory_path):
        """

        :param io_directory_path: path to the input-output directory
        :type io_directory_path: `str`
        """
        self.file_system = FileSystem(io_directory_path)

    def get_image_data(self, lens_name, band_name):
        """Get the image data.

        :param lens_name: lens name
        :type lens_name: `str`
        :param band_name: band name
        :type band_name: `str`
        """
        image_path = self.file_system.get_image_file_path(lens_name, band_name)
        image_data = ImageData(image_path)

        return image_data
