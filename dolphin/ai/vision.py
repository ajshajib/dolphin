# -*- coding: utf-8 -*-
"""This module takes an image data and create semantic segmentation for it."""

import numpy as np
from tensorflow.keras.models import load_model
from scipy.ndimage import zoom

from .ai import AI


class Vision(AI):
    """This class takes an image data and create semantic segmentation for it."""

    def __init__(self, io_directory_path, source_type="quasar"):
        """

        :param data_file_path: path to a data file
        :type data_file_path: `str`
        """
        if source_type not in ["quasar", "galaxy"]:
            raise ValueError(
                f"Invalid source type: {source_type}. It should be either 'quasar' or 'galaxy'."
            )
        super(Vision, self).__init__(io_directory_path)

        self._source_type = source_type
        self.nn_model_path = self.file_system.get_trained_nn_model_file_path(
            source_type=source_type
        )
        self.nn_model = load_model(self.nn_model_path, compile=False)

    def create_segmentation_for_all_lenses(self, band_name):
        """Create semantic segmentation for all lenses.

        :param band_name: band name
        :type band_name: `str`
        """
        lens_list = self.file_system.get_lens_list()

        for lens_name in lens_list:
            self.create_segmentation_for_single_lens(lens_name, band_name)

        print(f"Semantic segmentation for {len(lens_list)} lenses has been created.")

    def create_segmentation_for_single_lens(self, lens_name, band_name):
        """Create semantic segmentation for a single lens.

        :param lens_name: lens name
        :type lens_name: `str`
        :param band_name: band name
        :type band_name: `str`
        """
        image_data = self.get_image_data(lens_name, band_name)
        image = image_data.get_image()

        segmentation = self.get_semantic_segmentation_from_nn(image)

        self.save_segmentation(lens_name, band_name, segmentation)

        return segmentation

    def save_segmentation(self, lens_name, band_name, segmentation):
        """Save the segmentation to a file.

        :param lens_name: lens name
        :type lens_name: `str`
        :param band_name: band name
        :type band_name: `str`
        :param segmentation: semantic segmentation
        :type segmentation: `numpy.ndarray`
        """
        self.file_system.save_semantic_segmentation(lens_name, band_name, segmentation)

    @staticmethod
    def resize_image(image):
        """Resize the image to (128, 128, 1).

        :param image: image data
        :type image: `numpy.ndarray`
        :return: resized image
        :rtype: `numpy.ndarray`
        """
        # Target shape for spatial dimensions
        target_shape = (128, 128)

        zoom_factors = [
            target_shape[0] / image.shape[0],
            target_shape[1] / image.shape[1],
        ]
        resampled_image = zoom(
            image, zoom_factors, order=3
        )  # order=3 for bicubic interpolation
        return resampled_image

    @staticmethod
    def resize_segmentation_to_original_size(segmentation, original_size):
        """Resize the prediction to the original size.

        :param segmentation: predicted segmentation from the NN
        :type segmentation: `numpy.ndarray`
        :param original_size: original size of the image
        :type original_size: int
        :return: resized prediction
        :rtype: `numpy.ndarray`
        """
        segmentation_shape = segmentation.shape
        segmentation_reshaped = np.zeros((original_size, original_size))

        for i in range(original_size):
            for j in range(original_size):
                segmentation_reshaped[i, j] = segmentation[
                    int(i / float(original_size) * segmentation_shape[0]),
                    int(j / float(original_size) * segmentation_shape[1]),
                ]

        return segmentation_reshaped

    def get_semantic_segmentation_from_nn(self, image):
        """Get semantic segmentation for the image from the trained neural network.

        :param image: image data
        :type image: `numpy.ndarray`
        :return: semantic segmentation
        :rtype: `numpy.ndarray`
        """
        resized_image = self.resize_image(image)
        image_input = np.expand_dims(resized_image, axis=0)

        # Get predictions from the model
        prediction = self.nn_model.predict(image_input)  # Shape: (1, 128, 128, 5)

        segmentation = np.argmax(prediction[0], axis=-1)  # Shape: (128, 128)

        # Resize the segmentation to the original size
        reshaped_segmentation = self.resize_segmentation_to_original_size(
            segmentation, image.shape[0]
        )

        if self._source_type == "galaxy":
            # Setting the satellite label to 4 to match with the case of quasar
            reshaped_segmentation[reshaped_segmentation == 3] = 4

        return reshaped_segmentation
