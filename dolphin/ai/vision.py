# -*- coding: utf-8 -*-
"""This module takes an image data and create semantic segmentation for it."""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.ndimage import zoom
import numpy as np
import os
from .ai import AI


class Vision(AI):
    """This class takes an image data and create semantic segmentation for it."""

    def __init__(self, io_directory_path, source_type="quasar"):
        """

        :param data_file_path: path to a data file
        :type data_file_path: `str`
        """
        super(Vision, self).__init__(io_directory_path)

        # To-DO: Load the trained NN model.
        
        if source_type == "quasar":
            data_file_path = os.path.join("dolphin", "dolphin", "ai", "Final_Galaxy_Quasar_unet_model.h5")
            self.nn_model = load_model(data_file_path)  # This is a placeholder for the trained NN model.
        # elif source_type == "galaxy":
        #   self.nn_model = None  # This is a placeholder for the trained NN model.
        else:
            raise ValueError("Invalid source type.")  # This is a placeholder for the trained NN model.

    def create_segmentation_for_all_lenses(self, band_name):
        """Create semantic segmentation for all lenses.

        :param band_name: band name
        :type band_name: `str`
        """
        lens_list = self.file_system.get_lens_list()

        for lens_name in lens_list:
            self.create_segmentation_for_single_lens(lens_name, band_name)

        print(f"Semantic segmentation for {len(list)} lenses has been created.")

    def create_segmentation_for_single_lens(self, lens_name, band_name):
        """Create semantic segmentation for a single lens.

        :param lens_name: lens name
        :type lens_name: `str`
        :param band_name: band name
        :type band_name: `str`
        """
        image_data = self.get_image_data(lens_name, band_name)
        image = image_data.get_image()
        reshaped_image = self.resize_image(image)

        segmentation = self.get_semantic_segmentation_from_nn(reshaped_image)
        reshaped_segmentation = self.resize_segmentation_to_original_size(
            segmentation, image.shape[0]
        )

        self.save_segmentation(lens_name, band_name, reshaped_segmentation)

    def save_segmentation(self, lens_name, band_name, segmentation):
        """Save the segmentation to a file.

        :param lens_name: lens name
        :type lens_name: `str`
        :param band_name: band name
        :type band_name: `str`
        :param segmentation: semantic segmentation
        :type segmentation: `numpy.ndarray`
        """
        segmentation_path = self.file_system.get_semantic_segmentation_file_path(
            lens_name, band_name
        )
        self.file_system.save_semantic_segmentation(segmentation_path, segmentation)

    @staticmethod


    def resize_image(image):
        """Resize the image to (128, 128, 1).

        :param image: image data
        :type image: `numpy.ndarray`
        :return: resized image
        :rtype: `numpy.ndarray`
        """
        target_shape = (128, 128)

        # Convert RGB to grayscale if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
        # Average over the color channels to convert to grayscale
            image = np.mean(image, axis=2)
        elif len(image.shape) == 3 and image.shape[2] == 1:
        # Squeeze the single channel dimension
            image = image.squeeze()

        # Calculate zoom factors for resizing
        zoom_factors = [target_shape[0] / image.shape[0], target_shape[1] / image.shape[1]]
        resized_image = zoom(image, zoom_factors, order=1)

        # Add a single channel dimension
        resized_image = resized_image[..., np.newaxis]

        return resized_image

        pass

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

        #def get_semantic_segmentation_from_nn(self, image):
    def get_semantic_segmentation_from_nn(self, image):
        
        """Get semantic segmentation for the image from the trained neural network.

        :param image: image data
        :type image: `numpy.ndarray`
        :return: semantic segmentation
        :rtype: `numpy.ndarray`
        """
        # Resize the image to match the model input size
        resized_image = self.resize_image(image)  # Shape: (128, 128, 1)

        # Expand dimensions to include the batch axis: (1, 128, 128, 1)
        image_batch = np.expand_dims(resized_image, axis=0)  

        # Get predictions from the model
        prediction = self.nn_model.predict(image_batch)  # Shape: (1, 128, 128, 1)
        return np.squeeze(prediction)  



        
