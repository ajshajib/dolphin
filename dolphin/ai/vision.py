# -*- coding: utf-8 -*-
"""This module provides capabilities to create semantic segmentation for image data
using trained neural networks."""

__author__ = "ajshajib"

import numpy as np
from tensorflow.keras.models import load_model
from scipy.ndimage import zoom
from scipy.ndimage import label, center_of_mass
from .ai import AI


class Vision(AI):
    """This class takes image data and creates semantic segmentation for it using an AI
    model."""

    def __init__(self, io_directory_path, source_type="quasar"):
        """Initialize the Vision object.

        :param io_directory_path: path to the input-output directory
        :type io_directory_path: `str`
        :param source_type: the type of astronomical source being modeled. Must be either 'quasar' or 'galaxy'.
        :type source_type: `str`
        """
        if source_type not in ["quasar", "galaxy"]:
            raise ValueError(
                f"Invalid source type: {source_type}. It should be either 'quasar' or 'galaxy'."
            )
        super().__init__(io_directory_path)

        self._source_type = source_type
        self.nn_model_path = self.file_system.get_trained_nn_model_file_path(
            source_type=source_type
        )
        self.nn_model = load_model(self.nn_model_path, compile=False)

    def create_segmentation_for_all_lenses(self, band_name):
        """Create semantic segmentation maps for all lenses in the lens list.

        :param band_name: the observing band to process
        :type band_name: `str`
        :return: None
        :rtype: `None`
        """
        lens_list = self.file_system.get_lens_list()

        for lens_name in lens_list:
            self.create_segmentation_for_single_lens(lens_name, band_name)

        print(f"Semantic segmentation for {len(lens_list)} lenses has been created.")

    def create_segmentation_for_single_lens(self, lens_name, band_name):
        """Create and save semantic segmentation for a single lens system.

        :param lens_name: name of the lens system
        :type lens_name: `str`
        :param band_name: the observing band
        :type band_name: `str`
        :return: the generated semantic segmentation mask
        :rtype: `numpy.ndarray`
        """
        image_data = self.get_image_data(lens_name, band_name)
        image = image_data.get_image()

        segmentation = self.get_semantic_segmentation_from_nn(image)

        self.save_segmentation(lens_name, band_name, segmentation)

        return segmentation

    def relabel_central_satellite_to_lens(self, segmentation):
        """Relabel the blob labeled as 4 (satellite deflector) that is closest to the
        image center as label 1 (central deflector).

        This correction is applied because the AI model can sometimes misclassify the
        central deflector as a satellite deflector when no true satellite deflector is
        present near the image center.

        :param segmentation: 2D segmentation map containing integer class labels
        :type segmentation: `numpy.ndarray`
        :return: modified segmentation map with the closest label-4 blob relabeled to 1
        :rtype: `numpy.ndarray`
        """
        mask_label_4 = segmentation == 4
        labeled_blobs, num_features = label(mask_label_4)

        if num_features == 0:
            return segmentation  # No label 4 regions found

        center_y, center_x = np.array(segmentation.shape) / 2
        min_dist = float("inf")
        closest_blob = None

        for i in range(1, num_features + 1):
            blob_center = center_of_mass(mask_label_4, labeled_blobs, i)
            dist = np.linalg.norm(
                np.array([center_y, center_x]) - np.array(blob_center)
            )
            if dist < min_dist:
                min_dist = dist
                closest_blob = i

        if closest_blob is not None:
            segmentation[(labeled_blobs == closest_blob) & (mask_label_4)] = 1

        return segmentation

    def save_segmentation(self, lens_name, band_name, segmentation):
        """Save the generated segmentation mask to a file.

        :param lens_name: name of the lens system
        :type lens_name: `str`
        :param band_name: the observing band
        :type band_name: `str`
        :param segmentation: semantic segmentation mask array
        :type segmentation: `numpy.ndarray`
        :return: None
        :rtype: `None`
        """
        self.file_system.save_semantic_segmentation(lens_name, band_name, segmentation)

    @staticmethod
    def resize_image(image):
        """Resize the input image to (128, 128) using bicubic interpolation.

        :param image: input image array
        :type image: `numpy.ndarray`
        :return: resampled image with shape (128, 128)
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
        """Resize the predicted segmentation mask back to the original image dimensions.

        :param segmentation: predicted segmentation from the NN (usually 128x128)
        :type segmentation: `numpy.ndarray`
        :param original_size: the desired original size (assumes a square image)
        :type original_size: `int`
        :return: resized segmentation mask
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

        :param image: input image data
        :type image: `numpy.ndarray`
        :return: semantic segmentation mask resized to the original image shape
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

        if 1 not in reshaped_segmentation and 4 in reshaped_segmentation:
            reshaped_segmentation = self.relabel_central_satellite_to_lens(
                reshaped_segmentation
            )

        return reshaped_segmentation
