# -*- coding: utf-8 -*-
"""This module contains helper functions to aid in
data preprocessing."""

__author__ = "brady-ryan"

import numpy as np
from astropy.io import fits
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground

from dolphin.processor.files import FileSystem
from pathlib import Path


def get_background(io_directory, lens_name, data_band):
    """Estimate the background mean and RMS using `photutils`.

    :param io_directory: path to the input/output directory. Should not end with slash.
    :type io_directory: `str`
    :param lens_name: name of the system to create a cutout of
    :type lens_name: `str`
    :param data_band: data band to analze
    :type data_band: `str`

    :return: tuple of background mean and RMS as determined by `photutils`
    :rtype: `tuple` (`float`, `float`)
    """
    file_system = FileSystem(io_directory)
    data_dir = Path(file_system.get_data_directory())
    full_image = data_dir / f"{lens_name}" / f"full_image_{lens_name}_{data_band}.fits"
    full_data = fits.getdata(full_image)

    sigma_clip = SigmaClip(sigma=3.0)
    background_estimator = MedianBackground()
    background_class = Background2D(
        np.copy(full_data),
        (50, 50),
        filter_size=(3, 3),
        sigma_clip=sigma_clip,
        bkg_estimator=background_estimator,
    )
    background = background_class.background_median
    background_rms = background_class.background_rms_median

    return background, background_rms


def build_mask(shape, kwargs_mask=None):
    """Build a combined boolean mask from multiple geometric definitions. Options are
    "circle", "square", and "ellipse.".

    :param shape: tuple describing the shape of the image to mask
    :type shape: `tuple` (`int`, `int`)
    :param kwargs_mask: list of dictionaries corresponding to masking keywork arguments. Supported types,
        with all required keywords, are as follows: [{"type": "circle", "center": `tuple` (`int`, `int`),
        "radius": `int`}, {"type": "square", "center": `tuple` (`int`, `int`), "size": `int`},
        {"type": "ellipse", "center": `tuple` (`int`, `int`), "a": `int`, "b": `int`}]. To invert the boolean
        logic of a specific mask index, one must place "invert": True in that dictionary.
    :type kwargs_mask: `list` of `dict`
    """
    if kwargs_mask is None:
        return np.ones(shape)
    else:
        ny, nx = shape
        y, x = np.mgrid[0:ny, 0:nx]
        keep_mask = np.zeros(shape, dtype=bool)
        remove_mask = np.zeros(shape, dtype=bool)
        for param in kwargs_mask:
            if param["type"] == "circle":
                cx, cy = param["center"]
                r = param["radius"]
                mask = (x - cx) ** 2 + (y - cy) ** 2 <= r**2
            elif param["type"] == "square":
                cx, cy = param["center"]
                hx = param["size"]
                mask = (np.abs(x - cx) <= hx / 2) & (np.abs(y - cy) <= hx / 2)
            elif param["type"] == "ellipse":
                cx, cy = param["center"]
                a = param["a"]
                b = param["b"]
                mask = ((x - cx) ** 2 / a**2) + ((y - cy) ** 2 / b**2) <= 1
            else:
                raise ValueError(f"Unknown mask type: {param['type']}")

            if param.get("invert", False):
                remove_mask |= mask
            else:
                keep_mask |= mask

        if not any(not p.get("invert", False) for p in kwargs_mask):
            keep_mask[:] = True
        combined_mask = keep_mask & ~remove_mask

        return combined_mask.astype(float)
