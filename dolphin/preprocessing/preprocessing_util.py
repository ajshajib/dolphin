import os
import numpy as np
from astropy.io import fits
from dolphin.processor.files import FileSystem
from pathlib import Path
import subprocess


def make_image_catalog(io_directory, lens_name, data_band):
    """Run Source Extractor to obtain the full Source Extractor catalog.

    :param io_directory: path to the input/output directory. Should not end with slash.
    :type io_directory: `str`
    :param lens_name: name of the system to create a cutout of
    :type lens_name: `str`
    :param data_band: data band to analze
    :type data_band: `str`

    :return: full Source Extractor catalog
    :rtype: `BinTableHDU`
    """
    file_system = FileSystem(io_directory)
    full_image_file_name = (
        Path(file_system.get_data_directory())
        / lens_name
        / f"full_image_{lens_name}_{data_band}.fits"
    )
    subprocess.run(
        f"sex {full_image_file_name} -c default.sex "
        f"-CATALOG_NAME {lens_name}_{data_band}.cat "
        "-PARAMETERS_NAME default.param "
        "-FILTER_NAME default.conv "
        "-STARNNW_NAME default.nnw",
        shell=True,
        check=True,
    )

    catalog_str = f"{lens_name}_{data_band}.cat"
    catalog = fits.open(catalog_str)[2]

    # move catalog from working directory to respective data directory
    preprocessing_str = Path(file_system.get_preprocessing_path(lens_name))
    output_dir = f"{preprocessing_str}/{data_band}"
    os.makedirs(output_dir, exist_ok=True)
    catalog_destination = os.path.join(output_dir, catalog_str)
    os.replace(catalog_str, catalog_destination)

    return catalog


def get_background(io_directory, lens_name, data_band):
    """Estime the background mean and RMS from the Source Extractor catalog.

    :param io_directory: path to the input/output directory. Should not end with slash.
    :type io_directory: `str`
    :param lens_name: name of the system to create a cutout of
    :type lens_name: `str`
    :param data_band: data band to analze
    :type data_band: `str`

    :return: tuple of background mean and RMS as determined by Source Extractor
    :rtype: `tuple` (`float`, `float`)
    """
    file_system = FileSystem(io_directory)
    preprocessing_str = Path(file_system.get_preprocessing_path(lens_name))
    catalog_str = preprocessing_str / f"{data_band}" / f"{lens_name}_{data_band}.cat"

    mean, rms = None, None
    if os.path.exists(catalog_str):
        with fits.open(catalog_str) as hdul:
            header_text = hdul[1].data[0][0]
            for line in header_text:
                line = line.strip().split()
                if not line:
                    continue
                elif line[0] == "SEXBKGND" or line[0] == "SEXBKGND=":
                    mean = float(line[1])
                elif line[0] == "SEXBKDEV" or line[0] == "SEXBKDEV=":
                    rms = float(line[1])
    else:
        _ = make_image_catalog(io_directory, lens_name, data_band)
        with fits.open(catalog_str) as hdul:
            header_text = hdul[1].data[0][0]
            for line in header_text:
                line = line.strip().split()
                if not line:
                    continue
                elif line[0] == "SEXBKGND" or line[0] == "SEXBKGND=":
                    mean = float(line[1])
                elif line[0] == "SEXBKDEV" or line[0] == "SEXBKDEV=":
                    rms = float(line[1])

    return mean, rms


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
