# -*- coding: utf-8 -*-
"""This class contains helper functions to create a cutout image from the full science
mosaic."""

__author__ = "brady-ryan"

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import h5py
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from pathlib import Path
from lenstronomy.Data.pixel_grid import PixelGrid
import h5py

from mpl_toolkits.axes_grid1 import make_axes_locatable
from dolphin.processor.files import FileSystem
from dolphin.preprocessing import preprocessing_util

from astropy.nddata import Cutout2D

import ipywidgets as widgets
from IPython.display import display


class ImageCutout:
    """This class contains helper functions to create a cutout image from the full
    science mosaic."""

    def __init__(
        self,
        io_directory,
        lens_name,
        data_band,
        instrument,
        full_image_file,
        weight_image_file=None,
    ):
        """Initiate the class from the following inputs:

        :param io_directory: path to the input/output directory. Should not end with slash.
        :type io_directory: `str`
        :param lens_name: name of the system to create a cutout of
        :type lens_name: `str`
        :param data_band: data band of desired PSF
        :type data_band: `str`
        :param instrument: instrument which took the data
        :type intrument: `str`
        :param full_image_file: path to the full science image FITS file
        :type full_image_file: `str`
        :param weight_image_file: (optional) if analyzing HST data,
          the path to the full weight image FITS file
        :type weight_image_file: `str`
        """
        self.io_directory = io_directory
        self.file_system = FileSystem(io_directory)
        self.lens_name = lens_name
        self.data_band = data_band

        supported_instruments = ["HST", "JWST"]
        if instrument not in supported_instruments:
            raise ValueError(
                f"{instrument} is not supported! Options are: {supported_instruments}"
            )

        self.instrument = instrument
        self.image_file_name = full_image_file
        self.weight_file_name = weight_image_file

    def plot_mosaic(self, vmin=-1, vmax=1.5):
        """Plot the full science mosaic.

        :param vmin: lower limit of color map scale
        :type vmin: `float`
        :param vmax: upper limit of color map scale
        :type vmax: `float`

        :return: None
        """
        if self.instrument == "JWST":
            with fits.open(self.image_file_name) as hdul:
                data_full = hdul["SCI"].data
        else:
            with fits.open(self.image_file_name) as hdul:
                data_full = hdul[0].data

        _, ax = plt.subplots(figsize=(10, 10))
        plt_data = np.log10(data_full)

        im = ax.matshow(plt_data, origin="lower", vmin=vmin, vmax=vmax)
        ax.autoscale(False)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.show()

    def make_image_cutout(
        self,
        cutout_scale=100,
        cutout_center=None,
        save=False,
        use_noise_map=False,
        kwargs_mask=None,
        vmin=-0.75,
        vmax=1.5,
    ):
        """Create the science image cutout in the expected `dolphin` format. This
        includes generating `image_data`, `ra_at_xy_0`, `dec_at_xy_0`,
        `transform_pix2angle`, `exposure_time`, and either `background_rms` or
        `noise_map`, depending on the specified type with `use_noise_map`.

        :param cutout_scale: pixel length of one side of the cutout image
        :type cutout_scale: `int`
        :param cutout_center: specified coordinates (in degrees) of the cutout center. If `None`,
          then the target RA and DEC will be used as the cutout center.
        :type cutout_center: `tuple` (`float`, `float`)
        :param save: if `True`, creates the full HDF5 file in the proper location expected
          by `Dolphin`.
        :type save: `bool`
        :param use_noise_map: if `True`, uses a cutout of the noise map to estimate background
          quantities in the modeling. Otherwise, the scalar background RMS determined by `photutils`
          is used.
        :type use_noise_map: `bool`
        :param kwargs_mask: list of dictionaries corresponding to masking keywork arguments. Supported types,
          with all required keywords, are as follows: [{"type": "circle", "center": `tuple` (`int`, `int`),
          "radius": `int`}, {"type": "square", "center": `tuple` (`int`, `int`), "size": `int`},
          {"type": "ellipse", "center": `tuple` (`int`, `int`), "a": `int`, "b": `int`}]. To invert the boolean
          logic of a specific mask index, one must place `"invert": True` in that dictionary.
        :type kwargs_mask: `list` of `dict`
        :param vmin: lower limit of color map scale
        :type vmin: `float`
        :param vmax: upper limit of color map scale
        :type vmax: `float`

        :return: None
        """
        kwargs_data = {}
        if cutout_center is None:
            with fits.open(self.image_file_name) as hdul:
                header = hdul[0].header
            if self.instrument == "JWST":
                ra = header["TARG_RA"] * u.deg
                dec = header["TARG_DEC"] * u.deg
            else:
                ra = header["RA_TARG"] * u.deg
                dec = header["DEC_TARG"] * u.deg
            print(f"RA = {ra:.6f}")
            print(f"DEC = {dec:.6f}")
        else:
            ra = cutout_center[0] * u.deg
            dec = cutout_center[1] * u.deg

        center = SkyCoord(ra, dec)

        mean_bkd, sigma_bkd = preprocessing_util.get_background(self.image_file_name)

        if self.instrument == "JWST":
            with fits.open(self.image_file_name) as hdul:
                header = hdul["SCI"].header
                data_full = hdul["SCI"].data
                err_map = hdul["ERR"].data
                exposure_time = header.get("XPOSURE")

            wcs = WCS(header)
            image_data = Cutout2D(
                data_full, position=center, size=cutout_scale, wcs=wcs
            ).data
            image_reduced = image_data - mean_bkd
            if use_noise_map:
                noise_map = Cutout2D(
                    err_map, position=center, size=image_reduced.shape, wcs=wcs
                ).data

                kwargs_data["noise_map"] = noise_map
            else:
                kwargs_data["background_rms"] = sigma_bkd
        elif self.instrument == "HST":
            with fits.open(self.image_file_name) as hdul:
                header = hdul[0].header
                data_full = hdul[0].data
                exposure_time = header.get("EXPTIME")

            wcs = WCS(header)
            image_data = Cutout2D(
                data_full, position=center, size=cutout_scale, wcs=wcs
            ).data
            image_reduced = image_data - mean_bkd
            if use_noise_map:
                with fits.open(self.weight_file_name) as hdul:
                    wht_full = hdul[0].data
                wht_full[wht_full <= 0] = 10 ** (-10)
                full_noise_map = np.abs(data_full) / wht_full + sigma_bkd**2

                noise_map = Cutout2D(
                    full_noise_map, position=center, size=image_reduced.shape, wcs=wcs
                ).data

                kwargs_data["noise_map"] = noise_map
            else:
                kwargs_data["background_rms"] = sigma_bkd

        transform_pix2angle = wcs.pixel_scale_matrix * 3600.0

        ny, nx = image_reduced.shape
        x_c = nx // 2
        y_c = ny // 2

        dra, ddec = transform_pix2angle.dot([x_c, y_c])
        ra_at_xy_0 = -dra
        dec_at_xy_0 = -ddec

        kwargs_data["image_data"] = image_reduced
        kwargs_data["ra_at_xy_0"] = ra_at_xy_0
        kwargs_data["dec_at_xy_0"] = dec_at_xy_0
        kwargs_data["transform_pix2angle"] = transform_pix2angle
        kwargs_data["exposure_time"] = exposure_time

        mask = preprocessing_util.build_mask(image_reduced.shape, kwargs_mask)

        if use_noise_map:
            fig, ax = plt.subplots(1, 2, figsize=(8, 8))
            im_data = ax[0].matshow(
                np.log10(np.clip(image_reduced * mask, 1e-10, None)),
                vmin=vmin,
                vmax=vmax,
                origin="lower",
                cmap="cubehelix",
            )
            ax[0].autoscale(False)
            ax[0].set_title(f"{self.data_band} Cutout Data", fontsize=20)
            ax[0].xaxis.set_ticks_position("bottom")
            fig.colorbar(im_data, ax=ax[0], fraction=0.05)

            im_noise = ax[1].matshow(
                np.log10(noise_map * mask),
                vmin=vmin,
                origin="lower",
                cmap="cubehelix",
            )
            ax[1].autoscale(False)
            ax[1].set_title(f"{self.data_band} Noise Map", fontsize=20)
            ax[1].xaxis.set_ticks_position("bottom")
            fig.colorbar(im_noise, ax=ax[1], fraction=0.05)
        else:
            fig, ax = plt.subplots(figsize=(8, 8))
            im_data = ax.matshow(
                np.log10(np.clip(image_reduced * mask, 1e-10, None)),
                vmin=vmin,
                vmax=vmax,
                origin="lower",
                cmap="cubehelix",
            )
            ax.autoscale(False)
            ax.set_title(f"{self.data_band} Cutout Data", fontsize=20)
            ax.xaxis.set_ticks_position("bottom")
            fig.colorbar(im_data, ax=ax, fraction=0.05)

        plt.tight_layout()
        plt.show()

        if save:
            self.file_system.save_cutout_image(
                self.lens_name, self.data_band, kwargs_data
            )
            if kwargs_mask is not None:
                self.file_system.save_mask(self.lens_name, self.data_band, mask)

    def get_angular_coordinates(self, vmin=-1, vmax=1.5):
        """Helper function to obtain the intial position guesses of different model
        components. Interactively click on the image to obtain angular coordinates. The
        printed RA and DEC are in angular coordinates relative to the (0, 0) arcsecond
        pixel, as expected by `dolphin`.

        :param vmin: lower limit of color map scale
        :type vmin: float
        :param vmax: upper limit of color map scale
        :type vmax: float

        :return: clickable figure to print coordinates relative to (0,0) (RA, DEC)
        :rtype: `fig`
        """
        image_file = Path(
            self.file_system.get_image_file_path(self.lens_name, self.data_band)
        )

        with h5py.File(image_file, "r") as kwargs_data:
            image_data = kwargs_data["image_data"][:]
            ra_at_xy_0 = kwargs_data["ra_at_xy_0"][()]
            dec_at_xy_0 = kwargs_data["dec_at_xy_0"][()]
            transform_pix2angle = kwargs_data["transform_pix2angle"][:]

        ny, nx = image_data.shape

        kwargs_pixel = {
            "nx": nx,
            "ny": ny,
            "ra_at_xy_0": ra_at_xy_0,
            "dec_at_xy_0": dec_at_xy_0,
            "transform_pix2angle": transform_pix2angle,
        }
        pixel_grid = PixelGrid(**kwargs_pixel)

        fig, ax = plt.subplots(dpi=100)

        ax.imshow(
            np.log10(image_data),
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap="cubehelix",
        )

        ax.set_title("Click on the image to display RA/DEC")
        out = widgets.Output()
        display(out)

        # compute the RA/DEC of the center pixel once, outside onclick
        center_x = nx // 2
        center_y = ny // 2
        ra_center, dec_center = pixel_grid.map_pix2coord(center_x, center_y)

        def onclick(event):
            if event.inaxes != ax or event.xdata is None or event.ydata is None:
                return

            x = event.xdata
            y = event.ydata

            # get RA/DEC at the clicked pixel in the original frame
            ra, dec = pixel_grid.map_pix2coord(x, y)

            # re-reference to the center pixel
            ra_rel = ra - ra_center
            dec_rel = dec - dec_center

            ax.plot(x, y, "ro", ms=5, mew=2)
            fig.canvas.draw_idle()

            with out:
                print(f"Pixel: ({x:.2f}, {y:.2f})")
                print(f"RA  = {ra_rel:.4f} arcsec (from center)")
                print(f"DEC = {dec_rel:.4f} arcsec (from center)\n")
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("button_press_event", onclick)
        return fig
