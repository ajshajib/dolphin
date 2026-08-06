# -*- coding: utf-8 -*-
"""Tests for ImageCutout module."""

import numpy as np
import numpy.testing as npt
from pathlib import Path

from dolphin.preprocessing.image_cutout import ImageCutout

from unittest.mock import patch, MagicMock
import pytest
import h5py

_ROOT_DIR = Path(__file__).resolve().parents[2]
_TEST_IO_DIR = _ROOT_DIR / "io_directory_example"


class TestImageCutout(object):
    def setup_class(self):
        self.imagecutout = ImageCutout(
            _TEST_IO_DIR,
            lens_name="MOCK",
            data_band="F814W",
            instrument="HST",
            full_image_file="TEST",
            weight_image_file="TEST",
        )

        self.imagecutout2 = ImageCutout(
            _TEST_IO_DIR,
            lens_name="MOCK",
            data_band="F814W",
            instrument="JWST",
            full_image_file="TEST",
        )

    def test_invalid_instrument(self):
        """Test that an invalid instrument raises an error."""
        with pytest.raises(ValueError):
            _ = ImageCutout(
                _TEST_IO_DIR,
                lens_name="MOCK",
                data_band="F814W",
                instrument="INVALID",
                full_image_file="TEST",
            )

    @patch("dolphin.preprocessing.image_cutout.make_axes_locatable")
    @patch("dolphin.preprocessing.image_cutout.plt.show")
    @patch("dolphin.preprocessing.image_cutout.plt.colorbar")
    @patch("dolphin.preprocessing.image_cutout.plt.subplots")
    @patch("dolphin.preprocessing.image_cutout.fits.open")
    def test_plot_mosaic_hst(
        self,
        mock_fits_open,
        mock_subplots,
        mock_colorbar,
        mock_show,
        mock_make_axes,
    ):
        """Test :meth:`~dolphin.preprocessing.image_cutout.plot_mosaic`"""
        data = np.ones((5, 5))

        hdul = MagicMock()
        hdul.__enter__.return_value = hdul
        hdul.__exit__.return_value = None
        hdul.__getitem__.return_value.data = data
        mock_fits_open.return_value = hdul

        mock_ax = MagicMock()
        mock_fig = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_divider = MagicMock()
        mock_divider.append_axes.return_value = MagicMock()
        mock_make_axes.return_value = mock_divider

        self.imagecutout.plot_mosaic()

        mock_fits_open.assert_called_once_with(self.imagecutout.image_file_name)
        hdul.__getitem__.assert_called_once_with(0)

        mock_ax.matshow.assert_called_once()
        _ = mock_ax.matshow.call_args.args[0]
        mock_colorbar.assert_called_once()
        mock_show.assert_called_once()

    @patch("dolphin.preprocessing.image_cutout.make_axes_locatable")
    @patch("dolphin.preprocessing.image_cutout.plt.show")
    @patch("dolphin.preprocessing.image_cutout.plt.colorbar")
    @patch("dolphin.preprocessing.image_cutout.plt.subplots")
    @patch("dolphin.preprocessing.image_cutout.fits.open")
    def test_plot_mosaic_jwst(
        self,
        mock_fits_open,
        mock_subplots,
        mock_colorbar,
        mock_show,
        mock_make_axes,
    ):
        """Test :meth:`~dolphin.preprocessing.image_cutout.plot_mosaic`"""
        data = np.ones((5, 5))

        hdul = MagicMock()
        hdul.__enter__.return_value = hdul
        hdul.__exit__.return_value = None
        hdul.__getitem__.return_value.data = data
        mock_fits_open.return_value = hdul

        mock_ax = MagicMock()
        mock_fig = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        mock_divider = MagicMock()
        mock_divider.append_axes.return_value = MagicMock()
        mock_make_axes.return_value = mock_divider

        self.imagecutout2.plot_mosaic()

        hdul.__getitem__.assert_called_once_with("SCI")
        mock_show.assert_called_once()

    @patch("dolphin.preprocessing.image_cutout.preprocessing_util.build_mask")
    @patch("dolphin.preprocessing.image_cutout.preprocessing_util.get_background")
    @patch("dolphin.preprocessing.image_cutout.Cutout2D")
    @patch("dolphin.preprocessing.image_cutout.WCS")
    @patch("dolphin.preprocessing.image_cutout.plt.tight_layout")
    @patch("dolphin.preprocessing.image_cutout.plt.show")
    @patch("dolphin.preprocessing.image_cutout.plt.subplots")
    @patch("dolphin.preprocessing.image_cutout.fits.open")
    def test_make_image_cutout_hst(
        self,
        mock_fits_open,
        mock_subplots,
        mock_show,
        mock_tight_layout,
        mock_wcs,
        mock_cutout,
        mock_background,
        mock_build_mask,
    ):
        """Test :meth:`~dolphin.preprocessing.image_cutout.make_image_cutout`"""
        image = np.ones((20, 20))
        header = {
            "RA_TARG": 10.0,
            "DEC_TARG": 20.0,
            "EXPTIME": 1200.0,
        }

        hdul = MagicMock()
        hdul.__enter__.return_value = hdul
        hdul.__exit__.return_value = None
        hdul.__getitem__.return_value.data = image
        hdul.__getitem__.return_value.header = header
        mock_fits_open.return_value = hdul

        mock_background.return_value = (1.0, 0.5)
        mock_cutout.return_value.data = np.ones((10, 10))
        mock_wcs.return_value.pixel_scale_matrix = np.array(
            [[1.0e-5, 0.0], [0.0, 1.0e-5]]
        )

        mock_build_mask.return_value = np.ones((10, 10))

        mock_ax = MagicMock()
        mock_fig = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        self.imagecutout.file_system.save_cutout_image = MagicMock()

        self.imagecutout.make_image_cutout(save=True)

        hdul.__getitem__.assert_called_with(0)

        self.imagecutout.file_system.save_cutout_image.assert_called_once()

        kwargs = self.imagecutout.file_system.save_cutout_image.call_args.args[2]

        assert "image_data" in kwargs
        assert "background_rms" in kwargs
        assert kwargs["background_rms"] == 0.5
        assert kwargs["exposure_time"] == 1200.0
        assert "noise_map" not in kwargs

        mock_show.assert_called_once()

    @patch("dolphin.preprocessing.image_cutout.preprocessing_util.build_mask")
    @patch("dolphin.preprocessing.image_cutout.preprocessing_util.get_background")
    @patch("dolphin.preprocessing.image_cutout.Cutout2D")
    @patch("dolphin.preprocessing.image_cutout.WCS")
    @patch("dolphin.preprocessing.image_cutout.plt.tight_layout")
    @patch("dolphin.preprocessing.image_cutout.plt.show")
    @patch("dolphin.preprocessing.image_cutout.plt.subplots")
    @patch("dolphin.preprocessing.image_cutout.fits.open")
    def test_make_image_cutout_jwst(
        self,
        mock_fits_open,
        mock_subplots,
        mock_show,
        mock_tight_layout,
        mock_wcs,
        mock_cutout,
        mock_background,
        mock_build_mask,
    ):
        """Test :meth:`~dolphin.preprocessing.image_cutout.make_image_cutout`"""
        sci = MagicMock()
        sci.data = np.ones((20, 20))
        sci.header = {
            "TARG_RA": 10.0,
            "TARG_DEC": 20.0,
            "XPOSURE": 1500.0,
        }

        err = MagicMock()
        err.data = np.ones((20, 20)) * 2.0

        hdul = MagicMock()
        hdul.__enter__.return_value = hdul
        hdul.__exit__.return_value = None

        hdul.__getitem__.side_effect = lambda key: {
            0: MagicMock(header={"TARG_RA": 10.0, "TARG_DEC": 20.0}),
            "SCI": sci,
            "ERR": err,
        }[key]

        mock_fits_open.return_value = hdul

        mock_background.return_value = (1.0, 0.25)
        mock_cutout.return_value.data = np.ones((10, 10))
        mock_wcs.return_value.pixel_scale_matrix = np.array(
            [[1.0e-5, 0.0], [0.0, 1.0e-5]]
        )
        mock_build_mask.return_value = np.ones((10, 10))

        mock_axes = [MagicMock(), MagicMock()]
        mock_fig = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_axes)

        self.imagecutout2.file_system.save_cutout_image = MagicMock()

        self.imagecutout2.make_image_cutout(
            save=True,
            use_noise_map=True,
        )

        self.imagecutout2.file_system.save_cutout_image.assert_called_once()

        kwargs = self.imagecutout2.file_system.save_cutout_image.call_args.args[2]

        assert "image_data" in kwargs
        assert "noise_map" in kwargs
        assert "background_rms" not in kwargs
        assert kwargs["exposure_time"] == 1500.0

        mock_show.assert_called_once()

    @patch("dolphin.preprocessing.image_cutout.display")
    @patch("dolphin.preprocessing.image_cutout.widgets.Output")
    @patch("dolphin.preprocessing.image_cutout.PixelGrid")
    @patch("dolphin.preprocessing.image_cutout.plt.subplots")
    def test_get_angular_coordinates(
        self,
        mock_subplots,
        mock_pixel_grid,
        mock_output,
        mock_display,
    ):
        """Test :meth:`~dolphin.preprocessing.image_cutout.get_angular_coordinates`"""

        lens_name = "MOCK"
        data_band = "F814W"

        data_directory = Path(self.imagecutout.file_system.get_data_directory())
        (data_directory / lens_name).mkdir(exist_ok=True)

        image_file = Path(
            self.imagecutout.file_system.get_image_file_path(lens_name, data_band)
        )

        with h5py.File(image_file, "w") as f:
            f.create_dataset("image_data", data=np.ones((25, 25)))
            f.create_dataset("ra_at_xy_0", data=-1.23)
            f.create_dataset("dec_at_xy_0", data=4.56)
            f.create_dataset(
                "transform_pix2angle",
                data=np.array([[0.04, 0.0], [0.0, 0.04]]),
            )

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        pixel_grid = MagicMock()
        pixel_grid.map_pix2coord.return_value = (0.0, 0.0)
        mock_pixel_grid.return_value = pixel_grid

        fig = self.imagecutout.get_angular_coordinates()

        mock_pixel_grid.assert_called_once()
        kwargs = mock_pixel_grid.call_args.kwargs
        assert kwargs["nx"] == 25
        assert kwargs["ny"] == 25
        assert kwargs["ra_at_xy_0"] == -1.23
        assert kwargs["dec_at_xy_0"] == 4.56
        npt.assert_array_equal(
            kwargs["transform_pix2angle"],
            np.array([[0.04, 0.0], [0.0, 0.04]]),
        )

        mock_ax.imshow.assert_called_once()
        plotted_image = mock_ax.imshow.call_args.args[0]
        npt.assert_allclose(plotted_image, np.log10(np.ones((25, 25))))
        mock_display.assert_called_once()
        mock_fig.canvas.mpl_connect.assert_called_once()

        event_name, callback = mock_fig.canvas.mpl_connect.call_args.args
        assert event_name == "button_press_event"
        assert callable(callback)

        assert fig == mock_fig

    def test_save_image_cutout(self):
        """Test :meth:`~dolphin.processor.files.FileSystem.save_cutout_image`."""

        lens_name = "lens_system3"
        data_band = "F390W"
        image_cutout_temp = ImageCutout(
            _TEST_IO_DIR,
            lens_name,
            data_band,
            "HST",
            full_image_file="TEST",
            weight_image_file="TEST",
        )

        data_directory = Path(image_cutout_temp.file_system.get_data_directory())
        image_data = np.ones((21, 21))
        ra_at_xy_0 = 10.0
        dec_at_xy_0 = 20.0
        transform_pix2angle = np.array([[0.04, 0.0], [0.0, 0.04]])
        exposure_time = 1200.0
        background_rms = 0.5

        image_cutout_temp.file_system.save_cutout_image(
            lens_name=lens_name,
            data_band=data_band,
            kwargs_data={
                "image_data": image_data,
                "ra_at_xy_0": ra_at_xy_0,
                "dec_at_xy_0": dec_at_xy_0,
                "transform_pix2angle": transform_pix2angle,
                "exposure_time": exposure_time,
                "background_rms": background_rms,
            },
        )

        filename = data_directory / lens_name / f"image_{lens_name}_{data_band}.h5"

        # check file exists
        assert filename.exists()

        # check HDF5 contents
        with h5py.File(filename, "r") as f:
            assert "image_data" in f
            assert "ra_at_xy_0" in f
            assert "dec_at_xy_0" in f
            assert "transform_pix2angle" in f
            assert "exposure_time" in f
            assert "background_rms" in f

            np.testing.assert_array_equal(
                f["image_data"][:],
                image_data,
            )

            assert f["ra_at_xy_0"][()] == ra_at_xy_0
            assert f["dec_at_xy_0"][()] == dec_at_xy_0

            np.testing.assert_array_equal(
                f["transform_pix2angle"][:],
                transform_pix2angle,
            )

            assert f["exposure_time"][()] == exposure_time
            assert f["background_rms"][()] == background_rms

        # check noise map saving
        noise_map = np.ones((21, 21)) * 0.1
        image_cutout_temp.file_system.save_cutout_image(
            lens_name=lens_name,
            data_band=data_band,
            kwargs_data={
                "image_data": image_data,
                "ra_at_xy_0": ra_at_xy_0,
                "dec_at_xy_0": dec_at_xy_0,
                "transform_pix2angle": transform_pix2angle,
                "exposure_time": exposure_time,
                "noise_map": noise_map,
            },
        )

        # check HDF5 contents for noise map
        with h5py.File(filename, "r") as f:
            assert "noise_map" in f
            np.testing.assert_array_equal(
                f["noise_map"][:],
                noise_map,
            )
