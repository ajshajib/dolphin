# -*- coding: utf-8 -*-
"""This module provides a class for maintaining the file system and directory
architecture."""

__author__ = "ajshajib"

from pathlib import Path
import json
import numpy as np
import h5py
import gdown
from astropy.io import fits
import shutil
import glob
import re
import os
from regions import Regions
from warnings import warn


class FileSystem(object):
    """This class contains methods to handle the file system, directory paths, and
    standard IO operations."""

    def __init__(self, io_directory):
        """Initiates a FileSystem object with `io_directory` as the root base path.

        :param io_directory: path to the input/output base directory
        :type io_directory: `str`
        """
        self._root_path = Path(io_directory)
        self.root = io_directory

    @staticmethod
    def path2str(path):
        """Converts a `pathlib.Path` object into an absolute string path.

        :param path: path to a file or directory
        :type path: `pathlib.Path`
        :return: absolute string path
        :rtype: `str`
        """
        return str(path.resolve())

    def get_lens_list_file_path(self):
        """Get the file path for the `lens_list.txt` file.

        :return: path to the `lens_list.txt` file
        :rtype: `str`
        """
        return self.path2str(self._root_path / "lens_list.txt")

    def get_lens_list(self):
        """Get the list of lenses from the `lens_list.txt` file.

        Lines starting with `#` are ignored as comments.

        :return: list of lens names
        :rtype: `list` of `str`
        """
        lens_list = []

        for line in open(self.get_lens_list_file_path(), "r"):
            if not line.startswith("#"):
                lens_list.append(line.rstrip("\n").rstrip("\r"))

        return lens_list

    def get_config_file_path(self, lens_name):
        """Get the file path to the configuration YAML file for a given lens.

        :param lens_name: name of the lens
        :type lens_name: `str`
        :return: path to the configuration file
        :rtype: `str`
        """
        return self.path2str(self.get_settings_directory() / f"{lens_name}_config.yaml")

    def get_logs_directory(self):
        """Get the path to the logs directory.

        :return: path to the `logs` folder
        :rtype: `str`
        """
        logs_dir = self.path2str(self._root_path / "logs")
        return logs_dir

    def get_settings_directory(self):
        """Get the path to the settings directory as a `Path` object.

        :return: path to the `settings` folder
        :rtype: `pathlib.Path`
        """
        return self._root_path / "settings"

    def get_outputs_directory(self):
        """Get the path to the outputs directory.

        :return: path to the `outputs` folder
        :rtype: `str`
        """
        outputs_dir = self.path2str(self._root_path / "outputs")
        return outputs_dir

    def get_data_directory(self):
        """Get the path to the data directory.

        :return: path to the `data` folder
        :rtype: `str`
        """
        data_dir = self.path2str(self._root_path / "data")
        return data_dir

    def get_image_file_path(self, lens_name, band):
        """Get the file path for the HDF5 image data file for a specific lens and band.

        :param lens_name: name of the lens
        :type lens_name: `str`
        :param band: observing band name
        :type band: `str`
        :return: file path to the image data
        :rtype: `str`
        """
        return self.path2str(
            Path(self.get_data_directory())
            / f"{lens_name}"
            / f"image_{lens_name}_{band}.h5"
        )

    def get_psf_file_path(self, lens_name, band):
        """Get the file path for the HDF5 PSF data file for a specific lens and band.

        :param lens_name: name of the lens
        :type lens_name: `str`
        :param band: observing band name
        :type band: `str`
        :return: file path to the PSF data
        :rtype: `str`
        """
        return self.path2str(
            Path(self.get_data_directory())
            / f"{lens_name}"
            / f"psf_{lens_name}_{band}.h5"
        )

    def get_log_file_path(self, lens_name, model_id):
        """Get the file path for the log text file for a specific modeling run.

        :param lens_name: name of the lens
        :type lens_name: `str`
        :param model_id: identifier for the model run
        :type model_id: `str`
        :return: file path to the log file
        :rtype: `str`
        """
        return (
            self.path2str(Path(self.get_logs_directory()))
            + f"/log_{lens_name}_{model_id}.txt"
        )

    def get_output_file_path(self, lens_name, model_id, file_type="json"):
        """Get the file path for the output file of a specific modeling run.

        :param lens_name: name of the lens
        :type lens_name: `str`
        :param model_id: identifier for the model run
        :type model_id: `str`
        :param file_type: extension type of the file. Options: 'json', 'h5'.
        :type file_type: `str`
        :return: file path to the output file
        :rtype: `str`
        """
        return (
            self.path2str(Path(self.get_outputs_directory()))
            + f"/output_{lens_name}_{model_id}.{file_type}"
        )

    def save_output(self, lens_name, model_id, output, file_type="h5"):
        """Save the results output from the fitting sequence to disk.

        :param lens_name: name of the lens
        :type lens_name: `str`
        :param model_id: identifier for the model run
        :type model_id: `str`
        :param output: output dictionary containing modeling results
        :type output: `dict`
        :param file_type: type of file to save format. 'h5' or 'json'.
        :type file_type: `str`
        :return: None
        :rtype: `None`
        """
        if file_type == "h5":
            self.save_output_h5(lens_name, model_id, output)
        elif file_type == "json":
            self.save_output_json(lens_name, model_id, output)
        else:
            raise ValueError(f"File type {file_type} not recognized!")

    def save_output_json(self, lens_name, model_id, output):
        """Save the fitting sequence output as a JSON file.

        :param lens_name: name of the lens
        :type lens_name: `str`
        :param model_id: identifier for the model run
        :type model_id: `str`
        :param output: output dictionary containing modeling results
        :type output: `dict`
        :return: None
        :rtype: `None`
        """
        save_file = self.get_output_file_path(lens_name, model_id, file_type="json")
        with open(save_file, "w") as f:
            json.dump(self.encode_numpy_arrays(output), f, ensure_ascii=False, indent=4)

    def save_output_h5(self, lens_name, model_id, output):
        """Save the fitting sequence output as an HDF5 (.h5) file.

        :param lens_name: name of the lens
        :type lens_name: `str`
        :param model_id: identifier for the model run
        :type model_id: `str`
        :param output: output dictionary containing modeling results
        :type output: `dict`
        :return: None
        :rtype: `None`
        """
        save_file = self.get_output_file_path(lens_name, model_id, file_type="h5")

        with h5py.File(save_file, "w") as f:
            f.attrs["settings"] = json.dumps(
                self.encode_numpy_arrays(output["settings"]), ensure_ascii=False
            )
            f.attrs["kwargs_result"] = json.dumps(
                self.encode_numpy_arrays(output["kwargs_result"]), ensure_ascii=False
            )

            f.attrs["multi_band_list_out"] = json.dumps(
                self.encode_numpy_arrays(output["multi_band_list_out"]),
                ensure_ascii=False,
            )

            f.attrs["dolphin_version"] = output.get("dolphin_version", "unknown")
            f.attrs["lenstronomy_version"] = output.get(
                "lenstronomy_version", "unknown"
            )
            if "jaxtronomy_version" in output:
                f.attrs["jaxtronomy_version"] = output["jaxtronomy_version"]

            group = f.create_group("fit_output")
            for i, single_output in enumerate(output["fit_output"]):
                subgroup = group.create_group(f"{i}")
                subgroup.attrs["fitting_type"] = np.bytes_(single_output[0])

                if single_output[0] == "PSO":
                    subgroup.create_dataset("chi2", data=np.array(single_output[1][0]))
                    subgroup.create_dataset(
                        "position", data=np.array(single_output[1][1])
                    )
                    subgroup.create_dataset(
                        "velocity", data=np.array(single_output[1][2])
                    )
                    subgroup.create_dataset(
                        "param_list", data=np.array(single_output[2], dtype="S25")
                    )
                elif single_output[0] in ["emcee", "Nautilus"]:
                    subgroup.create_dataset(
                        "samples",
                        data=np.array(
                            single_output[1],
                        ),
                    )
                    subgroup.create_dataset(
                        "param_list", data=np.array(single_output[2], dtype="S25")
                    )
                    if single_output[0] == "emcee":
                        subgroup.create_dataset(
                            "log_likelihood",
                            data=np.array(
                                single_output[3],
                            ),
                        )
                    elif single_output[0] == "Nautilus":
                        subgroup.create_dataset(
                            "log_l", data=np.array(single_output[3])
                        )
                        subgroup.create_dataset(
                            "log_z", data=np.array(single_output[4])
                        )
                        subgroup.create_dataset(
                            "log_z_err", data=np.array(single_output[5])
                        )
                        results_group = subgroup.create_group("results_object")
                        results_group.create_dataset(
                            "points", data=np.array(single_output[6]["points"])
                        )
                        results_group.create_dataset(
                            "log_w", data=np.array(single_output[6]["log_w"])
                        )
                        results_group.create_dataset(
                            "log_l", data=np.array(single_output[6]["log_l"])
                        )
                else:
                    warn(
                        f"Fitting type {single_output[0]} not recognized for saving output!"
                        "Saved output will not be available for this fitting type."
                    )

    def load_output(self, lens_name, model_id, file_type="h5"):
        """Load output modeling results from a previously saved file.

        :param lens_name: name of the lens
        :type lens_name: `str`
        :param model_id: identifier for the model run
        :type model_id: `str`
        :param file_type: type of file. Options: 'h5' or 'json'. Default is 'h5'.
        :type file_type: `str`
        :return: the loaded output dictionary
        :rtype: `dict`
        """
        if file_type == "h5":
            output = self.load_output_h5(lens_name, model_id)
        elif file_type == "json":
            output = self.load_output_json(lens_name, model_id)
        else:
            raise ValueError(f"File type {file_type} not recognized!")

        return output

    def load_output_json(self, lens_name, model_id):
        """Load output modeling results from a JSON file.

        :param lens_name: name of the lens
        :type lens_name: `str`
        :param model_id: identifier for the model run
        :type model_id: `str`
        :return: the loaded output dictionary
        :rtype: `dict`
        """
        load_file = self.get_output_file_path(lens_name, model_id, file_type="json")

        with open(load_file, "r") as f:
            output = json.load(f)

        return self.decode_numpy_arrays(output)

    def load_output_h5(self, lens_name, model_id):
        """Load output modeling results from an HDF5 file.

        :param lens_name: name of the lens
        :type lens_name: `str`
        :param model_id: identifier for the model run
        :type model_id: `str`
        :return: the loaded output dictionary
        :rtype: `dict`
        """
        load_file = self.get_output_file_path(lens_name, model_id, file_type="h5")

        with h5py.File(load_file, "r") as f:
            settings = self.decode_numpy_arrays(json.loads(str(f.attrs["settings"])))

            kwargs_result = self.decode_numpy_arrays(
                json.loads(str(f.attrs["kwargs_result"]))
            )

            multi_band_list_out = self.decode_numpy_arrays(
                json.loads(str(f.attrs["multi_band_list_out"]))
            )

            dolphin_version = f.attrs.get("dolphin_version", "unknown")
            if isinstance(dolphin_version, bytes):
                dolphin_version = dolphin_version.decode("utf-8")

            lenstronomy_version = f.attrs.get("lenstronomy_version", "unknown")
            if isinstance(lenstronomy_version, bytes):
                lenstronomy_version = lenstronomy_version.decode("utf-8")

            jaxtronomy_version = f.attrs.get("jaxtronomy_version", None)
            if jaxtronomy_version is not None:
                if isinstance(jaxtronomy_version, bytes):
                    jaxtronomy_version = jaxtronomy_version.decode("utf-8")

            fit_output = []
            group = f["fit_output"]

            n = len(f["fit_output"].keys())
            for index in [f"{i}" for i in range(n)]:
                fitting_step = [
                    str(group[index].attrs["fitting_type"], encoding="utf-8")
                ]

                if fitting_step[0] == "PSO":
                    fitting_step.append(
                        [
                            group[index]["chi2"][:],
                            group[index]["position"][:],
                            group[index]["velocity"][:],
                        ]
                    )
                    fitting_step.append(
                        [
                            str(s, encoding="utf-8")
                            for s in group[index]["param_list"][:]
                        ]
                    )
                elif fitting_step[0] in ["emcee", "Nautilus"]:
                    fitting_step.append(group[index]["samples"][:])
                    fitting_step.append(
                        [
                            str(s, encoding="utf-8")
                            for s in group[index]["param_list"][:]
                        ]
                    )
                    if fitting_step[0] == "emcee":
                        fitting_step.append(group[index]["log_likelihood"][:])
                    elif fitting_step[0] == "Nautilus":
                        fitting_step.append(group[index]["log_l"][()])
                        fitting_step.append(group[index]["log_z"][()])
                        fitting_step.append(group[index]["log_z_err"][()])
                        results_object = {
                            "points": group[index]["results_object"]["points"][()],
                            "log_w": group[index]["results_object"]["log_w"][()],
                            "log_l": group[index]["results_object"]["log_l"][()],
                        }
                        fitting_step.append(results_object)

                fit_output.append(fitting_step)

            output = {
                "settings": settings,
                "kwargs_result": kwargs_result,
                "fit_output": fit_output,
                "multi_band_list_out": multi_band_list_out,
                "dolphin_version": dolphin_version,
                "lenstronomy_version": lenstronomy_version,
            }

            if jaxtronomy_version is not None:
                output["jaxtronomy_version"] = jaxtronomy_version

            return output

    @classmethod
    def encode_numpy_arrays(cls, obj):
        """Recursively encode a list or dictionary containing numpy arrays to allow JSON
        serialization. This function can also handle objects with a callable tolist()
        function and a 'shape' property, such as JAX arrays.

        :param obj: the object (list, dictionary, or array) to be encoded
        :type obj: `object`
        :return: the encoded object with `numpy.ndarray`s replaced by dictionaries
        :rtype: `object`
        """
        if isinstance(obj, np.ndarray):
            return {"__ndarray__": obj.tolist(), "shape": obj.shape}
        elif hasattr(obj, "tolist") and callable(obj.tolist) and hasattr(obj, "shape"):
            return {"__ndarray__": obj.tolist(), "shape": obj.shape}
        elif isinstance(obj, list):
            encoded = []
            for element in obj:
                encoded.append(cls.encode_numpy_arrays(element))
            return encoded
        elif isinstance(obj, dict):
            encoded = {}
            for key, value in obj.items():
                encoded[key] = cls.encode_numpy_arrays(value)
            return encoded
        else:
            return obj

    @classmethod
    def decode_numpy_arrays(cls, obj):
        """Recursively decode a list or dictionary, converting encoded dictionary
        representations back to numpy arrays.

        :param obj: the object containing encoded representations of arrays
        :type obj: `object`
        :return: the decoded object with true `numpy.ndarray` objects
        :rtype: `object`
        """
        if isinstance(obj, dict):
            if "__ndarray__" in obj:
                return np.asarray(obj["__ndarray__"]).reshape(obj["shape"])
            else:
                decoded = {}
                for key, value in obj.items():
                    decoded[key] = cls.decode_numpy_arrays(value)
                return decoded
        elif isinstance(obj, list):
            decoded = []
            for element in obj:
                decoded.append(cls.decode_numpy_arrays(element))
            return decoded
        else:
            return obj

    def get_semantic_segmentation_file_path(self, lens_name, band):
        """Get the file path for the semantic segmentation data for a specific lens and
        band.

        :param lens_name: name of the lens
        :type lens_name: `str`
        :param band: observing band name
        :type band: `str`
        :return: file path to the semantic segmentation numpy file
        :rtype: `str`
        """
        return self.path2str(
            Path(self.get_outputs_directory())
            / f"semantic_segmentation_{lens_name}_{band}.npy"
        )

    def load_semantic_segmentation(self, lens_name, band):
        """Load semantic segmentation data from its `.npy` file.

        :param lens_name: name of the lens
        :type lens_name: `str`
        :param band: observing band name
        :type band: `str`
        :return: the loaded semantic segmentation array
        :rtype: `numpy.ndarray`
        """
        semantic_segmentation = np.load(
            self.get_semantic_segmentation_file_path(lens_name, band)
        )

        return semantic_segmentation

    def save_semantic_segmentation(self, lens_name, band, semantic_segmentation):
        """Save a semantic segmentation array to a `.npy` file.

        :param lens_name: name of the lens
        :type lens_name: `str`
        :param band: observing band name
        :type band: `str`
        :param semantic_segmentation: the semantic segmentation mask to save
        :type semantic_segmentation: `numpy.ndarray`
        :return: None
        :rtype: `None`
        """
        save_file = self.get_semantic_segmentation_file_path(lens_name, band)
        np.save(save_file, semantic_segmentation)

    def get_mask_file_path(self, lens_name, band):
        """Get the file path for the mask data for a specific lens and band.

        :param lens_name: name of the lens
        :type lens_name: `str`
        :param band: observing band name
        :type band: `str`
        :return: file path to the mask numpy file
        :rtype: `str`
        """
        return self.path2str(
            self.get_settings_directory() / "masks" / f"mask_{lens_name}_{band}.npy"
        )

    def load_mask(self, lens_name, band):
        """Load mask data from its `.npy` file.

        :param lens_name: name of the lens
        :type lens_name: `str`
        :param band: observing band name
        :type band: `str`
        :return: the loaded mask array
        :rtype: `numpy.ndarray`
        """
        mask = np.load(self.get_mask_file_path(lens_name, band))

        return mask

    def save_mask(self, lens_name, band, mask):
        """Save a mask array to a `.npy` file.

        :param lens_name: name of the lens
        :type lens_name: `str`
        :param band: observing band name
        :type band: `str`
        :param mask: the mask array to save
        :type mask: `numpy.ndarray`
        :return: None
        :rtype: `None`
        """
        save_file = self.get_mask_file_path(lens_name, band)
        np.save(save_file, mask)

    def get_trained_nn_model_file_path(self, source_type="galaxy"):
        """Get the local file path for the trained neural network model. Downloads the
        model from Google Drive if it doesn't exist locally.

        :param source_type: the type of lens source. Expected 'galaxy' or 'quasar'. Default is 'galaxy'.
        :type source_type: `str`
        :return: absolute file path to the downloaded model `.h5` file
        :rtype: `str`
        """
        assert source_type in ["galaxy", "quasar"]
        path = self.path2str(
            self._root_path
            / "trained_nn"
            / f"lensed_{source_type}_segmentation_model.h5"
        )

        # Check if the directory exists, create if not
        if not Path(path).parent.is_dir():
            Path(path).parent.mkdir()

        # Check if the file exists
        if not Path(path).is_file():
            # Download the model using gdown
            index = ["galaxy", "quasar"].index(source_type)
            file_id = [
                "1iM4ayxdmAz_Qc-Z8-nT-uXlBmA1X3tYU",
                "1SVANCzCGCXLgKskd_6F7AZAM-UY7nowU",
            ][index]

            print("AI model not found in local storage. Downloading from the web...")
            gdown.download(
                id=file_id,
                output=path,
                quiet=False,
            )

        return path

    def get_photometry_file_path(self, lens_name, model_id):
        """Get the file path for :class:`~dolphin.analysis.photometry.Photometry`
        outputs.

        :param lens_name: name of the system to analyze
        :type lens_name: `str`
        :param model_id: model ID of the lens system being analyzed
        :type model_id: `str`
        :return: path to the :class:`~dolphin.analysis.photometry.Photometry` output HDF5 file
        :rtype: `str`
        """

        return self.path2str(
            Path(self.get_outputs_directory()) / f"photometry_{lens_name}_{model_id}.h5"
        )

    def save_photometry_to_hdf5(
        self, photometry_class, flux_chain, magnitude_chain=None, morphology_chain=None
    ):
        """Save linear inversion outputs in HDF5 format for later analysis.

        :param photometry_class: :class:`~dolphin.analysis.photometry.Photometry` class instance
        :type photometry_class: `class`
        :param flux_chain: Flux chain as computed from :meth:`~dolphin.analysis.photometry.Photometrydo_linear_inversion`
        :type flux_chain: `np.ndarray`
        :param magnitude_chain: (Optional) AB magnitude chain as computed from :meth:`~dolphin.analysis.photometry.Photometry.calculate_ab_magnitude`
        :type magnitude_chain: `np.ndarray`
        :param morphology_chain: (Optional) Morphology chain as computed from :meth:`~dolphin.analysis.photometry.Photometry.do_linear_inversion`
        :type morphology_chain: `dict`
        """

        filename = self.get_photometry_file_path(
            photometry_class.lens_name, photometry_class.model_id
        )

        with h5py.File(filename, "w") as f:
            f.attrs["lens_name"] = photometry_class.lens_name
            f.attrs["filters"] = photometry_class.band_list

            for data_band in photometry_class.band_list:
                group = f.create_group(data_band)

                for component, flux in flux_chain[data_band].items():
                    subgrp = group.create_group(component)
                    subgrp.create_dataset("flux", data=flux)

                    if magnitude_chain is not None:
                        subgrp.create_dataset(
                            "magnitude",
                            data=magnitude_chain[data_band][component],
                        )

            if photometry_class.do_morphology:
                morphology_group = f.create_group("lens_light_morphology")

                for data_band in photometry_class.band_list:
                    filter_group = morphology_group.create_group(data_band)
                    filter_group.create_dataset(
                        "phi", data=np.array(morphology_chain[data_band]["phi"])
                    )
                    filter_group.create_dataset(
                        "q", data=np.array(morphology_chain[data_band]["q"])
                    )
                    filter_group.create_dataset(
                        "r_eff", data=np.array(morphology_chain[data_band]["r_eff"])
                    )

    def load_flux_chain(self, photometry_class):
        """Load flux chain as computed by
        :meth:`~dolphin.analysis.photometry.Photometry.do_linear_inversion`.

        :param photometry_class: :class:`~dolphin.analysis.photometry.Photometry` class instance
        :type photometry_class: `class`
        :return: dictionary containing flux chains. Format: ``{filter: {"image1": array, "image2": array,
            "lens": array, ...}}``
        :rtype: `dict`
        """

        filename = self.get_photometry_file_path(
            photometry_class.lens_name, photometry_class.model_id
        )

        flux_chain = {}

        with h5py.File(filename, "r") as f:
            filters = list(f.attrs["filters"])

            for data_band in filters:
                group = f[data_band]
                flux_chain[data_band] = {}

                for component in group.keys():
                    flux_chain[data_band][component] = group[component]["flux"][:]

        return flux_chain

    def load_magnitude_chain(self, photometry_class):
        """Load magnitude chain.

        :param photometry_class: :class:`~dolphin.analysis.photometry.Photometry` class instance
        :type photometry_class: `class`
        :return: dictionary containing AB magnitude chains. Format: ``{filter: {"image1": array, "image2": array,
            "lens": array, ...}}``
        :rtype: `dict`
        """

        filename = self.get_photometry_file_path(
            photometry_class.lens_name, photometry_class.model_id
        )

        magnitude_chain = {}

        with h5py.File(filename, "r") as f:
            filters = list(f.attrs["filters"])

            for data_band in filters:
                group = f[data_band]
                magnitude_chain[data_band] = {}

                for component in group.keys():
                    magnitude_chain[data_band][component] = group[component][
                        "magnitude"
                    ][:]

        return magnitude_chain

    def load_morphology_chain(self, photometry_class):
        """Load morphology chains as computed by
        :meth:`~dolphin.analysis.photometry.Photometry.do_linear_inversion`.

        :return: dictionary containing morphological parameter chains for each
            filter. Format:
            ``{filter: {"phi": array, "q": array, "r_eff": array}}``
        :rtype: `dict`
        """

        filename = self.get_photometry_file_path(
            photometry_class.lens_name, photometry_class.model_id
        )
        with h5py.File(filename, "r") as f:
            if "lens_light_morphology" not in f:
                return None

            filters = list(f.attrs["filters"])
            morphology = f["lens_light_morphology"]
            morphology_chain = {}

            for data_band in filters:
                morphology_chain[data_band] = {
                    "phi": morphology[data_band]["phi"][:],
                    "q": morphology[data_band]["q"][:],
                    "r_eff": morphology[data_band]["r_eff"][:],
                }

        return morphology_chain

    def get_preprocessing_path(self, preprocessing_class):
        """Get the file path for preprocessing outputs.

        :param data_band: band of the data being analyzed
        :type lens_name: `str`
        :param preprocessing_class: instance of :class:`~dolphin.preprocessing.psf.PSF` class
        :type preprocessing_class: `class`

        :return: path to the :class:`~dolphin.preprocessing.psf.PSF` PSF directory
        :rtype: `str`
        """

        return self.path2str(
            Path(self.get_data_directory())
            / f"{preprocessing_class.lens_name}"
            / "preprocessing"
        )

    def save_star_cutouts(self, psf_class, star_exposures, star_weights, noise_maps):
        """Save the star cutouts, star weight maps, and star noise maps determined by
        :meth:`~dolphin.preprocessing.psf.PSF.get_psf_candidates`.

        :param psf_class: instance of :class:`~dolphin.preprocessing.psf.PSF` class
        :type psf_class: `class`
        :param star_exposures: array containing the data for each star cutout
        :type star_exposures: `np.ndarray`
        :param star_weights: array containing the weight data for each star cutout
        :type star_weights: `np.ndarray`
        :param noise_maps: array containing star cutouts weighted by the weight files and
          background noise
        :type noise_maps: `np.ndarray`
        """

        preprocessing_str = Path(self.get_preprocessing_path(psf_class))
        star_dir = preprocessing_str / psf_class.data_band / "stars"
        weight_dir = preprocessing_str / psf_class.data_band / "weights"
        noise_dir = preprocessing_str / psf_class.data_band / "noise_maps"

        # Delete the old directory if one already exists
        for directory in [star_dir, weight_dir, noise_dir]:
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True)

        for i in range(len(star_exposures)):
            fits.PrimaryHDU(star_exposures[i]).writeto(
                star_dir / f"star_{i}.fits", overwrite=True
            )

            fits.PrimaryHDU(star_weights[i]).writeto(
                weight_dir / f"weight_{i}.fits", overwrite=True
            )

            fits.PrimaryHDU(noise_maps[i]).writeto(
                noise_dir / f"noise_map_{i}.fits", overwrite=True
            )

    def save_psf_and_variance_map(self, psf_class, psf_guess, variance_map):
        """Save the PSF and variance map in the appropriate Dolphin workflow format.

        :param psf_class: instance of :class:`~dolphin.preprocessing.psf.PSF` class
        :type psf_class: `class`
        :param psf_guess: PSF guess determined by either
          :meth:`~dolphin.preprocessing.psf.PSF.make_psf_psfr` or
          :meth:`~dolphin.preprocessing.psf.PSF.make_psf_starred`
        :type psf_guess: `np.ndarray`
        :param psf_error_map: PSF error map determined by either
          :meth:`~dolphin.preprocessing.psf.PSF.make_psf_psfr` or
          :meth:`~dolphin.preprocessing.psf.PSF.make_psf_starred`
        :type psf_error_map: `np.ndarray`
        """
        data_dir = Path(self.get_data_directory()) / f"{psf_class.lens_name}"
        filename = data_dir / f"psf_{psf_class.lens_name}_{psf_class.data_band}.h5"

        with h5py.File(filename, "w") as f:
            f.create_dataset("kernel_point_source", data=psf_guess)

            f.create_dataset("psf_variance_map", data=variance_map)

    def load_catalog_table(self, psf_class):
        """Get the SExtractor catalog if already made.

        :return: fits table SExtractor catalog as determined by
            :meth:`~dolphin.preprocessing.psf.PSF.get_kwargs_cut`
        :rtype: `table`
        """

        preprocessing_str = Path(self.get_preprocessing_path(psf_class))
        catalog_path = (
            preprocessing_str
            / psf_class.data_band
            / f"{psf_class.lens_name}_{psf_class.data_band}.cat"
        )

        with fits.open(catalog_path) as hdul:
            catalog = hdul[2].copy()

        return catalog

    def load_psf_candidate_attributes(self, psf_class):
        """Reload the saved star cutouts, corresponding masks, weight maps, and noise
        maps needed by :class:`~dolphin.preprocessing.psf.PSF`.

        :return: A tuple containing the saved star cutouts, matched masks, weight maps, and saved noise maps.
        :rtype: `tuple` (`np.ndarray`, `np.ndarray`, `np.ndarray`, `np.ndarray`)
        """
        preprocessing_str = Path(self.get_preprocessing_path(psf_class))
        star_path_str = f"{preprocessing_str}/{psf_class.data_band}/stars/star_*.fits"
        star_list = sorted(
            glob.glob(star_path_str),
            key=lambda x: int(re.search(r"star_(\d+)", x).group(1)),
        )
        star_data_list = []
        for file in star_list:
            with fits.open(file) as hdul:
                data = hdul[0].data
                image_shape = data.shape
                star_data_list.append(np.array(data))
        star_data_list = np.array(star_data_list)

        mask_data_list = []
        # Automatically match masks by star number
        for file in star_list:
            star_num = int(re.search(r"star_(\d+)", file).group(1))
            mask_path_str = (
                f"{preprocessing_str}/{psf_class.data_band}/masks/mask_{star_num}.reg"
            )
            if os.path.exists(mask_path_str):
                print(f"Using mask {mask_path_str} for star {star_num}!")
                regions = Regions.read(mask_path_str)
                mask = np.zeros(image_shape, dtype=bool)
                for region in regions:
                    mask_region = region.to_mask(mode="center")
                    region_mask = mask_region.to_image(image_shape)
                    if region_mask is not None:
                        mask |= region_mask.astype(bool)
                mask = ~mask
            else:
                print(f"No mask found for star {star_num}, using default (all True).")
                mask = np.ones(image_shape, dtype=bool)
            mask_data_list.append(mask)
        mask_data_list = np.array(mask_data_list)

        weight_path_str = (
            f"{preprocessing_str}/{psf_class.data_band}/weights/weight_*.fits"
        )
        weight_list = sorted(
            glob.glob(weight_path_str),
            key=lambda x: int(re.search(r"weight_(\d+)", x).group(1)),
        )
        weight_map_list = []
        for file in weight_list:
            with fits.open(file) as hdul:
                data = hdul[0].data
                weight_map_list.append(np.array(data))
        weight_map_list = np.array(weight_map_list)

        noise_path_str = (
            f"{preprocessing_str}/{psf_class.data_band}/noise_maps/noise_map_*.fits"
        )
        noise_list = sorted(
            glob.glob(noise_path_str),
            key=lambda x: int(re.search(r"noise_map_(\d+)", x).group(1)),
        )
        noise_maps = []
        for file in noise_list:
            with fits.open(file) as hdul:
                data = hdul[0].data
                noise_maps.append(np.array(data))
        noise_maps = np.array(noise_maps)

        return star_data_list, mask_data_list, weight_map_list, noise_maps

    def load_saved_psf(self, psf_class):
        """Load the saved PSF and variance map generated by
        :class:`~dolphin.preprocessing.psf.PSF`.

        :return: a tuple containing the saved PSF and variance map
        :rtype: `tuple` (`array`, `array`)
        """

        data_dir = Path(self.get_data_directory())
        psf_file = (
            data_dir
            / psf_class.lens_name
            / f"psf_{psf_class.lens_name}_{psf_class.data_band}.h5"
        )
        variance_map = None

        with h5py.File(psf_file, "r") as file:
            psf_data = file["kernel_point_source"][()]
            if "psf_variance_map" in file:
                variance_map = file["psf_variance_map"][()]

        return psf_data, variance_map
