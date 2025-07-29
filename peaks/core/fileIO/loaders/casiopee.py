"""Functions to load data from the Casiopee beamline at Soleil."""

import itertools
import os
from os.path import isfile, join

import natsort
import numpy as np
import pint_xarray
from tqdm.notebook import tqdm

from peaks.core.fileIO.base_arpes_data_classes.base_ses_class import SESDataLoader
from peaks.core.fileIO.loc_registry import register_loader

ureg = pint_xarray.unit_registry


@register_loader
class CASIOPEEArpesLoader(SESDataLoader):
    """Data loader for Casiopee ARPES system at Soleil.

    Notes
    ------------
    Not thoroughly tested yet

    """

    _loc_name = "Soleil_Casiopee_ARPES"
    _loc_description = "ARPES branch of Casiopee beamline at Soleil"
    _loc_url = "https://www.synchrotron-soleil.fr/en/beamlines/cassiopee"
    _analyser_slit_angle = 0 * ureg("deg")

    _manipulator_name_conventions = {
        "polar": "theta",
        "tilt": "tilt",
        "azi": "phi",
        "x1": "x",
        "x2": "z",
        "x3": "y",
    }
    _SES_metadata_units = {
        f"manipulator_{dim}": ("mm" if dim in ["x1", "x2", "x3"] else "deg")
        for dim in _manipulator_name_conventions.keys()
    }

    @classmethod
    def _load_data(cls, fpath, lazy):
        """Load SOLEIL Casiopee data."""

        if os.path.splitext(fpath)[1] in [".txt", ".zip", ".ibw"]:
            # Then should be standard SES format
            return super()._load_data(fpath, lazy)

        # Otherwise, need to use custom SOLEIL Casiopee loader functions
        # Load and cache the metadata
        metadata_dict_ses_keys = cls._load_metadata(fpath, return_in_SES_format=True)
        # Cahce it in the metadata cache to avoid having to load it again later
        cls._metadata_cache[fpath] = metadata_dict_ses_keys

        # Extract the relevant filenames into two lists
        file_list_ROI = natsort.natsorted(
            [
                item
                for item in os.listdir(fpath)
                if "ROI1" in item and isfile(join(fpath, item)) and item.endswith(".txt")
            ]
        )
        file_list_i = natsort.natsorted(
            [
                item
                for item in os.listdir(fpath)
                if "_i" in item and isfile(join(fpath, item))
            ]
        )

        # Extract kinetic energy and theta_par values
        eV_values = np.fromstring(metadata_dict_ses_keys["Dimension 1 scale"], sep=" ")
        # Precision in the file isn't good enough, so do a linear scaling between end points
        eV_values = np.linspace(eV_values[0], eV_values[-1], len(eV_values))
        eV_units = cls._parse_SES_units_from_name(
            metadata_dict_ses_keys.get("Dimension 1 name", "")
        )
        theta_par_values = np.fromstring(
            metadata_dict_ses_keys["Dimension 2 scale"], sep=" "
        )
        theta_par_units = cls._parse_SES_units_from_name(
            metadata_dict_ses_keys.get("Dimension 2 name", "")
        )
        theta_par_label = "theta_par" if theta_par_units == "deg" else "y_scale"

        # Loop through images and extract potential mapping coordinates polar and hv (one of polar or hv will vary, one
        # will be static)
        polar_values = []
        hv_values = []

        def _SES_find(lines, item):
            # Loop over lines to extract the line starting with the desired keyword.
            for line in lines:
                if line.startswith(item) and "=" in line:
                    line_contents = line.split("=")[-1].strip()
                    break
                elif line.startswith(item) and ":" in line:
                    line_contents = line.split(":")[-1].strip()
                    break

            return line_contents

        for file_i in file_list_i:
            with open(fpath + "/" + file_i) as f:
                lines = f.readlines()
                polar = float(_SES_find(lines, "theta (deg)"))
                hv = float(_SES_find(lines, "hv (eV)"))
            polar_values.append(polar)
            hv_values.append(hv)

        polar_values = np.array(polar_values)
        hv_values = np.array(hv_values)

        if abs(hv_values[-1] - hv_values[0]) > 1:  # Should be an hv scan
            dims = ["hv", theta_par_label, "eV"]
            coords = {
                "hv": hv_values,
                theta_par_label: theta_par_values,
                "eV": eV_values,
            }
            units = {
                "hv": "eV",
                theta_par_label: theta_par_units,
                "eV": eV_units,
                "spectrum": "counts",
            }
            spectrum = np.zeros((len(hv_values), len(theta_par_values), len(eV_values)))
        else:
            dims = ["polar", theta_par_label, "eV"]
            coords = {
                "polar": polar_values,
                theta_par_label: theta_par_values,
                "eV": eV_values,
            }
            units = {
                "polar": "deg",
                theta_par_label: theta_par_units,
                "eV": eV_units,
                "spectrum": "counts",
            }
            spectrum = np.zeros(
                (len(polar_values), len(theta_par_values), len(eV_values))
            )

        # Define the shape of the individual 2D data slices of the 3D data to be extracted
        slice_shape = (len(theta_par_values), len(eV_values))

        # Loop through the individual 2D data slices of the 3D data and extract the spectrum
        for i, file in tqdm(
            enumerate(file_list_ROI),
            total=len(file_list_ROI),
            desc="Loading data slices",
        ):
            spectrum[i, :, :] = cls._load_CASSIOPEE_slice(fpath, file, slice_shape)

        # If scan is hv scan, add the KE_delta coord
        if "hv" in dims:
            KE_delta = hv_values - hv_values[0]
            coords["KE_delta"] = ("hv", KE_delta)

        return {"spectrum": spectrum, "coords": coords, "dims": dims, "units": units}

    @staticmethod
    def _load_CASSIOPEE_slice(folder, file, slice_shape):
        """This function loads a single 2D slice of 3D data (either a Fermi map or hv scan) that was obtained at the
        CASSIOPEE beamline at SOLEIL.

        Parameters
        ------------
        folder : str
            Path to the folder of the 3D data that a data slice will be loaded from.

        file : str
            Name of the file within the folder of the 3D data corresponding to the data slice to be loaded.

        slice_shape : tuple
            Shape of the numpy.ndarray data slice to be extracted.

        Returns
        ------------
        slice_data : numpy.ndarray
            A single 2D data slice of the 3D data stored in folder.

        Examples
        ------------
        Example usage is as follows::

            from peaks.core.fileIO.loaders.CASSIOPEE import _load_CASSIOPEE_slice

            folder = 'C:/User/Documents/Research/FS1'

            # Extract the sixth slice of data from the FS1 file obtained at the CASSIOPEE beamline at SOLEIL
            data_slice = _load_CASSIOPEE_slice(folder, 'FS1_1_ROI6_.txt')

        """

        # Open the file within the folder of the 3D data corresponding to the data slice to be loaded, loop through the
        # lines, and stop when the scan data is reached
        with open(folder + "/" + file) as f:
            lines = f.readlines()
            for counter, line in enumerate(lines):  # noqa: B007
                if line.startswith("inputA="):
                    break

            # Define numpy.ndarray to store data slice
            slice_data = np.zeros(slice_shape)

            # Extract the data slice
            for i in range(len(lines) - (counter + 2)):
                slice_data[:, i] = np.fromstring(lines[counter + 2 + i], sep="\t")[1:]

            return slice_data

    @classmethod
    def _load_metadata(cls, fpath, return_in_SES_format=False):
        """Load SOLEIL Casiopee metadata."""

        if os.path.splitext(fpath)[1] in [".txt", ".zip", ".ibw"]:
            # Then should be standard SES format
            return super()._load_metadata(fpath, return_in_SES_format)

        # Otherwise, need to use custom SOLEIL Casiopee loader functions

        # Extract the relevant filenames into two lists
        file_list_ROI = natsort.natsorted(
            [
                item
                for item in os.listdir(fpath)
                if "ROI1" in item and isfile(join(fpath, item))
            ]
        )

        # Open the first file and extract the lines containing metadata
        with open(fpath + "/" + file_list_ROI[0]) as f:
            metadata_lines = [line for line in itertools.islice(f, 0, 45)]

        metadata_dict_SES_keys = cls._SES_metadata_to_dict_w_SES_keys(metadata_lines)

        # Check if there is additional run mode information in the metadata
        try:
            run_mode_info_start_index = metadata_lines.index("[Run Mode Information]")
            run_mode_info_stop_index = (
                metadata_lines[run_mode_info_start_index:].index("")
                + run_mode_info_start_index
            )
            metadata_dict_SES_keys["Run Mode Information"] = metadata_lines[
                run_mode_info_start_index + 1 : run_mode_info_stop_index
            ]

        except ValueError:
            pass

        if return_in_SES_format:
            return metadata_dict_SES_keys

        # Convert the metadata to the peaks convention and return
        return cls._SES_metadata_dict_keys_to_peaks_keys(metadata_dict_SES_keys)
