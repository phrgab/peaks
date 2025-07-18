import re
import zipfile

import numpy as np
import pint_xarray

from peaks import BaseIBWDataLoader
from peaks.core.fileIO.base_arpes_data_classes.base_arpes_data_class import (
    BaseARPESDataLoader,
)
from peaks.core.fileIO.loc_registry import register_loader
from peaks.core.utils.misc import analysis_warning

ureg = pint_xarray.unit_registry


@register_loader
class SESDataLoader(BaseARPESDataLoader):
    """Generic data loader for SES data.

    Notes
    ------------
    This class is intended to be subclassed to provide specific loaders for different
    locs using SES data formats.

    Subclasses should define the `_SES_metadata_key_mappings` class variable to map any
    custom metadata fields or fixed values to the standard peaks metadata keys. See the
    docstrings for `_SES_metadata_dict_keys_to_peaks_keys` for more information.

    Subclasses should also define the `_SES_metadata_units` dictionary to map any fixed
    units that are not determined as part of the metadata loader.

    """

    _loc_name = "SES"
    _loc_description = "Loader for data acquired using the Scienta Omicron SES software."
    _loc_url = "https://scientaomicron.com"
    # Dictionary to overwrite or add to default metadata keys:
    _SES_metadata_key_mappings = {}  # mappings from SES metadata key to peaks key
    _SES_metadata_units = {}  # standard SES units
    _analyser_name_conventions = {
        "deflector_parallel": "ThetaX",
        "deflector_perp": "ThetaY",
    }

    @classmethod
    def _load_data(cls, fpath, lazy, **kwargs):
        """Load SES data from different file types."""
        handlers = {
            "txt": cls._load_from_txt,
            "zip": cls._load_from_zip,
            "ibw": cls._load_from_ibw,
        }
        ext = fpath.split(".")[-1]
        if ext not in handlers:
            raise ValueError(
                f"File extension {ext} is not supported for loader {cls.__name__}"
            )

        # Load data
        scan_no = kwargs.pop("scan_no", None)
        if ext == "zip" and scan_no is not None:
            return handlers[ext](fpath, scan_no=scan_no)
        else:
            return handlers[ext](fpath)

    @classmethod
    def _load_from_txt(cls, fpath):
        """Load data from a .txt file."""
        # Give a legacy warning
        analysis_warning(
            "Loading SES data from .txt file. This is a legacy format and only basic "
            "loading is supported. It is strongly encouraged to save data using the "
            ".ibw format from the SES software.",
            "warning",
            "Legacy format - SES .txt file loader",
        )

        # Get the metadata here returning with keys in SES format - this as needed for
        # parsing the core data.
        metadata_dict_ses_keys = cls._load_metadata(fpath, return_in_SES_format=True)
        # Cahce it in the metadata cache to avoid having to load it again later
        cls._metadata_cache[fpath] = metadata_dict_ses_keys

        # Load file data and core data-related metadata
        file_data = np.loadtxt(
            fpath, skiprows=int(metadata_dict_ses_keys["metadata_lines_length"])
        )
        spectrum = file_data[:, 1:]
        eV_values = file_data[:, 0]
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
        dims = ["eV", theta_par_label]
        coords = {"eV": eV_values, theta_par_label: theta_par_values}
        units = {"eV": eV_units, theta_par_label: theta_par_units, "spectrum": "counts"}

        return {
            "spectrum": spectrum,
            "dims": dims,
            "coords": coords,
            "units": units,
        }

    @classmethod
    def _load_from_zip(cls, fpath, scan_no=0):
        """Load data from standard SES .zip format.
        Adapted from the PESTO file loader by Craig Polley.

        Pass a `scan_no` to load a specific scan from the .zip file if multiple regions
        have been scanned together in SES."""
        # Open the file and load the data
        with zipfile.ZipFile(fpath) as z:
            files = z.namelist()
            file_bin = [file for file in files if ".bin" in file]
            file_ini = [file for file in files if "Spectrum_" in file and ".ini" in file]

            filename = file_bin[scan_no]
            filename_ini = file_ini[scan_no]

            # Extract coordinate information
            with z.open(filename_ini) as f:
                # Read and decode lines in file
                lines = f.readlines()
                lineText = [line.decode() for line in lines]

            # Convert relevant metadata to dictionary
            scan_metadata = cls._SES_metadata_to_dict_w_SES_keys(lineText)

            # Extract kinetic energy axis
            num_KE = int(scan_metadata["width"])
            KE_start = float(scan_metadata["widthoffset"])
            KE_step = float(scan_metadata["widthdelta"])
            KE_end = KE_start + (KE_step * (num_KE - 1))
            KE_values = np.linspace(KE_start, KE_end, num_KE)
            KE_units = cls._parse_SES_units_from_name(
                scan_metadata.get("widthlabel", "")
            )

            # Extract theta_par axis
            num_theta_par = int(scan_metadata["height"])
            theta_par_start = float(scan_metadata["heightoffset"])
            theta_par_step = float(scan_metadata["heightdelta"])
            theta_par_end = theta_par_start + (theta_par_step * (num_theta_par - 1))
            theta_par_values = np.linspace(theta_par_start, theta_par_end, num_theta_par)
            theta_par_units = cls._parse_SES_units_from_name(
                scan_metadata.get("heightlabel", "")
            )

            # Extract deflector axis
            num_defl_perp = int(scan_metadata["depth"])
            defl_perp_start = float(scan_metadata["depthoffset"])
            defl_perp_step = float(scan_metadata["depthdelta"])
            defl_perp_end = defl_perp_start + (defl_perp_step * (num_defl_perp - 1))
            defl_perp_values = np.linspace(defl_perp_start, defl_perp_end, num_defl_perp)
            defl_perp_units = cls._parse_SES_units_from_name(
                scan_metadata.get("depthlabel", "")
            )

            # Extract spectrum and reshape into a data cube to be consistent with loading
            with z.open(filename, "r") as f:
                spectrum = np.frombuffer(f.read(), dtype=np.dtype(np.float32))
                spectrum = spectrum.reshape(
                    num_KE, num_theta_par, num_defl_perp, order="F"
                )

        return {
            "spectrum": spectrum,
            "dims": ["eV", "theta_par", "deflector_perp"],
            "coords": {
                "eV": KE_values,
                "theta_par": theta_par_values,
                "deflector_perp": defl_perp_values,
            },
            "units": {
                "eV": KE_units,
                "theta_par": theta_par_units,
                "deflector_perp": defl_perp_units,
                "spectrum": "counts",
            },
        }

    @classmethod
    def _load_from_ibw(cls, fpath):
        """Load data from an Igor binary wave (ibw) file."""

        # Load the data from the ibw file using the default IBW loader
        data = BaseIBWDataLoader._load_data(fpath, lazy=False)

        # Load the metadata if needed for parsing the data
        pos_or_point_scan_dim = [
            axis
            for axis in data["dims"]
            if "position" in axis.lower() or "point" in axis.lower()
        ]
        has_binding_energy_dim = [
            axis for axis in data["dims"] if "binding" in axis.lower()
        ]
        has_hv_dim = [
            axis
            for axis in data["dims"]
            if "hv" in axis.lower() or "photon" in axis.lower()
        ]
        if has_binding_energy_dim or pos_or_point_scan_dim or has_hv_dim:
            metadata_dict_SES_keys = cls._load_metadata(fpath, return_in_SES_format=True)

            if pos_or_point_scan_dim:
                # Parse the manipulator (or other external) data for the scan

                # Extract the run mode information
                rmi = metadata_dict_SES_keys.get("Run Mode Information")
                # Remove the scan name line if present
                raw_manipulator_data = rmi[1:] if "Name" in rmi[0] else rmi

                # Parse the data into arrays
                split_data = [line.split("\x0b") for line in raw_manipulator_data]
                header = split_data[0]
                numeric_data = split_data[1:]
                numeric_array = np.array(numeric_data, dtype=float)
                manipulator_data = {
                    header[i]: numeric_array[:, i] for i in range(len(header))
                }

                # In the SES data format the Point and Position columns define the scan
                # and are duplicated in the raw axis columns
                # Check which arrays match
                matching_pairs = []
                keys = list(header)
                all_matched_keys = set()
                for i in range(len(keys)):
                    for j in range(i + 1, len(keys)):
                        key1 = keys[i]
                        key2 = keys[j]
                        if np.array_equal(
                            manipulator_data[key1], manipulator_data[key2]
                        ):
                            matching_pairs.append((key1, key2))
                            all_matched_keys.update([key1, key2])
                axis_mapping = {}
                for i in matching_pairs:
                    a, b = i
                    if "position" in a.lower() or "point" in a.lower():
                        axis_mapping[a] = b
                    else:
                        axis_mapping[b] = a

                inverse_manipulator_name_mapping = {
                    v: k
                    for k, v in cls._manipulator_name_conventions.items()
                    if v is not None
                }
                for axis in inverse_manipulator_name_mapping.keys():
                    # Check if it varies or is static
                    if abs(np.ptp(manipulator_data[axis])) < 1e-3:  # Same within noise
                        metadata_dict_SES_keys[axis] = manipulator_data[axis].mean()
                    else:
                        if axis in all_matched_keys:
                            # This is a scannable - just return min and max for some
                            # basic info in metadata
                            metadata_dict_SES_keys[axis] = np.asarray(
                                [
                                    np.min(manipulator_data[axis]),
                                    np.max(manipulator_data[axis]),
                                ]
                            )

                        else:
                            # This is likely an independent axis - return the full array
                            # to allow parsing if required
                            metadata_dict_SES_keys[axis] = manipulator_data[axis]

        # Handle some mapping of dimension names
        dim_new_name = data["dims"].copy()
        units = {}
        units["spectrum"] = "counts"
        for i, dim in enumerate(data["dims"]):
            if "kinetic" in dim.lower():
                dim_new_name[i] = "eV"
                units["eV"] = "eV"
            elif "binding" in dim.lower():
                if metadata_dict_SES_keys.get("Energy Unit") == "Kinetic":
                    if data["coords"][dim][0] > 0:  # Positive convention for BE
                        data["coords"][dim] = np.linspace(
                            float(metadata_dict_SES_keys.get("Low Energy")),
                            float(metadata_dict_SES_keys.get("High Energy")),
                            len(data["coords"][dim]),
                        )

                    else:
                        data["coords"][dim] = np.linspace(
                            float(metadata_dict_SES_keys.get("High Energy")),
                            float(metadata_dict_SES_keys.get("Low Energy")),
                            len(data["coords"][dim]),
                        )
                    dim_new_name[i] = "eV"
                    units["eV"] = "eV"
                    cls._SES_metadata_key_mappings["analyser_eV_type"] = (
                        lambda x: "Kinetic"
                    )
                else:
                    dim_new_name[i] = "eV"
                    units["eV"] = "eV"
                    cls._SES_metadata_key_mappings["analyser_eV_type"] = (
                        lambda x: "Binding"
                    )
                    analysis_warning(
                        "Data energy axis has been loaded as binding energy.",
                        title="Loading info",
                        warn_type="danger",
                    )
            elif "photon" in dim.lower() or "hv" in dim.lower():
                KE_delta = data["coords"][dim] - data["coords"][dim][0]
                units["hv"] = "eV"
                dim_new_name[i] = "hv"
                metadata_dict_SES_keys["Excitation Energy"] = [
                    data["coords"][dim][0],
                    data["coords"][dim][-1],
                ]
            elif "deg" in dim.lower():
                dim_new_name[i] = "theta_par"
                units["theta_par"] = "deg"
            elif "scale" in dim.lower():
                dim_new_name[i] = "y_scale"
                units["y_scale"] = cls._parse_SES_units_from_name(dim)
            elif "iteration" in dim.lower():
                dim_new_name[i] = "scan_no"
            elif "position" in dim.lower() or "point" in dim.lower():
                axis_name = axis_mapping[dim]
                peaks_axis_name = inverse_manipulator_name_mapping.get(
                    axis_name, axis_name
                )
                dim_new_name[i] = peaks_axis_name
                units[f"manipulator_{peaks_axis_name}"] = cls._SES_metadata_units.get(
                    peaks_axis_name
                )

        # Replace dimension names, co-ordinate labels, and units where required
        data["coords"] = {
            dim_new_name[i]: data["coords"][dim] for i, dim in enumerate(data["dims"])
        }
        if "hv" in data["coords"]:
            data["coords"]["KE_delta"] = ("hv", KE_delta)
            units["KE_delta"] = "eV"
        data["dims"] = dim_new_name
        data["units"].update(units)

        # Cache the metadata if it was loaded
        if "metadata_dict_SES_keys" in locals():
            cls._metadata_cache[fpath] = metadata_dict_SES_keys

        return data

    @classmethod
    def _load_metadata(cls, fpath, return_in_SES_format=False):
        """Load metadata from an SES file."""
        # Check for a cached version (only cache in this loader with keys in SES format)
        metadata_dict_SES_keys = cls._metadata_cache.get(fpath)

        # If no metadata in the cache, load it
        if not metadata_dict_SES_keys:
            handlers = {
                "txt": cls._load_SES_metadata_txt,
                "zip": cls._load_SES_metadata_zip,
                "ibw": cls._load_SES_metadata_ibw,
            }
            ext = fpath.split(".")[-1]
            metadata_lines = handlers[ext](fpath)
            metadata_dict_SES_keys = cls._SES_metadata_to_dict_w_SES_keys(metadata_lines)
            # Check if there is additional run mode information in the metadata
            try:
                run_mode_info_start_index = metadata_lines.index(
                    "[Run Mode Information]"
                )
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

    @staticmethod
    def _load_SES_metadata_txt(fpath):
        """Extract the lines containing metadata in an SES format .txt file.

        Returns
        ------------
        metadata_lines : list
            Lines extracted from the file containing the metadata.

        """
        # Open the file and extract the lines containing metadata
        with open(fpath) as f:
            metadata_lines = []
            while True:
                line = f.readline()
                metadata_lines.append(line)
                # When the line starting with '[Data' is encountered, metadata has ended
                # and scan data has begun
                if "[Data" in line:
                    break
            # Manually add the lengths of the metadata section as needed for .txt loader
            metadata_lines.append(f"metadata_lines_length={len(metadata_lines)}\n")
        return metadata_lines

    @staticmethod
    def _load_SES_metadata_zip(fpath):
        """Extract the lines containing metadata in an SES format .zip file.

        Returns
        ------------
        metadata_lines : list
            Lines extracted from the file containing the metadata.

        """
        # Open the file and extract the lines containing metadata
        with zipfile.ZipFile(fpath, "r") as z:
            files = z.namelist()
            file_ini = [
                file for file in files if "Spectrum_" not in file and ".ini" in file
            ]

            if not file_ini:
                raise FileNotFoundError(
                    "No .ini file found in the .zip archive. Cannot extract metadata."
                )
            filename = file_ini[0]
            with z.open(filename, "r") as f:
                lines_to_decode = f.readlines()
                metadata_lines = [line.decode() for line in lines_to_decode]
        return metadata_lines

    @staticmethod
    def _load_SES_metadata_ibw(fpath):
        """Extract the lines containing metadata in an SES format .ibw file."""
        # Load wavenote using the BaseIBWDataLoader
        wavenote = BaseIBWDataLoader._load_metadata(fpath).get("wavenote")

        return wavenote.split("\r")

    @staticmethod
    def _parse_SES_units_from_name(name):
        """Parse the units from a key in the SES metadata."""
        return (
            re.search(r"\[(.*?)\]", name).group(1)
            if re.search(r"\[(.*?)\]", name)
            else None
        )

    @staticmethod
    def _SES_metadata_to_dict_w_SES_keys(metadata_lines):
        """Convert metadata lines to a dictionary of key-value pairs with the keys being
        the metadata entries as they appear in the SES metadata
        (i.e. not yet in :class:`peaks` convention).
        """

        return {
            line.split("=" if "=" in line else ":")[0].strip(): line.split(
                "=" if "=" in line else ":"
            )[1].strip()
            for line in metadata_lines
            if ("=" in line or ":" in line)
        }

    @classmethod
    def _SES_metadata_dict_keys_to_peaks_keys(cls, metadata_dict_SES_keys):
        """Extract metadata in peaks conventions and assign units where appropriate.

        Parameters
        ------------
        metadata_dict_SES_keys : dict
            Dictionary of metadata key-value pairs with keys in SES format.

        Returns
        ------------
        metadata_dict : dict
            Dictionary of metadata key-value pairs with keys in peaks format.

        Notes
        ------------
        The extraction process is based on the mappings defined in the dictionaries
        `standard_keys` and `standard_units`. The entries of these dictionaries are
        updated from the class variables `_SES_metadata_key_mappings` and
        `_SES_metadata_units` respectively, and so subclasses can overwrite and  extend
        these defaults by specifying the appropriate mappings in these class variables.

        The extraction process supports different types of keys:
            - If a single key is given, the value is extracted directly.
            - If a list of keys is given, the function tries to extract values for all
            keys and returns a list or array of all non-None values.
            - If a callable is given, the function calls it, passing it the metadata
            dictionary and returns the result.
            - A constant value can be given by defining a simple lambda function, for
            example `lambda x: -1`.

        """

        def _parse_timestamp(metadata_dict_SES_keys):
            """Parse the timestamp from the metadata."""
            # Extract the date and time from the metadata
            date = metadata_dict_SES_keys.get("Date")
            time = metadata_dict_SES_keys.get("Time")

            # If the date and time are present, combine them and return the timestamp
            if date and time:
                return f"{date} {time}"
            return None

        # Define the mapping assuming the standard keys for SES metadata
        standard_keys = {
            "analyser_model": "Instrument",
            "analyser_slit_width": None,
            "analyser_slit_width_identifier": None,
            "analyser_eV": ["Low Energy", "High Energy"],
            "analyser_eV_type": lambda x: x.get("Energy Unit") or x.get("Energy Scale"),
            "analyser_step_size": "Energy Step",
            "analyser_PE": "Pass Energy",
            "analyser_sweeps": "Number of Sweeps",
            "analyser_dwell": "Step Time",
            "analyser_lens_mode": "Lens Mode",
            "analyser_acquisition_mode": "Acquisition Mode",
            "analyser_polar": None,
            "analyser_tilt": None,
            "analyser_azi": None,
            "analyser_deflector_parallel": ["Thetax_Low", "Thetax_High", "ThetaX"],
            "analyser_deflector_perp": ["Thetay_Low", "Thetay_High", "ThetaY"],
            "timestamp": _parse_timestamp,
            "manipulator_polar": cls._manipulator_name_conventions.get("polar", None),
            "manipulator_tilt": cls._manipulator_name_conventions.get("tilt", None),
            "manipulator_azi": cls._manipulator_name_conventions.get("azi", None),
            "manipulator_x1": cls._manipulator_name_conventions.get("x1", None),
            "manipulator_x2": cls._manipulator_name_conventions.get("x2", None),
            "manipulator_x3": cls._manipulator_name_conventions.get("x3", None),
            "photon_hv": "Excitation Energy",
            "local_location_identifier": "Location",
        }
        # Define standard units
        standard_units = {
            "analyser_model": None,
            "analyser_slit_width": None,
            "analyser_slit_width_identifier": None,
            "analyser_eV": "eV",
            "analyser_step_size": "eV",
            "analyser_PE": "eV",
            "analyser_sweeps": None,
            "analyser_dwell": "ms",
            "analyser_lens_mode": None,
            "analyser_acquisition_mode": None,
            "analyser_polar": "deg",
            "analyser_tilt": "deg",
            "analyser_azi": "deg",
            "analyser_deflector_parallel": "deg",
            "analyser_deflector_perp": "deg",
            "photon_hv": "eV",
        }

        # If custom key mappings or units have been supplied, update the standard keys
        standard_keys.update(cls._SES_metadata_key_mappings)
        standard_units.update(cls._SES_metadata_units)

        # Extract metadata values and give them units where appropriate
        # - if a single key is given, extract the value directly
        # - if a list of keys is given, try and extract and return all non-None values
        # - if a callable is given, call the function with the metadata dictionary and
        # return the result
        # - add units if given and extracted value is not None
        metadata_dict = {}
        units_failure = []

        def try_parse_to_float(value):
            try:
                return float(value)
            except (ValueError, TypeError):
                return value

        # Handle extracting metadata based on arguments in the standard_keys dictionary
        for peaks_key, ses_key in standard_keys.items():
            value = None
            if isinstance(ses_key, str):
                value = try_parse_to_float(metadata_dict_SES_keys.get(ses_key))
            elif callable(ses_key):
                value = ses_key(try_parse_to_float(metadata_dict_SES_keys))
            elif isinstance(ses_key, list):
                value = [
                    try_parse_to_float(metadata_dict_SES_keys.get(key))
                    for key in ses_key
                    if metadata_dict_SES_keys.get(key) is not None
                ]

                if len(value) == 1:
                    value = value[0]
                elif len(value) == 0:
                    value = None
                else:
                    try:
                        value = np.asarray(value)
                    except (ValueError, TypeError):
                        value = str(value)

            # Try to add units if they are given and the value is not None
            if value is not None and standard_units.get(peaks_key):
                try:
                    value = value * ureg(standard_units[peaks_key])
                except TypeError:
                    units_failure.append(peaks_key)  # Keep track of errors
            metadata_dict[peaks_key] = value

        # Warn if any units failed to be added
        if units_failure:
            analysis_warning(
                f"Failed to add units to the following metadata items: {units_failure}",
                "warning",
                "Failed to add units to metadata",
            )

        return metadata_dict
