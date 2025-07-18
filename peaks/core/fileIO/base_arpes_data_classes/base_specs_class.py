import os
import re
from datetime import datetime
from itertools import takewhile

import numpy as np
import pint_xarray

from peaks.core.fileIO.base_arpes_data_classes.base_arpes_data_class import (
    BaseARPESDataLoader,
)
from peaks.core.fileIO.loc_registry import register_loader
from peaks.core.utils.misc import analysis_warning

ureg = pint_xarray.unit_registry


@register_loader
class SpecsDataLoader(BaseARPESDataLoader):
    """Data loader for Specs ARPES data measured using Specs Prodigy.

    Define _scan_axis_resolution_order to define preferences for the primary dimension of
    a 3D scan where more than one user axis varies."""

    _loc_name = "Specs"
    _loc_description = "SPECS Phoibos Analysers with Prodigy control"
    _loc_url = "https://www.specs-group.com/"
    _analyser_slit_angle = None
    _scan_axis_resolution_order = []
    _SPECS_metadata_key_mappings = {}
    _SPECS_metadata_units = {}
    _analyser_include_in_metadata_warn = [
        "eV",
        "PE",
        "sweeps",
        "dwell",
        "azi",
    ]

    @classmethod
    def _load_data(cls, fpath, lazy):
        data_parsers = {
            ".xy": cls._parse_data_from_xy_file,
            ".sp2": cls._parse_data_from_sp2_file,
        }
        file_extension = os.path.splitext(fpath)[-1]
        if file_extension in data_parsers:
            return data_parsers[file_extension](fpath)
        else:
            raise ValueError(
                f"File extension {file_extension} not supported for data parsing."
            )

    @classmethod
    def _parse_data_from_xy_file(cls, fpath):
        # Open the file and load lines
        with open(fpath) as f:
            lines = f.readlines()
        # Split into metadata-type and data
        meta_lines = [line for line in lines if line.startswith("#")]
        data = np.loadtxt(lines)

        # Parse the metadata, including extracting any user-dependent axis variations
        metadata_dict = cls._parse_metalines(meta_lines)

        # Parse the energies and counts
        counts = data[:, 1]
        energies = data[:, 0]
        # Extract the non-repeating KE values
        start_KEs = np.where(energies == energies[0])
        number_of_KEs = start_KEs[0][1]
        KE_values = energies[0:number_of_KEs]
        metadata_dict["eV scale"] = (
            KE_values  # Add to metadata dict for use when caching
        )

        # Cache the metadata to use later if required
        cls._metadata_cache[fpath] = metadata_dict
        if "scan_parameters" in metadata_dict:
            parameters_dict = metadata_dict.pop("scan_parameters")
        if metadata_dict.get("Count Rate") == "Counts per Second":
            spectrum_units = "counts/s"
        else:
            spectrum_units = "counts"

        # Extract the theta_par values
        # (only need to take from the first cycle as they repeat after)
        cycle0_lines = list(
            takewhile(lambda line: not line.startswith("# Cycle: 1"), meta_lines)
        )
        theta_par_values = [
            float(line.split(": ")[1])
            for line in cycle0_lines
            if line.startswith("# NonEnergyOrdinate:")
        ]

        if parameters_dict:
            # Try and parse the names from local names to peaks names
            inverse_manipulator_name_mappings = {
                v: k for k, v in cls._manipulator_name_conventions.items()
            }
            for key in list(parameters_dict.copy().keys()):
                peaks_name = inverse_manipulator_name_mappings.get(key[0], key[0])
                parameters_dict[(peaks_name, key[1])] = parameters_dict.pop(key)

        # Reshape the data
        if len(parameters_dict) == 1:
            user_axis, values = next(iter(parameters_dict.items()))
            spectrum = counts.reshape(
                (len(values), len(theta_par_values), number_of_KEs)
            )
            return {
                "spectrum": spectrum,
                "dims": [user_axis[0], "theta_par", "eV"],
                "coords": {
                    user_axis[0]: values,
                    "theta_par": theta_par_values,
                    "eV": KE_values,
                },
                "units": {
                    "eV": "eV",
                    "theta_par": "deg",
                    user_axis[0]: user_axis[1],
                    "spectrum": spectrum_units,
                },
            }
        elif len(parameters_dict) == 0:
            spectrum = counts.reshape((len(theta_par_values), number_of_KEs))
            return {
                "spectrum": spectrum,
                "dims": ["theta_par", "eV"],
                "coords": {"theta_par": theta_par_values, "eV": KE_values},
                "units": {"eV": "eV", "theta_par": "deg", "spectrum": spectrum_units},
            }
        else:
            # If all the user axes have the same range, then we can parse this still
            # with secondary coords
            if len(set(len(v) for v in parameters_dict.values())) == 1:
                user_axes = list(parameters_dict.keys())
                user_axis_names = [axis[0] for axis in user_axes]
                user_axis_dimensions = [axis[1] for axis in user_axes]
                user_axis_len = len(parameters_dict[user_axes[0]])
                spectrum = counts.reshape(
                    (
                        user_axis_len,
                        len(theta_par_values),
                        number_of_KEs,
                    )
                )
                # Check if a resolution order for these axes has been defined to
                # determine the primary dim
                primary_dim = None
                if cls._scan_axis_resolution_order:
                    for dim in cls._scan_axis_resolution_order:
                        if dim.lower() in [name.lower() for name in user_axis_names]:
                            primary_dim = dim
                            break
                if not primary_dim:
                    primary_dim = user_axis_names[0]
                primary_dim_label = user_axes[user_axis_names.index(primary_dim)]

                # Primary coords
                coords = {
                    "theta_par": theta_par_values,
                    "eV": KE_values,
                    primary_dim: parameters_dict[primary_dim_label],
                }
                # Secondary coords
                for axis in user_axes:
                    if axis != primary_dim_label:
                        coords[axis[0]] = (primary_dim, parameters_dict[axis])
                return {
                    "spectrum": spectrum,
                    "dims": [primary_dim, "theta_par", "eV"],
                    "coords": coords,
                    "units": {
                        "eV": "eV",
                        "theta_par": "deg",
                        **{
                            user_axis_names[i]: user_axis_dimensions[i]
                            for i in range(len(user_axis_names))
                        },
                        "spectrum": spectrum_units,
                    },
                }
            else:
                raise ValueError(
                    "Multiple user-varying parameters detected. Scan type not currently "
                    "supported by loader."
                )

    @classmethod
    def _parse_data_from_sp2_file(cls, fpath):
        metadata_dict_SPECS_keys = cls._load_metadata(fpath, return_in_SPECS_format=True)

        # Open the file and load lines
        rows = []
        with open(fpath, "rb") as f:
            for i, row in enumerate(f):  # noqa: B007
                if i >= metadata_dict_SPECS_keys["data_start_line"]:
                    rows.append(int(row.decode(errors="ignore").strip("\n")))

        data = np.asarray(rows).reshape(
            int(metadata_dict_SPECS_keys["SIZE_Y"].split("#")[0]),
            int(metadata_dict_SPECS_keys["SIZE_X"].split("#")[0]),
        )

        # Extract the scale values
        x0, x1, x_units = cls._extract_numbers_and_unit_from_range(
            metadata_dict_SPECS_keys["X Range"]
        )
        y0, y1, y_units = cls._extract_numbers_and_unit_from_range(
            metadata_dict_SPECS_keys["Y Range"]
        )
        return {
            "spectrum": data,
            "dims": [
                "y",
                "eV",
            ],
            "coords": {
                "eV": np.linspace(x0, x1, data.shape[1]),
                "y": np.linspace(y0, y1, data.shape[0]),
            },
            "units": {"eV": x_units, "y": y_units, "spectrum": "counts"},
        }

    @staticmethod
    def _extract_calib2D_info(metadata_dict_SPECS_keys):
        """Extract the calibration information from the metadata."""
        # ToDo: Complete this method and implement transformation logic
        # Extract the calib2d info from the metadata
        calib2d = {}
        calib2d["eRange"] = [
            float(metadata_dict_SPECS_keys["ERange"].split(" ")[0]),
            float(metadata_dict_SPECS_keys["ERange"].split(" ")[1]),
        ]
        calib2d["De1"] = float(metadata_dict_SPECS_keys["De1"].split(" ", 1)[0])
        calib2d["eShift"] = (
            metadata_dict_SPECS_keys.get("EShift").split("#")[0].strip().split(" ")
        )
        calib2d["eShift"] = [float(val) for val in calib2d["eShift"]]
        calib2d["aRange"] = [
            float(metadata_dict_SPECS_keys["ARange"].split(" ")[0]),
            float(metadata_dict_SPECS_keys["ARange"].split(" ")[1]),
        ]
        calib2d["aInner"] = float(metadata_dict_SPECS_keys["aInner"].split(" ")[0])

    @staticmethod
    def _extract_numbers_and_unit_from_range(s):
        pattern = r"\[([-+]?\d*\.\d+|\d+) \.\. ([-+]?\d*\.\d+|\d+)\] (\w+)"
        match = re.search(pattern, s)
        if match:
            num1 = float(match.group(1))
            num2 = float(match.group(2))
            unit = match.group(3)
            return num1, num2, unit
        else:
            raise ValueError("String format is incorrect")

    @classmethod
    def _load_metadata(cls, fpath, return_in_SPECS_format=False):
        """Load metadata from an Prodigy file."""
        # Check if there is a cached version (cache with keys in SPECS format plus a
        # "scan_parameters" key with custom axis metadata)
        metadata_dict_SPECS_keys = cls._metadata_cache.get(fpath)

        # If no metadata in the cache, load it
        if not metadata_dict_SPECS_keys:
            metadata_parsers = {
                ".xy": cls._parse_metadata_from_xy_file,
                ".sp2": cls._parse_metadata_from_sp2_file,
            }
            file_extension = os.path.splitext(fpath)[-1]
            if file_extension in metadata_parsers:
                metadata_dict_SPECS_keys = metadata_parsers[file_extension](fpath)
            else:
                raise ValueError(
                    f"File extension {file_extension} not supported for metadata parsing"
                )

        if return_in_SPECS_format:
            return metadata_dict_SPECS_keys

        # Convert the metadata to the peaks convention and return
        return cls._SPECS_metadata_dict_keys_to_peaks_keys(metadata_dict_SPECS_keys)

    @classmethod
    def _parse_metadata_from_xy_file(cls, fpath):
        with open(fpath) as f:
            lines = f.readlines()
        metadata_dict_SPECS_keys = cls._parse_metalines(
            [line for line in lines if line.startswith("#")]
        )
        # Get energy axis info to add to the file
        energy_lines = list(takewhile(lambda line: line.startswith("#"), lines))
        eV = [line.split(" ") for line in energy_lines if not line.startswith("#")]
        metadata_dict_SPECS_keys["eV scale"] = eV[0][0]

        return metadata_dict_SPECS_keys

    @classmethod
    def _parse_metadata_from_sp2_file(cls, fpath):
        meta_dict_SPECS_keys = {}
        stop_on_next_line = False  # noqa: F841
        with open(fpath, "rb") as f:
            for i, row in enumerate(f):  # noqa: B007
                row_ = row.decode(errors="ignore")
                if isinstance(row_, str) and "=" in row_:
                    meta = row_.split("=", 1)
                    meta_dict_SPECS_keys[meta[0].split("#", 1)[1].strip()] = meta[
                        1
                    ].strip()
                    if meta[0].strip() == "SIZE_X":
                        x_pixels = int(meta[1].strip())  # noqa: F841
                if (
                    meta_dict_SPECS_keys.get("SIZE_X") is not None
                    and meta_dict_SPECS_keys.get("SIZE_X").split("#")[0].strip()
                    == row_.split(" ")[0].strip()
                ):
                    break
            meta_dict_SPECS_keys["data_start_line"] = i + 1
        return meta_dict_SPECS_keys

    @classmethod
    def _parse_metalines(self, meta_lines):
        # Iterate over the first part of the metadata file up to where the data starts
        initial_metadata_lines = list(
            takewhile(
                lambda line: not line.startswith("# NonEnergyOrdinate"), meta_lines
            )
        )
        meta_dict = {
            line.split(":", 1)[0].split("#")[1].strip(): line.split(":", 1)[1].strip()
            for line in initial_metadata_lines
            if len(line.split(":")) > 1
        }

        # Extract any user-varying parameter (e.g. for a FS map)
        parameter_regex = re.compile(
            r'# Parameter: "([^"]+)\s*\[(\w+)\]" = (-?\d+\.?\d*)'
        )  # Compile regex pattern
        parameters_dict = {}
        for line in meta_lines:
            if match := parameter_regex.match(line):
                name, dim, value = (
                    match.group(1).strip(),
                    match.group(2).strip(),
                    float(match.group(3)),
                )
                key = (name, dim)
                if key not in parameters_dict:
                    parameters_dict[key] = []
                parameters_dict[key].append(value)

        meta_dict["scan_parameters"] = parameters_dict

        return meta_dict

    @classmethod
    def _SPECS_metadata_dict_keys_to_peaks_keys(cls, metadata_dict_SPECS_keys):
        """Extract metadata values in peaks conventions and assign units where needed.

        Parameters
        ------------
        metadata_dict_SPECS_keys : dict
            Dictionary of metadata key-value pairs with keys in SPECS format.

        Returns
        ------------
        metadata_dict : dict
            Dictionary of metadata key-value pairs with keys in peaks format.

        Notes
        ------------
        The extraction process is based on the mappings defined in the dictionaries
        `standard_keys` and `standard_units`. The entries of these dictionaries are
        updated from the class variables `_SPECS_metadata_key_mappings` and
        `_SPECS_metadata_units` respectively, and so subclasses can overwrite and extend
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

        def _parse_timestamp(metadata_dict_SPECS_keys):
            """Parse the timestamp from the metadata."""
            # Extract the date and time from the metadata
            date_string = metadata_dict_SPECS_keys.get("Acquisition Date")
            if date_string is None:
                return None
            date_format = "%m/%d/%y %H:%M:%S %Z"
            parsed_date = datetime.strptime(date_string, date_format)
            formatted_date = parsed_date.strftime("%Y-%m-%d %H:%M:%S")
            return formatted_date

        def _parse_analyser_slit_width(metadata_dict_SPECS_keys):
            slit_info = metadata_dict_SPECS_keys.get("Analyzer Slit")
            if slit_info and isinstance(slit_info, str):
                try:
                    return try_parse_to_float(slit_info.split(":")[1].split("x")[0])
                except (IndexError, AttributeError):
                    return None

        def _parse_energy_scale(metadata_dict_SPECS_keys):
            """Parse the energy scale from the metadata."""
            # Extract the energy scale from the metadata
            energy_scale = metadata_dict_SPECS_keys.get("eV scale")
            try:
                return [min(energy_scale), max(energy_scale)]
            except (TypeError, ValueError):
                return None

        def _parse_energy_step(metadata_dict_SPECS_keys):
            """Parse the energy step from the metadata."""
            # Extract the energy step from the metadata
            energy_scale = metadata_dict_SPECS_keys.get("eV scale")
            try:
                return np.ptp(energy_scale) / (len(energy_scale) - 1)
            except (TypeError, ValueError):
                return None

        # Define the mapping assuming the standard keys for SES metadata
        standard_keys = {
            "analyser_model": "Analyzer",
            "analyser_slit_width": _parse_analyser_slit_width,
            "analyser_slit_width_identifier": "Analyzer Slit",
            "analyser_eV": lambda x: (
                try_parse_to_float(x.get("Kinetic Energy"))
                if x.get("Scan Mode") == "SnapshotFAT"
                else _parse_energy_scale(x)
            ),
            "analyser_eV_type": "Energy Axis",
            "analyser_step_size": lambda x: (
                _parse_energy_step(x) if x.get("Scan Mode") != "SnapshotFAT" else None
            ),
            "analyser_PE": "Pass Energy",
            "analyser_sweeps": "Number of Scans",
            "analyser_dwell": "Dwell Time",
            "analyser_lens_mode": "Analyzer Lens",
            "analyser_acquisition_mode": "Scan Mode",
            "analyser_polar": None,
            "analyser_tilt": None,
            "analyser_azi": None,
            "analyser_deflector_parallel": None,
            "analyser_deflector_perp": None,
            "timestamp": _parse_timestamp,
            "manipulator_polar": cls._manipulator_name_conventions.get("polar", None),
            "manipulator_tilt": cls._manipulator_name_conventions.get("tilt", None),
            "manipulator_azi": cls._manipulator_name_conventions.get("azi", None),
            "manipulator_x1": cls._manipulator_name_conventions.get("x1", None),
            "manipulator_x2": cls._manipulator_name_conventions.get("x2", None),
            "manipulator_x3": cls._manipulator_name_conventions.get("x3", None),
            "photon_hv": "Excitation Energy",
        }
        # Define standard units
        standard_units = {
            "analyser_model": None,
            "analyser_slit_width": "mm",
            "analyser_slit_width_identifier": None,
            "analyser_eV": "eV",
            "analyser_step_size": "eV",
            "analyser_PE": "eV",
            "analyser_sweeps": None,
            "analyser_dwell": "s",
            "analyser_lens_mode": None,
            "analyser_acquisition_mode": None,
            "analyser_polar": "deg",
            "analyser_tilt": "deg",
            "analyser_azi": "deg",
            "photon_hv": "eV",
        }

        # If custom key mappings or units have been supplied, update the standard keys
        standard_keys.update(cls._SPECS_metadata_key_mappings)
        standard_units.update(cls._SPECS_metadata_units)

        # Extract metadata values and give them units where appropriate
        # - if a single key is given, extract the value directly
        # - if a list of keys is given, try and extract all and return non-None values
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
        for peaks_key, local_key in standard_keys.items():
            value = None
            if isinstance(local_key, str):
                value = try_parse_to_float(metadata_dict_SPECS_keys.get(local_key))
            elif callable(local_key):
                value = local_key(try_parse_to_float(metadata_dict_SPECS_keys))
            elif isinstance(local_key, list):
                value = [
                    try_parse_to_float(metadata_dict_SPECS_keys.get(key))
                    for key in local_key
                    if metadata_dict_SPECS_keys.get(key) is not None
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
