import re
from datetime import datetime

import numpy as np
import pint_xarray

from peaks.core.fileIO.base_arpes_data_classes.base_arpes_data_class import (
    BaseARPESDataLoader,
)
from peaks.core.fileIO.loc_registry import register_loader
from peaks.core.utils.misc import analysis_warning

ureg = pint_xarray.unit_registry


@register_loader
class MBSDataLoader(BaseARPESDataLoader):
    """Generic data loader for MBS data.

    Notes
    ------------
    This class is intended to be subclassed to provide specific loaders for different locs using MBS data formats.

    Subclasses should define the `_MBS_metadata_key_mappings` class variable to map any custom metadata fields or
    fixed values to the standard peaks metadata keys. See the docstrings for `_MBS_metadata_dict_keys_to_peaks_keys`
    for more information.

    Subclasses should also define the `_MBS_metadata_units` dictionary to map any fixed units that are not determined
    as part of the metadata loader.

    """

    _loc_name = "MBS"
    _loc_description = (
        "Loader for data acquired using the MBS A1 soft acquisition software."
    )
    _loc_url = "https://www.mbscientific.se/"
    # Dictionary to overwrite or add to default metadata keys:
    _MBS_metadata_key_mappings = {}  # mappings from MBS metadata key to peaks key
    _MBS_metadata_units = {}  # standard MBS units
    _analyser_name_conventions = {
        "deflector_parallel": "DeflY",
        "deflector_perp": "DeflX",
    }

    @classmethod
    def _load_data(cls, fpath, lazy):
        """Load MBS data from different file types."""
        handlers = {
            "txt": cls._load_from_txt,
            "krx": cls._load_from_krx,
        }
        ext = fpath.split(".")[-1]
        if ext not in handlers:
            raise ValueError(
                f"File extension {ext} is not supported for loader {cls.__name__}. Should be one of "
                f"{list(handlers.keys())}"
            )

        # Get the metadata here returning with keys in MBS format - this as needed for parsing the core data.
        metadata_dict_MBS_keys = cls._load_metadata(fpath, return_in_MBS_format=True)
        # Cahce it in the metadata cache to avoid having to load it again later
        cls._metadata_cache[fpath] = metadata_dict_MBS_keys

        # Load data
        return handlers[ext](fpath, metadata_dict_MBS_keys)

    @classmethod
    def _load_from_txt(cls, fpath, metadata_dict_MBS_keys):
        """Load data from a .txt file."""
        # Give a legacy warning
        analysis_warning(
            "Loading MBS data from .txt file. This is a legacy format and only basic loading is supported. "
            "It is strongly encouraged to save data using the .krx format from the MBS software.",
            "warning",
            "Legacy format - MBS .txt file loader",
        )

        # Load file data
        file_data = np.loadtxt(
            fpath, skiprows=int(metadata_dict_MBS_keys["metadata_lines_length"])
        )
        spectrum = file_data[:, 1:]
        eV_values = file_data[:, 0]
        eV_units = "eV"
        y_scale_values = np.arange(
            float(metadata_dict_MBS_keys["ScaleMin"]),
            float(metadata_dict_MBS_keys["ScaleMax"]),
            float(metadata_dict_MBS_keys["ScaleMult"]),
        )
        y_scale_name, y_scale_units = cls._parse_axis_name_and_units(
            metadata_dict_MBS_keys["ScaleName"]
        )
        y_scale_name = (
            "theta_par" if y_scale_name.lower() in ["angle", "y angle"] else "y_scale"
        )

        return {
            "spectrum": spectrum,
            "dims": ["eV", y_scale_name],
            "coords": {"eV": eV_values, y_scale_name: y_scale_values},
            "units": {
                "eV": eV_units,
                y_scale_name: y_scale_units,
                "spectrum": "counts",
            },
        }

    @classmethod
    def _load_from_krx(cls, fpath, metadata_dict_MBS_keys):
        """Load data from krx format."""

        with open(fpath, "rb") as f:
            # Determine whether the file is 32-bit or 64-bit. The data type is little endian, so read initially as
            # 32 bit, but if either of the first 2 32-bit words are 0, then the file is 64-bit.
            dtype_identifier = np.fromfile(f, dtype="<i4", count=2)
            if 0 in dtype_identifier:  # File is 64 bit
                dtype = "<i8"  # 8-byte signed integer (little endian)
            else:  # File is 32 bit
                dtype = "<i4"  # 4-byte signed integer (little endian)

            # Set reading to the start of the file
            f.seek(0)

            # Read the pointer array size, which is the first word of the array
            pointer_array_size = np.fromfile(f, dtype=dtype, count=1)

            # Determine the number of images in the file
            num_images = int(pointer_array_size[0] / 3)

            # Sequentially read the image positions and sizes
            image_pos = []
            Y_size = []
            X_size = []
            for _i in range(num_images):
                # Extract position in array of the images in 32-bit integers
                image_pos.append(np.fromfile(f, dtype=dtype, count=1)[0])
                # Extract Y size of array (angular direction)
                Y_size.append(np.fromfile(f, dtype=dtype, count=1)[0])
                # Extract X size of array (energy direction)
                X_size.append(np.fromfile(f, dtype=dtype, count=1)[0])

            # Read more file information to identify the scan type
            scan_identifier = np.fromfile(f, dtype=dtype, count=1)[
                0
            ]  # 5 for spin, 4 for ARPES
            # Calculate the array size
            array_size = X_size[0] * Y_size[0]
            # Set file position to the first header
            f.seek((image_pos[0] + array_size + 1) * 4)

            # Extract the kinetic energy values
            KE_values = np.linspace(
                float(metadata_dict_MBS_keys["Start K.E."]),
                float(metadata_dict_MBS_keys["End K.E."]),
                X_size[0],
            )

            # Extract the theta_par values
            if scan_identifier != 5:  # ARPES scans, extract analyser MCP angular scale
                y_scale_values = np.linspace(
                    float(metadata_dict_MBS_keys["ScaleMin"]),
                    float(metadata_dict_MBS_keys["ScaleMax"]),
                    Y_size[0],
                )
            else:  # Spin ARPES scans, extract spin MCP angular scale
                y_scale_values = np.linspace(
                    float(metadata_dict_MBS_keys["S0ScaleMin"]),
                    float(metadata_dict_MBS_keys["S0ScaleMax"]),
                    Y_size[0],
                )
            y_scale_name, y_scale_units = cls._parse_axis_name_and_units(
                metadata_dict_MBS_keys["ScaleName"]
            )
            y_scale_name = (
                "theta_par"
                if y_scale_name.lower() in ["angle", "y angle"]
                else "y_scale"
            )

            # If there is a single image, load 2D spectrum
            if num_images == 1:
                # Set read position in the file to the image location
                f.seek(image_pos[0] * 4)
                # Read the image spectrum (images written as 32-bit words even in 64-bit format .krx file)
                spectrum = np.fromfile(f, dtype="<i4", count=array_size)
                # Reshape spectrum into the desired data structure
                spectrum = np.reshape(spectrum, [Y_size[0], X_size[0]])

                return {
                    "spectrum": spectrum,
                    "dims": [
                        y_scale_name,
                        "eV",
                    ],
                    "coords": {"eV": KE_values, y_scale_name: y_scale_values},
                    "units": {
                        "eV": "eV",
                        y_scale_name: y_scale_units,
                        "spectrum": "counts",
                    },
                }

            # If there are multiple images, load the spectrum as a data cube
            else:
                # Define the spectrum in the order [mapping_dim, theta_par, eV]
                spectrum = np.zeros((num_images, Y_size[0], X_size[0]))

                # Loop through the images and fill in spectrum
                for i, pos in enumerate(image_pos):
                    # Set the read position in the file to the current image location
                    f.seek(pos * 4)

                    # Read the current image data (images written as 32-bit words even in 64-bit format .krx file)
                    current_image_data = np.fromfile(f, dtype="<i4", count=array_size)

                    # Reshape the current image data into the desired data structure, and fill entries in spectrum
                    spectrum[i, :, :] = np.reshape(
                        current_image_data, [Y_size[0], X_size[0]]
                    )

                # If scan type is an ARPES map of some form, extract the deflector angular values
                if scan_identifier == 4:  # ARPES Deflector scan
                    mapping_values = np.linspace(
                        float(metadata_dict_MBS_keys["MapStartX"]),
                        float(metadata_dict_MBS_keys["MapEndX"]),
                        num_images,
                    )
                    mapping_label = "deflector_perp"
                # If scan type is a spin scan, extract the spin rotation angle values
                elif scan_identifier == 5:  # Spin scan
                    mapping_values = []
                    for i in range(num_images):
                        current_spin_rot_angle = float(
                            metadata_dict_MBS_keys["SpinComp#" + str(i)]
                            .split(",", 1)[1]
                            .split(">", 1)[0]
                        )
                        mapping_values.append(current_spin_rot_angle)
                    mapping_label = "spin_rot_angle"

                return {
                    "spectrum": spectrum,
                    "dims": [
                        mapping_label,
                        y_scale_name,
                        "eV",
                    ],
                    "coords": {
                        "eV": KE_values,
                        y_scale_name: y_scale_values,
                        mapping_label: mapping_values,
                    },
                    "units": {
                        "eV": "eV",
                        y_scale_name: y_scale_units,
                        mapping_label: "deg",
                        "spectrum": "counts",
                    },
                }

    @classmethod
    def _load_metadata(cls, fpath, return_in_MBS_format=False):
        """Load metadata from an MBS file."""
        # Check if there is a cached version (only cache in this loader with keys in MBS format)
        metadata_dict_MBS_keys = cls._metadata_cache.get(fpath)

        # If no metadata in the cache, load it
        if not metadata_dict_MBS_keys:
            handlers = {
                "txt": cls._load_MBS_metadata_txt,
                "krx": cls._load_MBS_metadata_krx,
            }
            ext = fpath.split(".")[-1]
            metadata_lines = handlers[ext](fpath)
            metadata_dict_MBS_keys = cls._MBS_metadata_to_dict_w_MBS_keys(metadata_lines)

        if return_in_MBS_format:
            return metadata_dict_MBS_keys

        # Convert the metadata to the peaks convention and return
        return cls._MBS_metadata_dict_keys_to_peaks_keys(metadata_dict_MBS_keys)

    @staticmethod
    def _load_MBS_metadata_txt(fpath):
        """Extract the lines containing metadata in an MBS format .txt file.

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
                metadata_lines.append(line.rstrip("\n"))
                # When the line starting with '[Data' is encountered, metadata has ended and scan data has begun
                if "DATA:" in line:
                    break
            # Manually add the lengths of the metadata section as needed for .txt loader
            metadata_lines.append(f"metadata_lines_length\t{len(metadata_lines)}")
        return metadata_lines

    @staticmethod
    def _load_MBS_metadata_krx(fpath):
        """Extract the lines containing metadata in an MBS format .krx file.

        Returns
        ------------
        metadata_lines : list
            Lines extracted from the file containing the metadata.

        """
        # Open the file in read mode and extract metalines
        with open(fpath, "rb") as f:
            # Determine whether the file is 32-bit or 64-bit. The data type is little endian, so read initially as
            # 32 bit, but if either of the first 2 32-bit words are 0, then the file is 64-bit.
            dtype_identifier = np.fromfile(f, dtype="<i4", count=2)
            if 0 in dtype_identifier:  # File is 64 bit
                dtype = "<i8"  # 8-byte signed integer (little endian)
            else:  # File is 32 bit
                dtype = "<i4"  # 4-byte signed integer (little endian)

            # Set reading to the start of the file
            f.seek(0)

            # Read the pointer array size, which is the first word of the array. Pointer array size is not used here,
            # but is read so that the next variable read is image position
            pointer_array_size = np.fromfile(f, dtype=dtype, count=1)  # noqa: F841

            # Extract first image position and array size to know where in the file to find metadata
            image_pos = np.fromfile(f, dtype=dtype, count=1)[0]
            Y_size = np.fromfile(f, dtype=dtype, count=1)[0]
            X_size = np.fromfile(f, dtype=dtype, count=1)[0]
            array_size = X_size * Y_size

            # Set file position to the first header
            f.seek((image_pos + array_size + 1) * 4)

            # Read the header (allowing up to 1800 bytes) containing metadata and convert into ascii format
            header = f.read(1800).decode("ascii")

            # Shorten header to required length (i.e. up to where scan data starts)
            header = header.split("\r\nDATA:")[0]

            return header.split("\r\n")

    @staticmethod
    def _parse_axis_name_and_units(string):
        match = re.match(r"([^\(]+)\(([^)]+)\)", string)
        if match:
            name, units = match.groups()
            return name.strip(), units.strip().lower()
        else:
            return string.strip(), None

    @staticmethod
    def _MBS_metadata_to_dict_w_MBS_keys(metadata_lines):
        """Convert metadata lines to a dictionary of key-value pairs with the keys being the metadata entries
        as they appear in the MBS metadata (i.e. not yet in :class:`peaks` convention).

        Returns
        ------------
        dict
            Dictionary of metadata key-value pairs with keys in MBS format.
        """

        meta_dict = {
            line.split("\t", 1)[0]: line.split("\t", 1)[1]
            for line in metadata_lines
            if "\t" in line
        }
        for k, v in meta_dict.items():
            try:
                meta_dict[k] = float(v)
            except ValueError:
                pass

        return meta_dict

    @classmethod
    def _MBS_metadata_dict_keys_to_peaks_keys(cls, metadata_dict_MBS_keys):
        """Extract metadata values in peaks conventions and assign units where appropriate.

        Parameters
        ------------
        metadata_dict_MBS_keys : dict
            Dictionary of metadata key-value pairs with keys in MBS format.

        Returns
        ------------
        metadata_dict : dict
            Dictionary of metadata key-value pairs with keys in peaks format.

        Notes
        ------------
        The extraction process is based on the mappings defined in the dictionaries `standard_keys` and
        `standard_units`. The entries of these dictionaries are updated from the class variables
        `_MBS_metadata_key_mappings` and `_MBS_metadata_units` respectively, and so subclasMBS can overwrite and
        extend these defaults by specifying the appropriate mappings in these class variables.

        The extraction process supports different types of keys:
            - If a single key is given, the value is extracted directly.
            - If a list of keys is given, the function tries to extract values for all keys and returns a list or
            array of all non-None values.
            - If a callable is given, the function calls it, passing it the metadata dictionary and returns the result.
            - A constant value can be given by defining a simple lambda function, for example `lambda x: -1`.

        """

        def _parse_KE(metadata_dict):
            ana_mode = metadata_dict.get("AcqMode")
            if ana_mode == "Fixed":
                return metadata_dict.get("Center K.E.")
            return np.asarray(
                [metadata_dict.get("Start K.E."), metadata_dict.get("End K.E.")]
            )

        # Define the mapping assuming the standard keys for MBS metadata
        standard_keys = {
            "analyser_model": lambda x: "A1",
            "analyser_slit_width": None,
            "analyser_slit_width_identifier": None,
            "analyser_eV": _parse_KE,
            "analyser_eV_type": lambda x: "kinetic",
            "analyser_step_size": "Step Size",
            "analyser_PE": lambda x: float(x.get("Pass Energy")[2:]),
            "analyser_sweeps": "No Scans",
            "analyser_dwell": "Frames Per Step",
            "analyser_lens_mode": "Lens Mode",
            "analyser_acquisition_mode": "AcqMode",
            "analyser_polar": None,
            "analyser_tilt": None,
            "analyser_azi": None,
            "analyser_deflector_parallel": "DeflY",
            "analyser_deflector_perp": "DeflX",
            "timestamp": lambda x: datetime.strptime(
                x.get("TIMESTAMP:"), "%d/%m/%Y %H:%M"
            ).strftime("%Y-%m-%d %H:%M:%S"),
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
        standard_keys.update(cls._MBS_metadata_key_mappings)
        standard_units.update(cls._MBS_metadata_units)

        # Extract metadata values and give them units where appropriate
        # - if a single key is given, extract the value directly
        # - if a list of keys is given, try and extract for all and return all non-None values
        # - if a callable is given, call the function with the metadata dictionary and return the result
        # - add units if given and extracted value is not None
        metadata_dict = {}
        units_failure = []

        def try_parse_to_float(value):
            try:
                return float(value)
            except (ValueError, TypeError):
                return value

        # Handle extracting the metadata based on the arguments in the standard_keys dictionary
        for peaks_key, MBS_key in standard_keys.items():
            value = None
            if isinstance(MBS_key, str):
                value = try_parse_to_float(metadata_dict_MBS_keys.get(MBS_key))
            elif callable(MBS_key):
                value = MBS_key(try_parse_to_float(metadata_dict_MBS_keys))
            elif isinstance(MBS_key, list):
                value = [
                    try_parse_to_float(metadata_dict_MBS_keys.get(key))
                    for key in MBS_key
                    if metadata_dict_MBS_keys.get(key) is not None
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
