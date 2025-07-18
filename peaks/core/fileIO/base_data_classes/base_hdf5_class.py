import h5py
import numpy as np
import pint
from termcolor import colored

from peaks.core.fileIO.base_data_classes.base_data_class import BaseDataLoader, ureg


class BaseHDF5DataLoader:
    """Helper class for automating parsing data and metadata from hdf5 files.

    Notes
    -----
    Subclasses should define the _hdf5_metadata_key_mappings and
    _hdf5_metadata_fixed_units class attributes.
    The extraction process supports different types of keys:
        - If a single key is given, the value is attempted to be extracted directly
        using that key.
        - If a `list` or `tuple` of keys is given, the function tries to extract values
        for each key in turn, returning
        the first value which returns a result.
        - If a callable is given, the function is passed the h5py file object. It should
        return either a key (string)
        or directly return a value (float, int etc.).
        - A constant value (:class:`pint.Quantity`, `float`, `int`) can be provided by
        giving that value directly.
        - A constant string can be given by passing a string that starts with
        "FIXED_VALUE:", e.g. "FIXED_VALUE:LH".

    Note, if a key is passed as a string, units will attempt to be parsed from the
    relevant hdf5 dataset attributes. If units are present in the file, this takes
    precedence over any fixed units defined in the `_hdf5_metadata_fixed_units` class
    attribute. If units are not present in the file, the fixed units will be used if
    supplied. If passing a callable in the metadata mapping, the recommended approach is
    to use the function to parse the relevant key to call, and then finally pass the key
    to the `_extract_hdf5_value` method to extract the value from the hdf5 file.
    In this way, consistent handling of units is ensured.
    """

    # mappings from desired metadata keys to hdf5 field addresses
    _hdf5_metadata_key_mappings = {}
    # mappings from metadata keys to fixed units,
    # otherwise these will be attempted to be determined from the hdf5 field attributes
    _hdf5_metadata_fixed_units = {}

    @classmethod
    def _make_dataarray(cls, data):
        """Ensure data is in a :class:`xarray.DataArray` format for HDF5 loaders, which
        typically already convert to :class:`xarray.DataArray` in `_load_data` method.

        Parameters
        ----------
        data : dict or xarray.DataArray
            If a dictionary, follow convention in :meth:`BaseDataLoader._load_data`.
        """
        if isinstance(data, dict):
            return BaseDataLoader._make_dataarray(data)
        return data

    @classmethod
    def _load_metadata(cls, fpath):
        """Load metadata from a Diamond hdf5 file.

        Parameters
        ----------
        fpath : str
            Path to the file to be loaded.

        Returns
        -------
        metadata : dict
            Dictionary containing the extracted metadata.

        """
        # Open the file (read only)
        with h5py.File(fpath, "r") as f:
            metadata = {}
            for (
                peaks_key,
                hdf5_key,
            ) in cls._hdf5_metadata_key_mappings.items():
                if isinstance(hdf5_key, (list, tuple)):
                    # Iterate through all keys until a value is returned
                    metadata_entry = None
                    for key in hdf5_key:
                        metadata_entry = cls._extract_hdf5_value(f, key)
                        if metadata_entry is not None:
                            break
                    metadata[peaks_key] = metadata_entry
                elif hdf5_key is not None:
                    metadata[peaks_key] = cls._extract_hdf5_value(f, hdf5_key)
                else:
                    metadata[peaks_key] = None
            return metadata

    @classmethod
    def _extract_hdf5_value(cls, f, key, return_extreme_values=True):
        """Extract a value from a hdf5 file, adding units if possible.

        Parameters
        ----------
        f : h5py.File
            The h5py file object (should be open).
        key : pint.Quantity, str, float, int, list, np.ndarray, callable
            The key to extract from the file.
        return_extreme_values : bool, optional
            Whether to return the extreme values of an array if it is an array or list.
            Defaults to True. If False, returns entire array.

        Returns
        -------
        extracted_value : Union[Quantity, str, float, int, list, np.ndarray, None]
            The extracted value. Preference given to pint.Quantity's where possible.

        """
        if isinstance(key, (float, int, list, np.ndarray, pint.Quantity)):
            return key
        elif callable(key):
            # Call the function and pass it back here.
            # This works if a key returned as a string from the function, otherwise also
            # to check for units
            return cls._extract_hdf5_value(f, key(f))
        elif isinstance(key, str):
            # Check for a direct string flag
            if key.startswith("FIXED_VALUE:"):
                return key.split("FIXED_VALUE:")[1]
            # Otherwise this should be a key. Try and extract value from the hdf5 file
            try:
                value = f[key][()]
                value = value.decode() if isinstance(value, bytes) else value
                if isinstance(value, (np.ndarray, list)):
                    if len(value) == 1:
                        while isinstance(value, (np.ndarray, list)):
                            value = value[0]
                    elif return_extreme_values:
                        value = np.array([np.min(value), np.max(value)])
                        if value[0] == value[-1]:
                            value = value[0]

                value = value.decode() if isinstance(value, bytes) else value
                units = f[key].attrs.get("units")
                units = (
                    units[0]
                    if isinstance(units, (np.ndarray, list)) and len(units) == 1
                    else units
                )
                units = (
                    units.decode() if isinstance(units, (bytes, np.bytes_)) else units
                )
                if not units:
                    # If can't parse from the file, check for a fixed units string
                    units = cls._hdf5_metadata_fixed_units.get(key)

                return value * ureg(units) if units else value
            except KeyError:
                return None
        if key is None:
            return None
        else:
            raise ValueError(
                f"Invalid key in metadata mapping key: {type(key)}. "
                f"Expected pint.Quantity, str, float, int, or callable."
            )

    @classmethod
    def _print_hdf5_structure(
        cls, name, obj, parent_group, indent_level=0, is_last=False, branch=""
    ):
        """Recursive function to print the structure of the HDF5 file with colored keys
        and default (black) data, and indentation lines.

        Parameters
        ----------
        name : str
            The name of the current object.

        obj : h5py.Group, h5py.Dataset, h5py.SoftLink, h5py.ExternalLink
            The object to explore.

        parent_group : h5py.Group
            The parent group of the current object.

        indent_level : int, optional
            The current indentation level of the object.

        is_last : bool, optional
            Whether the current object is the last child of the parent group.

        branch : str, optional
            The current branch of the object.

        """
        # Color definitions
        group_color = "cyan"
        dataset_color = "yellow"
        attribute_color = "green"
        link_color = "magenta"

        # Set the line connector based on whether the node is the last child
        connector = "└── " if is_last else "├── "

        # Build the branch line for current level
        new_branch = branch + ("    " if is_last else "│   ")

        if isinstance(obj, h5py.Group):
            # Print groups in cyan
            group_name = colored(
                f"{name.split('/')[-1]}:NX{obj.attrs.get('NX_class', 'unknown')}",
                group_color,
            )
            print(f"{branch}{connector}{group_name}")

            # Recursively explore the contents of the group
            keys = list(obj.keys())
            for i, key in enumerate(keys):
                sub_obj = obj[key]
                cls._print_hdf5_structure(
                    key,
                    sub_obj,
                    obj,
                    indent_level + 1,
                    is_last=(i == len(keys) - 1),
                    branch=new_branch,
                )

        elif isinstance(obj, h5py.Dataset):
            # Print datasets in yellow, but display data in default black if large data
            # is compacted
            data_info = (
                f"shape={obj.shape}, dtype={obj.dtype}"
                if obj.size > 10
                else f"{obj[()]}"
            )
            dataset_name = colored(f"{name.split('/')[-1]} =", dataset_color)
            print(f"{branch}{connector}{dataset_name} {data_info}")

            # Print dataset attributes in green, display attribute values in black
            for attr_name, attr_value in obj.attrs.items():
                attr_display = colored(f"@{attr_name} =", attribute_color)
                print(f"{new_branch}  {attr_display} {attr_value}")

        elif isinstance(obj, h5py.SoftLink) or isinstance(obj, h5py.ExternalLink):
            # Handle links
            link_name = colored(
                f"{name.split('/')[-1]} -> {obj.path} (Link)", link_color
            )
            print(f"{branch}{connector}{link_name}")

    @classmethod
    def hdf5_explorer(cls, file_path):
        """Function to explore the structure of an HDF5 file, printing the keys, groups,
        datasets and attributes.

        Parameters
        ----------
        file_path : str
            The path to the HDF5 file to explore.

        Examples
        --------
        Example usage is as follows::

            import peaks as pks

            # Explore the structure of an HDF5 file
            pks.hdf5_explorer('data.h5')

        Notes
        -----
        Colored output to distinguish between groups, datasets, attributes and links:
            - Cyan: For groups (NX classes like NXentry, NXdata, etc.)
            - Yellow: For datasets (the actual data arrays)
            - Green: For attributes (metadata attached to datasets or groups, typically
            prefixed with @)
            - Magenta: For soft links (references to other datasets/groups within file)
            - Black: For data values (the actual contents of datasets and attributes)

        Note, soft links may resolve automatically and so not show as links in the output
        """
        with h5py.File(file_path, "r") as f:
            # Manually explore the root level items
            keys = list(f.keys())
            for i, key in enumerate(keys):
                obj = f[key]
                cls._print_hdf5_structure(key, obj, f, is_last=(i == len(keys) - 1))
