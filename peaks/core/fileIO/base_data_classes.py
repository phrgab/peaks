from datetime import datetime
import numpy as np
import xarray as xr
import pint_xarray  # noqa: F401
import os
import inspect
from pydantic import create_model
from typing import Optional
from igor2 import binarywave
import h5py
import pint
import pint_xarray

from .loc_registry import LOC_REGISTRY, IdentifyLoc
from peaks.core.metadata.base_metadata_models import (
    BaseScanMetadataModel,
    AxisMetadataModelWithReference,
    TemperatureMetadataModel,
    PhotonMetadataModel,
)
from peaks.core.utils.misc import analysis_warning
from .loc_registry import register_loader

ureg = pint_xarray.unit_registry


class BaseDataLoader:
    """Base Class for data loaders

    Notes
    -----
    At a minimum, subclasses should implement the `_load_data` and `_load_metadata` methods. They are also expected
    to define the `_loc_name` class variable, which is a string identifier for the location at which the data was
    obtained. This is used to determine which loader to use when loading data.

    Storing of metadata is based aorund the use of Pydantic models and then stored in the attrs of the DataArray.
    The metadata models should generally be taken or derived from the models in the base_metadata_models.py file.
    The base class provides a timestamp from the last modification time of the file. If possible, this can be updated
    via a more robust method in the implemented `_load_metadata` method, returning a timestamp in the metadata
    dictionary with key `timestamp`. In genreal, subclasses should also define specific metadata parsers to map the
    metadata dictionary returned from `_load_metatdata` to the relevant Pydantic models and to apply these to the
    DataArray. Any subclass should define the _metadata_parsers class variable as a list of these methods to be called.

    See Also
    --------
    BaseDataLoader._apply_specific_metadata
    base_metadata_models.py
    """

    # Define core attributes as class variables
    _loc_name = "Base"  # Identifier for the location/loader
    _loc_description = "Base data loader class - generally not used directly"  # Description of the location
    _loc_url = None  # Link to some descriptor of the location
    _desired_dim_order = []  # List of the desired dimension order in final da
    _dtype = None  # Desired dtype for the main data
    _dorder = None  # Desired array order for the main data
    _metadata_cache = {}  # Cache for metadata
    _metadata_parsers = []  # List of metadata parsers to apply

    # Properties to access class variables
    @property
    def loc_name(self):
        """Return the location name."""
        return self._loc_name  # Identifier

    @property
    def loc_description(self):
        """Return the location description."""
        return self._loc_description

    @property
    def loc_url(self):
        """Return the location URL."""
        return self._loc_url

    @property
    def desired_dim_order(self):
        """Return the desired dimension order for the data."""
        return self._desired_dim_order

    @property
    def dtype(self):
        """Return the desired dtype for the main data."""
        return self._dtype  # dtype to force for main data

    @property
    def dorder(self):
        """Return the desired array order for the main data."""
        return self._dorder  # Order to force for main data

    @property
    def metadata_key_mappings(self):
        """Return the metadata key mappings."""
        return self._metadata_key_mappings

    @property
    def metadata_warn_if_missing(self):
        """Return the metadata key mappings."""
        return self._metadata_warn_if_missing

    @property
    def metadata_cache(self):
        """Return the metadata cache."""
        return self._metadata_cache

    # Public methods
    @classmethod
    def load(cls, fpath, lazy=None, loc=None, metadata=True, quiet=False):
        """Top-level method to load data and return a DataArray.

        Parameters
        ------------
        fpath : str, list
            Full file path of file to load.

        lazy : str, bool, optional
            Whether to load data in a lazily evaluated dask format. Set explicitly using True/False Boolean.
            Defaults to `None` where a file is only loaded in the dask format if its spectrum is above threshold
            set in `opts.FileIO.lazy_size`

        loc : str, optional
            Location identifier for where data was acquired. Defaults to `None`, where the location will be attempted
            to be automatically determined.

        metadata : bool, optional
            Whether to attempt to load metadata into the attributes of the :class:`xarray.DataArray`. Defaults to True.

        quiet : bool, optional
            Whether to suppress analysis warnings when loading data. Defaults to False.

        Returns
        ------------
        da : xarray.DataArray
            The loaded data as an xarray DataArray.

        Examples
        ------------
        Example usage is as follows::

            from peaks.core.fileIO.base_data_classes import BaseDataLoader

            fpath = 'C:/User/Documents/Research/disp1.xy'

            # Load the data
            da = BaseDataLoader.load(fpath)

        Notes
        -----
        This method will generally be run from a different class than the loader class for the specific file.
        If building a data loader by subclassing this, make sure so put any loc-specific logic in the
        _load method.
        """

        # Make sure the metadata cache for this file is empty
        cls._metadata_cache.pop(fpath, None)

        # Parse the loc
        loc = loc if loc else cls._get_loc(fpath)
        cls._check_valid_loc(loc)  # Check a valid loc
        # Trigger the loader for the correct loc
        loader_class = cls.get_loader(loc)
        return loader_class._load(fpath, lazy, metadata, quiet)

    @classmethod
    def load_metadata(cls, fpath, loc=None, return_as_dict=False, quiet=True):
        """Top-level method to load metadata and return it in a dictionary.

        Parameters
        ------------
        fpath : str
            Path to the file to be loaded.

        loc : str, optional
            The location at which the data was obtained. Defaults to `None` where the location will be attempted to be
            determined automatically.

        return_as_dict : bool, optional
            Whether to return the metadata as a simple dictionary of metadata_keys: values or as a dictionary
            of the parsed (i.e. structured) metadata models. Defaults to False.

        quiet : bool, optional
            Whether to suppress missing metadata warnings when loading data. Defaults to True.

        Returns
        ------------
        metadata_dict : dict
            The loaded metadata as a dictionary.

        Examples
        ------------
        Example usage is as follows::

            from peaks.core.fileIO.base_data_classes import BaseDataLoader

            fpath = 'C:/User/Documents/Research/disp1.xy'

            # Determine the location at which the data was obtained
            loc = BaseDataLoader.load_metadata(fpath)

        Notes
        -----
        This will generally be run from a different class than the loader class for the specific file.
        If building a data loader by subclassing this, make sure so put any loc-specific logic in the
        _load_metadata method.
        """

        # Parse the loc - if the base class is used, determine the loc automatically and route to the right subclass
        if cls._loc_name == "Base":
            loc = loc if loc else cls._get_loc(fpath)
            cls._check_valid_loc(loc)  # Check a valid loc
            if loc != "Base":
                return cls.get_loader(loc).load_metadata(fpath)
        # Otherwise, use the loc defined in the subclass
        loc = cls._loc_name

        # Parse some baseline metadata from the file
        # Extract a timestamp from last modification time - overwrite in subclass if more robust method available
        timestamp = os.path.getmtime(fpath)
        # Convert the timestamp to a human-readable format
        readable_timestamp = datetime.fromtimestamp(timestamp).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        metadata_dict = {"timestamp": readable_timestamp, "fpath": fpath}

        # Method to extract any specific metadata, which should be updated in subclasses
        metadata_dict.update(
            {
                k: v
                for k, v in cls.get_loader(loc)._load_metadata(fpath).items()
                if v is not None
            }
        )
        if return_as_dict:
            return metadata_dict
        parsed_metadata = {}
        parsed_metadata.update(cls._parse_general_metadata(metadata_dict))
        parsed_metadata.update(cls._parse_specific_metadata(metadata_dict, quiet))
        return parsed_metadata

    @staticmethod
    def get_loader(loc):
        """Get the loader class for the given location.

        Parameters
        ------------
        loc : str
            The location at which the data was obtained.

        Returns
        ------------
        loader_class : class
            The loader class for the given location.

        Raises
        ------------
        ValueError
            If no loader is found for the given location.

        Examples
        ------------
        Example usage is as follows::

                from peaks.core.fileIO.base_data_classes import BaseDataLoader

                loc = 'I05'

                # Get the loader class for the given location
                loader_class = BaseDataLoader.get_loader(loc)
        """
        loader_class = LOC_REGISTRY.get(loc)
        if loader_class:
            return loader_class
        else:
            raise ValueError(
                f"No loader found for location {loc}. Expected one of {set(LOC_REGISTRY.keys())}."
            )

    # Private methods
    @staticmethod
    def _get_loc(fpath):
        """This function determines the location at which the data was obtained.

        Parameters
        ------------
        fpath : str
            Path to the file to be loaded.

        Returns
        ------------
        loc : str
            The name of the location (typically a beamline).
        """
        file_extension = os.path.splitext(fpath)[1]
        # Define the handlers for the different file extensions
        # No extension
        handlers = {
            "": IdentifyLoc._no_extension,
        }
        # For ones with file extensions, generate the handlers from the methods in IdentifyLocation
        handlers.update(
            {
                f".{method_name.split("_handler_")[1]}": method
                for method_name, method in inspect.getmembers(
                    IdentifyLoc, predicate=inspect.isfunction
                )
                if method_name.startswith("_handler")
            }
        )
        handler = handlers.get(file_extension, IdentifyLoc._default_handler)

        return handler(fpath)

    @staticmethod
    def _check_valid_loc(loc):
        """Check if the location is valid."""
        if loc not in LOC_REGISTRY.keys():  # Check a valid loc
            raise ValueError(
                f"No loader defined for location {loc}. Specify one of {LOC_REGISTRY.keys()} or leave"
                f" loc argument empty to attempt to determine the location automatically."
            )

    @classmethod
    def _load(cls, fpath, lazy, metadata, quiet):
        """Generic method for loading the data."""
        data = cls._load_data(fpath, lazy)  # Load the actual data from the file
        da = cls._make_dataarray(data)  # Convert to DataArray
        # Add a name to the DataArray
        da.name = fpath.split("/")[-1].split(".")[0]
        if metadata:  # Load metadata if requested
            parsed_metadata = cls.load_metadata(fpath, loc=cls._loc_name, quiet=quiet)
            da.attrs.update(parsed_metadata)
            cls._metadata_cache.pop(
                fpath, None
            )  # Clear any metadata cache for this file
        # Apply any specific conventions and add a history of the load
        da = cls._apply_conventions(da)
        cls._add_load_history(da, fpath)

        return da

    @classmethod
    def _load_data(cls, fpath, lazy):
        """Load the data. To be implemented by subclasses.

        Parameters
        ------------
        fpath : str
            Path to the file to be loaded.

        lazy : bool or str
            Whether to load data in a lazily evaluated dask format. Set explicitly using True/False Boolean.

        Notes
        -----
        The data should be returned as a dictionary containing the following keys:
            spectrum : the data to be converted
            dims : a list of dims in order corresponding to the data axis order
            coords : associated coords
            units : dict containing units for the data and dimensions

        If metadata needs to be loaded
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @classmethod
    def _load_metadata(cls, fpath):
        """Load the metadata. Should return a dictionary `metadata_dict` mapping relevant metadata keys to values.
        Loaders that subclass this class should implement this method.

        Parameters
        ------------
        fpath : str
            Path to the file to be loaded.

        Returns
        ------------
        metadata_dict : dict
            Dictionary mapping metadata keys to values.
            Keys should be `peaks` notation, generally of the form subclass_item (e.g. `analyser_eV`)
            Values should be :class:`pint.Quantity` objects where possible.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @classmethod
    def _parse_general_metadata(cls, metadata_dict):
        """Apply general metadata to the DataArray - implemented irrespective of the loader."""

        fpath = metadata_dict.get("fpath")
        general_scan_metadata = BaseScanMetadataModel(
            name=fpath.split("/")[-1].split(".")[0],
            filepath=fpath,
            loc=cls._loc_name,
            timestamp=metadata_dict.get("timestamp"),
            scan_command=metadata_dict.get("scan_command"),
        )
        return {"scan": general_scan_metadata}

    @classmethod
    def _parse_specific_metadata(cls, metadata_dict, quiet):
        """Method to orchastrate applying loader-specific metadata to the DataArray.

        Parameters
        ------------
        metadata_dict : dict
            Dictionary containing all available metadata as returned by `_load_metadata`.

        quiet : bool
            Whether to suppress missing metadata warnings when loading data.

        Returns
        ------------
        parsed_metadata : dict
            Dictionary containing all parsed metadata as Pydantic models.

        Notes
        -----
        Subclasses should implement a method `_parse_xxxxxx_metadata` to map metadata from the general metadata
        dictionary to the specific pydantic metadata model and apply this the the dataarray. The class attribute
        _metadata_parsers should provide a list of all of these methods to be called.

        These methods should return a tuple of ({'key': MetadataModel()}, list) where 'key' is the key that should
        be used in the :class:xarray.DataArray attributes for holding the relevant metadata, MetadataModel is the
        model as generated from `base_metadata_models.py` and list is a list of metadata keys that should be checked
        if they exist and a warning be raised if they are missing.
        """
        metadata_warning_list = []
        parsed_metadata = {}
        for parser in cls._metadata_parsers:
            _parser = getattr(cls, parser, None)
            if _parser is None:
                raise NotImplementedError(
                    f"Method {parser} not found in subclass {cls.__name__} or its parent classes. Implement this or "
                    f"change the entries of class attribute `_metadata_parsers`."
                )
            metadata_to_apply, metadata_to_add_to_warning_list = _parser(metadata_dict)
            parsed_metadata.update(metadata_to_apply)
            metadata_warning_list.extend(metadata_to_add_to_warning_list)
        if not quiet:
            cls._warn_metadata(metadata_dict, metadata_warning_list)
        return parsed_metadata

    @classmethod
    def _warn_metadata(cls, metadata_dict, metadata_warning_list=None):
        """Warn if any of the metadata fields in missing_metadata_warning_list are missing."""
        missing_metadata_to_warn = [
            item for item in metadata_warning_list if metadata_dict.get(item) is None
        ]
        if missing_metadata_to_warn:
            analysis_warning(
                f"Unable to extract metadata for: {missing_metadata_to_warn}. If you expected this to be in the "
                f"available metadata, update the {cls.__name__} metadata loader in {cls.__module__} to account for new "
                "file format.",
                title="Loading info",
                warn_type="warning",
            )

    @classmethod
    def _make_dataarray(cls, data):
        """Convert data into an xarray DataArray.

        Parameters
        ------------
        data : dict
            Dictionary containing:
                spectrum : the data to be converted
                dims : a list of dims in order corresponding to the data axis order
                coords : associated coords
                units : dict containing units for the data and dimensions
        """
        da = xr.DataArray(
            data["spectrum"],
            dims=data.get("dims", None),
            coords=data.get("coords", None),
            name="spectrum",  # Temporarily name as spectrum to allow unit quantification
        )
        return da.pint.quantify(data.get("units"))

    @classmethod
    def _apply_conventions(cls, da):
        """Apply relevant conventions to the DataArray."""

        # Ensure that all the DataArray dimension coordinates are ordered low to high
        for dim in da.dims:
            if len(da[dim]) > 1:
                if da[dim].data[-1] - da[dim].data[0] < 0:
                    da = da.reindex({dim: da[dim][::-1]})
        # Reorder dimensions if needed
        if cls.desired_dim_order:
            da = da.transpose(*cls._desired_dim_order, ..., missing_dims="ignore")
        # Set the appropriate data type and order
        if cls.dtype:
            da = da.astype(cls._dtype, order=cls._dorder)
        return da

    @classmethod
    def _add_load_history(cls, da, fpath):
        """Add a history of the load to the DataArray."""
        da.history.add(
            {
                "record": "Data loaded",
                "loc": cls._loc_name,
                "loader": cls.__name__,
                "file_name": fpath,
            }
        )


class BaseHDF5DataLoader:
    """Helper class for automating parsing data and metadata from hdf5 files.

    Notes
    -----
    Subclasses should define the _hdf5_metadata_key_mappings and _hdf5_metadata_fixed_units class attributes.
    The extraction process supports different types of keys:
        - If a single key is given, the value is attempted to be extracted directly using that key.
        - If a `list` or `tuple` of keys is given, the function tries to extract values for each key in turn, returning
        the first value which returns a result.
        - If a callable is given, the function is passed the h5py file object. It should return either a key (string)
        or directly return a value (float, int etc.).
        - A constant value (:class:`pint.Quantity`, `float`, `int`) can be provided by giving that value directly.
        - A constant string can be given by passing a string that starts with "FIXED_VALUE:", e.g. "FIXED_VALUE:LH".

    Note, if a key is passed as a string, units will attempt to be parsed from the relevant hdf5 dataset attributes.
    If units are present in the file, this takes precedence over any fixed units defined in the
    `_hdf5_metadata_fixed_units` class attribute. If units are not present in the file, the fixed units will be used
    if supplied. If passing a callable in the metadata mapping, the recommended approach is to use the function to
    parse the relevant key to call, and then finally pass the key to the `_extract_hdf5_value` method to extract the
    value from the hdf5 file. In this way, consistent handling of units is ensured.
    """

    # mappings from desired metadata keys to hdf5 field addresses
    _hdf5_metadata_key_mappings = {}
    # mappings from metadata keys to fixed units,
    # otherwise these will be attempted to be determined from the hdf5 field attributes
    _hdf5_metadata_fixed_units = {}

    @classmethod
    def _make_dataarray(cls, data):
        """Ensure data is in a :class:`xarray.DataArray` format for HDF5 loaders, which typically already convert to
        :class:`xarray.DataArray` in the `_load_data` method.

        Parameters
        ----------
        data : dict or xarray.DataArray
            If a dictionary, should follow convention in :meth:`BaseDataLoader._load_data`.
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
            The extracted value. Preference is given to pint.Quantity objects where possible.

        """
        if isinstance(key, (float, int, list, np.ndarray, pint.Quantity)):
            return key
        elif callable(key):
            # Call the function and pass it back here.
            # This works if a key returned as a string from the function, otherwise also to check for units
            return cls._extract_hdf5_value(f, key(f))
        elif isinstance(key, str):
            # Check for a direct string flag
            if key.startswith("FIXED_VALUE:"):
                return key.split("FIXED_VALUE:")[1]
            # Otherwise this should be a key. Try and extract the value from the hdf5 file
            try:
                value = f[key][()]
                if isinstance(value, (np.ndarray, list)):
                    if len(value) == 1:
                        value = value[0]
                    elif return_extreme_values:
                        value = np.array([np.min(value), np.max(value)])
                        if np.ptp(value) == 0:
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


@register_loader
class BaseIBWDataLoader(BaseDataLoader):
    """Base class for data loaders for Igor Binary Wave files."""

    _loc_name = "ibw"
    _loc_description = "General loader for Igor Binary Wave files"
    _loc_url = "https://www.wavemetrics.com/products"
    _metadata_parsers = ["_parse_wavenote_metadata"]

    @classmethod
    def _load_data(cls, fpath, lazy):
        """Load the data from an Igor Binary Wave file."""

        # Open the file and load its contents
        file_contents = binarywave.load(fpath)

        # Extract spectrum
        spectrum = file_contents["wave"]["wData"]

        # Extract relevant information on the dimensions of the data
        dim_size = file_contents["wave"]["bin_header"]["dimEUnitsSize"]
        dim_units = file_contents["wave"]["dimension_units"].decode()

        # Extract scales of dimensions
        dim_start = file_contents["wave"]["wave_header"]["sfB"]  # Initial value
        dim_step = file_contents["wave"]["wave_header"]["sfA"]  # Step size
        dim_points = file_contents["wave"]["wave_header"]["nDim"]  # Number of points
        dim_end = dim_start + (dim_step * (dim_points - 1))

        # Loop through the dimensions and extract the relevant dimension names and coordinates
        dims = []
        coords = {}
        counter = 0
        for i in range(spectrum.ndim):
            dim = dim_units[counter : counter + dim_size[i]]
            dims.append(dim)
            coords[dim] = np.linspace(
                dim_start[i], dim_end[i], dim_points[i], endpoint=False
            )
            counter += dim_size[i]

        return {"spectrum": spectrum, "dims": dims, "coords": coords, "units": {}}

    @classmethod
    def _load_metadata(cls, fpath):
        """Load metadata from an Igor Binary Wave file."""
        # Load just the wavenote
        # Define maximum number of dimensions
        max_dims = 4

        # Read the ibw bin header segment of the file (IBW version 2,5 only)
        with open(fpath, "rb") as f:
            # Determine file version and extract file information
            version = np.fromfile(f, dtype=np.dtype("int16"), count=1)[0]
            if version == 2:
                # The size of the WaveHeader2 data structure plus the wave data plus 16 bytes of padding.
                wfmSize = np.fromfile(f, dtype=np.dtype("uint32"), count=1)[0]
                # The size of the note text.
                noteSize = np.fromfile(f, dtype=np.dtype("uint32"), count=1)[0]
                # Reserved. Write zero. Ignore on read.
                pictSize = np.fromfile(f, dtype=np.dtype("uint32"), count=1)[0]
                # Checksum over this header and the wave header.
                checksum = np.fromfile(f, dtype=np.dtype("int16"), count=1)[0]
            elif version == 5:
                # Checksum over this header and the wave header.
                checksum = np.fromfile(f, dtype=np.dtype("short"), count=1)[0]
                # The size of the WaveHeader5 data structure plus the wave data.
                wfmSize = np.fromfile(f, dtype=np.dtype("int32"), count=1)[0]
                # The size of the dependency formula, if any.
                formulaSize = np.fromfile(f, dtype=np.dtype("int32"), count=1)[0]
                # The size of the note text.
                noteSize = np.fromfile(f, dtype=np.dtype("int32"), count=1)[0]
                # The size of optional extended data units.
                dataEUnitsSize = np.fromfile(f, dtype=np.dtype("int32"), count=1)[0]
                # The size of optional extended dimension units.
                dimEUnitsSize = np.fromfile(f, dtype=np.dtype("int32"), count=max_dims)
                # The size of optional dimension labels.
                dimLabelsSize = np.fromfile(f, dtype=np.dtype("int32"), count=4)
                # The size of string indices if this is a text wave.
                sIndicesSize = np.fromfile(f, dtype=np.dtype("int32"), count=1)[0]
                # Reserved. Write zero. Ignore on read.
                optionsSize1 = np.fromfile(f, dtype=np.dtype("int32"), count=1)[0]
                # Reserved. Write zero. Ignore on read.
                optionsSize2 = np.fromfile(f, dtype=np.dtype("int32"), count=1)[0]

        # Open the file and read the wavenote
        with open(fpath, "rb") as f:
            # Move the cursor to the end of the file
            f.seek(0, os.SEEK_END)
            # Get the current position of pointer
            pointer_location = f.tell()

            # Determine file version-dependent offset
            if version == 2:
                offset = noteSize
            elif version == 5:
                # Work out file location of wavenote
                offset = (
                    noteSize
                    + dataEUnitsSize.sum()
                    + dimEUnitsSize.sum()
                    + dimLabelsSize.sum()
                    + sIndicesSize.sum()
                    + optionsSize1.sum()
                    + optionsSize2.sum()
                )

            # Move the file pointer to the location pointed by pointer_location, considering the offset
            f.seek(pointer_location - offset)
            # read that bytes/characters to determine the wavenote
            wavenote = f.read(offset).decode()
            # NB No not cache the wavenote, as other loaders rely on this method and that screws things up

        return {"wavenote": wavenote}

    @classmethod
    def _parse_wavenote_metadata(cls, metadata_dict):
        """Parse metadata specific to the wavenote."""
        return {"wavenote": metadata_dict.get("wavenote")}, []


class BaseManipulatorDataLoader(BaseDataLoader):
    """Base class for data loaders for systems with manipulators.

    Notes
    -----
    THIS NEEDS UPDATING...
    Subclasses should define the `_load_manipulator_metadata` method to return a dictionary of relevant axis
    values with keys of the form `manipulator_axis` where `axis` is the names in the `_manipulator_axes` list,
    i.e. is given in :class:`peaks` convention. This method should return values as :class:`pint.Quantity` objects
    where possible to ensure units are appropriately captured and propagated. Alternatively, the main `_load_metadata`
    method can be overwritten to return the full metadata dictionary, including manipulator metadata.

    In general, subclasses will always include the 6 base axes, with a name of `None` if an axis cannot be moved.
    Then the reference positions of that axis can still be captured. Subclasses should add any additional axes
    desired via the `_add_manipulator_axes` class variable, providing a list of additional axes. Subclasses should
    also add `_update_manipulator_sign_conventions` and `_update_manipulator_name_conventions` dictionaries to update
    any sign conventions and name conventions from the default values (all axes positive and named `None`).

    `_manipulator_sign_conventions` should be a dictionary mapping the sign conventions required to get from the raw
    dimension values to standard conventions used for `peaks`. The `peaks` convention is that values of dimensions of
    the data are left matching the same sign as the experiment to make comparison with the live data more obvious,
    with any sign conversions required for data processing (e.g. $$k$$-conversion) happening under the hood. Default
    behaviour is that all axes are positive.

    `_manipulator_name_conventions` should be a dictionary mapping the `peaks` axis names to physical axis names
    on the manipulator. Default behaviour is that all axes are named as `None`, and so each physical axis should be
    added here.

    See Also
    --------
    BaseDataLoader
    BaseDataLoader._load_metadata
    """

    # Define class variables
    _loc_name = "Default Manipulator"
    _manipulator_axes = ["polar", "tilt", "azi", "x1", "x2", "x3"]
    _desired_dim_order = ["x3", "x2", "x1", "polar", "tilt", "azi"]
    _manipulator_sign_conventions = {}  # Mapping of axes to sign conventions
    _manipulator_name_conventions = {}  # Mapping of peaks axes to local names
    _manipulator_exclude_from_metadata_warn = (
        []
    )  # List of axes to ignore if the metadata is missing
    _metadata_parsers = [
        "_parse_manipulator_metadata"
    ]  # Specific metadata parsers to apply

    # Properties to access class variables
    @property
    def manipulator_axes(self):
        """Return the manipulator axes."""
        return self._manipulator_axes

    @property
    def manipulator_sign_conventions(self):
        """Return the manipulator sign conventions to map to `peaks` convention."""
        return self._manipulator_sign_conventions

    @property
    def manipulator_name_conventions(self):
        """Return the `peaks` --> physical manipulator name mapping."""
        return self._manipulator_name_conventions

    @classmethod
    def _parse_manipulator_metadata(cls, metadata_dict):
        """Parse metadata specific to the manipulator."""

        # Build manipulator metadata model
        fields = {
            axis: (Optional[AxisMetadataModelWithReference], None)
            for axis in cls._manipulator_axes
        }
        ManipulatorMetadataModel = create_model("ManipulatorMetadataModel", **fields)

        # Extract the relevant metadata and parse in a form for passing to the model
        manipulator_metadata_dict = {}
        for axis in cls._manipulator_axes:
            manipulator_metadata_dict[axis] = {
                "local_name": cls._manipulator_name_conventions.get(axis, None),
                "value": metadata_dict.get(f"manipulator_{axis}"),
                "reference_value": None,
            }
        # Populate the metadata model
        manipulator_metadata = ManipulatorMetadataModel(**manipulator_metadata_dict)

        # Parse list of axes that are names, so where a metadata warning should be given unless excluded by passing
        # in the class variable _manipulator_ignore_missing_metadata
        metadata_to_warn_if_missing = [
            f"manipulator_{axis}"
            for axis, name in cls._manipulator_name_conventions.items()
            if name and axis not in cls._manipulator_exclude_from_metadata_warn
        ]

        # Return the model, and a list of any metadata that should be warned if missing
        return {"manipulator": manipulator_metadata}, metadata_to_warn_if_missing


class BaseTemperatureDataLoader(BaseDataLoader):
    """Base class for data loaders for systems with sample temperature control.

    Subclasses should define the `_load_temperature_metadata` method to return a dictionary of relevant metadata
    values with keys of the form `temperature_item` where `item` is the names in the `_temperature_attributes` list,
    i.e. is given in :class:`peaks` convention. This method should return values as :class:`pint.Quantity` objects
    where possible to ensure units are appropriately captured and propagated. Alternatively, the main `_load_metadata`
    method can be overwritten to return the full metadata dictionary, including manipulator metadata.

    Subclasses should add any additional temperature attributes via the `_add_temperature_attributes` class variable,
     providing a list of additional attributes.

    See Also
    --------
    BaseDataLoader
    BaseDataLoader._load_metadata
    """

    # Define class variables
    _loc_name = "Default Temperature"
    _temperature_attributes = ["sample", "cryostat", "shield", "setpoint"]
    _temperature_exclude_from_metadata_warn = [
        "cryostat",
        "shield",
        "setpoint",
    ]  # List of attributes to ignore for metadata warnings
    _metadata_parsers = [
        "_parse_temperature_metadata"
    ]  # Specific metadata parsers to apply

    # Properties to access class variables
    @property
    def temperature_attributes(self):
        """Return the temperature attributes."""
        return self._temperature_attributes

    @classmethod
    def _parse_temperature_metadata(cls, metadata_dict):
        """Parse metadata specific to the temperature data."""

        # Build and populate the temperature metadata model
        temperature_metadata = TemperatureMetadataModel(
            sample=metadata_dict.get("temperature_sample"),
            cryostat=metadata_dict.get("temperature_cryostat"),
            shield=metadata_dict.get("temperature_shield"),
            setpoint=metadata_dict.get("temperature_setpoint"),
        )

        metadata_to_warn_if_missing = (
            f"temperature_{attribute}"
            for attribute in cls._temperature_attributes
            if attribute not in cls._temperature_exclude_from_metadata_warn
        )

        # Return the model, and a list of any metadata that should be warned if missing
        return {"temperature": temperature_metadata}, metadata_to_warn_if_missing


class BasePhotonSourceDataLoader(BaseDataLoader):
    """Base class for data loaders for experiments involving photon sources.

    Subclasses should define the `_load_photon_metadata` method to return a dictionary of relevant metadata
    values with keys of the form `photon_item` where `item` is the names in the `_photon_attributes` list,
    i.e. is given in :class:`peaks` convention. This method should return values as :class:`pint.Quantity` objects
    where possible to ensure units are appropriately captured and propagated. Alternatively, the main `_load_metadata`
    method can be overwritten to return the full metadata dictionary, including manipulator metadata.

    Subclasses should add any additional photon attributes via the `_add_photon_attributes` class variable,
     providing a list of additional attributes.

    See Also
    --------
    BaseDataLoader
    BaseDataLoader._load_metadata
    """

    _loc_name = "Default Photon Source"
    _photon_attributes = ["hv", "polarisation", "exit_slit"]
    _photon_exclude_from_metadata_warn = [
        "polarisation",
        "exit_slit",
    ]  # List of attributes to ignore for metadata warnings
    _metadata_parsers = ["_parse_photon_metadata"]  # Specific metadata parsers to apply

    # Properties to access class variables
    @property
    def photon_attributes(self):
        """Return the photon attributes."""
        return self._photon_attributes

    @classmethod
    def _parse_photon_metadata(cls, metadata_dict):
        """Parse metadata specific to the photon data."""

        # Build and populate the photon metadata model
        photon_metadata = PhotonMetadataModel(
            hv=metadata_dict.get("photon_hv"),
            polarisation=metadata_dict.get("photon_polarisation"),
            exit_slit=metadata_dict.get("photon_exit_slit"),
        )

        metadata_to_warn_if_missing = (
            f"photon_{attribute}"
            for attribute in cls._photon_attributes
            if attribute not in cls._photon_exclude_from_metadata_warn
        )

        # Return the model, and a list of any metadata that should be warned if missing
        return {"photon": photon_metadata}, metadata_to_warn_if_missing
