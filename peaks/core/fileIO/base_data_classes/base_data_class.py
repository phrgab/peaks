import inspect
import os
from datetime import datetime

import pint_xarray  # noqa: F401
import xarray as xr

from peaks.core.fileIO.loc_registry import LOC_REGISTRY, IdentifyLoc
from peaks.core.metadata.base_metadata_models import (
    BaseScanMetadataModel,
)
from peaks.core.utils.misc import analysis_warning

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
    DataArray. Any subclass should define the `_metadata_parsers` class variable as a list of these methods to be
    called.

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
    def load(cls, fpath, lazy=None, loc=None, metadata=True, quiet=False, **kwargs):
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
            Whether to attempt to load metadata from the file. Defaults to True.

        quiet : bool, optional
            Whether to suppress analysis warnings when loading data. Defaults to False.

        kwargs : dict
            Additional keyword arguments to pass to the individual loaders.

        Returns
        ------------
        da : xarray.DataArray
            The loaded data as an xarray DataArray.

        Examples
        ------------
        Normally used via the `pks.load` core function. If needed to be accessed directly, example usage is as follows::

            from peaks.core.fileIO.base_data_classes import BaseDataLoader

            fpath = 'C:/User/Documents/Research/disp1.xy'  # Path to the file to be loaded
            da = BaseDataLoader.load(fpath)  # Load the data

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
        return loader_class._load(fpath, lazy, metadata, quiet, **kwargs)

    @classmethod
    def load_metadata(
        cls,
        fpath,
        loc=None,
        return_as_dict=False,
        quiet=True,
        load_metadata_from_file=True,
    ):
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

        load_metadata_from_file : bool, optional
            Whether to load the actual metadata from the file. Defaults to True. If False, only the basic file
            metadata will be added, along with the empty metadata structure


        Returns
        ------------
        metadata_dict : dict
            The loaded metadata as a dictionary simple dictionary or a dictionary of parsed metadata models.

        Examples
        ------------
        Example usage is as follows::

            from peaks.core.fileIO.base_data_classes import BaseDataLoader

            fpath = 'C:/User/Documents/Research/disp1.xy'
            metadata = BaseDataLoader.load_metadata(fpath)  # Load the metadata

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
                return cls.get_loader(loc).load_metadata(fpath, return_as_dict, quiet)
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

        if load_metadata_from_file:
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
                f".{method_name.split('_handler_')[1]}": method
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
    def _load(cls, fpath, lazy, metadata, quiet, **kwargs):
        """Generic method for loading the data."""
        # Load the actual data from the file
        data = cls._load_data(fpath, lazy, **kwargs)
        da = cls._make_dataarray(data)  # Convert to DataArray
        # Add a name to the DataArray
        da.name = fpath.split("/")[-1].split(".")[0]
        parsed_metadata = cls.load_metadata(
            fpath,
            loc=cls._loc_name,
            quiet=quiet or not metadata,
            load_metadata_from_file=metadata,
        )
        da.attrs.update(parsed_metadata)
        cls._metadata_cache.pop(fpath, None)  # Clear any metadata cache for this file
        # Apply any specific conventions and add a history of the load
        da = cls._apply_conventions(da)
        cls._add_load_history(da, fpath)
        return da

    @classmethod
    def _load_data(cls, fpath, lazy, **kwargs):
        """Load the data. To be implemented by subclasses.

        Parameters
        ------------
        fpath : str
            Path to the file to be loaded.

        lazy : bool or str
            Whether to load data in a lazily evaluated dask format. Set explicitly using True/False Boolean.

        kwargs : dict
            Additional keyword arguments to pass to the individual loaders.

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
        return {"_scan": general_scan_metadata}

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
            if metadata_to_add_to_warning_list:
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
