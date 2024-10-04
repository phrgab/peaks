from datetime import datetime
import xarray as xr
import pint
import pint_xarray  # noqa: F401
import os
from os.path import isfile, join
import natsort
import h5py
from pydantic import BaseModel, create_model
from typing import Optional, Union

from .base_metadata_models import ureg, Quantity
from .registry import LOC_REGISTRY
from .loaders.SES import _load_SES_metalines, _SES_find
from .data_loading import _h5py_str


class BaseDataLoader:
    """Base Class for data loaders"""

    # Define some attributes or attribute placeholders
    loc_name = "Base"

    # List of the standard order required for the dimensions of the data
    desired_dim_order = []

    # Dict of units for core data (use `data` key) and any dims [metadata handled separately]
    # Supply unit as a `pint`-compatible unit string or as single or composite pint.Unit() object
    units = {}

    # Data type and if desired specific data order for main data
    dtype = None
    dorder = None

    class BaseScanMetadataModel(BaseModel):
        """Model to store basic scan identifier metadata."""

        name: str
        filepath: str
        loc: str
        timestamp: str

    @classmethod
    def load(cls, fname, lazy="auto", loc="auto", metadata=True):
        """Top-level method to load data and return a DataArray.

        Will generally be run from a different class than the loader class for the specific file,
        so put loc-specific logic in _load method.

        Parameters
        ------------
        fname : str
            Path to the file to be loaded.

        lazy : bool or str, optional
            Whether to load data in a lazily evaluated dask format. Set explicitly using True/False Boolean. Defaults
            to 'auto' where a file is only loaded in the :class:`dask` format if its spectrum is above threshold
            in :class:peaks.File.

        loc : str, optional
            The location at which the data was obtained. If not specified, the location will be attempted to be
            determined automatically.

        metadata : bool, optional
            Whether to attempt to load metadata into the attributes of the :class:`xarray.DataArray`. Defaults to True.
        """

        # Parse the loc
        loc = cls.get_loc(fname) if loc == "auto" else loc
        if loc not in LOC_REGISTRY.keys():  # Check a valid loc
            raise ValueError(
                f"No loader defined for location {loc}. Specify one of {LOC_REGISTRY.keys()} or leave"
                f" loc argument empty to attempt to determine the location automatically."
            )
        # Trigger the loader for the correct loc
        loader_class = cls.get_loader(loc)
        return loader_class._load(fname, lazy, metadata)

    @staticmethod
    def get_loader(loc):
        """Get the loader class for the given location."""
        loader_class = LOC_REGISTRY.get(loc)
        if loader_class:
            return loader_class
        else:
            raise ValueError(f"No loader found for location {loc}")

    @staticmethod
    def get_loc(fname):
        """This function determines the location at which the data was obtained.

        Parameters
        ------------
        fname : str
            Path to the file to be loaded.

        Returns
        ------------
        loc : str
            The name of the location (typically a beamline).

        Examples
        ------------
        Example usage is as follows::

            from peaks.core.fileIO.base_data_classes import BaseDataLoader

            fname = 'C:/User/Documents/Research/disp1.xy'

            # Determine the location at which the data was obtained
            loc = BaseDataLoader.get_loc(fname)

        """

        # Get file_extension to determine the file type
        filename, file_extension = os.path.splitext(fname)

        # Set loc to Default None value
        loc = None

        # If there is no extension, the data is in a folder. This is consistent with SOLEIL CASSIOPEE Fermi maps or
        # CLF Artemis data
        if file_extension == "":
            # Extract identifiable data from the file to determine if the location is SOLEIL CASSIOPEE
            file_list = natsort.natsorted(os.listdir(fname))
            file_list_ROI = [
                item
                for item in file_list
                if "ROI1_" in item and isfile(join(fname, item))
            ]
            if len(file_list_ROI) > 1:  # Must be SOLEIL CASSIOPEE
                loc = "SOLEIL CASSIOPEE"
            else:  # Likely CLF Artemis
                # Extract identifiable data from the file to determine if the location is CLF Artemis
                file_list_Neq = [item for item in file_list if "N=" in item]
                if len(file_list_Neq) > 0:  # Must be CLF Artemis
                    loc = "CLF Artemis"

        # If the file is .xy format, the location must be either MAX IV Bloch-spin, StA-Phoibos or StA-Bruker
        elif file_extension == ".xy":
            # Open the file and load the first line
            with open(fname) as f:
                line0 = f.readline()

            # If measurement was performed using Specs analyser, location must be MAX IV Bloch-spin or StA-Phoibos
            if "SpecsLab" in line0:
                # By default, assume StA-Phoibos
                loc = "StA-Phoibos"
                # Open the file and load the first 30 lines to to check if this is instead MAX IV Bloch-spin
                with open(fname) as f:
                    for i in range(30):
                        # If the 'PhoibosSpin' identifier is present in any of the lines, location must be MAX IV Bloch-spin
                        if "PhoibosSpin" in f.readline():
                            loc = "MAX IV Bloch-spin"

            # If not (and the identifier 'Anode' is present), then measurement was XRD using StA-Bruker
            elif "Anode" in line0:
                loc = "StA-Bruker"

        # If the file is .sp2 format, the location must be MAX IV Bloch-spin
        elif file_extension == ".sp2":
            loc = "MAX IV Bloch-spin"

        # If the file is .krx format, the location must be StA-MBS
        elif file_extension == ".krx":
            loc = "StA-MBS"

        # If the file is .txt format, the location must be StA-MBS, MAX IV Bloch, Elettra APE or SOLEIL CASSIOPEE
        elif file_extension == ".txt":
            # Open the file and load the first line
            with open(fname) as f:
                line0 = f.readline()

            # MAX IV Bloch, Elettra APE or SOLEIL CASSIOPEE .txt files follow the same SES data format, so we can identify
            # the location from the location line in the file
            if line0 == "[Info]\n":  # Identifier of the SES data format
                # Extract metadata information from file using the _load_SES_metalines function
                metadata_lines = _load_SES_metalines(fname)
                # Extract and determine the location using the _SES_find function
                location = _SES_find(metadata_lines, "Location=")
                if "bloch" in location.lower() or "maxiv" in location.lower():
                    loc = "MAX IV Bloch"
                elif "ape" in location.lower() or "elettra" in location.lower():
                    loc = "Elettra APE"
                elif "cassiopee" in location.lower() or "soleil" in location.lower():
                    loc = "SOLEIL CASSIOPEE"

            # If the file does not follow the SES format, it must be StA-MBS
            else:
                loc = "StA-MBS"

        # If the file is .zip format, the file must be of SES format. Thus, the location must be MAX IV Bloch, Elettra APE,
        # SOLEIL CASSIOPEE or Diamond I05-nano (defl map)
        elif file_extension == ".zip":
            # Extract metadata information from file using the _load_SES_metalines function
            metadata_lines = _load_SES_metalines(fname)
            # Extract and determine the location using the _SES_find function
            location = _SES_find(metadata_lines, "Location=")
            if "bloch" in location.lower() or "maxiv" in location.lower():
                loc = "MAX IV Bloch"
            elif "ape" in location.lower() or "elettra" in location.lower():
                loc = "Elettra APE"
            elif "cassiopee" in location.lower() or "soleil" in location.lower():
                loc = "SOLEIL CASSIOPEE"
            elif "i05" in location.lower() or "diamond" in location.lower():
                loc = "Diamond I05-nano"

        # If the file is .ibw format, the file is likely SES format. Thus location should be MAX IV Bloch, Elettra APE or
        # SOLEIL CASSIOPEE
        elif file_extension == ".ibw":
            # Extract metadata information from file using the _load_SES_metalines function
            metadata_lines = _load_SES_metalines(fname)
            # Extract and determine the location using the _SES_find function
            location = _SES_find(metadata_lines, "Location=")
            if "bloch" in location.lower() or "maxiv" in location.lower():
                loc = "MAX IV Bloch"
            elif "ape" in location.lower() or "elettra" in location.lower():
                loc = "Elettra APE"
            elif "cassiopee" in location.lower() or "soleil" in location.lower():
                loc = "SOLEIL CASSIOPEE"
            # If we are unable to find a location, define location as a generic ibw file
            else:
                loc = "ibw"

        # If the file is .nxs format, the location should be Diamond I05-nano, Diamond I05-HR or ALBA LOREA
        elif file_extension == ".nxs":
            # Open the file (read only)
            f = h5py.File(fname, "r")
            # .nxs files at Diamond and Alba contain approximately the same identifier format
            identifier = _h5py_str(f, "entry1/instrument/name")
            # From the identifier, determine the location
            if "i05-1" in identifier:
                loc = "Diamond I05-nano"
            elif "i05" in identifier:
                loc = "Diamond I05-HR"
            elif "lorea" in identifier:
                loc = "ALBA LOREA"

        # If the file is .nc format, it is a NetCDF file. Set location to NetCDF, since the location information should be
        # in the NetCDF attributes
        elif file_extension == ".nc":
            loc = "NetCDF"

        # If the file is .cif format, it should be a structure file
        elif file_extension == ".cif":
            loc = "Structure"

        # If the file is standard image format, it should be a LEED file
        elif (
            file_extension == ".bmp"
            or file_extension == ".jpeg"
            or file_extension == ".png"
        ):
            loc = "StA-LEED"

        # If the file is .iso format, it should be a LEED file
        elif file_extension == ".iso":
            loc = "StA-RHEED"

        return loc

    @classmethod
    def _load(cls, fname, lazy, metadata):
        """Generic method for loading the data."""
        data = cls.load_data(fname, lazy)  # Load the actual data from the file
        DataArray = cls.make_DataArray(data)  # Convert to DataArray
        # Add a name to the DataArray
        DataArray.name = fname.split("/")[-1].split(".")[0]
        if metadata:  # Load and metadata if requested
            metadata_dict = cls.load_metadata(fname)
            cls.apply_metadata(DataArray, metadata_dict)
        # Apply any specific conventions and add a history of the load
        DataArray = cls.apply_conventions(DataArray)
        DataArray = cls.apply_units_to_core_data(DataArray)
        cls.add_load_history(DataArray, fname)

        return DataArray

    @classmethod
    def load_data(cls, fname, lazy):
        """Load the data. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses should implement this method.")

    @classmethod
    def load_metadata(cls, fname):
        """Load the metadata. Should return a dictionary `metadata_dict` mapping relevant metadata keys to values.
        Loaders that subclass this class should implement this method, generally calling super
        on the base class methods and extending, adding relevant metadata fields to metadata_dict.
        """

        # For general loader, just extract a timestamp
        # Get this from last modification time - overwrite this in subclasses if a more robust method is available
        timestamp = os.path.getmtime(fname)
        # Convert the timestamp to a human-readable format
        readable_timestamp = datetime.fromtimestamp(timestamp).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        metadata_dict = {"timestamp": readable_timestamp, "fname": fname}
        return metadata_dict

    @classmethod
    def apply_metadata(cls, DataArray, metadata):
        """Apply metadata to the DataArray."""

        general_scan_metadata = cls.BaseScanMetadataModel(
            name=DataArray.name,
            filepath=metadata.get("fname"),
            loc=cls.loc_name,
            timestamp=metadata.get("timestamp"),
        )
        DataArray.attrs.update({"scan": general_scan_metadata})

    @classmethod
    def make_DataArray(cls, data):
        """Convert data into an xarray DataArray.

        Parameters
        ------------
        data : dict
            Dictionary containing the data to be converted, a list of dims in order corresponding to the data axis
            order, and a dict of associated coords."""
        DataArray = xr.DataArray(
            data["spectrum"],
            dims=data.get("dims", None),
            coords=data.get("coords", None),
        )
        return DataArray

    @classmethod
    def apply_conventions(cls, DataArray):
        """Apply relevant conventions to the DataArray."""

        # Ensure that all the DataArray dimension coordinates are ordered low to high
        for dim in DataArray.dims:
            if len(DataArray[dim]) > 1:
                if DataArray[dim].data[-1] - DataArray[dim].data[0] < 0:
                    DataArray = DataArray.reindex({dim: DataArray[dim][::-1]})
        # Reorder dimensions if needed
        if cls.desired_dim_order:
            DataArray = DataArray.transpose(
                *cls.desired_dim_order, ..., missing_dims="ignore"
            )
        # Set the appropriate data type and order
        if cls.dtype:
            DataArray = DataArray.astype(cls.dtype, order=cls.dorder)
        return DataArray

    @classmethod
    def apply_units_to_core_data(cls, DataArray):
        """Apply units to the data (main array and dimensions)."""
        data_units = {dim: cls.units.get(dim, None) for dim in DataArray.dims}
        data_units[DataArray.name] = cls.units.get("data", None)
        return DataArray.pint.quantify(data_units)

    @classmethod
    def apply_units_to_metadata(cls, metadata_value, unit):
        """Apply units to metadata values."""
        if isinstance(metadata_value, Quantity):
            return metadata_value
        try:
            if isinstance(unit, str):
                return metadata_value * ureg(unit)
            elif isinstance(unit, pint.Unit):
                return metadata_value * unit
            else:
                return metadata_value * ureg("")
        except ValueError:
            return metadata_value

    @classmethod
    def add_load_history(cls, DataArray, fname):
        """Add a history of the load to the DataArray."""
        DataArray.history.add(
            {
                "record": "Data loaded",
                "loc": cls.loc_name,
                "loader": cls.__name__,
                "file_name": fname,
            }
        )


class BaseManipulatorDataLoader(BaseDataLoader):
    """Base class for data loaders for systems with manipulators.

    Notes
    -----
    Subclasses should: make a new version of copy of the manipulator_units, manipulator_sign_conventions and
    manipulator_name_conventions dictionaries, and update them with the appropriate values for the specific
    manipulator. They should define the load_manipulator_metadata method to return a dictionary of relevant axis
    values with keys in :class:`peaks` convention, as per e.g. manipulator_units."""

    loc_name = "Default Manipulator"

    desired_dim_order = ["x3", "x2", "x1", "polar", "tilt", "azi"]

    # Dict of units for manipulator axes.
    # Supply unit as a `pint`-compatible unit string or as single or composite pint.Unit() object
    manipulator_units = {
        "x1": "mm",
        "x2": "mm",
        "x3": "mm",
        "polar": "deg",
        "tilt": "deg",
        "azi": "deg",
    }

    # Dict of any sign conventions required to get from the raw dimension values to standard conventions used for
    # `peaks`. The `peaks` convention is that values of dimensions of the data are left matching the same sign
    # as the experiment, with then any sign conversions required for e.g. k-conversion happening under the hood.
    manipulator_sign_conventions = {axis: 1 for axis in manipulator_units.keys()}
    # Dict of mappings of `peaks` axis name to physical axis name on the manipulator
    manipulator_name_conventions = {axis: None for axis in manipulator_units.keys()}

    # Define the manipulator metadata models
    class AxisMetadataModel(BaseModel):
        """Base model to store metadata for a single manipulator axis.

        Attributes
        ----------
        name : Optional[str]
            The local name of the axis on the actual system manipulator.
        value : Optional[Union[Quantity, str, None]]
            The value of the axis. If an array (e.g. for movement during the scan), this should return a string that
            describes the axis movement in the form x0:x_step:x1.
        reference_value : Optional[Union[Quantity, None]]
            The reference value of the axis. Supplied as None on initial load, but can be used later for keeping track
            e.g. of normal emission values.
        """

        name: Optional[str] = None
        value: Optional[Union[Quantity, str]] = None
        reference_value: Optional[Quantity] = None

        def set(self, value):
            """Set the value of the axis.

            Parameters
            ----------
            value : float, int, str, pint.Unit
                The value to set the axis to.
                If passed without units, will assume units are the same as the existing value if possible.
            """
            if self.value:
                if isinstance(self.value, pint.Quantity):
                    unit = self.value.units
                else:
                    unit = ""
            self.value = super.apply_units_to_metadata(value, unit)

        def set_reference(self, value):
            """Set the reference value of the axis.

            Parameters
            ----------
            value : float, int, str, pint.Unit
                The value to set the reference axis value to.
                If passed without units, will assume units are the same as the axis value if possible.
            """
            if self.value:
                if isinstance(self.reference_value, pint.Quantity):
                    unit = self.reference_value.units
                else:
                    unit = ""
            self.reference_value = super.apply_units_to_metadata(value, unit)

    @classmethod
    def create_manipulator_metadata_model(cls, axes):
        """Function to dynamically create a model to store manipulator metadata for each axis."""
        fields = {axis: (Optional[cls.AxisMetadataModel], None) for axis in axes}
        return create_model("ManipulatorMetadataModel", **fields)

    @classmethod
    def load_metadata(cls, fname):
        """Load the metadata. Should return a dictionary `metadata_dict` mapping relevant metadata keys to values.

        See Also
        --------
        BaseDataLoader.load_metadata
        """

        metadata_dict = super().load_metadata(fname)
        metadata_dict.update(cls.load_manipulator_metadata(fname))
        return metadata_dict

    @classmethod
    def load_manipulator_metadata(cls, fname):
        """Load manipulator metadata from the file."""
        raise NotImplementedError("Subclasses should implement this method.")

    @classmethod
    def apply_metadata(cls, DataArray, metadata):
        """Apply metadata to the DataArray."""
        super().apply_metadata(DataArray, metadata)

        # Build manipulator metadata model - check which named axes exist
        manipulator_axes = [
            key for key, value in cls.manipulator_name_conventions.items() if value
        ]
        ManipulatorMetadataModel = cls.create_manipulator_metadata_model(
            manipulator_axes
        )

        # Extract the relevant metadata
        manipulator_metadata_dict = {}
        for axis in manipulator_axes:
            axis_metadata = metadata.get(axis)
            axis_unit = cls.manipulator_units[axis]
            axis_metadata_w_units = super().apply_units_to_metadata(
                axis_metadata, axis_unit
            )
            manipulator_metadata_dict[axis] = {
                "name": cls.manipulator_name_conventions[axis],
                "value": axis_metadata_w_units,
                "reference_value": None,
            }
        # Populate the metadata model
        manipulator_metadata = ManipulatorMetadataModel(**manipulator_metadata_dict)
        DataArray.attrs.update({"manipulator": manipulator_metadata})


class BaseCryoDataLoader(BaseDataLoader):
    """Base class for data loaders for systems with sample cooling."""

    loc_name = "Default Cryo"

    temperature_units = {"sample": "K", "cryostat": "K", "shield": "K", "setpoint": "K"}
    name_conventions = {key: key for key in temperature_units.keys()}

    class TemperatureMetadataModel(BaseModel):
        """Model to store temperature metadata."""

        sample: Optional[Union[Quantity, str]] = None
        cryostat: Optional[Union[Quantity, str]] = None
        shield: Optional[Union[Quantity, str]] = None
        setpoint: Optional[Union[Quantity, str]] = None

    @classmethod
    def load_metadata(cls, fname):
        """Load the metadata. Should return a dictionary `metadata_dict` mapping relevant metadata keys to values.

        See Also
        --------
        BaseDataLoader.load_metadata
        """

        metadata_dict = super().load_metadata(fname)
        metadata_dict.update(cls.load_temperature_metadata(fname))
        return metadata_dict

    @classmethod
    def load_temperature_metadata(cls, fname):
        """Load temperature metadata from the file.

        Returns
        -------
        dict :
            Dictionary of temperature values, with keys `temperature_sample`, `temperature_cryostat`,
            `temperature_shield`, `temperature_setpoint`.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @classmethod
    def apply_metadata(cls, DataArray, metadata):
        """Apply metadata to the DataArray."""
        super().apply_metadata(DataArray, metadata)

        # Populate the temperature metadata model
        temperature_metadata = cls.TemperatureMetadataModel(
            sample=cls.apply_units_to_metadata(
                metadata.get("temperature_sample"), cls.temperature_units["sample"]
            ),
            cryostat=cls.apply_units_to_metadata(
                metadata.get("temperature_cryostat"), cls.temperature_units["cryostat"]
            ),
            shield=cls.apply_units_to_metadata(
                metadata.get("temperature_shield"), cls.temperature_units["shield"]
            ),
            setpoint=cls.apply_units_to_metadata(
                metadata.get("temperature_setpoint"), cls.temperature_units["setpoint"]
            ),
        )
        DataArray.attrs.update({"temperature": temperature_metadata})


class BasePhotonDataLoader(BaseDataLoader):
    """Base class for data loaders for experiments involving photon sources."""

    loc_name = "Default Photon Source"

    photon_units = {"eV": "eV", "polarisation": None, "exit_slit": "um"}
    name_conventions = {key: key for key in photon_units.keys()}

    class PhotonMetadataModel(BaseModel):
        """Model to store photon-linked metadata."""

        eV: Optional[Union[Quantity, str]] = None
        polarisation: Optional[Union[int, float, str]] = None
        exit_slit: Optional[Union[Quantity, str]] = None

    @classmethod
    def load_metadata(cls, fname):
        """Load the metadata. Should return a dictionary `metadata_dict` mapping relevant metadata keys to values.

        See Also
        --------
        BaseDataLoader.load_metadata
        """

        metadata_dict = super().load_metadata(fname)
        metadata_dict.update(cls.load_photon_metadata(fname))
        return metadata_dict

    @classmethod
    def load_photon_metadata(cls, fname):
        """Load photon metadata from the file.

        Returns
        -------
        dict :
            Dictionary of values, with keys `photon_eV`, `photon_polarisation`,`photon_exit_slit`.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @classmethod
    def apply_metadata(cls, DataArray, metadata):
        """Apply metadata to the DataArray."""
        super().apply_metadata(DataArray, metadata)

        # Populate the photon metadata model
        photon_metadata = cls.PhotonMetadataModel(
            eV=cls.apply_units_to_metadata(
                metadata.get("photon_eV"), cls.photon_units["eV"]
            ),
            polarisation=metadata.get("photon_polarisation"),
            exit_slit=cls.apply_units_to_metadata(
                metadata.get("photon_exit_slit"), cls.photon_units["exit_slit"]
            ),
        )
        DataArray.attrs.update({"photon": photon_metadata})
