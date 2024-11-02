from peaks.core.fileIO.base_data_classes.base_data_class import BaseDataLoader
from peaks.core.metadata.base_metadata_models import (
    PhotonMetadataModel,
    PumpPhotonMetadataModel,
)


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
        return {"_photon": photon_metadata}, metadata_to_warn_if_missing


class BasePumpProbeClass(BasePhotonSourceDataLoader):
    _pump_photon_attributes = ["hv", "polarisation", "delay", "power"]
    _pump_photon_exclude_from_metadata_warn = _pump_photon_attributes
    _metadata_parsers = ["_parse_pump_photon_metadata"]

    @property
    def pump_photon_attributes(self):
        return self._pump_photon_attributes

    @classmethod
    def _parse_pump_photon_metadata(cls, metadata_dict):
        pump_photon_metadata = PumpPhotonMetadataModel(
            hv=metadata_dict.get("pump_hv"),
            polarisation=metadata_dict.get("pump_polarisation"),
            power=metadata_dict.get("pump_power"),
            delay=metadata_dict.get("pump_delay"),
            t0_position=metadata_dict.get("pump_t0_position"),
        )

        metadata_to_warn_if_missing = (
            f"pump_{attribute}"
            for attribute in cls._pump_photon_attributes
            if attribute not in cls._pump_photon_exclude_from_metadata_warn
        )

        return {"_pump": pump_photon_metadata}, metadata_to_warn_if_missing
