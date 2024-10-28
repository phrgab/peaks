from peaks.core.fileIO.base_data_classes.base_data_class import BaseDataLoader
from peaks.core.metadata.base_metadata_models import TemperatureMetadataModel


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
        return {"_temperature": temperature_metadata}, metadata_to_warn_if_missing
