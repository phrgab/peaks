from peaks.core.fileIO.base_data_classes.base_data_class import BaseDataLoader
from peaks.core.metadata.base_metadata_models import TemperatureMetadataModel


class BaseTemperatureDataLoader(BaseDataLoader):
    """Mixin providing sample temperature metadata.

    Provides metadata handling for sample temperature information, e.g. from
    cryostats or sample heaters. Designed to be used as a mixin alongside other
    base loader classes, see for example
    :class:`~peaks.core.fileIO.base_arpes_data_classes.base_arpes_data_class.BaseARPESDataLoader`.

    Concrete loaders should expose raw values through ``_load_metadata()`` using keys
    such as ``temperature_sample``, ``temperature_cryostat``, or
    ``temperature_heater_power``. ``_parse_temperature_metadata()`` then maps those
    values onto the :mod:`peaks`
    :class:`~peaks.core.metadata.base_metadata_models.TemperatureMetadataModel`.

    ``_temperature_attributes`` lists the recognised temperature-related fields.
    ``_temperature_exclude_from_metadata_warn`` controls which missing fields should be
    ignored when warning about incomplete metadata extraction.
    """

    # Define class variables
    _loc_name = "Default Temperature"
    _temperature_attributes = [
        "sample",
        "precooling_stage",
        "cryostat",
        "heater_power",
        "shield",
        "setpoint",
    ]
    _temperature_exclude_from_metadata_warn = [
        "precooling_stage",
        "cryostat",
        "heater_power",
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
        """Build the structured temperature metadata model from raw metadata."""

        # Build and populate the temperature metadata model
        temperature_metadata = TemperatureMetadataModel(
            sample=metadata_dict.get("temperature_sample"),
            precooling_stage=metadata_dict.get("temperature_precooling_stage"),
            cryostat=metadata_dict.get("temperature_cryostat"),
            heater_power=metadata_dict.get("temperature_heater_power"),
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
