from peaks.core.fileIO.base_data_classes.base_data_class import BaseDataLoader
from peaks.core.metadata.base_metadata_models import (
    PhotonMetadataModel,
    PumpPhotonMetadataModel,
)


class BasePhotonSourceDataLoader(BaseDataLoader):
    """Mixin providing photon source metadata.

    Provides metadata handling for photon beam information for synchrotron or
    lab-based light sources. Designed to be used as a mixin alongside other
    base loader classes, see for example
    :class:`~peaks.core.fileIO.base_arpes_data_classes.base_arpes_data_class.BaseARPESDataLoader`.

    Concrete loaders should expose raw values through ``_load_metadata()`` using keys
    such as ``photon_hv``, ``photon_polarisation``, and ``photon_exit_slit``.
    ``_parse_photon_metadata()`` then maps those values onto the :mod:`peaks`
    :class:`~peaks.core.metadata.base_metadata_models.PhotonMetadataModel`.

    ``_photon_attributes`` lists the recognised photon-source fields.
    ``_photon_exclude_from_metadata_warn`` controls which of them may be omitted
    without warning.
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
        """Return the photon beam attributes."""
        return self._photon_attributes

    @classmethod
    def _parse_photon_metadata(cls, metadata_dict):
        """Build the structured photon-source metadata model from raw metadata."""

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
    """Mixin adding metadata support for an optical pump in pump-probe experiments.

    Usage follows :class:`peaks.core.fileIO.base_data_classes.base_photon_source_classes.BasePhotonSourceDataLoader`.
    """

    _pump_photon_attributes = ["hv", "polarisation", "delay", "power"]
    _pump_photon_exclude_from_metadata_warn = _pump_photon_attributes
    _metadata_parsers = ["_parse_pump_photon_metadata"]

    @property
    def pump_photon_attributes(self):
        """Return the pump-photon metadata fields expected by this loader."""
        return self._pump_photon_attributes

    @classmethod
    def _parse_pump_photon_metadata(cls, metadata_dict):
        """Build the structured pump-photon metadata model from raw metadata."""
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
