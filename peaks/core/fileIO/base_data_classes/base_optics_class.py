from peaks.core.fileIO.base_data_classes.base_data_class import BaseDataLoader
from peaks.core.metadata.base_metadata_models import OpticsMetadataModel


class BaseOpticsDataLoader(BaseDataLoader):
    """Base class for data loaders for systems with focussing optics.

    Provides metadata handling for optical components such as zone plates, order sortign apertures, etc.
    Designed to be used as a mixin alongside other base loader classes, e.g. ``BaseARPESDataLoader``.

    Subclasses should define class variables describing their motor naming conventions:

    - ``_optics_name_conventions``: a dictionary mapping the three primary axes ``x1``, ``x2`` and ``x3`` to
      the local motor names used in the metadata, e.g. ``ZPx``, ``ZPy``, ``ZPz``.
    - ``_optics_additional_name_conventions``: (optional) a dictionary mapping any extra axes (e.g. coarse/fine splits)
      to their local motor names, e.g. ``OSA_x2`` --> ``OSAy``.

    The class provides a ``_parse_optics_metadata`` method that reads motor positions from the metadata dictionary
    and populates the model. ``_parse_optics_metadata`` should also be added to the ``_metadata_parsers`` list in subclasses.

    Examples
    --------
    See :class:`peaks.core.fileIO.loaders.diamond.I05NanoARPESLoader` for a working example using zone plate axes
    as the primary optics and order sorting aperture axes as additional optics.

    See Also
    --------
    :class:`peaks.core.fileIO.base_data_classes.base_data_class.BaseDataLoader`
    """

    # Define class variables
    _loc_name = "Default Optics"
    _optics_axes = ["x1", "x2", "x3"]
    _optics_name_conventions = {}
    _optics_additional_name_conventions = {}
    _desired_dim_order = ["x3", "x2", "x1"]
    _optics_exclude_from_metadata_warn = []
    _metadata_parsers = ["_parse_optics_metadata"]

    # Properties to access class variables
    @property
    def optics_axes(self):
        """Return the optics axes."""
        return self._optics_axes

    @property
    def optics_name_conventions(self):
        """Return the `peaks` --> local optics axis name mapping."""
        return self._optics_name_conventions

    @classmethod
    def _parse_optics_metadata(cls, metadata_dict):
        """Parse metadata specific to the optics data."""

        optics_metadata_dict = {}

        # Extract the primary axes metadata and parse in a form for passing to the model
        for axis in cls._optics_axes:
            local_motor = cls._optics_name_conventions.get(axis, None)
            optics_metadata_dict[axis] = {
                "local_name": local_motor,
                "value": metadata_dict.get(f"optics_{axis}"),
            }

        # Extract any additional axes metadata
        for axis, local_motor in cls._optics_additional_name_conventions.items():
            optics_metadata_dict[axis] = {
                "local_name": local_motor,
                "value": metadata_dict.get(f"optics_{axis}"),
            }

        # Populate the metadata model
        optics_metadata = OpticsMetadataModel(**optics_metadata_dict)

        metadata_to_warn_if_missing = [
            f"optics_{axis}"
            for axis in cls._optics_axes
            if axis not in cls._optics_exclude_from_metadata_warn
        ] + [
            f"optics_{axis}"
            for axis, local_motor in cls._optics_additional_name_conventions.items()
            if axis not in cls._optics_exclude_from_metadata_warn
        ]

        # Return the model, and a list of any metadata that should be warned if missing
        return {"_optics": optics_metadata}, metadata_to_warn_if_missing
