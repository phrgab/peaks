from typing import Optional

from pydantic import create_model

from peaks.core.fileIO.base_data_classes.base_data_class import BaseDataLoader
from peaks.core.metadata.base_metadata_models import AxisMetadataModelWithReference


class BaseManipulatorDataLoader(BaseDataLoader):
    """Mixin providing manipulator-axis metadata and sign/name conventions.

    Provides metadata handling for sample manipulators.
    Designed to be used as a mixin alongside other base loader classes, see
    for example
    :class:`~peaks.core.fileIO.base_arpes_data_classes.base_arpes_data_class.BaseARPESDataLoader`.

    Concrete loaders should expose manipulator values through ``_load_metadata()``
    using keys of the form  ``manipulator_<axis>``,  where ``<axis>`` is taken from
    ``_manipulator_axes``, for example ``manipulator_polar`` or ``manipulator_x1``.
    Those raw values are then converted into a structured ``ManipulatorMetadataModel``
    comprised of :class:`~peaks.core.metadata.base_metadata_models.AxisMetadataModelWithReference`
    entries by ``_parse_manipulator_metadata()``.

    Notes
    -----

    ``_manipulator_axes`` defines the axes supported by the loader. Loaders
    generally keep the six core axes ``polar``, ``tilt``, ``azi``, ``x1``, ``x2``, and ``x3``,
    as defined by :mod:`peaks` metadata conventions
    (https://research.st-andrews.ac.uk/kinggroup/peaks/latest/explanations/angle_conventions.html)
    and may append additional instrument-specific axes as required.

    ``_manipulator_name_conventions`` maps :mod:`peaks` axis names to the local motor names
    used at the instrument. ``_manipulator_sign_conventions`` records how those local
    axes relate to the shared :mod:`peaks` convention. The raw data coordinates are
    left in the experimental sign convention; these sign mappings are used later by
    downstream operations such as reference-angle handling and k-conversion.
    """

    # Define class variables
    _loc_name = "Default Manipulator"
    _manipulator_axes = ["polar", "tilt", "azi", "x1", "x2", "x3"]
    _desired_dim_order = ["x3", "x2", "x1", "polar", "tilt", "azi"]
    _manipulator_sign_conventions = {}  # Mapping of axes to sign conventions
    _manipulator_name_conventions = {}  # Mapping of peaks axes to local names
    _manipulator_exclude_from_metadata_warn = []  # List of axes to ignore if the metadata is missing
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
        """Return the manipulator sign conventions to map to :mod:`peaks` convention."""
        return self._manipulator_sign_conventions

    @property
    def manipulator_name_conventions(self):
        """Return the :mod:`peaks` --> physical manipulator name mapping."""
        return self._manipulator_name_conventions

    @classmethod
    def _parse_manipulator_metadata(cls, metadata_dict):
        """Build the structured manipulator metadata model from raw metadata."""

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
        return {"_manipulator": manipulator_metadata}, metadata_to_warn_if_missing

    @classmethod
    def _parse_manipulator_references(cls, da, specified_values):
        """Parse user-specified manipulator reference values for ``da``."""
        raise NotImplementedError("Subclasses should implement this method.")
