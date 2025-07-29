from typing import Optional

from pydantic import create_model

from peaks.core.fileIO.base_data_classes.base_data_class import BaseDataLoader
from peaks.core.metadata.base_metadata_models import AxisMetadataModelWithReference


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
        return {"_manipulator": manipulator_metadata}, metadata_to_warn_if_missing

    @classmethod
    def _parse_manipulator_references(cls, da, specified_values):
        """Methods to parse the reference values for the manipulator based on input from the user."""
        raise NotImplementedError("Subclasses should implement this method.")
