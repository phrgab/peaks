import pint
import pint_xarray

from peaks.core.fileIO.base_data_classes.base_manipulator_class import (
    BaseManipulatorDataLoader,
)
from peaks.core.fileIO.base_data_classes.base_temperature_class import (
    BaseTemperatureDataLoader,
)
from peaks.core.fileIO.base_data_classes.base_photon_source_class import (
    BasePhotonSourceDataLoader,
)

from peaks.core.metadata.base_metadata_models import (
    NamedAxisMetadataModel,
    ARPESMetadataModel,
    ARPESScanMetadataModel,
    ARPESDeflectorMetadataModel,
    ARPESSlitMetadataModel,
    ARPESAnalyserAnglesMetadataModel,
    ARPESAnalyserMetadataModel,
)

from peaks.core.utils.misc import analysis_warning

# Define the unit registry
ureg = pint_xarray.unit_registry


class BaseARPESDataLoader(
    BasePhotonSourceDataLoader, BaseTemperatureDataLoader, BaseManipulatorDataLoader
):
    """Base class for data loaders for ARPES systems. Assume a cryo-manipulator and photon source

    Subclasses should define the `_load_analyser_metadata` method to return a dictionary of relevant metadata
    values with keys of the form in `analyser_item` where `item` is the names in the `_analyser_attributes` list,
    i.e. is given in :class:`peaks` convention. This method should return values as :class:`pint.Quantity` objects
    where possible to ensure units are appropriately captured and propagated. Alternatively, the main `_load_metadata`
    method can be overwritten to return the full metadata dictionary, including manipulator metadata.

    Subclasses should add any additional analyser attributes via the `_add_analyser_attributes` class variable,
    providing a list of additional attributes.

    Subclasses should also define the `_analyser_slit_angle` class variable if this is fixed. If custom logic is
    required, this should be left as None and the logic handled within the metadata loading.

    See Also
    --------
    BaseDataLoader
    BaseDataLoader._load_metadata
    BasePhotonSourceDataLoader
    BaseTemperatureDataLoader
    BaseManipulatorDataLoader
    """

    # Define class variables
    _loc_name = "Default ARPES"
    _dtype = "float32"
    _dorder = "C"
    # Any sign and name conventions to apply to the analyser cf. `peaks` convention
    _analyser_sign_conventions = {}
    _analyser_name_conventions = {}
    # Define analyser slit angle in subclass, otherwise it must be passed to the metadata dict from the metadata loader
    _analyser_slit_angle = None
    # Core attributes
    _analyser_attributes = [
        "model",
        "slit_width",
        "slit_width_identifier",
        "eV",
        "step_size",
        "PE",
        "sweeps",
        "dwell",
        "lens_mode",
        "acquisition_mode",
        "eV_type",
        "scan_command",
        "polar",
        "tilt",
        "azi",
        "deflector_parallel",
        "deflector_perp",
    ]
    _desired_dim_order = [
        "scan_no",
        "hv",
        "temperature_sample",
        "temperature_cryostat",
        "x3",
        "x2",
        "x1",
        "polar",
        "tilt",
        "azi",
        "y_scale",
        "deflector_perp",
        "eV",
        "deflector_parallel",
        "theta_par",
    ]
    _analyser_include_in_metadata_warn = [
        "eV",
        "step_size",
        "PE",
        "sweeps",
        "dwell",
        "azi",
    ]  # List of analyser attributes to include in metadata warning
    _metadata_parsers = [
        "_parse_analyser_metadata",
        "_parse_manipulator_metadata",
        "_parse_photon_metadata",
        "_parse_temperature_metadata",
    ]  # List of metadata parsers to apply

    @property
    def analyser_attributes(self):
        return self._analyser_attributes

    @classmethod
    def _load(cls, fpath, lazy, metadata, quiet):
        # Raise an exception if the manipulator doesn't have at least the core 6 axes required in other data handling
        required_axis_list = ["polar", "tilt", "azi", "x1", "x2", "x3"]
        if not hasattr(cls, "_manipulator_axes") or not all(
            item in cls._manipulator_axes for item in required_axis_list
        ):
            raise ValueError(
                f"ARPES data loaders are expected to subclass a BaseManipulatorDataLoader and include all of "
                f"{required_axis_list} in the `_manipulator_axes` class variable. If the physical axis is missing, "
                f"this can be represented by not passing a name for that axis in the `_manipulator_name_conventions` "
                f"class variable, but the axis is required in `_manipulator_axes` to be able to define normal "
                f"emissions for that axis, required for e.g. ARPES k-conversions."
            )

        # Run the main _load method returning a DataArray
        da = super()._load(fpath, lazy, metadata, quiet)

        # Try to parse to counts/s if possible
        try:
            if da.data.unit == "counts":
                t = (
                    da.metadata.analyser.scan.dwell.to("s")
                    * da.metadata.analyser.scan.sweeps
                )
                da = da / t
        except (ValueError, AttributeError, TypeError):
            pass

        return da

    @classmethod
    def _parse_analyser_metadata(cls, metadata_dict):
        """Parse metadata specific to the analyser."""

        # Get the analyser slit angle from default option for loader if specified and not already in metadata dict
        if not metadata_dict.get("analyser_azi"):
            metadata_dict["analyser_azi"] = cls._analyser_slit_angle

        # Make and populate the analyser metadata model
        arpes_metadata = ARPESMetadataModel(
            analyser=ARPESAnalyserMetadataModel(
                model=metadata_dict.get("analyser_model"),
                slit=ARPESSlitMetadataModel(
                    width=metadata_dict.get("analyser_slit_width"),
                    identifier=metadata_dict.get("analyser_slit_width_identifier"),
                ),
            ),
            scan=ARPESScanMetadataModel(
                eV=metadata_dict.get("analyser_eV"),
                step_size=metadata_dict.get("analyser_step_size"),
                PE=metadata_dict.get("analyser_PE"),
                sweeps=metadata_dict.get("analyser_sweeps"),
                dwell=metadata_dict.get("analyser_dwell"),
                lens_mode=metadata_dict.get("analyser_lens_mode"),
                acquisition_mode=metadata_dict.get("analyser_acquisition_mode"),
                eV_type=metadata_dict.get("analyser_eV_type"),
            ),
            angles=ARPESAnalyserAnglesMetadataModel(
                polar=metadata_dict.get("analyser_polar"),
                tilt=metadata_dict.get("analyser_tilt"),
                azi=metadata_dict.get("analyser_azi"),
            ),
            deflector=ARPESDeflectorMetadataModel(
                parallel=NamedAxisMetadataModel(
                    value=metadata_dict.get("analyser_deflector_parallel"),
                    local_name=cls._analyser_name_conventions.get("deflector_parallel"),
                ),
                perp=NamedAxisMetadataModel(
                    value=metadata_dict.get("analyser_deflector_perp"),
                    local_name=cls._analyser_name_conventions.get("deflector_perp"),
                ),
            ),
        )

        # Return the model, and a list of any metadata that should be warned if missing
        return {"_analyser": arpes_metadata}, [
            f"analyser_{i}" for i in cls._analyser_include_in_metadata_warn
        ]

    # Methods to parse manipulator reference values (normal emissions etc.)
    # Define all the sets of axes/coordinates that go together here
    # Do as separate functions to overwriting individual components in subclasses easier
    @classmethod
    def _manipulator_reference_polar_group(cls, da):
        if not da.metadata.analyser.angles.azi:  # Slit angle 0, along the polar axis
            return {"polar", "deflector_perp"}
        else:  # Assume slit angle 90, perpendicular to the polar axis
            return {"polar", "deflector_parallel", "theta_par"}

    @classmethod
    def _manipulator_reference_tilt_group(cls, da):
        if not da.metadata.analyser.angles.azi:
            return {"tilt", "deflector_parallel", "theta_par"}
        else:
            return {"tilt", "deflector_perp"}

    @classmethod
    def _manipulator_reference_axis_groups(cls, da):
        return {
            axis: (
                getattr(cls, f"_manipulator_reference_{axis}_group")(da)
                if hasattr(cls, f"_manipulator_reference_{axis}_group")
                else {axis}
            )
            for axis in cls._manipulator_axes
        }

    @classmethod
    def _parse_manipulator_reference_default(
        cls, da, set_axis_key, set_axis_value, secondary_axes
    ):
        # Take the given reference angle
        reference_angles = []
        reference_angle_signs = []
        reference_angles.append(-set_axis_value)
        # Check sign convention for the primary axis
        reference_angle_signs.append(
            cls._manipulator_sign_conventions.get(set_axis_key, 1)
            * cls._analyser_sign_conventions.get(set_axis_key, 1)
        )

        # Iterate through the other axis and add values from metadata
        for axis in secondary_axes:
            if axis == "theta_par":
                # Always varying as a secondary axis, so ignore
                value = None
            elif hasattr(da.metadata.manipulator, axis):  # Manipulator axis specified
                value = getattr(da.metadata.manipulator, axis).value
            elif "deflector" in axis:
                value = getattr(
                    da.metadata.analyser.deflector, axis.split("_")[1]
                ).value

            if value:
                reference_angles.append(value)

                # Get relative sign convention to set_axis
                reference_angle_signs.append(
                    cls._manipulator_sign_conventions.get(axis, 1)
                    * cls._analyser_sign_conventions.get(axis, 1)
                    * cls._manipulator_sign_conventions.get(set_axis_key, 1)
                    * cls._analyser_sign_conventions.get(set_axis_key, 1)
                )

        # Handle units
        units = []
        for angle in reference_angles:
            if isinstance(angle, pint.Quantity):
                units.append(angle.units)
            else:
                units.append(None)

        # Check list only contains the same units and/or no units
        unit_set = set(units)
        if len(unit_set) == 1:
            # All the same units (or no units) so can just sum them
            pass
        elif len(unit_set - {None}) == 1:
            unit = unit_set - {None}
            reference_angles = [
                angle if isinstance(angle, pint.Quantity) else angle * unit.pop()
                for angle in reference_angles
            ]
        else:
            raise ValueError(
                "Cannot determine relevant units to use. Please pass the set value with units and/or "
                "ensure metadata has compatible units."
            )

        reference_angle = sum(
            [
                reference_angle * sign
                for reference_angle, sign in zip(
                    reference_angles, reference_angle_signs
                )
            ]
        )

        return reference_angle

    @classmethod
    def _set_units(cls, da, value, primary_axis):
        if isinstance(value, pint.Quantity):
            return value
        # If not already a pint.Quantity, try and parse units to match the primary axis
        primary_value = getattr(da.metadata.manipulator, primary_axis).value
        if isinstance(primary_value, pint.Quantity):
            units = primary_value.units
            return value * units
        return value  # Fall back to returning just the value if units can't otherwise be determined

    @classmethod
    def _parse_manipulator_reference_values(cls, da, specified_values={}):
        """Parse the normal emission angles for the manipulator axes.

        Parameters
        ------------
        da : xarray.DataArray
            The data array to parse the manipulator reference values for.

        specified_values : dict
            A dictionary of values to use to set the manipulator reference values.
        """

        # Separate the axes and other relevant variables into relevant groups
        axis_groups = cls._manipulator_reference_axis_groups(da)

        # Check that only a single setting entry is specified for each axis
        set_item_by_axis_group = {}
        other_axes_in_axis_group = {}
        for axis, group in axis_groups.items():
            specified_angle = [i for i in specified_values if i in group]
            if len(specified_angle) > 1:
                raise ValueError(f"Only one of {group} can be specified.")
            elif len(specified_angle) == 1:
                set_item_by_axis_group[axis] = specified_angle[0]
                other_axes_in_axis_group[axis] = group - set(specified_angle)

        # If the axes have been passed directly, set that as the normal emission
        reference_values = {}
        for axis, set_axis in set_item_by_axis_group.copy().items():
            if axis == set_axis:
                reference_values[axis] = cls._set_units(
                    da, specified_values[set_axis], axis
                )
                set_item_by_axis_group.pop(axis)
                other_axes_in_axis_group.pop(axis)

        # Now iterate over remaining axes and parse the normal emission from the passed parameters
        for axis, set_axis in set_item_by_axis_group.items():
            # Check if a custom parsing method is defined for the axis and if so use it
            if hasattr(cls, f"_parse_manipulator_reference_{axis}"):
                reference_values[axis] = cls._set_units(
                    da,
                    getattr(cls, f"_parse_manipulator_reference_{axis}")(
                        da,
                        set_axis,
                        specified_values[set_axis],
                        other_axes_in_axis_group[axis],
                    ),
                    axis,
                )
            else:  # Use the default
                reference_values[axis] = cls._set_units(
                    da,
                    cls._parse_manipulator_reference_default(
                        da,
                        set_axis,
                        specified_values[set_axis],
                        other_axes_in_axis_group[axis],
                    ),
                    axis,
                )

        return reference_values

    @classmethod
    def _get_angles_Ishida_Shin(cls, da, quiet=False):
        """Convert the angles from friendly manipulator names into the conventions from Y. Ishida and S. Shin,
        Functions to map photoelectron distributions in a variety of setups in angle-resolved photoemission spectroscopy,
        Rev. Sci. Instrum. 89, 043903 (2018), taking care of the sign conventions.

        Parameters
        ----------
        da : xarray.DataArray
            Data for converting to k-space.
        quiet : bool, optional
            Whether to suppress warnings. Defaults to False.

        Returns
        -------
        dict
            Angles of converted angles.
        """

        _manipulator_axes = ["polar", "tilt", "azi"]
        _deflector_axes = ["parallel", "perp"]
        _analyser_axes = ["polar", "tilt", "azi"]

        def _get_nested_attr(obj, attr_path, default=None):
            """Get a nested attribute from an object, with a default value if any attribute in the path
            does not exist."""
            try:
                for attr in attr_path.split("."):
                    obj = getattr(obj, attr)
                return obj
            except AttributeError:
                return default

        def _get_angle_in_rad(da, base_metadata_attr, axis):
            """Get the angle from the core data if exists or else metadata if it exists (return in radians)."""
            if hasattr(da, axis):
                data = getattr(da, axis).data.pint.to("rad").magnitude
            else:
                data = _get_nested_attr(da.metadata, f"{base_metadata_attr}.{axis}")
                if data:
                    data = data.to("rad").magnitude

            return data

        # Manipulator axes
        manipulator_angles = {}
        reference_angles = {}
        missing_values_to_warn = []

        # Iterate over manipulator axes
        for axis in _manipulator_axes:
            sign_convention = cls._manipulator_sign_conventions.get(axis, 1)

            # Get the angle from the data or metadata
            angle = (
                _get_angle_in_rad(da, "manipulator", f"{axis}.value") * sign_convention
            )
            if angle is None:
                missing_values_to_warn.append(f"{axis}")
                angle = 0.0
            manipulator_angles[axis] = angle

            # Get reference values for the manipulator axes
            reference_angle = (
                _get_angle_in_rad(da, "manipulator", f"{axis}.reference_value")
                * sign_convention
            )
            if reference_angle is None:
                missing_values_to_warn.append(f"{axis} <<reference value>>")
                reference_angles[f"{axis}_reference"] = (
                    0.0 if axis != "azi" else -manipulator_angles["azi"]
                )

        # Warn if any values are missing for the manipulator
        if not quiet and missing_values_to_warn:
            analysis_warning(
                f"Warning: Missing angle specifications for the manipulator axes: {set(missing_values_to_warn)}. "
                f"Values ignored for these axes. Ensure to set them in the metadata.",
                "warning",
                "Missing metadata",
            )

        # Deflector axes
        deflector_angles = {}
        for axis in _deflector_axes:
            sign_convention = cls._analyser_sign_conventions.get(f"deflector_{axis}", 1)
            deflector_angles[axis] = (
                _get_angle_in_rad(da, "analyser.deflector", f"{axis}.value")
                * sign_convention
            ) or 0.0

        # Analyser angles
        analyser_angles = {}
        for axis in _analyser_axes:
            sign_convention = cls._analyser_sign_conventions.get(axis, 1)
            analyser_angles[axis] = (
                _get_angle_in_rad(da, "analyser.angles", axis) * sign_convention or 0.0
            )

        # Convert to Ishida and Shin conventions
        theta_par = _get_angle_in_rad(da, "", "theta_par") or 0.0
        if analyser_angles["azi"] == 0:  # Type I
            if np.any(deflector_angles["parallel"]) or np.any(deflector_angles["perp"]):
                type = "Ip"
                alpha = theta_par + deflector_angles["parallel"]
                beta = deflector_angles["perp"]
                chi = manipulator_angles["polar"] + analyser_angles["polar"]
                chi_0 = reference_angles["polar_reference"]
                xi = manipulator_angles["tilt"] + analyser_angles["tilt"]
                xi_0 = reference_angles["tilt_reference"]
            else:
                type = "I"
                alpha = theta_par
                beta = manipulator_angles["polar"] + analyser_angles["polar"]
                beta_0 = reference_angles["polar_reference"]
                xi = manipulator_angles["tilt"] + analyser_angles["tilt"]
                xi_0 = reference_angles["tilt_reference"]
        else:  # Type II
            if np.any(deflector_angles["parallel"]) or np.any(deflector_angles["perp"]):
                type = "IIp"
                alpha = theta_par + deflector_angles["parallel"]
                beta = deflector_angles["perp"]
                chi = manipulator_angles["polar"] + analyser_angles["polar"]
                chi_0 = reference_angles["polar_reference"]
                xi = manipulator_angles["tilt"] + analyser_angles["tilt"]
                xi_0 = reference_angles["tilt_reference"]
            else:
                type = "II"
                alpha = theta_par
                beta = manipulator_angles["tilt"] + analyser_angles["tilt"]
                beta_0 = reference_angles["tilt_reference"]
                xi = manipulator_angles["polar"] + analyser_angles["polar"]
                xi_0 = reference_angles["polar_reference"]

        delta = analyser_angles["azi"]
        delta_0 = reference_angles["azi_reference"]

        return {
            "type": type,
            "alpha": alpha,
            "beta": beta,
            "chi": chi,
            "chi_0": chi_0,
            "xi": xi,
            "xi_0": xi_0,
            "delta": delta,
            "delta_0": delta_0,
        }
