import pint
import pint_xarray
import numpy as np
import copy

from h5py.h5r import get_region
from sphinx.ext.intersphinx import missing_reference

from peaks.core.fileIO.base_data_classes.base_manipulator_class import (
    BaseManipulatorDataLoader,
)
from peaks.core.fileIO.base_data_classes.base_temperature_class import (
    BaseTemperatureDataLoader,
)
from peaks.core.fileIO.base_data_classes.base_photon_source_classes import (
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
    def _load(cls, fpath, lazy, metadata, quiet, **kwargs):
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
        da = super()._load(fpath, lazy, metadata, quiet, **kwargs)

        # Try to parse to counts/s if possible
        try:
            if "count" in str(da.pint.units):
                t = (
                    da.metadata.analyser.scan.dwell.to("s")
                    * da.metadata.analyser.scan.sweeps
                )
                da /= t
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
    @classmethod
    def _get_sign_convention(cls, axis_key):
        """
        Retrieve the sign convention for a given axis.
        """
        if "deflector" in axis_key or "theta_par" in axis_key or "ana_" in axis_key:
            return cls._analyser_sign_conventions.get(axis_key, 1)
        else:
            return cls._manipulator_sign_conventions.get(axis_key, 1)

    @classmethod
    def _group_axes(cls, da):
        """
        Group axes based on the slit orientation.
        """
        if da.metadata.analyser.angles.azi == 0:
            # Slit angle 0, along the polar axis
            polar_group = {"polar", "deflector_perp", "ana_polar"}
            tilt_group = {"tilt", "deflector_parallel", "theta_par", "ana_tilt"}
        else:
            # Slit angle 90, perpendicular to the polar axis
            polar_group = {"polar", "deflector_parallel", "theta_par", "ana_polar"}
            tilt_group = {"tilt", "deflector_perp", "ana_tilt"}
        return {"polar": polar_group, "tilt": tilt_group, "azi": {"azi"}}

    @classmethod
    def _parse_reference_value(cls, da, axis_key, value, axis_group):
        """
        Parse the manipulator reference value for a given axis.
        """

        if axis_key in ["polar", "tilt", "azi"] and axis_key not in da.dims:
            # The axis is a core manipulator axis and is not a scannable, so this should be the actual normal emission
            return value

        reference_angles = []
        reference_signs = []

        # Primary axis value (user-specified)
        reference_angles.append(
            value if axis_key in ["polar", "tilt", "azi"] else -value
        )  # If one of the primary manipulator dims, no sign change is needed
        reference_signs.append(cls._get_sign_convention(axis_key))

        # Secondary axes (from metadata)
        for axis in axis_group:
            if axis == axis_key:
                continue  # Skip the primary axis
            angle_value = cls._get_axis_value(da, axis)
            if angle_value is not None:
                reference_angles.append(angle_value)
                reference_signs.append(cls._get_sign_convention(axis))

        # Convert angles to consistent units and sum
        total_reference_angle = cls._sum_angles(reference_angles, reference_signs)
        return total_reference_angle

    @classmethod
    def _get_axis_value(cls, da, axis):
        """
        Retrieve the value of an axis from the data array or metadata.
        """
        if hasattr(da, axis):
            data = getattr(da, axis).pint.to("deg")
            return (
                data.data
                if isinstance(data.data, pint.Quantity)
                else data.data * ureg("deg")
            )

        else:
            # Attempt to get from metadata
            if "deflector" in axis:
                return getattr(da.metadata.analyser.deflector, axis.split("_")[1]).value
            elif "ana_" in axis:
                return getattr(da.metadata.analyser.angles, axis.split("_")[1])
            elif hasattr(da.metadata.manipulator, axis):
                return getattr(da.metadata.manipulator, axis).value
            else:
                return None  # Axis value not found

    @classmethod
    def _sum_angles(cls, angles, signs):
        """
        Sum angles with appropriate sign conventions.
        """
        total_angle = 0.0 * ureg("deg")
        for angle, sign in zip(angles, signs):
            try:
                angle_value = angle.to("deg")
            except AttributeError:
                angle_value = angle * ureg("deg")
            total_angle += angle_value * sign
        return total_angle

    @classmethod
    def _parse_manipulator_references(cls, da, specified_values):
        """
        Parse manipulator reference values based on specified parameters.
        """
        axis_groups = cls._group_axes(da)
        reference_values = {}

        for axis, group in axis_groups.items():
            # Check if any axis in the group is specified
            specified_axis = None
            for ax in group:
                if ax in specified_values:
                    specified_axis = ax
                    break
            if specified_axis:
                value = specified_values[specified_axis]
                # Parse the reference value
                total_reference = cls._parse_reference_value(
                    da, specified_axis, value, group
                )

                # Apply final sign convention
                final_value = total_reference * cls._get_sign_convention(axis)
                try:
                    reference_values[axis] = final_value.to("deg")
                except AttributeError:
                    reference_values[axis] = final_value * ureg("deg")
        return reference_values

    @classmethod
    def _get_angles_Ishida_Shin(cls, da, quiet=False):
        """
        Convert angles to the conventions of Ishida and Shin.
        """

        missing_values = []
        missing_references = []

        # Helper functions
        def get_angle(da, axis, default=0.0 * ureg("rad")):
            """Get the angle value for a given axis.

            If the value is missing, return the default value

            Parameters
            ----------
            da : xarray.DataArray
                The data array.
            axis : str
                The axis name.
            default : pint.Quantity, optional
                The default value to return if the axis value is missing.

            Returns
            -------
            pint.Quantity
                The angle value.
            """
            value = cls._get_axis_value(da, axis)
            if value is None:
                if not quiet:
                    print(
                        f"Warning: Missing value for axis '{axis}'. Using default {default}."
                    )
                missing_values.append(axis)
                return default
            else:
                return value.to("rad") * cls._get_sign_convention(axis)

        def get_reference_angle(da, axis, default=0.0 * ureg("rad")):
            """Get the reference angle value for a given axis.

            If the value is missing, return the default value.

            Parameters
            ----------
            da : xarray.DataArray
                The data array.
            axis : str
                The axis name.
            default : pint.Quantity, optional
                The default value to return if the axis value is missing.

            Returns
            -------
            pint.Quantity
                The angle value.

            """
            value = getattr(da.metadata.manipulator, axis).reference_value
            if value is None:
                if not quiet:
                    print(
                        f"Warning: Missing reference value for axis '{axis}'. Using default {default}."
                    )
                missing_references.append(axis)
                return default
            else:
                return value.to("rad") * cls._get_sign_convention(axis)

        # Retrieve manipulator angles
        manipulator_angles = {
            axis: get_angle(da, axis) for axis in ["polar", "tilt", "azi"]
        }

        # Retrieve analyser angles
        analyser_angles = {
            axis: get_angle(da, f"ana_{axis}") for axis in ["polar", "tilt", "azi"]
        }

        # Retrieve deflector angles
        deflector_angles = {
            axis: get_angle(da, f"deflector_{axis}") for axis in ["parallel", "perp"]
        }

        # Determine the type based on analyser azi angle
        analyser_azi = analyser_angles.get("azi", 0)
        if analyser_azi == 0:
            if any(deflector_angles.values()):
                type_ = "Ip"
                alpha = get_angle(da, "theta_par") + deflector_angles["parallel"]
                beta = deflector_angles["perp"]
                chi = manipulator_angles["polar"] + analyser_angles["polar"]
                xi = manipulator_angles["tilt"] + analyser_angles["tilt"]
                beta_0 = None
                chi_0 = get_reference_angle(da, "polar")
                xi_0 = get_reference_angle(da, "tilt")
            else:
                type_ = "I"
                alpha = get_angle(da, "theta_par")
                beta = manipulator_angles["polar"] + analyser_angles["polar"]
                chi = None
                xi = manipulator_angles["tilt"] + analyser_angles["tilt"]
                beta_0 = get_reference_angle(da, "polar")
                chi_0 = None
                xi_0 = get_reference_angle(da, "tilt")
        else:
            if any(deflector_angles.values()):
                type_ = "IIp"
                alpha = get_angle(da, "theta_par") + deflector_angles["parallel"]
                beta = deflector_angles["perp"]
                chi = manipulator_angles["polar"] + analyser_angles["polar"]
                xi = manipulator_angles["tilt"] + analyser_angles["tilt"]
                beta_0 = None
                chi_0 = get_reference_angle(da, "polar")
                xi_0 = get_reference_angle(da, "tilt")
            else:
                type_ = "II"
                alpha = get_angle(da, "theta_par")
                beta = manipulator_angles["tilt"] + analyser_angles["tilt"]
                chi = None
                xi = manipulator_angles["polar"] + analyser_angles["polar"]
                beta_0 = get_reference_angle(da, "tilt")
                chi_0 = None
                xi_0 = get_reference_angle(da, "polar")

        delta = manipulator_angles["azi"]
        delta_0 = get_reference_angle(da, "azi", default=delta)

        angle_dict = {
            "type": type_,
            "alpha": alpha,
            "beta": beta,
            "beta_0": beta_0,
            "chi": chi,
            "chi_0": chi_0,
            "xi": xi,
            "xi_0": xi_0,
            "delta": delta,
            "delta_0": delta_0,
        }

        return angle_dict
