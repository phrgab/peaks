"""
Helper functions and xarray accessors for providing the user interface to metadata
"""

import copy
import pprint

import numpy as np
import pint
import pint_xarray
import xarray as xr
from pydantic import BaseModel

from peaks.core.fileIO.base_data_classes.base_data_class import BaseDataLoader
from peaks.core.utils.datatree_utils import _map_over_dt_containing_single_das

ureg = pint_xarray.unit_registry


def display_metadata(da_or_model, mode="ANSI"):
    # Recursive function to display dictionary with colored keys
    colours = [
        "\x1b[38;2;187;85;0m",  # orange
        "\x1b[38;2;0;90;181m",  # blue
        "\x1b[38;2;212;17;89m",  # magenta
        "\x1b[38;2;0;133;119m",  # green
    ]
    RESET = "\x1b[0m"

    # Recursive function to display dictionary with cycling colors for each indent level
    def display_colored_dict(d, indent_level=0, col_cycle=0):
        indent = "    " * indent_level
        current_color = colours[col_cycle % len(colours)]  # Cycle through colors
        lines = []
        for key, value in d.items():
            if isinstance(value, dict):  # Nested dictionary (recursive case)
                lines.append(f"{indent}{current_color}{key}{RESET}:")
                lines.extend(
                    display_colored_dict(value, indent_level + 1, col_cycle + 1)
                )
            else:  # Base case (simple value)
                lines.append(f"{indent}{current_color}{key}{RESET}: {value}")
        return lines

    def display_colored_dict_html(d, indent_level=0, col_cycle=0):
        indent = "&nbsp;" * 4 * indent_level
        colours = ["#DDCC77", "#88CCEE", "#CC6677", "#44AA99"]
        current_color = colours[col_cycle % len(colours)]  # Cycle through colors
        lines = []
        for key, value in d.items():
            if isinstance(value, dict):  # Nested dictionary (recursive case)
                lines.append(
                    f"{indent}<span style='color:{current_color}'>{key}:</span>"
                )
                lines.extend(
                    display_colored_dict_html(value, indent_level + 1, col_cycle + 1)
                )
            else:  # Base case (simple value)
                lines.append(
                    f"{indent}<span style='color:{current_color}'>{key}:</span> {value}"
                )
        return lines

    # Display the model with colored keys
    try:
        metadata = {
            key.lstrip("_"): value.dict()
            for key, value in da_or_model.attrs.items()
            if key.startswith("_") and key not in ["_analysis_history"]
        }
    except AttributeError:
        metadata = da_or_model if isinstance(da_or_model, dict) else da_or_model.dict()
    if mode.upper() == "ANSI":
        return "\n".join(display_colored_dict(metadata))
    elif mode.upper() == "HTML":
        return "<br>".join(display_colored_dict_html(metadata))


def compare_metadata(da_or_model1, da_or_model2):
    # Function to extract metadata dictionary
    def get_metadata(da_or_model):
        try:
            metadata = {
                key.lstrip("_"): value.dict()
                for key, value in da_or_model.attrs.items()
                if key.startswith("_") and key not in ["_analysis_history"]
            }
        except AttributeError:
            metadata = da_or_model.dict()
        return metadata

    metadata1 = get_metadata(da_or_model1)
    metadata2 = get_metadata(da_or_model2)

    # Helper function to unwrap 'value' keys in dictionaries
    def unwrap_value(val):
        while isinstance(val, dict) and "value" in val and len(val) == 1:
            val = val["value"]
        return val

    # Helper function to compare values considering their types
    def values_equal(val1, val2):
        val1 = unwrap_value(val1)
        val2 = unwrap_value(val2)
        # Handle NoneType
        if val1 is None and val2 is None:
            return True
        if val1 is None or val2 is None:
            return False

        # Handle Pint quantities
        if isinstance(val1, pint.Quantity) and isinstance(val2, pint.Quantity):
            try:
                # Compare magnitudes after converting to common units
                return np.array_equal(
                    val1.to_base_units().magnitude,
                    val2.to_base_units().magnitude,
                )
            except pint.DimensionalityError:
                return False

        # Handle NumPy arrays
        if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            return np.array_equal(val1, val2)

        # Handle lists or tuples
        if isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)):
            if len(val1) != len(val2):
                return False
            return all(values_equal(v1, v2) for v1, v2 in zip(val1, val2, strict=True))

        # Handle dictionaries
        if isinstance(val1, dict) and isinstance(val2, dict):
            return compare_dicts(val1, val2) == {}

        # Default comparison
        return val1 == val2

    # Recursive function to compare two dictionaries
    def compare_dicts(d1, d2):
        differences = {}
        keys = set(d1.keys()).union(d2.keys())
        for key in keys:
            in_d1 = key in d1
            in_d2 = key in d2
            if in_d1 and in_d2:
                val1 = unwrap_value(d1[key])
                val2 = unwrap_value(d2[key])
                if isinstance(val1, dict) and isinstance(val2, dict):
                    sub_diff = compare_dicts(val1, val2)
                    if sub_diff:  # Only include if there is a difference
                        differences[key] = sub_diff
                else:
                    if not values_equal(val1, val2):
                        differences[key] = {"value1": val1, "value2": val2}
            elif in_d1:
                differences[key] = {"value1": d1[key], "value2": None}
            else:
                differences[key] = {"value1": None, "value2": d2[key]}
        return differences

    return compare_dicts(metadata1, metadata2)


@xr.register_dataset_accessor("metadata")
@xr.register_dataarray_accessor("metadata")
class Metadata:
    """Accessor for metadata on xarray DataArrays."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __repr__(self):
        return display_metadata(self._obj)

    def __getattr__(self, name):
        # Return a MetadataItem for the attribute
        if f"_{name}" in self._obj.attrs:
            return MetadataItem(self._obj.attrs[f"_{name}"], path=name, obj=self._obj)
        elif name == "history":  # Add a shortcut to the history attribute
            return self._obj.history
        else:
            raise AttributeError(f"'Metadata' object has no attribute '{name}'")

    def __dir__(self):
        # Include dynamic attributes in the list of available attributes
        return super().__dir__() + list(self.keys()) + ["history"]

    def keys(self):
        return {
            k.lstrip("_"): v
            for k, v in self._obj.attrs.items()
            if k.startswith("_") and k != "_analysis_history"
        }.keys()

    # -------------- Special set methods --------------
    def set_normal_emission(self, norm_values=None, **kwargs):
        """Set the normal emission angles for the scan from a dictionary of key: value pairs to specify
        the requires angles. Like `get_normal_emission_for`, but sets the values rather than returning them.

        Parameters
        ----------
        norm_values : dict, optional
            A dictionary of the normal emission angles to set.

        **kwargs :
            Additional keyword arguments. Alternative way to pass the normal emission angles to set.
            Takes precedence over values supplied in `norm_values`.

        See Also
        --------
        get_normal_emission_for : Get the normal emission angles for the scan corresponding to a dictionary of
        key: value pairs.
        set_normal_emission_like : Set the normal emission angles for the scan to match another scan.
        assign_normal_emission : Assign the normal emission angles to a copy of the data.


        Examples
        --------
        Example usage is as follows::
            import peaks as pks

            # Load data
            FS = pks.load("path/to/data")

            # Set the normal emission angles in FS
            FS.metadata.set_normal_emission(theta_par=12, polar=5)

            # Alternatively, to return a copy of the data with the normal emission angles applied
            FS_with_norm = FS.metadata.assign_normal_emission(theta_par=12, polar=5)


        """

        # Get loc and loader class
        loc = self._obj.metadata.scan.loc
        loader = BaseDataLoader.get_loader(loc)

        if not hasattr(loader, "_parse_manipulator_references"):
            raise NotImplementedError(
                "The loader for this data does not support setting normal emission angles."
            )

        # Get and apply the new reference data to the current dataarray
        normal_emission = self.get_normal_emission_from_values(norm_values, **kwargs)
        normal_emission_dict = {
            k: {"reference_value": v} for k, v in normal_emission.items()
        }  # For passing to the metadata set methods

        self._obj.metadata.manipulator(normal_emission_dict)

    def get_normal_emission_from_values(self, norm_values=None, **kwargs):
        """Get the normal emission angles for the scan corresponding to a dictionary of key: value pairs to
        specify the required angles.

        Parameters
        ----------
        norm_values : dict
            A dictionary of the normal emission angles to set.

        **kwargs :
            Additional keyword arguments. Alternative way to pass the normal emission angles to set.
            Takes precedence over values supplied in `norm_values`.

        See Also
        --------
        set_normal_emission_like : Set the normal emission angles for the scan to match another scan.
        assign_normal_emission : Assign the normal emission angles to a copy of the data.


        Examples
        --------
        Example usage is as follows::
            import peaks as pks

            # Load data
            FS = pks.load("path/to/data")

            # Set the normal emission angles in FS
            FS.metadata.set_normal_emission(theta_par=12, polar=5)

            # Alternatively, to return a copy of the data with the normal emission angles applied
            FS_with_norm = FS.metadata.assign_normal_emission(theta_par=12, polar=5)


        """
        if norm_values is None:
            norm_values = {}
        # Update the normal emission angles with any additional kwargs
        norm_values.update(kwargs)

        # Get relevant loader class
        loc = self._obj.metadata.scan.loc
        loader = BaseDataLoader.get_loader(loc)
        if not hasattr(loader, "_parse_manipulator_references"):
            raise NotImplementedError(
                "The loader for this data does not support setting normal emission angles."
            )

        # If a pint Quantity is passed in tuple format, convert it
        for key, value in norm_values.items():
            if isinstance(value, str):
                norm_values[key] = ureg.Quantity(value)

        # Get the normal emission angles
        normal_emission = loader._parse_manipulator_references(self._obj, norm_values)

        return normal_emission

    def assign_normal_emission(self, norm_values, **kwargs):
        raise NotImplementedError("This method is not yet implemented.")

    def set_normal_emission_like(self, da):
        """Set the normal emission angles for the scan to match another scan.

        Parameters
        ----------
        da : xarray.DataArray
            The data array to match the normal emission angles to.
        """

        # Get any set reference data in da
        current_reference_data = self._get_normal_emission_dict(da)
        # Apply the new reference data to the current dataarray
        self._apply_normal_emission(self._obj, current_reference_data)

    @staticmethod
    def _apply_normal_emission(da, normal_emission_dict):
        """Apply the normal emission angles to the scan. Used as part of the `set_normal_emission_like` method.

        Parameters
        ----------
        da : xarray.DataArray
            The data array to apply the normal emission angles to.

        normal_emission_dict : dict
            A dictionary of the normal emission angles to apply.
        """
        scan_name = normal_emission_dict["scan_name"]
        normal_emission_data = {
            k: v for k, v in normal_emission_dict.items() if k != "scan_name"
        }
        if normal_emission_dict:
            # Apply the new reference data to the current dataarray
            da.metadata.manipulator(normal_emission_data)
            # Patch the analysis history
            current_history = da._analysis_history.records[-1].record
            new_history = current_history.replace(
                "were manually updated from a dictionary",
                f"were set to match the reference values from scan {scan_name}",
            )
            da._analysis_history.records[-1].record = new_history

    @staticmethod
    def _get_normal_emission_dict(da):
        """Get the normal emission angles for the scan.

        Parameters
        ----------
        da : xarray.DataArray
            The data array to get the normal emission angles of.

        Returns
        -------
        dict
            A dictionary of the normal emission angles and also scan name.
        """
        # Get the axis names in da
        axes = list(da._manipulator.dict().keys())
        # Get any set reference data in da
        current_reference_data = {
            axis: {
                "reference_value": getattr(da.metadata.manipulator, axis).reference_value
            }
            for axis in axes
            if getattr(da.metadata.manipulator, axis).reference_value is not None
        }
        current_reference_data["scan_name"] = da.name
        return current_reference_data

    def set_EF_correction(self, EF_correction):
        """Sets the Fermi level correction for a :class:`xarray.DataArray`.

        Parameters
        ----------
        EF_correction : float, int, dict or xarray.Dataset
            Fermi level correction to apply. This should be:

            - a dictionary of the form {'c0': 16.82, 'c1': -0.001, ...} specifying the coefficients of a
            polynomial fit to the Fermi edge.
            - a float or int, this will be taken as a constant shift in energy.
            - an xarray.Dataset containing the fit_result as returned by the `peaks` `.fit_gold` method

          Returns
         -------
         None
             Adds the Fermi level correction to the data attributes.
        """

        # Do some checks on the EF_correction format
        if isinstance(EF_correction, xr.Dataset):
            EF_correction = copy.deepcopy(EF_correction.attrs.get("EF_correction"))
            correction_type = "gold fit result"
        else:
            correction_type = type(EF_correction)
        if not isinstance(EF_correction, (float, int, dict)):
            raise ValueError(
                "EF_correction must be a float, int, dict of fit coefficients or xarray.Dataset containing the fit_result "
                "of the `.fit_gold` function."
            )
        if isinstance(EF_correction, dict):
            expected_keys = [f"c{i}" for i in range(len(EF_correction))]
            if not all(key in EF_correction for key in expected_keys):
                raise ValueError(
                    f"EF_correction dictionary must contain keys {expected_keys} for the polynomial fit."
                )
            for value in EF_correction.values():
                if not isinstance(value, (float, int)):
                    raise ValueError(
                        "EF_correction dictionary must contain only floats or ints as values in the form "
                        "{'c0': 16.82, 'c1': -0.001, ...} specifying the coefficients of a polynomial fit to the Fermi edge."
                    )

        self._obj.metadata.calibration.set(
            "EF_correction", copy.deepcopy(EF_correction), add_history=False
        )
        self._obj.history.add(
            f"EF_correction set to {EF_correction} from a passed {correction_type}."
        )

    def set_EF_correction_like(self, da_to_set_like):
        """Sets the Fermi level correction for a :class:`xarray.DataArray` to be the same as another DataArray.

        Parameters
        ----------
        da_to_set_like : xarray.DataArray
            DataArray to copy the Fermi level correction from.

        Returns
        -------
        None
            Adds the Fermi level correction to the data attributes.
        """
        EF_correction = copy.deepcopy(da_to_set_like.metadata.get_EF_correction())
        self._obj.metadata.calibration.set(
            "EF_correction",
            EF_correction,
            add_history=False,
        )
        self._obj.history.add(
            f"EF_correction set to {EF_correction} to match scan `{da_to_set_like.name}`."
        )

    def get_EF_correction(self):
        """Get the Fermi level correction for a :class:`xarray.DataArray`.

        Returns
        -------
        dict or float or int
            Fermi level correction dictionary.
        """
        EF_correction = self._obj.metadata.calibration.EF_correction
        if isinstance(EF_correction, MetadataItem):
            return EF_correction.__dict__["_data"]
        return EF_correction


@xr.register_datatree_accessor("metadata")
class MetadataDT:
    """Accessor for metadata on xarray DataTrees."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __call__(self, metadata_dict=None, **kwargs):
        if metadata_dict is None:
            metadata_dict = {}
        metadata_dict.update(kwargs)

        def apply_metadata_to_ds(ds):
            # Apply the metadata to the dataset if it exists at dataset level
            if len(ds.metadata.keys()) > 0:
                for key, value in metadata_dict.items():
                    getattr(ds.metadata, key)(value)
            else:
                # Otherwise map over all dataarrays in the dataset
                for key, value in metadata_dict.items():
                    ds.map(
                        lambda da, key=key, value=value: (
                            getattr(da.metadata, key)(value),
                            da,
                        )[1],
                        keep_attrs=False,
                    )

            return ds

        for _key, _value in metadata_dict.items():
            self._obj.map_over_datasets(apply_metadata_to_ds)

        return self

    def set_normal_emission(self):
        raise NotImplementedError("This method is not yet implemented.")

    def set_normal_emission_like(self, da):
        # Get any set reference data in da
        current_reference_data = Metadata._get_normal_emission_dict(da)

        def apply_normal_to_ds(ds):
            # Apply the metadata to the dataset if it exists at dataset level
            if len(ds.metadata.keys()) > 0:
                Metadata._apply_normal_emission(ds, current_reference_data)
            else:
                # Otherwise map over all dataarrays in the dataset
                ds.map(
                    lambda da: Metadata._apply_normal_emission(
                        da, current_reference_data
                    )
                    or da
                )
            return ds

        self._obj.map_over_datasets(apply_normal_to_ds)

    def set_EF_correction(self, EF_correction):
        _map_over_dt_containing_single_das(
            self._obj, lambda da: da.metadata.set_EF_correction(EF_correction)
        )

    def set_EF_correction_like(self, da_to_set_like):
        _map_over_dt_containing_single_das(
            self._obj, lambda da: da.metadata.set_EF_correction_like(da_to_set_like)
        )

    set_EF_correction.__doc__ = Metadata.set_EF_correction.__doc__
    set_EF_correction_like.__doc__ = Metadata.set_EF_correction_like.__doc__


class MetadataItem:
    """Class to handle metadata items."""

    def __init__(self, data, path="", obj=None):
        self._data = data
        self._path = path  # Path to the current attribute
        self._obj = obj  # Reference to the parent object

    def __repr__(self):
        return display_metadata(self._data)

    def __getattr__(self, name):
        if name in (
            "_data",
            "_path",
            "_obj",
            "__class__",
            "__dict__",
            "__weakref__",
            "__module__",
        ):
            return object.__getattribute__(self, name)
        data = self._data
        new_path = f"{self._path}.{name}" if self._path else name
        if isinstance(data, BaseModel):
            if hasattr(data, name):
                attr = getattr(data, name)
                if isinstance(attr, (BaseModel, dict)):
                    return MetadataItem(
                        attr, path=new_path, obj=self._obj
                    )  # Pass self._obj
                else:
                    return attr  # Return the value directly
            else:
                raise AttributeError(
                    f"'{data.__class__.__name__}' object has no attribute '{name}'"
                )
        elif isinstance(data, dict):
            if name in data:
                attr = data[name]
                if isinstance(attr, (BaseModel, dict)):
                    return MetadataItem(
                        attr, path=new_path, obj=self._obj
                    )  # Pass self._obj
                else:
                    return attr  # Return the value directly
            else:
                raise AttributeError(f"Dict has no key '{name}'")
        else:
            raise AttributeError(
                f"Cannot access attribute '{name}' on type {type(data)}"
            )

    def __setattr__(self, name, value):
        if name in ("_data", "_path", "_obj"):
            object.__setattr__(self, name, value)
        else:
            self.set(name, value)

    def set(self, name, value, add_history=True):
        data = self._data
        full_path = f"{self._path}.{name}" if self._path else name
        if isinstance(data, BaseModel):
            if hasattr(data, name):
                current_value = getattr(data, name)
                if isinstance(current_value, pint.Quantity) and not isinstance(
                    value, pint.Quantity
                ):
                    new_value = value * current_value.units
                else:
                    new_value = value
                setattr(data, name, new_value)
                if add_history:
                    self._obj.history.add(
                        f"Metadata attribute '{full_path}' was manually set to {new_value}",
                        ".metadata",
                    )
            else:
                raise AttributeError(
                    f"'{data.__class__.__name__}' object has no attribute '{name}'"
                )
        elif isinstance(data, dict):
            current_value = data.get(name)
            if isinstance(current_value, pint.Quantity) and not isinstance(
                value, pint.Quantity
            ):
                new_value = value * current_value.units
            else:
                new_value = value
            data[name] = new_value
            if add_history:
                self._obj.history.add(
                    f"Metadata attribute '{full_path}' was manually set to {new_value}",
                    ".metadata",
                )
        else:
            raise AttributeError(f"Cannot set attribute '{name}' on type {type(data)}")

    def __call__(self, value=None):
        if value is not None:
            data = self._data
            path = self._path
            if isinstance(data, BaseModel):
                if isinstance(value, dict):
                    self._update_model(data, value, path)
                else:
                    raise ValueError(
                        "Value must be a dictionary to update the pydantic model."
                    )
            elif isinstance(data, dict):
                if isinstance(value, dict):
                    self._update_dict(data, value, path)
                else:
                    raise ValueError("Value must be a dictionary to update the dict.")
            else:
                raise TypeError(
                    "Underlying data is neither a pydantic model nor a dict."
                )
            # After all updates, add a single history entry
            if self._obj is not None and hasattr(self._obj, "history"):
                # Format the passed dictionary for readability
                value_str = pprint.pformat(value)
                self._obj.history.add(
                    f"Metadata attributes for '{path}' were manually updated from a dictionary: {value_str}",
                    ".metadata",
                )
        else:
            print("No value provided to update the metadata.")
        return self

    def _update_model(self, model, updates, path):
        for key, value in updates.items():
            if hasattr(model, key):
                current_value = getattr(model, key)
                full_path = f"{path}.{key}" if path else key
                if isinstance(current_value, BaseModel):
                    if isinstance(value, dict):
                        self._update_model(current_value, value, full_path)
                    else:
                        raise ValueError(
                            f"Expected dict for updating '{key}', got {type(value)}"
                        )
                else:
                    if isinstance(current_value, pint.Quantity) and not isinstance(
                        value, pint.Quantity
                    ):
                        new_value = value * current_value.units
                    else:
                        new_value = value
                    setattr(model, key, new_value)
            else:
                raise AttributeError(
                    f"'{model.__class__.__name__}' object has no attribute '{key}'"
                )

    def _update_dict(self, data_dict, updates, path):
        for key, value in updates.items():
            current_value = data_dict.get(key)
            full_path = f"{path}.{key}" if path else key
            if isinstance(current_value, dict):
                if isinstance(value, dict):
                    self._update_dict(current_value, value, full_path)
                else:
                    raise ValueError(
                        f"Expected dict for updating '{key}', got {type(value)}"
                    )
            else:
                if isinstance(current_value, pint.Quantity) and not isinstance(
                    value, pint.Quantity
                ):
                    new_value = value * current_value.units
                else:
                    new_value = value
                data_dict[key] = new_value

    def __dir__(self):
        data = self._data
        if isinstance(data, BaseModel):
            return super().__dir__() + list(data.__fields__.keys())
        elif isinstance(data, dict):
            return super().__dir__() + list(data.keys())
        else:
            return super().__dir__()
