"""
Helper functions and xarray accessors for providing the user interface to metadata
"""

from datatree import register_datatree_accessor
from pydantic import BaseModel
import xarray as xr
import pint
import pint_xarray
from termcolor import colored
import pprint

from peaks.core.fileIO.base_data_classes.base_data_class import BaseDataLoader

ureg = pint_xarray.unit_registry


def display_metadata(da_or_model, mode="ANSI"):
    # Recursive function to display dictionary with colored keys
    colours = ["green", "blue", "red", "yellow"]

    # Recursive function to display dictionary with cycling colors for each indent level
    def display_colored_dict(d, indent_level=0, col_cycle=0):
        indent = "    " * indent_level
        current_color = colours[col_cycle % len(colours)]  # Cycle through colors
        lines = []
        for key, value in d.items():
            if isinstance(value, dict):  # Nested dictionary (recursive case)
                lines.append(f"{indent}{colored(key, current_color)}:")
                lines.extend(
                    display_colored_dict(value, indent_level + 1, col_cycle + 1)
                )
            else:  # Base case (simple value)
                lines.append(f"{indent}{colored(key, current_color)}: {value}")
        return lines

    def display_colored_dict_html(d, indent_level=0, col_cycle=0):
        indent = "&nbsp;" * 4 * indent_level
        colours = ["green", "blue", "red", "yellow"]
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
    except AttributeError as e:
        metadata = da_or_model.dict()
    if mode.upper() == "ANSI":
        return "\n".join(display_colored_dict(metadata))
    elif mode.upper() == "HTML":
        return "<br>".join(display_colored_dict_html(metadata))


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
                "reference_value": getattr(
                    da.metadata.manipulator, axis
                ).reference_value
            }
            for axis in axes
            if getattr(da.metadata.manipulator, axis).reference_value is not None
        }
        current_reference_data["scan_name"] = da.name
        return current_reference_data


@register_datatree_accessor("metadata")
class MetadataDT:
    """Accessor for metadata on xarray DataTrees."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __call__(self, metadata_dict={}, **kwargs):

        metadata_dict.update(kwargs)

        def apply_metadata_to_ds(ds):
            # Apply the metadata to the dataset if it exists at dataset level
            if len(ds.metadata.keys()) > 0:
                for key, value in metadata_dict.items():
                    getattr(ds.metadata, key)(value)
            else:
                # Otherwise map over all dataarrays in the dataset
                for key, value in metadata_dict.items():
                    ds.map(lambda da: getattr(da.metadata, key)(value) or da)
            return ds

        for key, value in metadata_dict.items():
            self._obj.map_over_subtree(apply_metadata_to_ds)

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

        self._obj.map_over_subtree(apply_normal_to_ds)


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
            print(f"No value provided to update the metadata.")
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
