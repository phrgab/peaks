"""
- Make a set method for updating each metadata entry, which handles units etc. and validation
- Make a general metadata function which parses a dict-like or attr-like syntax to set that metadata on the da or iterating over ds or dt and applying at each relevant node, e.g. `dt.metadata.set('analyser/slit/width', 0.2)` or `dt.metadata.set('analyser.slit.width', 0.3)`
- Should it also accept a fully qualified metadata Model? Or maybe a `dict`?
- Make a specific method for setting normal emissions which can parse relevant data but only when called on a scan, e.g. `da.metadata.set_normal_emission(theta_par=2)` which then parses to the `reference_angle`
- Make a `set_normal_emission_like` method which can then iterate over a tree, e.g. `dt.metadata.set_normal_emission_like(da)`
- When calling `.metadata`, should print the nice nested display [from peaks.core.fileIO.helper_functions]
"""

from datatree import register_datatree_accessor
from pydantic import BaseModel
import xarray as xr
import pint
from termcolor import colored
import pprint


def display_metadata(da_or_model):
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

    # Display the model with colored keys
    try:
        metadata = {
            key: value.dict()
            for key, value in da_or_model.attrs.items()
            if key not in ["analysis_history"]
        }
    except AttributeError:
        metadata = da_or_model.dict()
    return "\n".join(display_colored_dict(metadata))


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
        if name in self._obj.attrs:
            return MetadataItem(self._obj.attrs[name], path=name, obj=self._obj)
        else:
            raise AttributeError(f"'Metadata' object has no attribute '{name}'")

    def __dir__(self):
        # Include dynamic attributes in the list of available attributes
        return super().__dir__() + list(self.keys())

    def keys(self):
        return {
            k: v for k, v in self._obj.attrs.items() if k != "analysis_history"
        }.keys()


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
                self._obj.history.add(
                    f"Metadata attribute '{full_path}' was manually set to {new_value}",
                    ".metadata",
                )
            else:
                raise AttributeError(
                    f"Cannot set attribute '{name}' on type {type(data)}"
                )

    import pprint  # Import pprint at the top of your file

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
