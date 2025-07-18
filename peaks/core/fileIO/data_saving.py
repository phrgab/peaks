"""Functions to save data as a netCDF file."""

import copy
import json
import os

import numpy as np
import pint_xarray  # noqa: F401
import xarray as xr


def _serialise_attrs(attrs):
    """Serialises attributes of a :class:`xarray.DataArray` or similar for saving in
    NetCDF/Zarr files.

    Parameters
    ----------
    attrs : dict
        The attributes dictionary of the :class:`xarray.DataArray` to be saved.

    Returns
    -------
    dict
        The serialised attributes dictionary.
    """

    # Keep track of the metadata models for de-serialisation on loading again
    metadata_models = {}

    # Make data attributes serialisable
    for attr_name, attr in attrs.copy().items():
        try:  # Attrs should generally define json method to convert to/from json string
            attrs[attr_name] = attr.json(by_alias=True)
            metadata_models[attr_name] = (
                f"{attr.__class__.__module__}.{attr.__class__.__name__}"
            )

        except AttributeError:
            # Other type serialisations
            if not isinstance(attr, np.ndarray):
                try:  # Try to convert to a json string
                    attrs[attr_name] = json.dumps(attr)
                    metadata_models[attr_name] = "json"
                except TypeError:  # Fall back to string representation
                    attrs[attr_name] = str(attr)
                    metadata_models[attr_name] = None
    attrs["metadata_models"] = json.dumps(metadata_models)

    return attrs


def _enforce_extension(fpath, required_extension):
    """Ensure that the file path has the required extension, and add it as a default if
    no extension passed.

    Parameters
    ----------
    fpath : str
        The file path to be checked.

    required_extension : str
        The required extension for the file path.

    Returns
    -------
    str : The file path with the required extension.

    Raises
    ------
    ValueError : If the file path has an incorrect extension.
    """

    f_ext = os.path.splitext(fpath)[1]
    if not f_ext:
        return f"{fpath}{required_extension}"
    elif f_ext != required_extension:
        raise ValueError(f"File path must have a {required_extension} extension.")
    else:
        return fpath


def _save_da(data, fpath):
    """Save a :class:`xarray.DataArray` or :class:`xarray.Dataset` as a NetCDF file.

    Parameters
    ----------
    data : xarray.DataArray or xarray.DataSet
        The data to be saved.

    fpath : str
        The path to the file to be created.
    """
    # Ensure the file path has the correct extension
    fpath = _enforce_extension(fpath, ".nc")

    # Copy the original attributes of the data to reset them after saving
    original_attrs = copy.deepcopy(data.attrs)

    # Add a history entry for the data saving to the analysis_history
    data.history.add(f"Data saved as a NetCDF file to {fpath}.")

    # Prepare the data to be saved by serialising the data attributes
    data.attrs.update(_serialise_attrs(data.attrs))

    # Save the data and reset the attributes to their original state
    data.pint.dequantify().to_netcdf(fpath)
    data.attrs = original_attrs


def _save_dt(data, fpath):
    """Save a :class:`xarray.DataTree` as a zarr file.

    Parameters
    ----------
    data : xarray.DataTree
        The data to be saved.

    fpath : str
        The path to the file to be created.
    """
    # Ensure the file path has the correct extension
    fpath = _enforce_extension(fpath, ".zarr")

    def _quantify_da_in_dt(data):
        return data.pint.quantify()

    def _add_history_entry(da):
        da.history.add(f"Data saved as part of a DataTree Zarr file to {fpath}.")
        return da

    def _serialise_da_in_dt(data):
        """Serialises the attributes of a :class:`xarray.DataArray` or similar for saving
        in NetCDF/Zarr files."""
        data.attrs.update(_serialise_attrs(data.attrs))
        return data

    def _parse_nodes_in_dt(ds):
        """Parse the attributes of the current datatree node

        Parameters
        ----------
        ds : xarray.DataArray or xarray.Dataset
            The data to be parsed

        Returns
        -------
        xarray.Dataset : The parsed data
        """
        # Add history entries to the Dataset or DataArray as appropriate
        if "_analysis_history" in ds.attrs:
            ds = _add_history_entry(ds)
        else:
            ds = ds.map(_add_history_entry)
        # Serialise attributes of each DataArray, mapping over the DataTree & Datasets
        ds = _serialise_da_in_dt(ds).map(_serialise_da_in_dt)

        # Dequantify the Dataset
        ds = ds.pint.dequantify()
        return ds

    # Iterate through the DataTree and store the original attributes of each node
    original_attrs = []
    for node in data.subtree:
        if node.attrs:
            original_attrs.append(copy.deepcopy(node.attrs))
        else:
            original_attrs.append({})

    # Map over the DataTree to serialise the attributes and dequantify the DataArrays
    data = data.map_over_datasets(_parse_nodes_in_dt)

    # Save data as a zarr file
    data.to_zarr(fpath)

    # Reset the attributes of the data to their original state
    for node, attrs in zip(data.subtree, original_attrs, strict=True):
        node.attrs = attrs

    # Re-quanitfy the data
    data = data.map_over_datasets(_quantify_da_in_dt)


def save(data, fpath):
    """This function saves data in the :class:`xarray.DataArray` or
    :class:`xarray.DataSet` format as a NetCDF file or a :class:`xarray.DataTree` as a
    zarr file. These formats are restricted in what types of attributes can be saved.
    This function attempts to save peaks metadata attributes in a way that can be parsed
    again when reloading but if custom attributes are used or the file is not saved and
    loaded with the same version of `peaks`, there may be some loss of metadata.

    Parameters
    ------------
    data : xarray.DataArray or xarray.DataSet or xarray.DataTree
        The data to be saved

    fpath : str
        Path to the file to be created. Note: the extension of the file must either be omitted, or must be specified
        as .nc for a :class:`xarray.DataArray` or :class:`xarray.DataSet` and .zarr for a :class:`xarray.DataTree`.

    Examples
    ------------
    Example usage is as follows::

        import peaks as pks

        # Load some data and do some processing
        data = pks.load('my_file.ibw').k_convert()

        # Save the data as a NetCDF file
        pks.save(data, 'my_file.nc')

    """

    if isinstance(data, (xr.DataArray, xr.Dataset)):
        _save_da(data, fpath)
    elif isinstance(data, xr.DataTree):
        return _save_dt(data, fpath)
