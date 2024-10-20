"""Functions to load NetCDF files.

"""

import xarray as xr
import json

from peaks.core.options import _BaseDataConventions, _register_location


def _load_NetCDF_data(fname):
    """This function loads data stored in NetCDF files.

    Parameters
    ------------
    fname : str
        Path to the file to be loaded.

    Returns
    ------------
    data : xarray.DataArray
        The loaded data.

    Examples
    ------------
    Example usage is as follows::

        from peaks.core.fileIO.loaders.NetCDF import _load_NetCDF_data

        fname = 'C:/User/Documents/Research/FS1.nc'

        # Extract data from a NetCDF file
        data = _load_NetCDF_data(fname)

    """

    # load NetCDF file as xarray.DataArray
    data = xr.open_dataarray(fname)

    # Since NetCDF file attributes cannot be saved as None type, we want to revert any empty strings to type None
    # (except for analysis_history which if it happens to be empty, we want it to be an empty list)
    for attr in data.attrs:
        if data.attrs[attr] == "":
            if attr != "analysis_history":
                data.attrs[attr] = None
            else:
                data.attrs[attr] = []

    # Since NetCDF file attributes cannot be saved as a list, we represent the list analysis_history as a json string.
    # We want to revert analysis_history to a list.
    try:  # If data.attrs['analysis_history'] is a string, split it into a list
        data.attrs["analysis_history"] = json.loads(data.attrs["analysis_history"])
    except (
        AttributeError
    ):  # If data.attrs['analysis_history'] is not a string, do nothing (future proofing)
        pass
    except (
        KeyError
    ):  # If data.attrs['analysis_history'] does not exist, define it with no history
        data.attrs["analysis_history"] = []

    return data


@_register_location
class _NetCDF(_BaseDataConventions):
    loc_name = "NetCDF"
    loader = _load_NetCDF_data
