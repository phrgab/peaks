"""Functions to load NetCDF files.

"""

# Brendan Edwards 16/02/2021
# Brendan Edwards 09/02/2024

import xarray as xr


def _load_NetCDF_data(fname):
    """This function loads data stored in NetCDF files.

    Parameters
    ------------
    fname : str
        Path to the file to be loaded.

    Returns
    ------------
    data : xr.DataArray
        The loaded data.

    Examples
    ------------
    from peaks.core.fileIO.loaders.NetCDF import _load_NetCDF_data

    fname = 'C:/User/Documents/Research/FS1.nc'

    # Extract data from a NetCDF file
    data = _load_NetCDF_data(fname)

    """

    # load NetCDF file as xr.DataArray
    data = xr.open_dataarray(fname)

    # Since NetCDF file attributes cannot be saved as None type, we want to revert any empty strings to type None
    for attr in data.attrs:
        if data.attrs[attr] == '':
            data.attrs[attr] = None

    # Since NetCDF file attributes cannot be saved as a list, we represent the list analysis_history as a string with
    # items seperated by the identifier '<->' when saved in a NetCDF file. We want to revert analysis_history to a list.
    try:  # If data.attrs['analysis_history'] is a string, split it into a list
        data.attrs['analysis_history'] = data.attrs['analysis_history'].split('<->')[1:]
    except AttributeError:  # If data.attrs['analysis_history'] is not a string, do nothing (future proofing)
        pass
    except KeyError:  # If data.attrs['analysis_history'] does not exist, define it with no history
        data.attrs['analysis_history'] = []

    return data
