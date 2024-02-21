"""Functions to save data as a netCDF file.

"""

# Brendan Edwards 16/02/2021
# Brendan Edwards 09/02/2024

import xarray as xr
from peaks.core.utils.OOP_method import add_methods


@add_methods(xr.DataArray)
def save(data, fname):
    """This function saves data in the :class:`xarray.DataArray` format as a NetCDF file. Note: The NetCDF file format
    is very restricted in what types of DataArray attributes can be saved, allowing only basic str, int,
    float etc. While this function makes an attempt to save the attributes, there may be some loss of metadata.

    Parameters
    ------------
    data : xarray.DataArray
        The data to be saved as a NetCDF file.

    fname : str
        Path to the NetCDF file to be created. Note: the extension of the file must be .nc.

    Examples
    ------------
    Example usage is as follows::

        from peaks import *

        FM = load('FM.zip')

        # Extract a constant energy slice at eV = -1.2 +/- 0.005
        FS1 = FM.FS(E=-1.2, dE=0.01)

        # Save FS1 as a NetCDF file in the current folder directory
        FS1.save('FS1.nc')

        # Save FS1 as a NetCDF file in a defined folder directory
        FS1.save('C:/User/Documents/Research/FS1.nc')

    """

    # Copy the attrs of the inputted xarray.DataArray - needed since when we change the attrs in this function, which
    # will also change the attrs of the inputted xarray.DataArray. We do not want this, so revert changes at the end of
    # the function so that the altered attrs only apply to the saved NetCDF file. This method is less  memory-intensive
    # than just copying the full xarray.DataArray
    data_attrs = data.attrs.copy()

    # Since NetCDF cannot save lists, to save the list analysis_history we convert it to a string where the items are
    # seperated by the identifier '<->'. We can convert this string back to a list when the NetCDF file is loaded.
    analysis_history = ''
    for item in data.attrs['analysis_history']:
        analysis_history += '<->' + item
    data.attrs['analysis_history'] = analysis_history

    # Loop through the attrs to ensure that all are of a valid type to be saved into NetCDF format
    for attr in data.attrs:
        # Since NetCDF cannot save None type, we convert any None items to an empty string
        if data.attrs[attr] is None:
            data.attrs[attr] = ''
        # Convert any other non-compatible attrs to strings
        elif not (isinstance(data.attrs[attr], str) or
                  isinstance(data.attrs[attr], float) or
                  isinstance(data.attrs[attr], int)):
            data.attrs[attr] = str(data.attrs[attr])

    # Save data as a NetCDF file
    data.to_netcdf(fname)

    # Update the attrs of data to their original attrs prior to running this function
    data.attrs = data_attrs
