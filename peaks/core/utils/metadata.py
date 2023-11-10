"""Helper functions for acting on metadata.

"""

# Phil King 20/04/2021
# Brendan Edwards 16/10/2023

import xarray as xr
from peaks.core.utils.OOP_method import add_methods


@add_methods(xr.DataArray)
def update_hist(data, hist):
    """Updates the analysis history metadata of the supplied DataArray.

    Parameters
    ------------
    data : xr.DataArray
        The DataArray to which the analysis history is to be updated.

    hist : string
        Analysis history to be appended to DataArray analysis history metadata.


    Returns
    ------------
    data : xr.DataArray
        The DataArray with updated analysis history.


    Examples
    ------------
    from peaks import *

    disp = load('disp.ibw')

    disp = disp/2

    # Update the analysis history metadata of the dispersion
    disp = disp.update_hist('Dispersion data divided by 2')

    """
    
    # If analysis history is not in the DataArray attributes, define it
    if 'analysis_history' not in data.attrs:
        data.attrs['analysis_history'] = []
    
    # Update the DataArray analysis history (the following is done so that the original DataArray is not overwritten)
    analysis_history = []
    for item in data.attrs['analysis_history']:
        analysis_history.append(item)
    analysis_history.append(hist)
    data.attrs['analysis_history'] = analysis_history
    
    return data


@add_methods(xr.DataArray)
def _set_normals(data, **kwargs):
    """Function to set normal emissions into the attributes of DataArrays based on arguments passed in kwargs.

    Parameters
    ------------
    data: xr.DataArray
        The DataArray to write normal emission angles to.

    **kwargs : float (optional)
        Attributes to be overwritten in the DataArray in the format attr=float. Valid for polar, tilt, azi, norm_polar,
        norm_tilt and norm_azi. All other kwargs are ignored.

    Examples
    ------------
    from peaks import *

    disp = load('disp.ibw')

    # Sets the attributes of the dispersion
    disp._set_normals(polar=0, tilt=0, azi=0, norm_polar=0, norm_tilt=0, norm_azi=0)

    """

    if 'polar' in kwargs:
        data.attrs['polar'] = kwargs['polar']  # Set polar attribute of metadata
    if 'tilt' in kwargs:
        data.attrs['tilt'] = kwargs['tilt']  # Set tilt attribute of metadata
    if 'azi' in kwargs:
        data.attrs['azi'] = kwargs['azi']  # Set azi attribute of metadata
    if 'norm_polar' in kwargs:
        data.attrs['norm_polar'] = kwargs['norm_polar']  # Set norm_polar attribute of metadata
    if 'norm_tilt' in kwargs:
        data.attrs['norm_tilt'] = kwargs['norm_tilt']  # Set norm_tilt attribute of metadata
    if 'norm_azi' in kwargs:
        data.attrs['norm_azi'] = kwargs['norm_azi']  # Set norm_azi attribute of metadata
