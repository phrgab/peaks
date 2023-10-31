"""Helper functions for acting on metadata

"""

# Phil King 20/04/2021
# Brendan Edwards 16/10/2023

import xarray as xr
from peaks.core.utils.OOP_method import add_methods


@add_methods(xr.DataArray)
def update_hist(array, hist):
    """Updates the analysis history metadata of the supplied xarray

    Parameters
    ------------
    array : xr.DataArray
        The xarray to which the analysis history is to be updated

    hist : string
        Analysis history to be appended to xarray analysis history metadata


    Returns
    ------------
    array : xr.DataArray
        The xarray with updated analysis history


    Examples
    ------------
    from peaks import *

    my_data = load('my_file.ibw')

    my_data = my_data/2

    my_data = my_data.update_hist('Data divided by 2')  # Update the analysis history metadata of my_data

    """
    
    # If analysis history is not in the xarray metadata
    if 'analysis_history' not in array.attrs:
        array.attrs['analysis_history'] = []
    
    # Update the xarray analysis history (the following is done so that the original xarray is not overwritten)
    analysis_history = []
    for item in array.attrs['analysis_history']:
        analysis_history.append(item)
    analysis_history.append(hist)
    array.attrs['analysis_history'] = analysis_history
    
    return array
