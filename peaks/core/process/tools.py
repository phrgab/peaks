"""Functions that apply general operations on data.

"""

# Phil King 17/04/2021
# Brendan Edwards 01/11/2023

import xarray as xr
from peaks.core.utils.OOP_method import add_methods


@add_methods(xr.DataArray)
def norm(dispersion, *args, **kwargs):
    # Temp function
    norm_dispersion = dispersion.copy(deep=True)
    norm_dispersion = norm_dispersion/float(norm_dispersion.max())
    return norm_dispersion

