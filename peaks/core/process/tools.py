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


def _norm_wave():
    pass


def bgs():
    pass


def smooth():
    pass


def rotate():
    pass


def sym_nfold():
    pass


def sym_EF():
    pass


def sym_1d():
    pass


def sum_spec():
    pass


def interpolation():
    pass


def merge_data():
    pass


def _merge_two_xarrays():
    pass
