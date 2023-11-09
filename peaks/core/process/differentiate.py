"""Functions used for derivative operations on data.

"""

# Phil King 30/04/2021
# Brendan Edwards 09/11/2023

import xarray as xr
from peaks.core.utils.OOP_method import add_methods


@add_methods(xr.DataArray)
def deriv(data, dims):
    """Function to perform differentiations along the specified dimensions of data contained in a DataArray.

    Parameters
    ------------
    data : xr.DataArray
        The data to differentiate.

    dims : str, list
        Dimension(s) to perform differentiation(s) along. Use a str for a single differentiation, and a list of strs
        for multiple differentiations.

    Returns
    ------------
    deriv_data : xr.DataArray
        The differentiated data.

    Examples
    ------------
    from peaks import *

    disp = load('disp.ibw')

    disp.deriv('eV')  # Differentiate the dispersion along eV

    disp.deriv(['eV', 'eV'])  # Double differentiate the dispersion along eV

    disp.deriv(['eV', 'theta_par'])  # Differentiate the dispersion along eV and then along theta_par

    disp.smooth(eV=0.02).deriv('eV')  # Smooth and then differentiate the dispersion along eV

    """

    # Copy the input data to prevent overwriting issues
    deriv_data = data.copy(deep=True)

    # Ensure dims is of type list
    if not isinstance(dims, list):
        dims = [dims]

    # Save the attributes as these currently get killed by xarray's default differentiate function
    # Note: keep_attrs option still does not exist for differentiate in xarray version 2023.9.0
    attributes = deriv_data.attrs

    # List to store analysis history
    hist_list = []

    # Iterate through specified dimensions and perform differentiations
    for dim in dims:
        if dim not in deriv_data.dims:  # If supplied dimension is not a valid, raise an error
            raise Exception("{dim} is not a valid dimension of the inputted DataArray".format(dim=dim))
        deriv_data = deriv_data.differentiate(dim)  # Perform differentiation
        hist_list.append('Applied differentiation along {dim}'.format(dim=dim))  # Update analysis history list

    # Rewrite attributes
    deriv_data.attrs = attributes

    # Update analysis history
    for hist in hist_list:
        deriv_data.update_hist(hist)

    return deriv_data


def double_deriv():
    pass


def d2k():
    pass


def d2E():
    pass


def dkdE():
    pass


def dEdk():
    pass


def curvature():
    pass


def min_grad():
    pass
