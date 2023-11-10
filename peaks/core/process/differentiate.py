"""Functions used for derivative operations on data.

"""

# Phil King 30/04/2021
# Brendan Edwards 10/11/2023

import xarray as xr
from peaks.core.utils.OOP_method import add_methods


@add_methods(xr.DataArray)
def deriv(data, dims):
    """General function to perform differentiations along the specified dimensions of data contained in a DataArray.

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

    # Differentiate the dispersion along eV
    disp_deriv = disp.deriv('eV')

    # Double differentiate the dispersion along eV
    disp_deriv = disp.deriv(['eV', 'eV'])

    # Differentiate the dispersion along eV and then along theta_par
    disp_deriv = disp.deriv(['eV', 'theta_par'])

    # Smooth and then differentiate the dispersion along eV
    disp_deriv = disp.smooth(eV=0.05).deriv('eV')

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
            raise Exception("{dim} is not a valid dimension of the inputted DataArray.".format(dim=dim))
        deriv_data = deriv_data.differentiate(dim)  # Perform differentiation
        hist_list.append('Applied differentiation along {dim}'.format(dim=dim))  # Update analysis history list

    # Rewrite attributes
    deriv_data.attrs = attributes

    # Update analysis history
    for hist in hist_list:
        deriv_data.update_hist(hist)

    return deriv_data


@add_methods(xr.DataArray)
def d2E(data):
    """Shortcut function to perform a double differentiation along the eV dimension of data contained in a DataArray.

    Parameters
    ------------
    data : xr.DataArray
        The data to differentiate.

    Returns
    ------------
    deriv_data : xr.DataArray
        The differentiated data.

    Examples
    ------------
    from peaks import *

    disp = load('disp.ibw')

    # Double differentiate the dispersion along eV
    disp_deriv = disp.d2E()

    # Smooth and then double differentiate the dispersion along eV
    disp_deriv = disp.smooth(eV=0.05).d2E()

    """

    # If eV is not a valid dimension, raise an error
    if 'eV' not in data.dims:
        raise Exception("eV is not a valid dimension of the inputted DataArray.")

    # Double differentiate the data along eV axis
    deriv_data = data.deriv(['eV', 'eV'])

    return deriv_data


@add_methods(xr.DataArray)
def d2k(data):
    """Shortcut function to perform a double differentiation along the momentum (or angle) dimension of data contained
    in a DataArray.

    Parameters
    ------------
    data : xr.DataArray
        The data to differentiate.

    Returns
    ------------
    deriv_data : xr.DataArray
        The differentiated data.

    Examples
    ------------
    from peaks import *

    disp = load('disp.ibw')

    # Double differentiate the dispersion along the angle dimension
    disp_deriv = disp.d2k()

    # Smooth and then double differentiate the dispersion along the angle dimension
    disp_deriv = disp.smooth(theta_par=0.5).d2k()

    """

    # Work out correct variable for differentiation direction (i.e. is data in angle or k-space)
    coords = list(data.dims)
    if 'eV' in coords:
        coords.remove('eV')
    coord = coords[-1]  # Should always be the last one if data loading is consistent

    # Double differentiate the data along the momentum (or angle) axis
    deriv_data = data.deriv([coord, coord])

    return deriv_data


@add_methods(xr.DataArray)
def dEdk(data):
    """Shortcut function to perform sequential differentiations along the eV then momentum (or angle) dimensions of data
    contained in a DataArray.

    Parameters
    ------------
    data : xr.DataArray
        The data to differentiate.

    Returns
    ------------
    deriv_data : xr.DataArray
        The differentiated data.

    Examples
    ------------
    from peaks import *

    disp = load('disp.ibw')

    # Sequentially differentiate the dispersion along the eV then angle dimension
    disp_deriv = disp.dEdk()

    # Smooth and then sequentially differentiate the dispersion along the eV then angle dimension
    disp_deriv = disp.smooth(eV=0.05, theta_par=0.5).dEdk()

    """

    # Get inputted DataArray dimensions
    coords = list(data.dims)

    # If inputted DataArray is not 2D or higher, raise an error
    if len(coords) < 2:
        raise Exception("Inputted DataArray must be at least 2D.")

    # If eV is not a valid dimension, raise an error
    if 'eV' not in data.dims:
        raise Exception("eV is not a valid dimension of the inputted DataArray.")

    # Work out correct variable for the angle/momentum direction
    coords.remove('eV')
    coord = coords[-1]  # Should always be the last one if data loading is consistent

    # Sequentially differentiate the dispersion along the eV then angle/momentum dimension
    deriv_data = data.deriv(['eV', coord])

    return deriv_data


@add_methods(xr.DataArray)
def dkdE(data):
    """Shortcut function to perform sequential differentiations along the momentum (or angle) then eV dimensions of data
    contained in a DataArray.

    Parameters
    ------------
    data : xr.DataArray
        The data to differentiate.

    Returns
    ------------
    deriv_data : xr.DataArray
        The differentiated data.

    Examples
    ------------
    from peaks import *

    disp = load('disp.ibw')

    # Sequentially differentiate the dispersion along the angle then eV dimension
    disp_deriv = disp.dkdE()

    # Smooth and then sequentially differentiate the dispersion along the angle then eV dimension
    disp_deriv = disp.smooth(theta_par=0.5, eV=0.05).dkdE()

    """

    # Get inputted DataArray dimensions
    coords = list(data.dims)

    # If inputted DataArray is not 2D or higher, raise an error
    if len(coords) < 2:
        raise Exception("Inputted DataArray must be at least 2D.")

    # If eV is not a valid dimension, raise an error
    if 'eV' not in data.dims:
        raise Exception("eV is not a valid dimension of the inputted DataArray.")

    # Work out correct variable for the angle/momentum direction
    coords.remove('eV')
    coord = coords[-1]  # Should always be the last one if data loading is consistent

    # Sequentially differentiate the dispersion along the eV then angle/momentum dimension
    deriv_data = data.deriv([coord, 'eV'])

    return deriv_data


@add_methods(xr.DataArray)
def curvature(data, **parameter_kwargs):
    """Perform 2D curvature analysis of data contained in a DataArray (see Rev. Sci. Instrum.  82, 043712 (2011) for
    analysis procedure).

    Parameters
    ------------
    data : xr.DataArray
        The data to perform curvature analysis on.

    **parameter_kwargs : float
        Curvature analysis free parameters in the format axis=value, e.g. theta_par=0.1. Free parameters must be defined
        for both axes of the data.

    Returns
    ------------
    curv_data : xr.DataArray
        The data following curvature analysis.

    Examples
    ------------
    from peaks import *

    disp = load('disp.ibw')

    # Perform curvature analysis on the dispersion using free parameters for the theta_par and eV axes of 10 and 1.
    curv_data = disp.curvature(theta_par=10, eV=1)

    # Smooth and then perform curvature analysis on the dispersion using free parameters for the
    # theta_par and eV axes of 10 and 1.
    curv_data = disp.smooth(eV=0.03, theta_par=0.3).curvature(theta_par=10, eV=1)

    """

    # Check data is 2D
    if len(data.dims) != 2:
        raise Exception("Function only acts on 2D data.")

    # Check free parameters have been provided for both axes of the data
    for dim in data.dims:
        if dim not in parameter_kwargs:  # Raise error if a dimension of the data is not defined in parameter_kwargs
            raise Exception("Function requires free parameters to be defined for both axes of the data.")

    # Copy the input xarray to prevent overwriting issues
    curv_data = data.copy(deep=True)

    # Save the attributes as these get killed by the curvature analysis summation
    attributes = curv_data.attrs

    # Determine relevant axes and get associated free parameters
    dimx = curv_data.dims[0]
    dimy = curv_data.dims[1]
    Cx = parameter_kwargs[dimx]
    Cy = parameter_kwargs[dimy]

    # Determine various derivatives used in curvature analysis (0 and 1 in following notation represent dimx and dimy)
    dx = curv_data.deriv(dimx)  # d/dx
    d2x = curv_data.deriv([dimx, dimx])  # d^2/dx^2
    dy = curv_data.deriv(dimy)  # d/dy
    d2y = curv_data.deriv([dimy, dimy])  # d^2/dy^2
    dxdy = curv_data.deriv([dimx, dimy])  # d^2/dxdy

    # Perform 2D curvature analysis
    curv_data = (((1 + (Cx * (dx ** 2))) * Cy * d2y) - (2 * Cx * Cy * dx * dy * dxdy) + (
            (1 + (Cy * (dy ** 2))) * Cx * d2x)) / ((1 + (Cx * (dx ** 2)) + (Cy * (dy ** 2))) ** 1.5)

    # Rewrite attributes
    curv_data.attrs = attributes

    # Update analysis history
    hist = '2D curvature analysis performed with coefficients: {dimx}: {Cx}, {dimy}: {Cy}'.format(dimx=dimx, Cx=Cx,
                                                                                                  dimy=dimy, Cy=Cy)
    curv_data.update_hist(hist)

    return curv_data


def min_grad():
    pass
