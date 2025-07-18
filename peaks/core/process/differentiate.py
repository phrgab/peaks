"""Functions used for derivative operations on data."""

import numpy as np
from scipy.ndimage import gaussian_gradient_magnitude

from peaks.core.utils.misc import analysis_warning


def deriv(data, dims):
    """General function to perform differentiations along the specified dimensions of data.

    Parameters
    ------------
    data : xarray.DataArray
        The data to differentiate.

    dims : str, list
        Dimension(s) to perform differentiation(s) along. Use a str for a single differentiation, and a list of strs
        for multiple differentiations.

    Returns
    ------------
    deriv_data : xarray.DataArray
        The differentiated data.

    Examples
    ------------
    Example usage is as follows::

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
        if (
            dim not in deriv_data.dims
        ):  # If supplied dimension is not a valid, raise an error
            raise Exception(
                "{dim} is not a valid dimension of the inputted DataArray.".format(
                    dim=dim
                )
            )
        deriv_data = deriv_data.differentiate(dim)  # Perform differentiation
        hist_list.append(
            "Applied differentiation along {dim}".format(dim=dim)
        )  # Update analysis history list

    # Rewrite attributes
    deriv_data.attrs = attributes

    # Update analysis history
    for hist in hist_list:
        deriv_data.history.add(hist)

    return deriv_data


def d2E(data):
    """Shortcut function to perform a double differentiation along the eV dimension of data.

    Parameters
    ------------
    data : xarray.DataArray
        The data to differentiate.

    Returns
    ------------
    deriv_data : xarray.DataArray
        The differentiated data.

    Examples
    ------------
    Example usage is as follows::

        from peaks import *

        disp = load('disp.ibw')

        # Double differentiate the dispersion along eV
        disp_deriv = disp.d2E()

        # Smooth and then double differentiate the dispersion along eV
        disp_deriv = disp.smooth(eV=0.05).d2E()

    """

    # If eV is not a valid dimension, raise an error
    if "eV" not in data.dims:
        raise Exception("eV is not a valid dimension of the inputted DataArray.")

    # Double differentiate the data along eV axis
    deriv_data = data.deriv(["eV", "eV"])

    return deriv_data


def d2k(data):
    """Shortcut function to perform a double differentiation along the momentum (or angle) dimension of data.

    Parameters
    ------------
    data : xarray.DataArray
        The data to differentiate.

    Returns
    ------------
    deriv_data : xarray.DataArray
        The differentiated data.

    Examples
    ------------
    Example usage is as follows::

        from peaks import *

        disp = load('disp.ibw')

        # Double differentiate the dispersion along the angle dimension
        disp_deriv = disp.d2k()

        # Smooth and then double differentiate the dispersion along the angle dimension
        disp_deriv = disp.smooth(theta_par=0.5).d2k()

    """

    # Work out correct variable for differentiation direction (i.e. is data in angle or k-space)
    coords = list(data.dims)
    if "eV" in coords:
        coords.remove("eV")
    coord = coords[-1]  # Should always be the last one if data loading is consistent

    # Double differentiate the data along the momentum (or angle) axis
    deriv_data = data.deriv([coord, coord])

    return deriv_data


def dEdk(data):
    """Shortcut function to perform sequential differentiations along the eV then momentum (or angle) dimensions of
    data.

    Parameters
    ------------
    data : xarray.DataArray
        The data to differentiate.

    Returns
    ------------
    deriv_data : xarray.DataArray
        The differentiated data.

    Examples
    ------------
    Example usage is as follows::

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
    if "eV" not in data.dims:
        raise Exception("eV is not a valid dimension of the inputted DataArray.")

    # Work out correct variable for the angle/momentum direction
    coords.remove("eV")
    coord = coords[-1]  # Should always be the last one if data loading is consistent

    # Sequentially differentiate the dispersion along the eV then angle/momentum dimension
    deriv_data = data.deriv(["eV", coord])

    return deriv_data


def dkdE(data):
    """Shortcut function to perform sequential differentiations along the momentum (or angle) then eV dimensions of
    data.

    Parameters
    ------------
    data : xarray.DataArray
        The data to differentiate.

    Returns
    ------------
    deriv_data : xarray.DataArray
        The differentiated data.

    Examples
    ------------
    Example usage is as follows::

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
    if "eV" not in data.dims:
        raise Exception("eV is not a valid dimension of the inputted DataArray.")

    # Work out correct variable for the angle/momentum direction
    coords.remove("eV")
    coord = coords[-1]  # Should always be the last one if data loading is consistent

    # Sequentially differentiate the dispersion along the eV then angle/momentum dimension
    deriv_data = data.deriv([coord, "eV"])

    return deriv_data


def curvature(data, **parameter_kwargs):
    """Perform 2D curvature analysis of data (see Rev. Sci. Instrum.  82, 043712 (2011) for analysis procedure).

    Parameters
    ------------
    data : xarray.DataArray
        The data to perform curvature analysis on.

    **parameter_kwargs : float
        Curvature analysis free parameters in the format axis=value, e.g. theta_par=0.1. Free parameters must be defined
        for both axes of the data. Set a given axis free parameter to 0 to obtain 1D curvature analysis for the other
        axis.

    Returns
    ------------
    curv_data : xarray.DataArray
        The data following curvature analysis.

    Examples
    ------------
    Example usage is as follows::

        from peaks import *

        disp = load('disp.ibw')

        # Perform curvature analysis on the dispersion using free parameters for the theta_par and eV axes of 10 and 1
        disp_curv = disp.curvature(theta_par=10, eV=1)

        # Smooth and then perform curvature analysis on the dispersion using free parameters for the
        # theta_par and eV axes of 10 and 1
        disp_curv = disp.smooth(eV=0.03, theta_par=0.3).curvature(theta_par=10, eV=1)

    """

    # Check data is 2D
    if len(data.dims) != 2:
        raise Exception("Function only acts on 2D data.")

    # Check free parameters have been provided for both axes of the data
    for dim in data.dims:
        if (
            dim not in parameter_kwargs
        ):  # Raise error if a dimension of the data is not defined in parameter_kwargs
            raise Exception(
                "Function requires free parameters to be defined for both axes of the data."
            )

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
    curv_data = (
        ((1 + (Cx * (dx**2))) * Cy * d2y)
        - (2 * Cx * Cy * dx * dy * dxdy)
        + ((1 + (Cy * (dy**2))) * Cx * d2x)
    ) / ((1 + (Cx * (dx**2)) + (Cy * (dy**2))) ** 1.5)

    # Rewrite attributes
    curv_data.attrs = attributes

    # Update analysis history
    hist = "2D curvature analysis performed with coefficients: {dimx}: {Cx}, {dimy}: {Cy}".format(
        dimx=dimx, Cx=Cx, dimy=dimy, Cy=Cy
    )
    curv_data.history.add(hist)

    return curv_data


def min_gradient(data, **smoothing_kwargs):
    """Perform minimum gradient analysis of data, using Gaussian filtering (see Rev. Sci. Instrum 88 (2017) 07390 for
    analysis procedure).

    Parameters
    ------------
    data : xarray.DataArray
        The data to perform minimum gradient analysis on.

    **smoothing_kwargs : float
        Axes to smooth over in the format axis=FWHM, where FWHM is the relevant FWHM of the Gaussian for convolution
        in this direction, e.g. eV=0.1. Defaults to a single pixel. Note: The use of these broadening terms is
        inequivalent to the use in the smooth function,
        i.e. data.smooth(eV=0.01).min_grad(eV=0) != data.min_grad(eV=0.01).

    Returns
    ------------
    grad_data : xarray.DataArray
        The data following minimum gradient analysis.

    Examples
    ------------
    Example usage is as follows::

        from peaks import *

        disp = load('disp.ibw')

        FM = load('FM.zip')

        # Perform minimum gradient analysis on the dispersion using Gaussian filters with FWHMs for the theta_par and eV
        # axes of 0.2 deg and 0.05 eV
        disp_grad = disp.min_gradient(theta_par=0.2, eV=0.05)

        # Perform minimum gradient analysis on the Fermi map using Gaussian filters with FWHMs for the ana_polar,
        # theta_par and eV axes of 0.2 deg, 0.2 deg and 0.05 eV
        FM_grad = FM.min_gradient(ana_polar=0.2, theta_par=0.5, eV=0.05)

    """

    # Check data is not 1D
    if len(data.dims) < 2:
        raise Exception("Function cannot act on 1D data.")

    # Copy the input data to prevent overwriting issues
    grad_data = data.copy(deep=True)

    # Define the start of the analysis history update string
    hist = "Minimum gradient analysis performed using Gaussian filters with the following FWHM along given axes: "

    # Make the sigma array (used to store standard deviations) as zeros, then update using supplied definitions
    sigma = np.zeros(len(data.dims))

    # Iterate through coordinates and determine the standard deviations in pixels from the DataArray axis scaling
    for count, value in enumerate(data.dims):
        if (
            value not in smoothing_kwargs
        ):  # No broadening given for this dimension, so set to default of a single pixel
            sigma[count] = 1 / 2.35482005
            hist += str(value) + ": 1 pixel, "  # Update analysis history string
            # Display warning that the default value has been assumed
            analysis_warning(
                "No broadening parameter supplied for {dim} dimension. Set to default of a single pixel.".format(
                    dim=value
                ),
                title="Analysis info",
                warn_type="danger",
            )
        else:  # Determine broadening in pixels from axis scaling
            delta = abs(
                data[value].data[1] - data[value].data[0]
            )  # Pixel size in relevant units for axis
            # Must convert smoothing factor from FWHM to standard deviation (in pixels)
            sigma_px = (
                np.round(smoothing_kwargs[value] / delta) / 2.35482005
            )  # Coordinate sigma in pixels
            sigma[count] = sigma_px  # Update standard deviations array
            hist += (
                str(value) + ": " + str(smoothing_kwargs[value]) + ", "
            )  # Update analysis history string
            smoothing_kwargs.pop(
                value
            )  # Remove this axis from smoothing_kwargs for consistency check later

    # Extract the raw DataArray data
    array = grad_data.data

    # Apply gradient magnitude to raw DataArray data
    array_sm = gaussian_gradient_magnitude(array, sigma)

    # Extract the renormalised gradient modulus map
    grad_data /= array_sm

    # Check that all supplied smoothing_kwargs are used, giving a warning if not (this occurs if an axis does not exist)
    if len(smoothing_kwargs) != 0:
        analysis_warning(
            "Not all supplied axes are coordinates of DataArray: {coords} have been ignored.".format(
                coords=str(smoothing_kwargs)
            ),
            title="Analysis info",
            warn_type="danger",
        )

    # Update the analysis history
    grad_data.history.add(hist[:-2])

    return grad_data
