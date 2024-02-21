"""In-depth fitting models.

"""

# Tommaso Antonelli 13/03/2021
# Brendan Edwards 19/02/2024

import numpy as np
import xarray as xr


def fermi_function():
    pass


def _gauss_conv():
    pass


def _Shirley(data, num_avg=1, offset_start=0, offset_end=0, max_iterations=10):
    """Function to calculate the Shirley background of 1D data.

    Parameters
    ------------
    data : numpy.ndarray, list, xarray.DataArray
        The 1D data (y values) to find the Shirley background of.

    num_avg : int, optional
        The number of points to consider when calculating the average value of the data start and end points. Useful for
        noisy data. Defaults to 1.

    offset_start : float, optional
        The offset to subtract from the data start value. Useful when data range does not completely cover the start
        (left) tail of the peak. Defaults to 0.

    offset_end : float, optional
        The offset to subtract from the data end value. Useful when data range does not completely cover the end (right)
        tail of the peak. Defaults to 0.

    max_iterations : int, optional
        The maximum number of iterations to allow for convergence of Shirley background. Defaults to 10.

    Returns
    ------------
    Shirley_bkg : numpy.ndarray
        The Shirley background of the 1D data.

    Examples
    ------------
    Example usage is as follows::

        from peaks import *

        S2p_XPS = load('XPS1.ibw').DOS()

        # Extract Shirley background of the XPS scan
        S2p_XPS_Shirley_bkg = _Shirley(S2p_XPS)

        # Extract Shirley background of the XPS scan, using 3 points to calculate the average value of the data start
        and end points
        S2p_XPS_Shirley_bkg = _Shirley(S2p_XPS, num_avg=3)

        # Extract Shirley background of the XPS scan, applying an offset to the data start value
        S2p_XPS_Shirley_bkg = _Shirley(S2p_XPS, offset_start=0.3)

    """

    # Ensure data is of type numpy.ndarray
    if isinstance(data, np.ndarray):  # if data is a numpy array
        pass
    elif isinstance(data, xr.core.dataarray.DataArray):  # if data is a DataArray
        data = data.data
    elif isinstance(data, list):  # if data is a list
        data = np.array(data)
    else:
        raise Exception('Inputted data must be a 1D numpy.ndarray, list or xarray.DataArray.')

    # Ensure data is 1D
    if len(data.shape) != 1:
        raise Exception('Inputted data must be a 1D numpy.ndarray, list or xarray.DataArray.')

    # Ensure num_avg and max_iterations are integers
    try:
        num_avg = int(num_avg)
        max_iterations = int(max_iterations)
    except ValueError:
        raise Exception('The inputs num_avg and max_iterations must both be integers')

    # Ensure offset_start and offset_end are floats
    try:
        offset_start = float(offset_start)
        offset_end = float(offset_end)
    except ValueError:
        raise Exception('The inputs offset_start and offset_end must both be floats')

    # Get number of points in data and define tolerance
    num_points = len(data)
    tolerance = 1e-5

    # Determine start and end limits of Shirley background
    y_start = data[0:num_avg].mean() - offset_start
    y_end = data[(num_points - num_avg):num_points].mean() - offset_end

    # Initialise the bkg shape B, where total Shirley bkg is given by Shirley_bkg = y_end + B
    B = np.zeros(data.shape)

    # First B value is equal to y_start - y_end, i.e. Shirley_bkg[0] = y_start as expected
    B[0] = y_start - y_end

    # Define function to determine Shirley bkg
    dk = lambda i: sum(0.5 * (data[i:-1] + data[i + 1:] - 2 * y_end - B[i:-1] + B[i + 1:]))

    # Perform iterative procedure to converge to Shirley bkg, stopping if maximum number of iterations is reached
    num_iterations = 0
    while num_iterations < max_iterations:
        # Calculate new k = (y_start - y_end) / (int_(xl)^(xr) J(x') - y_end - B(x') dx')
        k_sum = sum(0.5 * (data[:-1] + data[1:] - 2 * y_end - B[:-1] + B[1:]))
        k = (y_start - y_end) / k_sum
        # Calculate new B
        y_sum = np.array(list(map(dk, range(0, num_points))))
        new_B = k * y_sum
        # If new_B is close to B (within tolerance), stop the loop
        if sum(abs(new_B - B)) < tolerance:
            B = new_B
            break
        else:
            B = new_B
        num_iterations += 1

    # Raise an error if the maximum allowed number of iterations is exceeded
    if num_iterations >= max_iterations:
        raise Exception('Maximum number of iterations exceeded before convergence of Shirley background was achieved.')

    # Determine Shirley bkg
    Shirley_bkg = y_end + B

    return Shirley_bkg


def create_fit_model():
    pass


def save_fit_model():
    pass
