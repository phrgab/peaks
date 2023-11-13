"""Functions that apply general operations on data.

"""

# Phil King 17/04/2021
# Brendan Edwards 09/11/2023

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter
from peaks.core.utils.OOP_method import add_methods
from peaks.core.utils.misc import analysis_warning


@add_methods(xr.DataArray)
def norm(dispersion, *args, **kwargs):
    # Temp function
    norm_dispersion = dispersion.copy(deep=True)
    norm_dispersion = norm_dispersion / float(norm_dispersion.max())
    return norm_dispersion


def _norm_wave():
    pass


def bgs():
    pass


@add_methods(xr.DataArray)
def smooth(data, **smoothing_kwargs):
    """Function to smooth data by applying a Gaussian smoothing operator.

    Parameters
    ------------
    data : xr.DataArray
        The data to smooth.

    **smoothing_kwargs : float
        Axes to smooth over in the format axis=FWHM, where FWHM is the relevant FWHM of the Gaussian for convolution
        in this direction, e.g. eV=0.1.

    Returns
    ------------
    smoothed_data : xr.DataArray
        The smoothed data.

    Examples
    ------------
    from peaks import *

    disp = load('disp.ibw')

    EDC1 = disp.EDC()

    # Smooth the dispersion by Gaussian filters with FWHMs for the theta_par and eV axes of 0.5 deg and 0.2 eV
    disp_smooth = disp.smooth(theta_par=0.5, eV=0.2)

    # Smooth the EDC by a Gaussian filter with FWHM for the eV axis of 0.2 eV
    EDC1_smooth = EDC1.smooth(eV=0.1)

    """

    # Check that some axes to smooth over were passed
    if len(smoothing_kwargs) == 0:
        raise Exception("Function requires axes to be smoothed over to be defined.")

    # Copy the input data to prevent overwriting issues
    smoothed_data = data.copy(deep=True)

    # Define the start of the analysis history update string
    hist = 'Data smoothed with the following FWHMs along given axes: '

    # Make the sigma array (used to store standard deviations) as zeros, then update using supplied definitions
    sigma = np.zeros(len(data.dims))

    # Iterate through coordinates and determine the standard deviations in pixels from the DataArray axis scaling
    for count, value in enumerate(data.dims):
        if value not in smoothing_kwargs:  # No broadening for this dimension
            sigma[count] = 0
        else:  # Determine broadening in pixels from axis scaling
            delta = abs(data[value].data[1] - data[value].data[0])  # Pixel size in relevant units for axis
            # Must convert smoothing factor from FWHM to standard deviation (in pixels)
            sigma_px = np.round(smoothing_kwargs[value] / delta) / 2.35482005  # Coordinate sigma in pixels
            sigma[count] = sigma_px  # Update standard deviations array
            hist += str(value) + ': ' + str(smoothing_kwargs[value]) + ', '  # Update analysis history string
            smoothing_kwargs.pop(value)  # Remove this axis from smoothing_kwargs for consistency check later

    # Extract the raw DataArray data
    array = smoothed_data.data

    # Apply gaussian convolution to raw DataArray data
    array_sm = gaussian_filter(array, sigma)

    # Update DataArray with smoothed data
    smoothed_data.data = array_sm

    # Check that all supplied smoothing_kwargs are used, giving a warning if not (this occurs if an axis does not exist)
    if len(smoothing_kwargs) != 0:
        analysis_warning(
            'Not all supplied axes are coordinates of DataArray: {coords} have been ignored.'.format(
                coords=str(smoothing_kwargs)), title='Analysis info', warn_type='danger')

    # Update the analysis history
    smoothed_data.update_hist(hist[:-2])

    return smoothed_data


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
