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


def sum_data(data):
    """Function to sum two or more DataArrays together, maintaining the metadata. If the metadata of the DataArrays
    differ, that of the first inputted DataArray will be used. If the coordinate grids of the DataArrays differ, all
    DataArrays will be interpolated onto the coordinate grid of the first inputted DataArray.

    Parameters
    ------------
    data : list
        Any number of xr.DataArrays to sum together.

    Returns
    ------------
    summed_data : xr.DataArray
        The single summed DataArray.

    Examples
    ------------
    from peaks import *

    disp1 = load('disp1.ibw')
    disp2 = load('disp2.ibw')

    # Extract the sum of the dispersions disp1 and disp2
    disp_sum = sum_data([disp1, disp2])

    disp1_proc = disp1.proc(EF_correction=55.67, norm_polar = 1.56)
    disp2_proc = disp2.proc(EF_correction=55.65, norm_polar = 1.43)

    # Extract the sum of the processed dispersions disp1_proc and disp2_proc, with different Fermi level corrections
    # and normal emissions
    disp_proc_sum = sum_data([disp1_proc, disp2_proc])

    """

    # Get the number of inputted DataArrays
    num_data = len(data)

    # Copy the first DataArray
    data_0_data = data[0].copy(deep=True)

    # Copy the attributes of the first DataArray
    data_0_attrs = data_0_data.attrs.copy()

    # Remove scan_name from the attributes, and assign scan name to data_0_name
    data_0_name = data_0_attrs.pop('scan_name')

    # Variable used to use to store the summed DataArray data (will be updated with other DataArrays)
    summed_data = data_0_data.copy(deep=True)

    # Variable used to use to store the summed DataArray name (will be updated with other DataArrays)
    summed_name = data_0_name

    # Flags used to determine whether the analysis history is updated with a caution informing the user of an
    # attributes/coordinates mismatch between the inputted DataArrays
    attrs_warn_flag = False
    coords_warn_flag = False

    # Iterate through the rest of the inputted DataArrays and sum together
    for i in range(1, num_data):
        # Get current DataArray information
        current_data = data[i]
        # Extract attributes of current DataArray
        current_attrs = current_data.attrs.copy()
        # Remove scan_name from the current DataArray attributes, and assign scan name to current_name
        current_name = current_attrs.pop('scan_name')

        # Ensure that the dimensions of the current DataArray match those of the first DataArray, raise an error if not
        if current_data.dims != data_0_data.dims:
            raise Exception('Inputted DataArrays must have the same dimensions.')

        # Ensure that the coordinates of the current DataArray match those of the first DataArray. If not, interpolate
        # the current DataArray onto the coordinate grid of the first DataArray
        for dim in current_data.dims:  # Loop through dimensions
            # Check if the coordinates of the current dimension do not match that of the first DataArray
            if not (current_data[dim].data == data_0_data[dim].data).all():
                # Interpolate the current DataArray onto the current dimension coordinate grid of the first DataArray
                current_data = current_data.interp({dim: data_0_data[dim]})
                coords_warn_flag = True  # Update warning flag
                # Display warning informing the user of the interpolation
                warning_str = ('The {dim} coordinates of scan {current_name} do not match those of scan {data_0_name}. '
                               'Interpolated scan {current_name} onto the {dim} coordinate grid of scan {'
                               'data_0_name}.').format(dim=dim, current_name=current_name, data_0_name=data_0_name)
                analysis_warning(warning_str, title='Analysis info', warn_type='danger')

        # Determine any attributes (except scan name) of the current DataArray that do not match the first DataArray
        mismatched_attrs = []  # Used to store mismatched attributes
        # Check if the current scan has the same attribute options (i.e. polar, temp_sample etc.) as the first DataArray
        if list(current_attrs) == list(data_0_attrs):
            # Loop through attributes (except scan name) of the current DataArray and extract any that do not match
            # the first DataArray
            for attr in current_attrs:
                if current_attrs[attr] != data_0_attrs[attr]:
                    mismatched_attrs.append(attr)

            # If any attributes (except scan name) of the current DataArray do not match the first DataArray, display
            # a warning telling the user that the attributes of the first DataArray will be saved
            if len(mismatched_attrs) > 0:
                attrs_warn_flag = True  # Update warning flag
                warning_str = ('The following attributes of scan {current_name} do not match those of scan {'
                               'data_0_name}: {mismatched_attrs}. '
                               'Attributes of scan {data_0_name} kept.').format(
                    current_name=current_name, data_0_name=data_0_name, mismatched_attrs=mismatched_attrs)
                analysis_warning(warning_str, title='Analysis info', warn_type='danger')

        # If the current scan does not have the same attribute options (i.e. polar, x0, temp_sample etc.) as the
        # first DataArray, display a warning telling the user that the attributes of the first DataArray will be saved
        else:
            attrs_warn_flag = True
            warning_str = ('The attribute options of scan {current_name} do not match those of scan {data_0_name}. '
                           'Attributes of scan {data_0_name} kept.').format(
                current_name=current_name, data_0_name=data_0_name, mismatched_attrs=mismatched_attrs)
            analysis_warning(warning_str, title='Analysis info', warn_type='danger')

        # Add the current DataArray to the running summed total
        summed_data += current_data.data

        # Append the current DataArray scan name to the summed scan name
        summed_name += ' + {current_name}'.format(current_name=current_name)

    # Update summed data scan name
    summed_data.attrs['scan_name'] = summed_name
    summed_data.name = summed_name

    # Update the analysis history
    hist_str = '{num_data} scans summed together.'.format(num_data=num_data)
    if attrs_warn_flag:  # If the there is an attributes mismatch, append information to analysis history
        hist_str += ' CAUTION: mismatch of some attributes - those of scan {data_0_name} kept.'.format(
            data_0_name=data_0_name)
    if coords_warn_flag:  # If the there is a coordinates mismatch, append information to analysis history
        hist_str += (' CAUTION: mismatch of some coordinates - interpolated data onto scan {data_0_name} coordinate '
                     'grid.').format(data_0_name=data_0_name)
    summed_data.update_hist(hist_str)

    return summed_data


def merge_data():
    pass


def _merge_two_xarrays():
    pass
