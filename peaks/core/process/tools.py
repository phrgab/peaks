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


@add_methods(xr.DataArray)
def rotate(data, rotation, **centre_kwargs):
    """Function to rotate 2D data around a given centre of rotation

    Parameters
    ------------
    data : xr.DataArray
        The data to be rotated.

    rotation : int, float
        The rotation angle in degrees.

    **centre_kwargs : float (optional)
        Used to define centre of rotation in the format dim = coord, e.g. k_par = 1.2 sets the k_par centre as 1.2.
        Default centre of rotation is (0, 0).

    Returns
    ------------
    rotated_data : xr.DataArray
        The rotated data.

    Examples
    ------------
    from peaks import *

    FM = load('FM.zip')

    FS1 = FM.FS(E=78.45, dE=0.01)

    # Rotate Fermi surface around a (0,0) centre of rotation by 13 degrees
    FS1_rotated_1 = FS1.rotate(13)

    # Rotate Fermi surface around a (theta_par=5, ana_polar=5) centre of rotation by 50 degrees
    FS1_rotated_2 = FS1.rotate(50, theta_par=5, ana_polar=5)

    """

    # If the rotation is a multiple of 360, no need to do anything
    if rotation % 360 == 0:
        return data

    # Check data is 2D
    if len(data.dims) != 2:
        raise Exception("Function only acts on 2D data.")

    # Copy the input data to prevent overwrite issues
    data_to_be_expanded = data.copy(deep=True)

    # Get dimensions of inputted data
    x_dim = data_to_be_expanded.dims[1]  # By default, dims[1] is plotted as x in xarray
    y_dim = data_to_be_expanded.dims[0]  # By default, dims[0] is plotted as y in xarray

    # Check for user-defined centre of rotations
    x_centre = centre_kwargs.get(x_dim)
    if not x_centre:
        x_centre = 0
    y_centre = centre_kwargs.get(y_dim)
    if not y_centre:
        y_centre = 0

    # Must now prepare to interpolate data onto an expanded coordinate grid that is determined by the rotation
    # (in order to not cut off data)
    # Obtain the corners of the inputted data coordinate grid
    x_min = float(data_to_be_expanded.coords[x_dim].min())
    x_max = float(data_to_be_expanded.coords[x_dim].max())
    y_min = float(data_to_be_expanded.coords[y_dim].min())
    y_max = float(data_to_be_expanded.coords[y_dim].max())
    corner_coords = [[x_min, y_max], [x_max, y_max], [x_min, y_min], [x_max, y_min]]

    # Calculate the positions these corners will be rotated to
    rotated_corner_coords = []
    for coord in corner_coords:
        rotated_x = x_centre + (np.cos(np.radians(rotation)) * (coord[0] - x_centre)) - (
                    np.sin(np.radians(rotation)) * (coord[1] - y_centre))
        rotated_y = y_centre + (np.sin(np.radians(rotation)) * (coord[0] - x_centre)) + (
                    np.cos(np.radians(rotation)) * (coord[1] - y_centre))
        rotated_corner_coords.append([rotated_x, rotated_y])

    # Determine rotated x and y limits
    rotated_x_min = min([coord[0] for coord in rotated_corner_coords])
    rotated_x_max = max([coord[0] for coord in rotated_corner_coords])
    rotated_y_min = min([coord[1] for coord in rotated_corner_coords])
    rotated_y_max = max([coord[1] for coord in rotated_corner_coords])

    # Determine whether the original or rotated coordinate grid has the larger limits (to ensure no data is lost), and
    # then define the limits of the expanded coordinate grid
    new_x_min = min(x_min, rotated_x_min)
    new_x_max = max(x_max, rotated_x_max)
    new_y_min = min(y_min, rotated_y_min)
    new_y_max = max(y_max, rotated_y_max)
    new_limits = {x_dim: [new_x_min, new_x_max], y_dim: [new_y_min, new_y_max]}

    # Determine expanded coordinate grids for us to interpolate the inputted DataArray onto
    x_step = data_to_be_expanded.coords[x_dim].data[1] - data_to_be_expanded.coords[x_dim].data[0]
    x_coord_grid = np.arange(new_limits[x_dim][0], new_limits[x_dim][1], x_step)
    y_step = data_to_be_expanded.coords[y_dim].data[1] - data_to_be_expanded.coords[y_dim].data[0]
    y_coord_grid = np.arange(new_limits[y_dim][0], new_limits[y_dim][1], y_step)

    # Interpolate inputted data onto the expanded coordinate grids
    data_to_be_rotated = data_to_be_expanded.interp({x_dim: x_coord_grid, y_dim: y_coord_grid})

    # Determine mapping onto rotated coordinate grid (note: y_dim is treated as the conventional x dimension,
    # owing to xarray convention)
    new_x_coord_grid = np.zeros((len(y_coord_grid), len(x_coord_grid)))
    new_y_coord_grid = np.zeros((len(y_coord_grid), len(x_coord_grid)))
    for i, y in enumerate(y_coord_grid):
        for j, x in enumerate(x_coord_grid):
            new_x_coord_grid[i][j] = x_centre + (np.sin(np.radians(rotation)) * (y - y_centre)) + (
                        np.cos(np.radians(rotation)) * (x - x_centre))
            new_y_coord_grid[i][j] = y_centre + (np.cos(np.radians(rotation)) * (y - y_centre)) - (
                        np.sin(np.radians(rotation)) * (x - x_centre))

    # Define mapping along x as xarray
    x_xarray = xr.DataArray(
        new_x_coord_grid,
        dims=[y_dim, x_dim],
        coords={y_dim: y_coord_grid, x_dim: x_coord_grid},
    )

    # Define mapping along y as xarray
    y_xarray = xr.DataArray(
        new_y_coord_grid,
        dims=[y_dim, x_dim],
        coords={y_dim: y_coord_grid, x_dim: x_coord_grid},
    )

    # Perform interpolation, and cut off any coordinates outside the plotting range
    rotated_data = data_to_be_rotated.interp({y_dim: y_xarray, x_dim: x_xarray}).sel(
        {y_dim: slice(rotated_y_min, rotated_y_max)}).sel({x_dim: slice(rotated_x_min, rotated_x_max)})

    # If there are coords units in the inputted data, add them back to the rotated data
    try:
        x_units = data_to_be_rotated.coords[x_dim].units
        rotated_data.coords[x_dim].attrs = {'units': x_units}
    except AttributeError:
        pass
    try:
        y_units = data_to_be_rotated.coords[y_dim].units
        rotated_data.coords[y_dim].attrs = {'units': y_units}
    except AttributeError:
        pass

    # Update analysis history
    rotated_data.update_hist('Rotated data by {rotation} degrees'.format(rotation=rotation))

    return rotated_data


@add_methods(xr.DataArray)
def sym(data, flipped=False, fillna=True, **sym_kwarg):
    """Function which primarily applies a symmetrisation around a given axis. It can alternatively be used to simply
    flip data around a given axis.

    Parameters
    ------------
    data : xr.DataArray
        The data to be symmetrised.

    flipped : Boolean (optional)
        Whether to return the flipped data rather than the sum of the original and flipped data. Defaults to False.

    fillna : Boolean (optional)
        Whether to fill NaNs with 0s. NaNs occur for regions where the original and flipped data do not overlap. If
        fillna is True, regions without overlap will appear with half intensity (since only one of the original or
        flipped data contributes). Defaults to True.

    **sym_kwarg : float (optional)
        Axis to symmetrise about in the format axis=coord, where coord is coordinate around which the symmetrisation is
        performed, e.g. theta_par=1.4. Defaults to eV=0.

    Returns
    ------------
    sym_data : xr.DataArray
        The symmetrised (or simply flipped) data.

    Examples
    ------------
    from peaks import *

    disp = load('disp.ibw')

    # Symmetrise the dispersion about theta_par=3
    disp_sym = disp.sym(theta_par=3)

    # Flip the dispersion about theta_par=3
    disp_sym = disp.sym(theta_par=3, flipped=True)

    """
    # Copy the input data to prevent overwrite issues
    sym_data = data.copy(deep=True)

    # Check that more than one sym_kwarg has not been passed. If so, raise an error
    if len(sym_kwarg) > 1:  # Check only called with single axis kwarg
        raise Exception("Function can only be called with single axis.")

    # If no axis has been provided in sym_kwarg, set to the default symmetrisation axis and coordinate of eV=0
    if len(sym_kwarg) == 0:
        sym_kwarg = {'eV': 0}

    # Get provided axis and coordinate to perform symmetrisation around
    sym_axis = next(iter(sym_kwarg))
    sym_coord = sym_kwarg[sym_axis]

    # Check that provided axis is a valid dimension of inputted DataArray, if not raise an error
    if sym_axis not in sym_data.dims:
        raise Exception("Provided symmetrisation axis is not a valid dimension of inputted DataArray.")

    # Check if the symmetrisation coordinate is within the range of the inputted DataArray, if not raise an error
    if sym_coord < min(sym_data[sym_axis].data) or sym_coord > max(sym_data[sym_axis].data):
        raise Exception(
            "Provided symmetrisation coordinate ({sym_axis}={sym_coord}) is not within the coordinate range of the "
            "inputted DataArray".format(sym_axis=sym_axis, sym_coord=sym_coord))

    # Generate flipped axis DataArray which maps the original axis to the flipped axis
    flipped_axis_values = (2 * sym_coord) - sym_data[sym_axis].data
    flipped_axis_xarray = xr.DataArray(flipped_axis_values, dims=[sym_axis], coords={sym_axis: sym_data[sym_axis].data})

    # Flip the inputted DataArray by interpolating it onto the flipped axis (and replace NaNs with 0)
    flipped_data = sym_data.interp({sym_axis: flipped_axis_xarray})

    # Fill NaNs with 0s if requested
    if fillna:
        flipped_data = flipped_data.fillna(0)

    # If only the flipped data is requested
    if flipped:
        # Assign sym_data to just the flipped data
        sym_data = flipped_data
        # Update the analysis history
        sym_data.update_hist('Flipped data about {sym_kwarg}'.format(sym_kwarg=sym_kwarg))

    # If the full symmetrisation is requested
    else:
        # Sum the original and flipped data
        sym_data += flipped_data
        # Update the analysis history
        sym_data.update_hist('Symmetrised data about {sym_kwarg}'.format(sym_kwarg=sym_kwarg))

    return sym_data


def sym_nfold():
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
