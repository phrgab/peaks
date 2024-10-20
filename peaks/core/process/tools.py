"""Functions that apply general operations on data.

"""

import copy
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift
from scipy.ndimage import gaussian_filter
from skimage.registration import phase_cross_correlation
from IPython.display import clear_output
from peaks.core.fitting.models import _shirley_bg
from peaks.core.utils.accessors import register_accessor
from peaks.utils import analysis_warning


@register_accessor(xr.DataArray)
def norm(data, dim=None, **kwargs):
    """Function to apply a normalisation to data.

    Parameters
    ------------
    data : xarray.DataArray
        The data to be normalised.

    dim : str, optional
         Normalise data by an integrated DC along direction defined by dim, e.g. dim='eV' would normalise the data by an
         integrated MDC. Set dim='all' to normalise by the mean of the data. Defaults to None where the data is
         normalised to unity (i.e. normalised by the maximum value). Takes precedence over keyword arguments.

    **kwargs : slice, optional
        Slice to normalise by. E.g. eV=slice(105, 105.1) normalises by an integrated MDC defined by the eV slice given.
        Multiple slices can be defined to define a ROI to normalise by.

    Returns
    ------------
    norm_data : xarray.DataArray
        The normalised data.

    Examples
    ------------
    Example usage is as follows::

        import peaks as pks

        # Load data
        disp = pks.load('disp.ibw')

        # Normalise the dispersion to unity
        disp_norm = disp.norm()

        # Normalise the dispersion by an integrated MDC
        disp_norm = disp.norm('eV')

        # Normalise the dispersion by an integrated EDC
        disp_norm = disp.norm('theta_par')

        # Normalise the dispersion by an EDC slice in the background
        disp_norm = disp.norm(theta_par=slice(15, 15.5))

        # Normalise by a region in the backgroun
        disp_norm = disp.norm(eV=slice(105, 105.1), theta_par=slice(-12, -8))

    """

    # Copy the input data to prevent overwriting issues
    norm_data = data.copy(deep=True)

    # If a dim has been provided, either normalise by the mean value of the data, or by an integrated DC
    if dim:
        # If dim is 'all', normalise by the mean value of the data
        if dim == "all":
            # Normalise by the mean value of the data
            norm_mean = norm_data.mean()
            norm_data /= norm_mean

            # Update analysis history
            norm_data.history.add(f"Data normalised by its mean value of {norm_mean}")

        # If not, normalise data by integrated DC along specified dim
        else:
            # Ensure dim is a valid dimension
            if dim not in norm_data.dims:
                raise Exception(
                    "{dim} is not a valid dimension of the inputted DataArray.".format(
                        dim=dim
                    )
                )

            # Normalise data by integrated DC along dim
            int_data = norm_data.mean(dim)
            norm_data /= int_data

            # Update analysis history
            norm_data.history.add(
                "Data normalised an integrated DC along {dim}".format(dim=dim)
            )

    # If no dim and no *kwargs have been provided, normalise to unity
    elif not dim and not kwargs:
        # Normalise data to unity
        norm_data /= norm_data.max()

        # Update analysis history
        norm_data.history.add("Data normalised to unity")

    # If kwargs have been provided, normalise by the slice defined in kwargs
    else:
        norm_slice = norm_data.sel(kwargs).mean(list(kwargs))
        norm_data /= norm_slice

        # Update analysis history
        norm_data.history.add(
            f"Data normalised by an integrated DC defined by the slice {slice}"
        )

    return norm_data


@register_accessor(xr.DataArray)
def bgs(
    data,
    subtraction=None,
    num_avg=1,
    offset_start=0,
    offset_end=0,
    max_iterations=10,
    **kwargs,
):
    """Function to subtract a background from data.

    Parameters
    ------------
    data : xarray.DataArray
        The data from which a background will be subtracted.

    subtraction : int, float, str
        The type of background to subtract:

            Set to a int/float to subtract that number, e.g. subtraction=3.4.

            Set to 'all' to subtract the mean of the data.

            Set to a str of a valid dimension to subtract an integrated DC along the direction defined by the dimension,
            e.g. subtraction='eV' would subtract an integrated MDC.

            Set to 'Shirley' to subtract a Shirley background, e.g. subtraction='Shirley'. Additional arguments num_avg,
            offset_start, offset_end and max_iterations can be defined to optimise the Shirley background.

            Takes precedence over keyword arguments.

    num_avg : int, optional
        Shirley background optimisation parameter. The number of points to consider when calculating the average value
        of the data start and end points. Useful for noisy data. Defaults to 1.

    offset_start : float, optional
        Shirley background optimisation parameter. The offset to subtract from the data start value. Useful when data
        range does not completely cover the start (left) tail of the peak. Defaults to 0.

    offset_end : float, optional
        Shirley background optimisation parameter. The offset to subtract from the data end value. Useful when data
        range does not completely cover the end (right) tail of the peak. Defaults to 0.

    max_iterations : int, optional
        Shirley background optimisation parameter. The maximum number of iterations to allow for convergence of Shirley
        background. Defaults to 10.

    **kwargs : slice, optional
        Slice to define background for subtraction by. E.g. eV=slice(105, 105.1) subtracts an integrated MDC defined
        by the eV slice given. Multiple slices can be defined to define a ROI to subtract the mean of.

    Returns
    ------------
    bgs_data : xarray.DataArray
        The background subtracted data.

    Examples
    ------------
    Example usage is as follows::

        from peaks import *

        disp = load('disp1.xy')

        S2p_XPS = load('XPS1.ibw').DOS()

        # Subtract a constant background of 3.5 from the data
        bgs_disp = disp.bkg(3.5)

        # Subtract a constant background equal to the mean value of the data from the data
        disp_norm = disp.bkg('all')

        # Subtract a background equal to an integrated MDC from the data
        disp_norm = disp.bkg('eV')

        # Subtract a background equal to an integrated EDC from the data
        disp_norm = disp.bkg('theta_par')

        # Subtract a Shirley background from the data
        bgs_S2p_XPS = S2p_XPS.bkg('Shirley')

        # Subtract a Shirley background from the data, using 3 points to calculate the average value of the data start
        # and end points
        bgs_S2p_XPS = S2p_XPS.bkg('Shirley', num_avg=3)

    """

    # Check a subtraction argument has been inputted
    if not subtraction and not kwargs:
        raise Exception(
            "No background subtraction argument has been inputted. Please specify either via the subtraction"
            " argument or by defining at least one slice in the kwargs with a dimension keyword."
        )
    # Copy the input data to prevent overwriting issues
    bgs_data = data.copy(deep=True)

    # If subtraction is an int or float, perform a constant background subtraction by a user-defined amount
    if isinstance(subtraction, int) or isinstance(subtraction, float):
        # Subtract a constant background from the data
        bgs_data -= subtraction

        # Update analysis history
        bgs_data.history.add(
            "A constant background of {val} has been subtracted from the data".format(
                val=str(subtraction)
            )
        )

    # If subtraction is a str, there a few options of what the desired operation is.
    elif isinstance(subtraction, str):
        # If subtraction is 'Shirley'
        if subtraction == "Shirley":
            if "eV" not in bgs_data.dims:
                raise ValueError(
                    "Shirley background subtraction can only be performed on data with an 'eV' dimension."
                )

            # Calculate the Shirley background using the function _Shirley
            Shirley_bkg = _shirley_bg(
                bgs_data.DOS(),  # Mean over all non-energy dimensions
                num_avg=num_avg,
                offset_start=offset_start,
                offset_end=offset_end,
                max_iterations=max_iterations,
            )
            Shirley_bkg = xr.DataArray(
                Shirley_bkg, dims="eV", coords={"eV": bgs_data.coords["eV"]}
            )
            # Subtract the Shirley background from the data
            bgs_data -= Shirley_bkg

            # Update analysis history
            bgs_data.history.add(
                "A Shirley background has been subtracted from the data"
            )

        # If subtraction is 'all'
        elif subtraction == "all":
            # Subtract a constant background equal to the mean value of the data from the data
            bgs_data -= bgs_data.data.mean()

            # Update analysis history
            bgs_data.history.add(
                "A constant background equal to the mean value of the data has been subtracted from the data"
            )

        # If subtraction is a valid dimension of the data
        elif subtraction in bgs_data.dims:
            # Integrate data along dimension defined by subtraction
            int_data = bgs_data.mean(subtraction)

            # Subtract data by integrated DC
            bgs_data.data = (bgs_data - int_data).data

            # Update analysis history
            bgs_data.history.add(
                "A background equal to an integrated DC along {subtraction} has been subtracted from the data".format(
                    subtraction=subtraction
                )
            )

        # Inputted argument for subtraction is not valid. Raise an error
        else:
            raise Exception(
                f"Subtraction type {subtraction} is not a valid argument. Expected int, float, 'all', 'Shirley', or a "
                f"valid dimension of the inputted DataArray, or for the subtraction to be defined by keyword arguemnts."
            )

    elif subtraction is None and kwargs:
        # If kwargs have been provided, subtract data dedined by the slice defined in kwargs
        bg_slice = bgs_data.sel(kwargs).mean(list(kwargs))
        bgs_data -= bg_slice

        # Update analysis history
        bgs_data.history.add(
            f"Data subtracted by an integrated DC defined by the slice {slice}"
        )
    # Invalid data type for subtraction. Raise an error
    else:
        raise Exception(
            "Invalid data type for subtraction. Please pass an int, float, 'all', 'Shirley', or a valid dimension of "
            "the inputted DataArray to the `subtraction` argument, or supply valid slices as keyword arguemnts."
        )

    return bgs_data


@register_accessor(xr.DataArray)
def bin_data(data, binning=None, boundary="trim", **binning_kwargs):
    """Shortcut function to bin data. Thin wrapper around :class:`xarray.DataArray.coarsen` but also with updating
    analysis history.

    Parameters
    ------------
    data : xarray.DataArray
        The data to be binned.

    binning : int, optional
        Size of bins to apply to all dimensions. Defaults to None (dimension-specific binning_kwargs must be defined).
        Takes priority over use of binning_kwargs.

    boundary : str, optional
        Determines how to handle boundaries. Defaults to 'trim' where bins are trimmed to fit the data.
        Other options are 'exact' and 'pad'; see :class:`xarray.DataArray.coarsen` for more information.

    **binning_kwargs : int, optional
        Used to define dimension-specific binning in the format dim = bin_size, e.g. theta_par = 2.

    Returns
    ------------
    binned_data : xarray.DataArray
        The binned data.

    Examples
    ------------
    Example usage is as follows::

        import peaks as pks

        disp = pks.load('disp.ibw')

        # Bin dispersion using bin sizes of 2 in both dimensions.
        disp_binned1 = disp.bin_data(2)

        # Bin dispersion using a bin size of 2 along theta_par, and a bin size of 3 along eV.
        disp_binned2 = disp.bin_data(theta_par=2, eV=3)

    """

    # If binning argument is defined, apply same size bins to all dimensions
    if binning:
        # If binning argument is not an integer, raise an error
        if not isinstance(binning, int):
            raise Exception("Binning value must be an integer.")

        # Overwrite any binning_kwargs since binning takes priority
        binning_kwargs = {}

        # Loop through all dimensions and define bin size in binning_kwargs
        for dim in data.dims:
            binning_kwargs[dim] = binning

    # If binning argument is not defined, get dimension-specific bins from binning_kwargs
    else:
        # If no binning_kwargs are supplied, raise an error
        if len(binning_kwargs) == 0:
            raise Exception(
                "No binning parameters set. Define either binning or binning_kwargs arguments."
            )

        # Loop through all items in binning_kwargs to ensure dictionary keys are valid dimensions and dictionary values
        # are integers, if not raise an error.
        for dim in binning_kwargs:
            if dim not in data.dims:
                raise Exception(
                    "{dim} is not a valid dimension of the inputted DataArray.".format(
                        dim=dim
                    )
                )
            if not isinstance(binning_kwargs[dim], int):
                raise Exception("Binning values must be integers.")

    # Apply binning to data
    binned_data = data.coarsen(binning_kwargs, boundary=boundary).mean()

    # Update analysis history
    binned_data = binned_data.history.assign(
        f"Binned data using the bins: {binning_kwargs}, boundary={boundary}"
    )

    return binned_data


@register_accessor(xr.DataArray)
def smooth(data, **smoothing_kwargs):
    """Function to smooth data by applying a Gaussian smoothing operator.

    Parameters
    ------------
    data : xarray.DataArray
        The data to smooth.

    **smoothing_kwargs : float
        Axes to smooth over in the format axis=FWHM, where FWHM is the relevant FWHM of the Gaussian for convolution
        in this direction, e.g. eV=0.1.

    Returns
    ------------
    smoothed_data : xarray.DataArray
        The smoothed data.

    Examples
    ------------
    Example usage is as follows::

        import peaks as pks

        disp = pks.load('disp.ibw')

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
    hist = "Data smoothed with the following FWHMs along given axes: "

    # Make the sigma array (used to store standard deviations) as zeros, then update using supplied definitions
    sigma = np.zeros(len(data.dims))

    # Iterate through coordinates and determine the standard deviations in pixels from the DataArray axis scaling
    for count, value in enumerate(data.dims):
        if value not in smoothing_kwargs:  # No broadening for this dimension
            sigma[count] = 0
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
    array = smoothed_data.data

    # Apply gaussian convolution to raw DataArray data
    array_sm = gaussian_filter(array, sigma)

    # Update DataArray with smoothed data
    smoothed_data.data = array_sm

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
    smoothed_data.history.add(hist[:-2])

    return smoothed_data


@register_accessor(xr.DataArray)
def rotate(data, rotation, **centre_kwargs):
    """Function to rotate 2D data around a given centre of rotation.

    Parameters
    ------------
    data : xarray.DataArray
        The data to be rotated.

    rotation : int, float
        The rotation angle in degrees.

    **centre_kwargs : float, optional
        Used to define centre of rotation in the format dim=coord, e.g. theta_par=1.2 sets the theta_par centre as 1.2.
        Default centre of rotation is (0, 0).

    Returns
    ------------
    rotated_data : xarray.DataArray
        The rotated data.

    Examples
    ------------
    Example usage is as follows::

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
    for dim in list(centre_kwargs):
        if dim not in list(data_to_be_expanded.dims):
            raise Exception(
                "{dim} is not a valid dimension of the inputted DataArray.".format(
                    dim=dim
                )
            )

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
        rotated_x = (
            x_centre
            + (np.cos(np.radians(rotation)) * (coord[0] - x_centre))
            - (np.sin(np.radians(rotation)) * (coord[1] - y_centre))
        )
        rotated_y = (
            y_centre
            + (np.sin(np.radians(rotation)) * (coord[0] - x_centre))
            + (np.cos(np.radians(rotation)) * (coord[1] - y_centre))
        )
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
    x_step = (
        data_to_be_expanded.coords[x_dim].data[1]
        - data_to_be_expanded.coords[x_dim].data[0]
    )
    x_coord_grid = np.arange(new_limits[x_dim][0], new_limits[x_dim][1], x_step)
    y_step = (
        data_to_be_expanded.coords[y_dim].data[1]
        - data_to_be_expanded.coords[y_dim].data[0]
    )
    y_coord_grid = np.arange(new_limits[y_dim][0], new_limits[y_dim][1], y_step)

    # Interpolate inputted data onto the expanded coordinate grids
    data_to_be_rotated = data_to_be_expanded.interp(
        {x_dim: x_coord_grid, y_dim: y_coord_grid}
    )

    # Determine mapping onto rotated coordinate grid (note: y_dim is treated as the conventional x dimension,
    # owing to xarray convention)
    new_x_coord_grid = np.zeros((len(y_coord_grid), len(x_coord_grid)))
    new_y_coord_grid = np.zeros((len(y_coord_grid), len(x_coord_grid)))
    for i, y in enumerate(y_coord_grid):
        for j, x in enumerate(x_coord_grid):
            new_x_coord_grid[i][j] = (
                x_centre
                + (np.sin(np.radians(rotation)) * (y - y_centre))
                + (np.cos(np.radians(rotation)) * (x - x_centre))
            )
            new_y_coord_grid[i][j] = (
                y_centre
                + (np.cos(np.radians(rotation)) * (y - y_centre))
                - (np.sin(np.radians(rotation)) * (x - x_centre))
            )

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
    rotated_data = (
        data_to_be_rotated.interp({y_dim: y_xarray, x_dim: x_xarray})
        .sel({y_dim: slice(rotated_y_min, rotated_y_max)})
        .sel({x_dim: slice(rotated_x_min, rotated_x_max)})
    )

    # If there are coords units in the inputted data, add them back to the rotated data
    try:
        x_units = data_to_be_rotated.coords[x_dim].units
        rotated_data.coords[x_dim].attrs = {"units": x_units}
    except AttributeError:
        pass
    try:
        y_units = data_to_be_rotated.coords[y_dim].units
        rotated_data.coords[y_dim].attrs = {"units": y_units}
    except AttributeError:
        pass

    # Update analysis history
    rotated_data.history.add(f"Rotated data by {rotation} degrees")

    return rotated_data


@register_accessor(xr.DataArray)
def sym(data, flipped=False, fillna=True, **sym_kwarg):
    """Function which primarily applies a symmetrisation of data around a given axis. It can alternatively be used to
    simply flip data around a given axis.

    Parameters
    ------------
    data : xarray.DataArray
        The data to be symmetrised.

    flipped : bool, optional
        Whether to return the flipped data rather than the sum of the original and flipped data. Defaults to False.

    fillna : bool, optional
        Whether to fill NaNs with 0s. NaNs occur for regions where the original and flipped data do not overlap. If
        fillna=True, regions without overlap will appear with half intensity (since only one of the original or
        flipped data contributes). Defaults to True.

    **sym_kwarg : float, optional
        Axis to symmetrise about in the format axis=coord, where coord is coordinate around which the symmetrisation is
        performed, e.g. theta_par=1.4. Defaults to eV=0.

    Returns
    ------------
    sym_data : xarray.DataArray
        The symmetrised (or simply flipped) data.

    Examples
    ------------
    Example usage is as follows::

        import peaks as pks

        disp = pks.load('disp.ibw')

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
        sym_kwarg = {"eV": 0}

    # Get provided axis and coordinate to perform symmetrisation around
    sym_axis = next(iter(sym_kwarg))
    sym_coord = sym_kwarg[sym_axis]

    # Check that provided axis is a valid dimension of inputted DataArray, if not raise an error
    if sym_axis not in sym_data.dims:
        raise Exception(
            "Provided symmetrisation axis is not a valid dimension of inputted DataArray."
        )

    # Check if the symmetrisation coordinate is within the range of the inputted DataArray, if not raise an error
    if sym_coord < min(sym_data[sym_axis].data) or sym_coord > max(
        sym_data[sym_axis].data
    ):
        raise Exception(
            "Provided symmetrisation coordinate ({sym_axis}={sym_coord}) is not within the coordinate range of the "
            "inputted DataArray".format(sym_axis=sym_axis, sym_coord=sym_coord)
        )

    # Generate flipped axis DataArray which maps the original axis to the flipped axis
    flipped_axis_values = (2 * sym_coord) - sym_data[sym_axis].data
    flipped_axis_xarray = xr.DataArray(
        flipped_axis_values, dims=[sym_axis], coords={sym_axis: sym_data[sym_axis].data}
    )

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
        sym_data.history.add(f"Flipped data about {sym_kwarg}")

    # If the full symmetrisation is requested
    else:
        # Sum the original and flipped data
        sym_data += flipped_data
        # Update the analysis history
        sym_data.history.add(f"Symmetrised data about {sym_kwarg}")

    return sym_data


@register_accessor(xr.DataArray)
def sym_nfold(data, nfold, expand=True, fillna=True, **centre_kwargs):
    """Function to perform an n-fold symmetrisation of data around a centre coordinate.

    Parameters
    ------------
    data : xarray.DataArray
        The data to be symmetrised.

    nfold : int
        The rotation order.

    expand : bool, optional
        Whether to expand the coordinate grid to view all symmetrised data.

    fillna : bool, optional
        Whether to fill NaNs with 0s. When the data is rotated, the coordinate grid is expanded. As such there will be
        regions with NaNs on the new coordinate grid. Setting fillna=False means such regions remain NaNs, leading to a
        loss of data when the different rotations are summed during the symmetrisation. Setting fillna=True sets the
        NaNs to 0, but when the different rotations are summed during the symmetrisation, such regions are scaled to
        allow for a consistent intensity of the symmetrised data. Some NaNs will remain for regions of the new
        coordinate grid where there is no data. Defaults to True

    **centre_kwargs : float, optional
        Used to define centre of rotation used for the symmetrisation in the format dim=coord, e.g. theta_par=1.2 sets
        the theta_par centre as 1.2. Default centre of rotation is (0, 0).

    Returns
    ------------
    sym_data : xarray.DataArray
        The symmetrised data.

    Examples
    ------------
    Example usage is as follows::

        from peaks import *

        FM1 = load('FM1.zip')
        FS1 = FM1.MDC(E=75.62, dE=0.02)

        # Perform a 3-fold symmetrisation of the Fermi surface around a (0,0) centre of rotation
        FS1_sym = FS1.sym_nfold(nfold=3)

        # Perform a 3-fold symmetrisation of the Fermi surface around a (theta_par=3, ana_polar=5) centre of rotation,
        # restricting the coordinate grid of the output to that of the input, and not replacing Nans.
        FS1_sym = FS1.sym(theta_par=3, ana_polar=5, expand=False, fillna=False)

    """

    # Copy the input data to prevent overwrite issues
    data_to_be_symmetrised = data.copy(deep=True)

    # Get dimensions of inputted data
    dim1 = data_to_be_symmetrised.dims[0]
    dim2 = data_to_be_symmetrised.dims[1]

    # Determine the rotations that will be applied and summed to produce the symmetrised data
    rotation_values = np.linspace(0, 360, nfold + 1)[0:-1]

    # Define lists to store rotated data, and coordinate limits
    rotated_data = []
    dim1_limits = []
    dim2_limits = []

    # Perform the required rotations, and determine coordinate limits of each rotated data
    for rotation in rotation_values:
        current_rotated_data = data_to_be_symmetrised.rotate(rotation, **centre_kwargs)
        dim1_limits.append(
            [
                current_rotated_data[dim1].data.min(),
                current_rotated_data[dim1].data.max(),
            ]
        )
        dim2_limits.append(
            [
                current_rotated_data[dim2].data.min(),
                current_rotated_data[dim2].data.max(),
            ]
        )
        rotated_data.append(current_rotated_data)

    # Determine the extremes of the coordinate limits in the rotated data
    dim1_min = round(min([item[0] for item in dim1_limits]), 4)
    dim1_max = round(max([item[1] for item in dim1_limits]), 4)
    dim2_min = round(min([item[0] for item in dim2_limits]), 4)
    dim2_max = round(max([item[1] for item in dim2_limits]), 4)

    # Determine the coordinate step size of the inputted data
    dim1_step = round(
        float(data_to_be_symmetrised[dim1][1] - data_to_be_symmetrised[dim1][0]), 4
    )
    dim2_step = round(
        float(data_to_be_symmetrised[dim2][1] - data_to_be_symmetrised[dim2][0]), 4
    )

    # Determine the number of coordinate points needed
    num_dim1 = int((dim1_max - dim1_min) / dim1_step) + 1
    num_dim2 = int((dim2_max - dim2_min) / dim2_step) + 1

    # Determine the extremal coordinate grid
    dim1_values = np.linspace(dim1_min, dim1_max, num_dim1)
    dim2_values = np.linspace(dim2_min, dim2_max, num_dim2)

    # Define list to store interpolate rotated data
    interp_rotated_data = []

    # If fillna=True, keep track of where we encounter NaNs, so that we can rescale the data accordingly later
    if fillna:
        NaN_counter = np.zeros((num_dim1, num_dim2))

    # Interpolate rotated data onto extremal coordinate grid
    for entry in rotated_data:
        current_rotated_data = entry.interp({dim1: dim1_values, dim2: dim2_values})
        # If fillna=True, fill NaNs with 0 and update NaN_counter accordingly
        if fillna:
            current_rotated_data = current_rotated_data.fillna(0)
            # current_NaN_counter becomes 1 where a nan has been filled, as 0 elsewhere
            current_NaN_counter = (
                current_rotated_data.where(current_rotated_data == 0, -1) + 1
            )
            NaN_counter += current_NaN_counter.data
        interp_rotated_data.append(current_rotated_data)

    # Sum the rotated data to get the symmetrised data
    sym_data = sum_data(interp_rotated_data)

    # Remove sum_data analysis_history entry, and clear warning (given during sum_data since analysis_history attrs will
    # differ
    sym_data.attrs["analysis_history"] = sym_data.attrs["analysis_history"][0:-1]
    clear_output()

    # Rescale the data according to the NaN_counter so regions are of consistent intensity
    if fillna:
        sym_data.data = (sym_data / (NaN_counter.max() - NaN_counter).data).data

    # If expand is False, perform an interpolation to keep the same coordinate grid as the inputted data
    if not expand:
        sym_data = sym_data.interp(
            {dim1: data_to_be_symmetrised[dim1], dim2: data_to_be_symmetrised[dim2]}
        ).fillna(0)

    # Update analysis history
    sym_data.history.add(f"Symmetrised data using a {nfold}-fold rotation")

    return sym_data


@register_accessor(xr.DataArray)
def degrid(data, width=0.1, height=0.1, cutoff=4):
    """Function which removes a mesh grid from 2D data by filtering its fast Fourier transform (FFT).

    Parameters
    ------------
    data : xarray.DataArray
        The 2D data to remove a mesh grid from.

    width : float, optional
        The width (as a fraction of the total width) of the central low-frequency region of the FFT (which defines most
        of the important data information) that is excluded from filtering analysis. Defaults to 0.1 (i.e 10 %).

    height : float, optional
        The height (as a fraction of the total height) of the central low-frequency region of the FFT (which defines
        most of the important data information) that is excluded from filtering analysis. Defaults to 0.1 (i.e. 10 %).

    cutoff : int, float, optional
        Data points (outside the central low-frequency region) of the FFT with an intensity larger than cutoff
        multiples of the mean of data (i.e. intensity > cutoff * (data).mean()) are removed. Defaults to 4.

    Returns
    ------------
    degrid_data : xarray.DataArray
        The data with the mesh grid removed.

    Examples
    ------------
    Example usage is as follows::

        from peaks import *

        SM = load('SM1.ibw')
        disp = SM.tot(spatial_int=True)

        # Remove the mesh grid from the dispersion using the default settings
        degrid_disp = disp.degrid()

        # Remove the mesh grid from the dispersion, excluding a larger central low-frequency region and considering
        # points above a lower cutoff intensity
        degrid_disp = disp.degrid(width=0.2, height=0.2, cutoff=3.2)

    """

    # Ensure data is a 2D scan, if not raise an error
    if len(data.dims) == 2:
        # Copy the input data to prevent overwrite issues
        data_to_degrid = data.copy(deep=True)
    else:
        raise Exception("Inputted data must be a 2D scan.")

    # Perform fast Fourier transform (FFT) of data_to_degrid
    FFT = fft2(data_to_degrid.data)

    # Rearrange the FFT by shifting the zero-frequency component to the centre of the np.array (equivalent to changing
    # range from (0, 2 Pi) to (-Pi, Pi) since FFT is periodic according to period defined by grid size). Useful for
    # visualisation and data manipulation
    FFT = fftshift(FFT)

    # Define n1 and n2 values, the discrete coordinates in FFT space
    n1_values = np.linspace(0, FFT.shape[0] - 1, FFT.shape[0])
    n2_values = np.linspace(0, FFT.shape[1] - 1, FFT.shape[1])

    # Shift n1 and n2 values in accordance with fftshift
    if len(n1_values) % 2 == 0:
        n1_values -= (FFT.shape[0]) / 2
    else:
        n1_values -= (FFT.shape[0] - 1) / 2

    if len(n2_values) % 2 == 0:
        n2_values -= (FFT.shape[1]) / 2
    else:
        n2_values -= (FFT.shape[1] - 1) / 2

    # Represent the magnitude of the FFT as an xarray.DataArray for plotting/manipulation purposes.
    FFT_DataArray = xr.DataArray(
        abs(FFT), dims=("n1", "n2"), coords={"n1": n1_values, "n2": n2_values}
    )

    # Define the centre region that we want to leave untouched
    n1_centre = [-1 * FFT.shape[0] * width / 2, FFT.shape[0] * width / 2]
    n2_centre = [-1 * FFT.shape[1] * height / 2, FFT.shape[1] * height / 2]

    # Get FFT_DataArray with just the central low-frequency region (which defines most of the important information)
    centre_FFT_DataArray = (
        FFT_DataArray.sel(
            n1=slice(n1_centre[0], n1_centre[1]), n2=slice(n2_centre[0], n2_centre[1])
        )
        .interp({"n1": n1_values, "n2": n2_values})
        .fillna(0)
    )

    # Subtract the central low-frequency region from FFT_DataArray, leaving the not so important information which we
    # can filter
    centre_subtracted_FFT_DataArray = FFT_DataArray - centre_FFT_DataArray

    # Redefine cutoff as cutoff multiples of the mean of the FFT
    cutoff = float(FFT_DataArray.mean()) * cutoff

    # Create a filter where any points of centre_subtracted_FFT_DataArray for which the intensity is above the cutoff is
    # assigned to 0. All other points are assigned to 1
    FFT_filter = xr.DataArray(
        np.where(centre_subtracted_FFT_DataArray.data > cutoff, 0, 1),
        dims=("n1", "n2"),
        coords={"n1": n1_values, "n2": n2_values},
    )

    # Ensure n=0 components are unaffected by changing their values in the filter to 1
    FFT_filter[int(np.where(n1_values == 0)[0])] = 1
    FFT_filter[:, int(np.where(n2_values == 0)[0])] = 1

    # Define degrid_data, and assign its contents to the inverse FFT of the original FFT of the data, multiplied by the
    # filter to remove intense high frequency Fourier components
    degrid_data = data_to_degrid.copy(deep=True)
    degrid_data.data = abs(ifft2(FFT * FFT_filter.data))

    # Update the analysis history
    degrid_data.history.add(
        "Data has been degridded by removing intense high frequency Fourier components"
    )

    # Determine the grid that was removed by calculating the difference between the inputted and degridded data
    grid = data_to_degrid - degrid_data

    # Set up a subplot to plot the Fourier analysis
    fig, axes = plt.subplots(figsize=(15, 4), ncols=4)

    # Plot the magnitude of the FFT, and the central low-frequency region that is excluded from filtering analysis
    FFT_DataArray.plot(
        ax=axes[0], y="n2", robust=True, add_colorbar=False, cmap="binary"
    )
    axes[0].plot(
        [n1_centre[0], n1_centre[1]],
        [n2_centre[0], n2_centre[0]],
        c="yellow",
        linewidth=2,
    )
    axes[0].plot(
        [n1_centre[0], n1_centre[1]],
        [n2_centre[1], n2_centre[1]],
        c="yellow",
        linewidth=2,
    )
    axes[0].plot(
        [n1_centre[0], n1_centre[0]],
        [n2_centre[0], n2_centre[1]],
        c="yellow",
        linewidth=2,
    )
    axes[0].plot(
        [n1_centre[1], n1_centre[1]],
        [n2_centre[0], n2_centre[1]],
        c="yellow",
        linewidth=2,
    )
    axes[0].set_title("Data transformed to \nFourier space")

    # Plot the FFT with the central central low-frequency region removed
    centre_subtracted_FFT_DataArray.plot(
        ax=axes[1], y="n2", robust=True, add_colorbar=False, cmap="binary"
    )
    axes[1].set_title("Important low frequency data \nnot considered for filter")

    # Plot the filter
    FFT_filter.plot(ax=axes[2], y="n2", add_colorbar=False, cmap="binary_r")
    axes[2].set_title("Filter determined by \npoints above cutoff")

    # Plot the FFT multiplied by the filter (thus with intense high frequency Fourier components removed)
    (FFT_DataArray * FFT_filter).plot(
        ax=axes[3], y="n2", robust=True, add_colorbar=False, cmap="binary"
    )
    axes[3].set_title("Apply filter to data to \nremove points above cutoff")

    plt.tight_layout()

    # Set up a subplot to plot the results of the degridding
    fig, axes = plt.subplots(figsize=(15, 7), ncols=3)

    # Plot the inputted data
    data_to_degrid.plot(ax=axes[0], add_colorbar=False, cmap="binary")
    axes[0].set_title("Input data")

    # Plot the degridded data
    degrid_data.plot(ax=axes[1], add_colorbar=False, cmap="binary")
    axes[1].set_title("Degridded data")

    # Plot the difference between the inputted and degridded data, representing the mesh grid removed
    grid.plot(ax=axes[2], add_colorbar=False, cmap="binary")
    axes[2].set_title("Difference")

    plt.tight_layout()

    return degrid_data


def sum_data(data):
    """Function to sum two or more DataArrays together, maintaining the metadata. If the metadata of the DataArrays
    differ, that of the first inputted DataArray will be used. If the coordinate grids of the DataArrays differ, all
    DataArrays will be interpolated onto the coordinate grid of the first inputted DataArray.

    Parameters
    ------------
    data : list
        Any number of :class:`xarray.DataArray` items to sum together.

    Returns
    ------------
    summed_data : xarray.DataArray
        The single summed :class:`xarray.DataArray`.

    Examples
    ------------
    Example usage is as follows::

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

    # Ensure scan name is an attribute
    if "scan_name" not in data_0_attrs:
        data_0_attrs["scan_name"] = "Unknown"

    # Remove scan_name from the attributes, and assign scan name to data_0_name
    data_0_name = data_0_attrs.pop("scan_name")

    # Remove history from the attributes
    data_history = []
    data_history.append(data_0_attrs.pop("analysis_history", "NONE"))

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
        # Ensure scan name is an attribute
        if "scan_name" not in current_attrs:
            current_attrs["scan_name"] = "Unknown"
        # Remove scan_name & history from the current DataArray attributes
        current_name = current_attrs.pop("scan_name")
        data_history.append(current_attrs.pop("analysis_history", "NONE"))

        # Ensure that the dimensions of the current DataArray match those of the first DataArray, raise an error if not
        if current_data.dims != data_0_data.dims:
            raise Exception("Inputted DataArrays must have the same dimensions.")

        # Ensure that the coordinates of the current DataArray match those of the first DataArray. If not, interpolate
        # the current DataArray onto the coordinate grid of the first DataArray
        for dim in current_data.dims:  # Loop through dimensions
            # Check if the coordinates of the current dimension do not match that of the first DataArray
            if not (current_data[dim].data == data_0_data[dim].data).all():
                # Interpolate the current DataArray onto the current dimension coordinate grid of the first DataArray
                current_data = current_data.interp({dim: data_0_data[dim]})
                coords_warn_flag = True  # Update warning flag
                # Display warning informing the user of the interpolation
                warning_str = (
                    "The {dim} coordinates of scan {current_name} do not match those of scan {data_0_name}. "
                    "Interpolated scan {current_name} onto the {dim} coordinate grid of scan {"
                    "data_0_name}."
                ).format(dim=dim, current_name=current_name, data_0_name=data_0_name)
                analysis_warning(warning_str, title="Analysis info", warn_type="danger")

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
                warning_str = (
                    "The following attributes of scan {current_name} do not match those of scan {"
                    "data_0_name}: {mismatched_attrs}. "
                    "Attributes of scan {data_0_name} kept."
                ).format(
                    current_name=current_name,
                    data_0_name=data_0_name,
                    mismatched_attrs=mismatched_attrs,
                )
                analysis_warning(warning_str, title="Analysis info", warn_type="danger")

        # If the current scan does not have the same attribute options (i.e. polar, x0, temp_sample etc.) as the
        # first DataArray, display a warning telling the user that the attributes of the first DataArray will be saved
        else:
            attrs_warn_flag = True
            warning_str = (
                "The attribute options of scan {current_name} do not match those of scan {data_0_name}. "
                "Attributes of scan {data_0_name} kept."
            ).format(
                current_name=current_name,
                data_0_name=data_0_name,
                mismatched_attrs=mismatched_attrs,
            )
            analysis_warning(warning_str, title="Analysis info", warn_type="danger")

        # Add the current DataArray to the running summed total
        summed_data += current_data.data

        # Append the current DataArray scan name to the summed scan name
        summed_name += " + {current_name}".format(current_name=current_name)

    # Update summed data scan name
    summed_data.attrs["scan_name"] = summed_name
    summed_data.name = summed_name

    # Update the analysis history
    hist_str = "{num_data} scans summed together.".format(num_data=num_data)
    if (
        attrs_warn_flag
    ):  # If the there is an attributes mismatch, append information to analysis history
        hist_str += " CAUTION: mismatch of some attributes - those of scan {data_0_name} kept".format(
            data_0_name=data_0_name
        )
    if (
        coords_warn_flag
    ):  # If the there is a coordinates mismatch, append information to analysis history
        hist_str += (
            " CAUTION: mismatch of some coordinates - interpolated data onto scan {data_0_name} coordinate "
            "grid"
        ).format(data_0_name=data_0_name)
    total_hist = {"record": hist_str}
    for i in range(len(data_history)):
        total_hist[f"original scan {i} analysis history"] = data_history[i]
    summed_data = summed_data.history.assign(total_hist)

    return summed_data


def merge_data(data, dim="theta_par", sel=slice(None, None), offsets=None):
    """Function to merge two or more DataArrays together along a given dimension.

    Parameters
    ------------
    data : list
        Any number of N-dimensional DataArrays to merge together.

    dim : str, optional
        The dimension to merge along. Defaults to 'theta_par'.

    sel : slice, optional
        Selection of DataArrays in the list data along the dimension defined in dim, e.g. setting dim='theta_par' and
        sel=slice(-9,8.5) would perform the selection .sel(theta_par=slice(-9,8.5)) on each DataArray in the list data.
        Defaults to slice(None,None).

    offsets : list, numpy.ndarray, float, int, optional
        The offsets along the dimension defined in dim that must be applied to the DataArrays in the list data. If
        offsets is of type list (or numpy.ndarray), len(offsets) must equal len(data), and the offsets will be applied
        to their corresponding DataArray in data. If offsets is of type float (or int), a list of offsets will be
        generated with evenly spaced offsets starting at 0, e.g. if offsets=12, a list with length equal to len(data)
        will be created of the form [0, 12, 24, ...]. Defaults to None where no offsets are applied.

    Returns
    ------------
    merged_data : xarray.DataArray
        The single merged :class:`xarray.DataArray`.

    Examples
    ------------
    Example usage is as follows::

        import peaks as pks

        # Load some dispersions into a list
        disps = []
        for i in range(4):
            disps.append(pks.load(f'disp{i+1}.ibw'))

        # Load some XPS scans into a list
        XPS_data = [pks.load(f'XPS_{i}.ibw') for i in range(1,3)]

        # Merge the dispersions (measured at subsequent polar values with 10 degree intervals) along 'theta_par',
        # applying the offsets [0, 10, 20, 30] as a list
        merged_disp = pks.merge_data([disp1, disp2, disp3, disp4], offsets=[0, 10, 20, 30])

        # As above, but defining offsets=10 to produce the same result
        merged_disp = pks.merge_data([disp1, disp2, disp3, disp4], offsets=10)

        # As above, but cutting the detector edges of the data by slicing theta_par between -9 and 9 to obtain a better
        # merge result
        merged_disp = pks.merge_data([disp1, disp2, disp3, disp4], sel=slice(-9,9), offsets=10)

        # Merge the XPS scans (measured over different energy ranges)
        merged_XPS = pks.merge_data([XPS_1, XPS_2], coord='eV')

    """

    # Ensure dim is a valid dimension
    if dim not in data[0].dims:
        raise Exception(
            "{dim} is not a valid dimension of the inputted data.".format(dim=dim)
        )

    # Copy the input data to prevent overwriting issues, and perform the selection along dim defined by sel
    data_to_merge = []
    data_history = []
    for item in data:
        if isinstance(item, xr.core.dataarray.DataArray):
            data_to_merge.append(item.copy(deep=True).sel({dim: sel}))
            if "analysis_history" in item.attrs:
                data_history.append(item.attrs["analysis_history"])
            else:
                data_history.append("None")
        else:
            raise Exception("Data must be a list of xarray.DataArrays.")

    # Apply offsets to inputted data if required
    if offsets:
        # If offsets is a list (or numpy.ndarray), ensure len(offsets) == len(data)
        if isinstance(offsets, list) or isinstance(offsets, np.ndarray):
            if not len(offsets) == len(data):
                raise Exception(
                    "If offsets is provided as a list (or numpy.ndarray), the length of offsets and data must match."
                )

        # If offsets is a float or int, make offsets a list of the form [0, offsets*1, offsets*2, ...]
        elif isinstance(offsets, float) or isinstance(offsets, int):
            offsets = [i * offsets for i in range(len(data))]

        # If offset is not of type list, numpy.ndarray, float or int, raise an error
        else:
            raise Exception(
                "Invalid type for offsets. Argument must be a list, numpy.ndarray, float or int."
            )

        # Loop through data, apply offsets, and maintain coordinate units if any are present
        for i, current_data in enumerate(data_to_merge):
            data_to_merge[i].coords[dim] = current_data.coords[dim] - offsets[i]
            try:
                data_to_merge[i].coords[dim].attrs["units"] = (
                    data[i].coords[dim].attrs["units"]
                )
            except KeyError:
                pass

    # Ensure data_to_merge is arranged in order of increasing coordinates along dim
    min_values = []
    for item in data_to_merge:
        min_values.append([item.coords[dim].data.min(), item])
    min_values.sort()
    data_to_merge = [entry[1] for entry in min_values]

    # Initially define merged_data as the first entry of data_to_merge, and extract the scan name
    merged_data = data_to_merge[0]
    scan_name = merged_data.attrs["scan_name"]

    # Loop through the remaining entries of data_to_merge, merge the data with merged_data (using the function
    # _merge_two_DataArrays), and update the scan name
    for current_data in data_to_merge[1:]:
        merged_data = _merge_two_DataArrays(merged_data, current_data, dim)
        scan_name += " & " + current_data.attrs["scan_name"]

    # Update analysis history
    history_str = f"Merged {scan_name} along {dim} "
    if offsets is not None:
        history_str += f"with offsets {offsets} "
    if sel.start is not None or sel.stop is not None:
        history_str += f"with data cropped to {dim}={sel} "
    history_str = history_str.rstrip() + "."
    history_record = {"record": history_str}
    for i in range(len(data_history)):
        history_record[f"original scan {i} analysis history"] = data_history[i]
    merged_data.attrs.pop("analysis_history", None)
    merged_data = merged_data.history.assign(history_record)

    return merged_data


def _merge_two_DataArrays(DataArray1, DataArray2, dim):
    """Function to merge two N-dimensional DataArrays together along a given dimension.

    Parameters
    ------------
    DataArray1 : xarray.DataArray
        The first DataArray to be merged, with the lowest coordinates along dim.

    DataArray2 : xarray.DataArray
        The second DataArray to be merged, with the highest coordinates along dim

    dim : str
        The dimension to merge along.

    Returns
    ------------
    merged_data : xarray.DataArray
        The single merged :class:`xarray.DataArray`.

    Examples
    ------------
    Example usage is as follows::

        from peaks import *

        from peaks.core.process.tools import _merge_two_DataArrays

        disp1 = load('disp1.nc')
        disp2 = load('disp2.nc')

        # Merge the dispersions along 'theta_par'
        merged_disp = merge_data(disp1, disp2, 'theta_par')

    """

    # Ensure dims of the inputted DataArrays are the same
    if not DataArray1.dims == DataArray2.dims:
        raise ValueError("The dimensions of the inputted DataArrays do not match.")

    # Ensure dim is a valid dimension
    if dim not in DataArray1.dims:
        raise ValueError(
            "{dim} is not a valid dimension of the inputted data.".format(dim=dim)
        )

    # Copy the input data to prevent overwriting issues
    DataArray1 = DataArray1.copy(deep=True)
    DataArray2 = DataArray2.copy(deep=True)

    # Determine overlap region of the two DataArrays
    overlap_limits = (
        DataArray2.coords[dim].data.min(),
        DataArray1.coords[dim].data.max(),
    )

    # Define coordinate axis along dim for merged_data
    coord_step = abs(DataArray1.coords[dim][1] - DataArray1.coords[dim][0])
    coord_limits = (
        DataArray1.coords[dim].data.min(),
        DataArray2.coords[dim].data.max(),
    )
    coord_num_points = int((coord_limits[1] - coord_limits[0]) / coord_step)
    coord_values = np.linspace(coord_limits[0], coord_limits[1], coord_num_points)

    # Interpolate DataArrays onto new coordinate grid
    DataArray1 = DataArray1.interp({dim: coord_values}).fillna(0)
    DataArray2 = DataArray2.interp({dim: coord_values}).fillna(0)

    # The overlap region will now be slightly different due to the new coordinate system. Find the indexes of the new
    # overlap region
    overlap_limits_indexes = (
        (np.abs(coord_values - overlap_limits[0])).argmin(),
        (np.abs(coord_values - overlap_limits[1])).argmin(),
    )

    # Determine the total counts within the overlap region of the two DataArrays
    DataArray1_overlap_intensity = float(
        DataArray1.isel(
            theta_par=slice(overlap_limits_indexes[0], overlap_limits_indexes[1])
        ).sum()
    )
    DataArray2_overlap_intensity = float(
        DataArray2.isel(
            theta_par=slice(overlap_limits_indexes[0], overlap_limits_indexes[1])
        ).sum()
    )

    # Scale DataArray2 such that the total counts within the overlap region of the two DataArrays is equal
    ratio = DataArray1_overlap_intensity / DataArray2_overlap_intensity
    DataArray2.data *= ratio

    # Before we sum the two DataArrays, we want to apply linear intensity reductions in the overlap region so that we
    # gradually go from the DataArray1 spectrum to the DataArray2 spectrum, obtaining equal contributions in the center
    # of the overlap region. Achieve this by defining a scaling for DataArray1 (called DataArray1_scaling) which is 1
    # over the theta_par region consisting of just DataArray1 (left_region), linearly decreases from 1 to 0 over the
    # overlap region, and is 0 over the theta_par region consisting of just DataArray2 (left_region). An equivalent
    # scaling for DataArray2 (called DataArray2_scaling) is obtained as DataArray2_scaling = 1 - DataArray1_scaling
    left_region = np.ones(overlap_limits_indexes[0])
    overlap_region = np.linspace(
        1, 0, (overlap_limits_indexes[1] - overlap_limits_indexes[0] + 1)
    )
    right_region = np.zeros(len(coord_values) - overlap_limits_indexes[1] - 1)
    DataArray1_scaling = np.concatenate((left_region, overlap_region, right_region))
    DataArray2_scaling = 1 - DataArray1_scaling

    # Represent the scaling as DataArrays so that they are associated with the correct dimension (makes function
    # applicable to any N-dimensional data)
    DataArray1_scaling = xr.DataArray(
        DataArray1_scaling, dims=[dim], coords={dim: coord_values}
    )
    DataArray2_scaling = xr.DataArray(
        DataArray2_scaling, dims=[dim], coords={dim: coord_values}
    )

    # Normalise DataArray1 and DataArray2 by their respective scaling along dim
    DataArray1 *= DataArray1_scaling
    DataArray2 *= DataArray2_scaling

    # Sum DataArray1 and DataArray2
    merged_data = sum_data([DataArray1, DataArray2])

    # Remove sum_data analysis_history entry
    merged_data.attrs["analysis_history"] = copy.deepcopy(
        merged_data.attrs["analysis_history"][0:-1]
    )

    return merged_data


@register_accessor(xr.DataArray)
def estimate_sym_point(data, dims=None, upsample_factor=100):
    """Function to estimate the centrepoint of data that should be symmetric about an axis or axes.
    Used for e.g. estimating normal emission of an ARPES scan.

    Parameters
    ------------
    data : xarray.DataArray
        The data to estimate the centrepoint of.

    dims : string, tuple of string, list of string, optional
        The dimensions to estimate the centrepoint along.
        Defaults to None, which estimates the centrepoint along all dims of the array

    upsample_factor : int, optional
        The upsample factor used in the phase cross correlation algorithm.
        Defaults to 100.

    Returns
    ------------
    centre : dict
        The estimated centrepoint of the data along the specified dims.

    Examples
    ------------
    Example usage is as follows::

        import peaks as pks

        # Find the centre of the dispersion data along the theta_par dimension
        disp = pks.load('disp.ibw')
        centre = estimate_sym_point(disp, dims='theta_par')

        # Find the centre of a Fermi surface map
        FS = pks.load('FS.ibw')
        centre = estimate_sym_point(FS, dims=['theta_par', 'polar'])

    """

    # Ensure dims is of list type
    if isinstance(dims, str):
        dims = [dims]
    elif isinstance(dims, tuple):
        dims = list(dims)
    elif dims is None:
        dims = data.dims

    # Flip the data along the specified dimensions
    dim_numbers = [data.dims.index(keyword) for keyword in dims if keyword in data.dims]
    data_array = data.data
    data_array_flipped = np.flip(data_array, axis=tuple(dim_numbers))

    # Perform phase cross correlation to estimate the offsets between the arrays
    shift, _, _ = phase_cross_correlation(
        data_array, data_array_flipped, upsample_factor=upsample_factor
    )

    # Work out the offsets in real co-ordinates
    centre = {}
    for i, dim in enumerate(dims):
        coord = data.coords[dim].values
        coord_delta = coord[1] - coord[0]
        coord_midpoint = (coord[0] + coord[-1]) / 2
        midpoint = coord_midpoint + ((shift[dim_numbers[i]] / 2) * coord_delta)
        centre[dim] = midpoint

    return centre


def drift_correction(reference_data, moving_data, orig_pos=None, **kwargs):
    """Estimate new position to correct for drift between two spatial maps.

    Parameters
    ------------
    reference_data : xarray.DataArray
        The original data, should be a 1D or 2D DataArray

    moving_data : xarray.DataArray
        The new data which should be over the same relative range and be of the same dimensions as orig_data,
        but which can be over a different absolute range

    orig_pos : dict, optional
         Dictionary specifying the original positions

    **kwargs : optional
        Additional keyword arguments to be passed to :class:`skimage.registration.phase_cross_correlation` for
        the subpixel image registration

    Returns
    ------------
    shift : dict
        Dictionary of shifts required to register ``moving_data`` with ``reference_data``.
    new_pos : dict, optional
        If orig_pos supplied, dictionary of position in new_map corresponding to orig_pos in orig_map
    """

    reference_data = reference_data.squeeze()
    moving_data = moving_data.squeeze()
    if len(reference_data.shape) >= 3 or len(moving_data.shape) >= 3:
        raise ValueError("Supplied data has more than 2 dimensions. Select a 2D slice.")

    # Set default upsample factor
    kwargs.setdefault("upsample_factor", 10)

    # Calculate shifts in pixel space
    shifts, _, _ = phase_cross_correlation(
        reference_data.data, moving_data.data, **kwargs
    )

    # Convert to real axis units
    dim_shift = {}
    for i, shift in enumerate(shifts):
        dim = reference_data.dims[i]
        orig_coord = reference_data[dim].data
        new_coord = moving_data[dim].data
        dim_delta = orig_coord[1] - orig_coord[0]
        dim_shift[dim] = shift * dim_delta.data - (new_coord[0] - orig_coord[0])

    if orig_pos:
        new_pos = {dim: orig_pos[dim] - dim_shift[dim] for dim in orig_pos}
        return dim_shift, new_pos
    return dim_shift
