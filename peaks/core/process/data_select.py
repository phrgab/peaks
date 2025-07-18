"""Functions used to extract selections of data."""

import copy

import numpy as np
import pint_xarray  # noqa: F401
import xarray as xr
from matplotlib.path import Path

from peaks.core.utils.interpolation import (
    _fast_bilinear_interpolate,
    _fast_bilinear_interpolate_rectilinear,
    _is_linearly_spaced,
)
from peaks.core.utils.misc import dequantify_quantify_wrapper

ureg = pint_xarray.unit_registry


def drop_nan_borders(data):
    """
    Trims the edges of an :class:`xarray.DataArray` or :class:`xarray.Dataset` that are all NaN values.

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        The data to trim.

    Returns
    --------
    trimmed_data : xarray.DataArray or xarray.Dataset
        Trimmed :class:`xarray` object with NaN edges removed.

    Examples
    --------
    Example usage is as follows::

            import peaks as pks

            # Load a data file
            data = load("data.nxs")

            # Trim all zero edges from the data
            trimmed_data = data.drop_nan_borders()
    """

    # Iterate over each dimension to trim NaN edges
    for dim in data.dims:
        # Create a boolean mask where data is not NaN along the dimension
        not_nan = ~np.isnan(data).any(dim=[d for d in data.dims if d != dim])

        # Find the first and last index where the data is not all NaN
        non_nan_indices = np.where(not_nan)[0]
        if non_nan_indices.size > 0:
            start = non_nan_indices[0]
            end = non_nan_indices[-1] + 1
            data = data.isel({dim: slice(start, end)})

    return data


def _drop_nan_borders_2D(data):
    """Function to drop rows and columns containing only NaNs from a 2D array."""
    # Drop rows and columns containing only NaNs
    # Depreciate in future in favour of drop_nan_borders
    rows_to_keep = ~np.all(np.isnan(data), axis=1)
    cols_to_keep = ~np.all(np.isnan(data), axis=0)
    return data[rows_to_keep, :][:, cols_to_keep]


def drop_zero_borders(data):
    """
    Trims the edges of an :class:`xarray.DataArray` or :class:`xarray.Dataset` that are all zero values.

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        The data to trim.

    Returns
    --------
    trimmed_data : xarray.DataArray or xarray.Dataset
        Trimmed :class:`xarray` object with NaN edges removed.

    Examples
    --------
    Example usage is as follows::

            import peaks as pks

            # Load a data file
            data = load("data.nxs")

            # Trim all zero edges from the data
            trimmed_data = data.drop_zero_borders()
    """

    # Iterate over each dimension to trim zero edges
    for dim in data.dims:
        # Create a boolean mask where data is not zero along the dimension
        not_zero = (data != 0).any(dim=[d for d in data.dims if d != dim])

        # Find the first and last index where the data is not all zero
        non_zero_indices = np.where(not_zero)[0]
        if non_zero_indices.size > 0:
            start = non_zero_indices[0]
            end = non_zero_indices[-1] + 1
            data = data.isel({dim: slice(start, end)})

    return data


@dequantify_quantify_wrapper
def DC(data, coord="eV", val=0, dval=0, ana_hist=True):
    """General function to extract DCs from data along any coordinate.

    Parameters
    ------------
    data : xarray.DataArray
        The data to extract a DC from.

    coord : str, optional
        Coordinate to extract DC at. Defaults to eV.

    val : float, list, numpy.ndarray, tuple, optional
        DC value(s) to select. If tuple, must be in the format (start, end, step). Defaults to 0.

    dval : float, optional
        Integration range (represents the total range, i.e. integrates over +/- dval/2). Defaults to 0.

    ana_hist : bool, optional
        Defines whether the function appends information to the analysis history metadata. Defaults to True.

    Returns
    ------------
    dc : xarray.DataArray
        Extracted DC(s).

    Examples
    ------------
    Example usage is as follows::

        from peaks import *

        disp = load('disp.ibw')

        # Extract an EDC at theta_par = 3.5 +/- 0.25
        DC1 = disp.DC(coord='theta_par', val=3.5, dval=0.5)

        # Extract an MDC at eV = -0.5 +/- 0.1
        DC2 = disp.DC('eV', -0.5, 0.2)

        # Extract MDCs at eV = -0.5 +/- 0.1 and -0.4 +/- 0.1
        DC3 = disp.DC('eV', [-0.5, -0.4], 0.2)

        # Extract MDCs between eV = -0.2 and 0.1 in steps of 0.05 with +/- 0.01 integrations
        DC4 = disp.DC('eV', (-0.2, 0.1, 0.05), 0.02)

    """

    # If val is a 3 element tuple of the format (start, end, step), make val an numpy.ndarray of the relevant values
    if isinstance(val, tuple):
        if len(val) == 3:
            delta = 0.000000001  # small value to add to end so that values include end number (if appropriate)
            val = np.arange(val[0], val[1] + delta, val[2])
        else:
            raise Exception("Tuple argument must be in the format (start, end, step).")

    # Ensure val is of type list
    if isinstance(val, np.ndarray):
        val = list(val)
    elif not isinstance(val, list):
        val = [val]

    # Convert window to pixels
    num_pixels = int((dval / abs(data[coord].data[1] - data[coord].data[0])) + 1)

    # Extract single pixel DC(s)
    dc = data.sel({coord: val}, method="nearest")

    # Apply extra binning if required
    if num_pixels > 1:
        for i in dc[coord]:
            dc.loc[{coord: i}] = data.sel(
                {coord: slice(i - dval / 2, i + dval / 2)}
            ).mean(coord, keep_attrs=True)

    # If returning a single DC, want to remove the non-varying coordinate as a dimension
    try:
        dc = dc.squeeze(coord)
    except ValueError:
        pass

    # Update the analysis history if ana_hist is True (will be False when DC is called from e.g. EDC, MDC, FS)
    if ana_hist:
        hist = "DC(s) extracted, integration window: " + str(dval)
        dc = dc.history.assign(hist)

    return dc


def MDC(data, E=0, dE=0):
    """Extract MDCs (i.e. slices at constant energy) from data. Broadcasts to higher dimensions as necessary.

    Parameters
    ------------
    data : xarray.DataArray
        The dispersion to extract an MDC from.

    E : float, list, numpy.ndarray, tuple, optional
        Energy (or energies) of MDC(s) to extract. If tuple, must be in the format (start, end, step). Defaults to 0.

    dE : float, optional
        Integration range (represents the total range, i.e. integrates over +/- dE/2). Defaults to 0.

    Returns
    ------------
    mdc : xarray.DataArray
        Extracted MDC(s).

    Examples
    ------------
    Example usage is as follows::

        import peaks as pks

        # Load data
        disp = pks.load('disp.ibw')
        FS1 = pks.load('FS.ibw').k_convert()  # Convert to k-space and BE

        # Extract single MDCs from the dispersion
        MDC1 = disp.MDC(E=-1.2, dE=0.01)  # at eV = -1.2 +/- 0.005
        MDC2 = disp.MDC(55.6, 0.06)  # at eV = 55.6 +/- 0.03
        MDC3 = disp.MDC()  # a single non-integrated MDC at eV value closest to 0

        # Extract multiple MDCs from the dispersion
        MDC4 = disp.MDC([55.6, 55.7], 0.06)  # at eV = 55.6 & 55.7, +/- 0.03
        MDC5 = disp.MDC((-0.2, 0.1, 0.05), 0.02)  # eV = -0.2 and 0.1 in steps of 0.05 with +/- 0.01 integrations

        # Extract a Fermi surface, data already in binding energy
        FS_map = FS1.MDC()  # No integration required - defaults are good!
        FS_map2 = FS1.MDC(dE=0.02)  # Integration over +/- 0.01 eV
    """

    # Call function to extract relevant MDC from dispersion
    mdc = data.DC(coord="eV", val=E, dval=dE, ana_hist=False)

    # Update the analysis history
    hist = "MDC(s) extracted, integration window: " + str(dE)
    mdc = mdc.history.assign(hist)

    return mdc


def EDC(data, k=0, dk=0):
    """Extract EDCs from data.

    Parameters
    ------------
    data : xarray.DataArray
        The dispersion to extract an EDC from.

    k : float, list, numpy.ndarray, tuple, optional
        k or theta_par value(s) of EDC(s) to extract. If tuple, must be in the format (start, end, step). Defaults to 0.

    dk : float, optional
        Integration range (represents the total range, i.e. integrates over +/- dk/2). Defaults to 0.

    Returns
    ------------
    edc : xarray.DataArray
        Extracted EDC(s).

    Examples
    ------------
    Example usage is as follows::

        from peaks import *

        disp = load('disp.ibw')

        # Extract an EDC at k (or theta_par) = 0.5 +/- 0.005
        EDC1 = disp.EDC(k=0.5, dk=0.01)

        # Extract an EDC at k (or theta_par) = -0.2 +/- 0.03
        EDC2 = disp.EDC(-0.2, 0.06)

        # Extract a single non-integrated EDC at k (or theta_par) value closest to 0
        EDC3 = disp.EDC()

        # Extract EDCs at k (or theta_par) = -0.2 +/- 0.03 and -0.1 +/- 0.03
        EDC4 = disp.EDC([-0.2, -0.1], 0.06)

        # Extract EDCs between k (or theta_par) = 0.7 and 1.2 in steps of 0.1 with +/- 0.01 integrations
        EDC5 = disp.EDC((0.7, 1.2, 0.1), 0.02)

    """

    # Work out correct variable for dispersive direction (i.e. is data in angle or k-space)
    coords = list(data.dims)
    coords.remove("eV")
    coord = coords[-1]  # Should always be the last one if data loading is consistent

    # Call function to extract relevant EDC from dispersion
    edc = data.DC(coord=coord, val=k, dval=dk, ana_hist=False)

    # Update the analysis history
    hist = "EDC(s) extracted, integration window: " + str(dk)
    edc = edc.history.assign(hist)

    return edc


def DOS(data):
    """Integrate over all but the energy axis to return the best approximation to the DOS possible from the data.

    Parameters
    ------------
    data : xarray.DataArray
        Data to extract DOS from.

    Returns
    ------------
    dos : xarray.DataArray
        Extracted DOS.

    Examples
    ------------
    Example usage is as follows::

        from peaks import *

        disp = load('disp.ibw')

        FM = load('FM.zip')

        # Extract DOS of a dispersion
        disp_DOS = disp.DOS()

        # Extract DOS of a Fermi map
        FM_DOS = FM.DOS()

    """

    # Get relevant dimensions to integrate over
    int_dim = list(filter(lambda i: i != "eV", data.dims))

    # Calculate the DOS
    dos = data.mean(int_dim, keep_attrs=True)

    # Update the analysis history
    hist = "Integrated along axes: " + str(int_dim)
    dos = dos.history.assign(hist)

    return dos


def tot(data, spatial_int=False):
    """Integrate spatial map data over all non-spatial (energy and angle/k) or all spatial dimensions.

    Parameters
    ------------
    data : xarray.DataArray
        Spatial map data.

    spatial_int : bool, optional
        Determines whether integration is performed over spatial or non-spatial dimensions. Defaults to False.

    Returns
    ------------
    data_tot : xarray.DataArray
        The integrated data.

    Examples
    ------------
    Example usage is as follows::

        from peaks import *

        SM = load('SM.ibw')

        # Extract energy and angle integrated spatial map
        SM_int = SM.tot()

        # Extract spatially integrated dispersion
        SM_int_spatial = SM.tot(spatial_int=True)

    """

    # Integrate over spatial dimensions
    if spatial_int:
        data_tot = data.mean(["x1", "x2"], keep_attrs=True)
        hist = "Integrated along axes: " + str(["x1", "x2"])

    # Integrate over non-spatial dimensions
    else:
        # Get relevant dimensions to integrate over
        int_dim = list(filter(lambda n: n != "x1" and n != "x2", data.dims))
        hist = "Integrated along axes: " + str(int_dim)
        data_tot = data.mean(int_dim, keep_attrs=True)

    # Update the analysis history
    data_tot = data_tot.history.assign(hist)

    return data_tot


@dequantify_quantify_wrapper
def radial_cuts(data, num_azi=361, num_points=200, radius=2, **centre_kwargs):
    """Extract radial cuts of a Fermi surface slice or cube as a function of azimuthal angle, about some central point.

    Parameters
    ------------
    data : xarray.DataArray
        Data to extract radial cuts from.

    num_azi : float, optional
        Number of evenly spaced azi values between 0 and 360 degrees to take radial cuts. Defaults to 361.

    num_points : float, optional
        Number of evenly spaced points to sample along a cut. Defaults to 200.

    radius : float, optional
        Maximum radius to take cuts up to. Defaults to 2.

    **centre_kwargs : float, optional
        Used to define centre of rotations in the format dim = coord, e.g. k_par = 1.2 sets the k_par centre as 1.2.
        Default centre of rotation is (0, 0).

    Returns
    ------------
    data_to_return : xarray.DataArray
        Radial cuts against azi.

    Examples
    ------------
    Example usage is as follows::

        import peaks as pks

        FM = pks.load('FM.zip')

        FS1 = FM.MDC(E=-0, dE=0.01)

        # Extract radial cuts (radius = 15) at azi values in 2 degree increments, using a centre of
        # rotation (theta_par, ana_polar) = (-2, -5)
        FS1_radial_cuts_1 = FS1.radial_cuts(num_azi=181, num_points=300, radius=15, theta_par=-2, ana_polar=-5)

        # Extract radial cuts (radius = 2) at azi values in 1 degree increments, using a centre of
        # rotation (coord_1, coord_2) = (0, 0)
        FS1_radial_cuts_2 = FS1.radial_cuts()

    """

    angle_dims = list(set(data.dims) - {"eV"})

    # Check remaining data is 2D
    if len(angle_dims) != 2:
        raise Exception(
            "Radial cuts can only be taken on data with two angle/k-space dimensions,"
            "with optionally an additional energy dimension."
        )

    # Define the coordinate system
    ang0_coord = angle_dims[0]
    ang1_coord = angle_dims[1]

    # Check for user-defined centre of rotations
    ang0_centre = centre_kwargs.get(ang0_coord)
    if not ang0_centre:
        ang0_centre = 0
    ang1_centre = centre_kwargs.get(ang1_coord)
    if not ang1_centre:
        ang1_centre = 0

    # Define coordinates to be sampled
    azi_angles = np.linspace(0, 360, num_azi)
    k_values = np.linspace(0, radius, num_points)

    # Calculate the values for interpolation
    ang0_values = np.linspace(
        0 + ang0_centre,
        (np.cos(np.radians(azi_angles)) * radius) + ang0_centre,
        num_points,
    )
    ang1_values = np.linspace(
        0 + ang1_centre,
        (np.sin(np.radians(azi_angles)) * radius) + ang1_centre,
        num_points,
    )

    # Check if we have a rectilinear grid to determine the interpolation function
    if _is_linearly_spaced(data[ang0_coord].data) and _is_linearly_spaced(
        data[ang1_coord].data
    ):
        interpolation_fn = _fast_bilinear_interpolate_rectilinear
    else:
        interpolation_fn = _fast_bilinear_interpolate

    # Do the interpolation, broadcasting over energy dimension if required
    interpolated_data = xr.apply_ufunc(
        interpolation_fn,
        ang0_values,
        ang1_values,
        data[ang0_coord].data,
        data[ang1_coord].data,
        data,
        dask="parallelized",
        input_core_dims=[
            ["k", "azi"],
            ["k", "azi"],
            [ang0_coord],
            [ang1_coord],
            [ang0_coord, ang1_coord],
        ],
        output_core_dims=[["k", "azi"]],
        output_dtypes=[data.dtype],
        vectorize=True,
        dask_gufunc_kwargs={"allow_rechunk": True},
    )

    # Update co-ordinates
    interpolated_data["k"] = k_values
    interpolated_data["azi"] = azi_angles

    # Update attributes and analysis history
    interpolated_data.attrs = copy.deepcopy(data.attrs)
    interpolated_data.history.add("Radial cuts taken as a function of azi")

    return interpolated_data


@dequantify_quantify_wrapper
def extract_cut(data, start_point, end_point, num_points=None):
    """Extract cut between two end-points, e.g. dispersion or MDC from a Fermi map data cube.

    Parameters
    ------------
    data : xarray.DataArray
        Data to extract radial cuts from.

    start_point : dict
        Dictionary containing the coordinates of the start of the desired cut to extract.

    end_point : dict
        Dictionary containing the coordinates of the end of the desired cut to extract.

    num_points : int, optional
        Number of points to sample along the cut. Defaults to None, in which case spacing is determined
        based on the original data.

    Returns
    ------------
    data_to_return : xarray.DataArray
        Extracted cut

    Examples
    ------------
    Example usage is as follows::

        import peaks as pks

        FM = pks.load('FM.zip')

        # Extract cut from (0, 0) to (15, 12) with 100 points
        cut = FM.extract_dispersion({'theta_par': 0, 'polar': 0}, {'theta_par': 15, 'polar': 12}, num_points=100)

    """

    # Check the start and end points specification
    if set(start_point.keys()) != set(end_point.keys()):
        raise ValueError(
            "Start and end points must be specified as dictionaries `start_point={dim0: start_point, dim1: start_point}`"
            " and `end_point={dim0: end, dim1: end_point}`, with the same dimensions specified in both."
        )
    if not all([key in data.dims for key in start_point.keys()]):
        raise ValueError(
            "Ensure the dimensions specified in the start and end points are present in the data."
        )

    # Define the coordinate system
    dim0, dim1 = start_point.keys()

    # Calculate projection vector along the cut from the start point
    start_coord0, start_coord1 = start_point[dim0], start_point[dim1]
    end_coord0, end_coord1 = end_point[dim0], end_point[dim1]
    distance = np.sqrt(
        (end_coord0 - start_coord0) ** 2 + (end_coord1 - start_coord1) ** 2
    )
    # If num_points is not specified, calculate the number of points based on the step size in the original data
    if num_points is None:
        step_size = np.sqrt(
            (data[dim0].data[1] - data[dim0].data[0]) ** 2
            + (data[dim1].data[1] - data[dim1].data[0]) ** 2
        )
        num_points = int(np.ceil(distance / step_size))
    projection = np.linspace(0, distance, num_points)

    # Calculate the values for interpolation
    ang0_values = np.linspace(start_point[dim0], end_point[dim0], num_points)
    ang1_values = np.linspace(start_point[dim1], end_point[dim1], num_points)

    # Check if we have a rectilinear grid to determine the interpolation function
    if _is_linearly_spaced(data[dim0].data) and _is_linearly_spaced(data[dim1].data):
        interpolation_fn = _fast_bilinear_interpolate_rectilinear
    else:
        interpolation_fn = _fast_bilinear_interpolate

    # Do the interpolation, broadcasting over remaining dimensions if required
    interpolated_data = xr.apply_ufunc(
        interpolation_fn,
        ang0_values,
        ang1_values,
        data[dim0].data,
        data[dim1].data,
        data,
        input_core_dims=[
            ["proj"],
            ["proj"],
            [dim0],
            [dim1],
            [dim0, dim1],
        ],
        output_core_dims=[["proj"]],
        output_dtypes=[data.dtype],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
    )

    # Update co-ordinates
    interpolated_data["proj"] = projection

    # Update attributes and analysis history
    interpolated_data.attrs = copy.deepcopy(data.attrs)
    interpolated_data.history.add(
        f"Slice extracted from data from {start_point} to {end_point}. "
        f"Data returned vs. the projected distance."
    )

    interpolated_data = interpolated_data.pint.quantify()

    return interpolated_data


def mask_data(data, ROI, return_integrated=True):
    """This function applies a polygon region of interest (ROI) as a mask to multidimensional data. By default, the
    function will then extract the mean over the two dimensions defined by the ROI. For a rectangular ROI, this is
    equivalent to a simple .sel over those dimensions followed by a mean, but an arbitrary polygon can be used to define
    the ROI.

    Parameters
    ------------
    data : xarray.DataArray
        The multidimensional data to apply the ROI selection to.

    ROI : dict
        A dictionary of two lists which contains the vertices of the polygon for the ROI definition, in the form
        {'dim1': [pt1, pt2, pt3, ...], 'dim2'=[pt1', pt2', pt3', ...]}. As many points can be specified as required,
        but this should be given with the same number of points for each dimension.

    return_integrated : bool, optional
        Whether to mean the data confined within ROI region over the ROI dimensions, or instead return the masked data.
        Defaults to True.

    Returns
    ------------
    ROI_selected_data : xarray.DataArray
        The input data with the ROI applied as a mask, and (if return_integrated=True) the mean taken over those
        remaining dimensions.

    Examples
    ------------
    Example usage is as follows::

        from peaks import *

        SM = load('SM.ibw')

        # Define ROI used to mask data
        ROI = {'theta_par': [-8, -5.5, -3.1, -5.6], 'eV': [95.45, 95.45, 95.77, 95.77]}

        # Extract SM consisting of the integrated spectral weight confined within the ROI
        ROI_SM = SM.mask_data(ROI)

        # Extract SM consisting of the input data with the ROI applied as a mask
        ROI_SM = SM.mask_data(ROI, return_integrated=False)

    """

    # Check function has been fed with suitable dictionary for ROI generation
    err_str = (
        "ROI must be a dictionary containing two entries for the relevant axes. Each of these entries should "
        "be a list of the vertices of the polygon for the labelled axis of the ROI. These must be of equal "
        "length for the two axes."
    )
    if not isinstance(ROI, dict) or len(ROI) != 2:
        raise Exception(err_str)

    else:  # Seems correct format
        dims = list(ROI)  # Determine relevant dimensions for ROI

        # Check lengths match
        if len(ROI[dims[0]]) != len(ROI[dims[1]]):
            raise Exception(err_str)

    # Define ROI_path to make a polygon path defining the ROI
    ROI_path = []
    for i in range(len(ROI[dims[0]])):
        ROI_path.append((ROI[dims[0]][i], ROI[dims[1]][i]))
    p = Path(ROI_path)  # Make a polygon defining the ROI

    # Restrict the data cube down to the minimum possible size (making this a copy to avoid overwriting problems)
    data_bounded = data.sel(
        {
            dims[0]: slice(min(ROI[dims[0]]), max(ROI[dims[0]])),
            dims[1]: slice(min(ROI[dims[1]]), max(ROI[dims[1]])),
        }
    ).copy(deep=True)

    # Broadcast coordinate data
    b, dim0 = xr.broadcast(data_bounded, data_bounded[dims[0]])
    b, dim1 = xr.broadcast(data_bounded, data_bounded[dims[1]])

    # Convert coordinate data into a format for passing to matplotlib path function for identifying which points are
    # in the relevant ROI
    points = np.vstack((dim0.data.flatten(), dim1.data.flatten())).T

    # Check which of these points fall within our ROI
    grid = p.contains_points(points, radius=0.01)

    # Reshape to make a data mask
    mask = grid.reshape(data_bounded.shape)

    if return_integrated:  # Data should be averaged over the ROI dimensions
        ROI_selected_data = data_bounded.where(mask).mean(dims, keep_attrs=True)
        hist = (
            "Data averaged over region of interest defined by polygon with vertices: "
            + str(ROI)
        )
    else:  # Masked data to be returned
        ROI_selected_data = data_bounded.where(mask)

        # Trim any rows or columns of only NaNs
        try:  # The top method works for 2D, but seems to fail for higher dimensions
            ROI_selected_data = _drop_nan_borders_2D(ROI_selected_data)
        except Exception:
            ROI_selected_data = drop_nan_borders(ROI_selected_data)

        hist = (
            "Data masked by region of interest defined by polygon with vertices: "
            + str(ROI)
        )

    # Update analysis history
    ROI_selected_data = ROI_selected_data.history.assign(hist)

    return ROI_selected_data


def disp_from_hv(da, hv):
    """Function to extract a dispersion at a given hv from an hv scan, correcting for the kinetic energy offsets
    (KE_delta) that arise from using the hv scan loading method.

    Parameters
    ------------
    da : xarray.DataArray
        The hv scan extract a single dispersion from.

    hv : float
         The photon energy at which to extract a dispersion at.

    Returns
    ------------
    hv_disp : xarray.DataArray
        The extracted dispersion.

    Examples
    ------------
    Example usage is as follows::

        from peaks import *

        hv_scan = load('i05-1-54321.nxs')

        # Extract a dispersion from the hv_scan at 79 eV
        disp_79eV = hv_scan.disp_from_hv(79)

    """

    # Ensure the inputted data is an hv scan
    if "hv" not in da.dims:
        raise Exception(
            "The scan type of the inputted data is incompatible with disp_from_hv, which only extracts a "
            "single dispersion from an hv scan."
        )

    # Extract the relevant hv slice
    hv_scan = da.sel(hv=hv, method="nearest")

    # If the inputted data is in binding energy, we are done
    if "binding" in da.metadata.analyser.scan.eV_type.lower():
        return hv_scan

    # Rescale eV axis to get the correct kinetic energy
    orig_units = hv_scan.eV.units
    if isinstance(hv_scan.KE_delta.data, np.ndarray):
        hv_scan["eV"] = hv_scan.eV.data + hv_scan.KE_delta.data
    else:
        hv_scan["eV"] = (
            hv_scan.eV.data + hv_scan.KE_delta.pint.to(orig_units).pint.dequantify().data
        )  # Handle unit conversion
        hv_scan = hv_scan.pint.quantify(eV=orig_units)  # Add the units back

    # Delete the now redundant hv and KE_delta coordinates
    del hv_scan["hv"]
    del hv_scan["KE_delta"]

    # Update the hv attribute
    hv_scan.metadata.photon.set("hv", float(hv), add_history=False)

    # Update analysis history
    hv_scan = hv_scan.history.assign(
        "Dispersion extracted from hv scan at hv={hv} eV".format(hv=hv)
    )

    return hv_scan
