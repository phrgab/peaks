"""Functions used to extract selections of data.

"""

# Phil King 24/04/2021
# Brendan Edwards 03/12/2021
# Brendan Edwards 17/10/2023

import numpy as np
import xarray as xr
from matplotlib.path import Path
from peaks.core.utils.OOP_method import add_methods


@add_methods(xr.DataArray)
def DC(data, coord='eV', val=0, dval=0, ana_hist=True):
    """General function to extract DCs from data along any coordinate.

    Parameters
    ------------
    data : xr.DataArray
        The data to extract a DC from.

    coord : str (optional)
        Coordinate to extract DC at. Defaults to eV.

    val : float, list, np.ndarray, tuple (optional)
        DC value(s) to select. If tuple, must be in the format (start, end, step). Defaults to 0.

    dval : float (optional)
        Integration range (represents the total range, i.e. integrates over +/- dval/2). Defaults to 0.

    ana_hist : Boolean (optional)
        Defines whether the function appends information to the analysis history metadata. Defaults to True.

    Returns
    ------------
    dc : xr.DataArray
        Extracted DC(s).

    Examples
    ------------
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

    # If val is a 3 element tuple of the format (start, end, step), make val an np.ndarray of the relevant values
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
    dc = data.sel({coord: val}, method='nearest')

    # Apply extra binning if required
    if num_pixels > 1:
        for i in dc[coord]:
            dc.loc[{coord: i}] = data.sel({coord: slice(i - dval / 2, i + dval / 2)}).mean(coord, keep_attrs=True)

    # If returning a single DC, want to remove the non-varying coordinate as a dimension
    try:
        dc = dc.squeeze(coord)
    except:
        pass

    # Update the analysis history if ana_hist is True (will be False when DC is called from e.g. EDC, MDC, FS)
    if ana_hist:
        hist = 'DC extracted, integration window: ' + str(dval)
        dc.update_hist(hist)

    return dc


@add_methods(xr.DataArray)
def MDC(data, E=0, dE=0):
    """Extract MDCs from DataArrays.

    Parameters
    ------------
    data : xr.DataArray
        The dispersion to extract an MDC from.

    E : float, list, np.ndarray, tuple (optional)
        Energy (or energies) of MDC(s) to extract. If tuple, must be in the format (start, end, step). Defaults to 0.

    dE : float (optional)
        Integration range (represents the total range, i.e. integrates over +/- dE/2). Defaults to 0.

    Returns
    ------------
    mdc : xr.DataArray
        Extracted MDC(s).

    Examples
    ------------
    from peaks import *

    disp = load('disp.ibw')

    # Extract an MDC at eV = -1.2 +/- 0.005
    MDC1 = disp.MDC(E=-1.2, dE=0.01)

    # Extract an MDC at eV = 55.6 +/- 0.03
    MDC2 = disp.MDC(55.6, 0.06)

    # Extract a single non-integrated MDC at eV value closest to 0
    MDC3 = disp.MDC()

    # Extract MDCs at eV = 55.6 +/- 0.03 and 55.7 +/- 0.03
    MDC4 = disp.MDC([55.6, 55.7], 0.06)

    # Extract MDCs between eV = -0.2 and 0.1 in steps of 0.05 with +/- 0.01 integrations
    MDC5 = disp.MDC((-0.2, 0.1, 0.05), 0.02)

    """

    # Call function to extract relevant MDC from dispersion
    mdc = data.DC(coord='eV', val=E, dval=dE, ana_hist=False)

    # Update the analysis history
    hist = 'MDC extracted, integration window: ' + str(dE)
    mdc.update_hist(hist)

    return mdc


@add_methods(xr.DataArray)
def EDC(data, k=0, dk=0):
    """Extract EDCs from DataArrays.

    Parameters
    ------------
    data : xr.DataArray
        The dispersion to extract an EDC from.

    k : float, list, np.ndarray, tuple (optional)
        k or theta_par value(s) of EDC(s) to extract. If tuple, must be in the format (start, end, step). Defaults to 0.

    dk : float (optional)
        Integration range (represents the total range, i.e. integrates over +/- dk/2). Defaults to 0.

    Returns
    ------------
    edc : xr.DataArray
        Extracted EDC(s).

    Examples
    ------------
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
    coords.remove('eV')
    coord = coords[-1]  # Should always be the last one if data loading is consistent

    # Call function to extract relevant EDC from dispersion
    edc = data.DC(coord=coord, val=k, dval=dk, ana_hist=False)

    # Update the analysis history
    hist = 'EDC extracted, integration window: ' + str(dk)
    edc.update_hist(hist)

    return edc


@add_methods(xr.DataArray)
def FS(data, E=0, dE=0):
    """Extract constant energy slices, e.g. Fermi surfaces, from 3D DataArrays.

    Parameters
    ------------
    data : xr.DataArray
        The 3D Fermi map to extract an FS from.

    E : float, list, np.ndarray, tuple (optional)
        Energy (or energies) of slice(s) to extract. If tuple, must be in the format (start, end, step). Defaults to 0.

    dE : float (optional)
        Integration range (represents the total range, i.e. integrates over +/- dE/2). Defaults to 0.

    Returns
    ------------
    fs : xr.DataArray
        Extracted constant energy slice(s).

    Examples
    ------------
    from peaks import *

    FM = load('FM.zip')

    # Extract a constant energy slice at eV = -1.2 +/- 0.005
    FS1 = FM.FS(E=-1.2, dE=0.01)

    # Extract a constant energy slice at eV = 95.56 +/- 0.03
    FS2 = FM.FS(95.56, 0.06)

    # Extract a single non-integrated constant energy slice at eV value closest to 0
    FS3 = FM.FS()

    # Extract constant energy slices at eV = 95.56 +/- 0.03 and 95.60 +/- 0.03
    FS4 = FM.FS([95.56, 95.60], 0.06)

    # Extract constant energy slices between eV = -0.2 and 0.1 in steps of 0.05 with +/- 0.01 integrations
    FS5 = FM.FS((-0.2, 0.1, 0.05), 0.02)

    """

    # Check data is 3D
    if len(data.dims) != 3:
        raise Exception("Function only acts on 3D data.")

    # Call function to extract relevant constant energy slice from Fermi map
    fs = data.DC(coord='eV', val=E, dval=dE, ana_hist=False)

    # Update the analysis history
    hist = 'Constant energy slice extracted, integration window: ' + str(dE)
    fs.update_hist(hist)

    return fs


@add_methods(xr.DataArray)
def DOS(data):
    """Integrate over all but the energy axis to return the best approximation to the DOS possible from the data.

    Parameters
    ------------
    data : xr.DataArray
        Data to extract DOS from.

    Returns
    ------------
    dos : xr.DataArray
        Extracted DOS.

    Examples
    ------------
    from peaks import *

    disp = load('disp.ibw')

    FM = load('FM.zip')

    # Extract DOS of a dispersion
    disp_DOS = disp.DOS()

    # Extract DOS of a Fermi map
    FM_DOS = FM.DOS()

    """

    # Get relevant dimensions to integrate over
    int_dim = list(filter(lambda i: i != 'eV', data.dims))

    # Calculate the DOS
    dos = data.mean(int_dim, keep_attrs=True)

    # Update the analysis history
    hist = 'Integrated along axes: ' + str(int_dim)
    dos.update_hist(hist)

    return dos


@add_methods(xr.DataArray)
def tot(data, spatial_int=False):
    """Integrate spatial map data over all non-spatial (energy and angle/k) or all spatial dimensions.

    Parameters
    ------------
    data : xr.DataArray
        Spatial map data.

    spatial_int : Boolean (optional)
        Determines whether integration is performed over spatial or non-spatial dimensions. Defaults to False.

    Returns
    ------------
    data_tot : xr.DataArray
        The integrated data.

    Examples
    ------------
    from peaks import *

    SM = load('SM.ibw')

    # Extract energy and angle integrated spatial map
    SM_int = SM.tot()

    # Extract spatially integrated dispersion
    SM_int_spatial = SM.tot(spatial_int=True)

    """

    # Integrate over spatial dimensions
    if spatial_int:
        data_tot = data.mean(['x1', 'x2'], keep_attrs=True)
        hist = 'Integrated along axes: ' + str(['x1', 'x2'])

    # Integrate over non-spatial dimensions
    else:
        # Get relevant dimensions to integrate over
        int_dim = list(filter(lambda n: n != 'x1' and n != 'x2', data.dims))
        hist = 'Integrated along axes: ' + str(int_dim)
        data_tot = data.mean(int_dim, keep_attrs=True)

    # Update the analysis history
    data_tot.update_hist(hist)

    return data_tot


@add_methods(xr.DataArray)
def radial_cuts(data, num_azi=361, num_points=200, radius=2, **centre_kwargs):
    """Extract radial cuts of a Fermi surface as a function of azimuthal angle.

    Parameters
    ------------
    data : xr.DataArray
        Data to extract radial cuts from.

    num_azi : float (optional)
        Number of evenly spaced azi values between 0 and 360 degrees to take radial cuts. Defaults to 361.

    num_points : float (optional)
        Number of evenly spaced points to sample along a cut. Defaults to 200.

    radius : float (optional)
        Maximum radius to take cuts up to. Defaults to 2.

    **centre_kwargs : float (optional)
        Used to define centre of rotations in the format dim = coord, e.g. k_par = 1.2 sets the k_par centre as 1.2.
        Default centre of rotation is (0, 0).

    Returns
    ------------
    data_to_return : xr.DataArray
        Radial cuts against azi.

    Examples
    ------------
    from peaks import *

    FM = load('FM.zip')

    FS1 = FM.FS(E=-0, dE=0.01)

    # Extract radial cuts (radius = 15) at azi values in 2 degree increments, using a centre of
    # rotation (theta_par, ana_polar) = (-2, -5)
    FS1_radial_cuts_1 = FS1.radial_cuts(num_azi=181, num_points=300, radius=15, theta_par=-2, ana_polar=-5)

    # Extract radial cuts (radius = 2) at azi values in 1 degree increments, using a centre of
    # rotation (coord_1, coord_2) = (0, 0)
    FS1_radial_cuts_2 = FS1.radial_cuts()

    """

    # Check data is 2D
    if len(data.dims) != 2:
        raise Exception("Function only acts on 2D data.")

    # Define the coordinate system
    x_coord = data.dims[0]
    y_coord = data.dims[1]

    # Check for user-defined centre of rotations
    x_centre = centre_kwargs.get(x_coord)
    if not x_centre:
        x_centre = 0
    y_centre = centre_kwargs.get(y_coord)
    if not y_centre:
        y_centre = 0

    # Define coordinates to be sampled
    azi_angles = np.linspace(0, 360, num_azi)
    k_values = np.linspace(0, radius, num_points)
    spectrum = []

    # For each azi angle, interpolate the data onto a radial cut and append result to spectrum
    for angle in azi_angles:
        x_values = np.linspace(0 + x_centre, (np.cos(np.radians(angle)) * radius) + x_centre, num_points)
        y_values = np.linspace(0 + y_centre, (np.sin(np.radians(angle)) * radius) + y_centre, num_points)
        x_xarray = xr.DataArray(x_values, dims='k')
        y_xarray = xr.DataArray(y_values, dims='k')
        interpolated_data = data.interp({x_coord: x_xarray, y_coord: y_xarray})
        spectrum.append(interpolated_data.data)

    # Create xarray of radial cuts against azi
    data_to_return = xr.DataArray(np.array(spectrum).transpose(), dims=("k", "azi"),
                                  coords={"k": k_values, "azi": azi_angles})
    data_to_return.attrs = data.attrs

    # Update the analysis history
    data_to_return.update_hist('Radial cuts taken as a function of azi')

    return data_to_return


@add_methods(xr.DataArray)
def mask_data(data, ROI, return_integrated=True):
    """This function takes a multidimensional DataArray, and applies a polygon region of interest (ROI) as a mask. By
    default, the function will then extract the mean over the two dimensions defined by the ROI. For a rectangular
    ROI, this is equivalent to a simple .sel over those dimensions followed by a mean, but an arbitrary polygon can
    be used to define the ROI.

    Parameters
    ------------
    data : xr.DataArray
        The multidimensional data to apply the ROI selection to.

    ROI : dict
        A dictionary of two lists which contains the vertices of the polygon for the ROI definition, in the form
        {'dim1': [pt1, pt2, pt3, ...], 'dim2'=[pt1', pt2', pt3', ...]}. As many points can be specified as required,
        but this should be given with the same number of points for each dimension.

    return_integrated : Boolean (optional)
        Whether to mean the data confined within ROI region over the ROI dimensions, or instead return the masked data.
        Defaults to True.

    Returns
    ------------
    ROI_selected_data : xr.DataArray
        The input data with the ROI applied as a mask, and (if return_integrated=True) the mean taken over those
        remaining dimensions.

    Examples
    ------------
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
    err_str = ('ROI must be a dictionary containing two entries for the relevant axes. Each of these entries should '
               'be a list of the vertices of the polygon for the labelled axis of the ROI. These must be of equal '
               'length for the two axes.')
    if type(ROI) != dict or len(ROI) != 2:
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
    data_bounded = data.sel({dims[0]: slice(min(ROI[dims[0]]), max(ROI[dims[0]])),
                             dims[1]: slice(min(ROI[dims[1]]), max(ROI[dims[1]]))}).copy(deep=True)

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
        hist = 'Data averaged over region of interest defined by polygon with vertices: ' + str(ROI)
    else:  # Masked data to be returned
        ROI_selected_data = data_bounded.where(mask)
        hist = 'Data masked by region of interest defined by polygon with vertices: ' + str(ROI)

    # Update analysis history
    ROI_selected_data.update_hist(hist)

    return ROI_selected_data


def disp_from_hv():
    pass
