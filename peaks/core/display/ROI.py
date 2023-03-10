#A collection of functions designed for spatially-resolved ARPES data display/processing
#Phil King 15/5/21

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.path import Path
from peaks.utils.metadata import update_hist
from peaks.utils.OOP_method import add_methods


# Function to select a 2D region of interest from data, defined by a specified polygon
@add_methods(xr.DataArray)
def ROI_select(data, ROI_in, return_masked=False):
    '''This function takes a multi-dimensional data array, takes the mean over 2 of these directions defined by a polygon region of interest. If this is rectangular, it is equivalent to a simple .sel over those dimensions followed by a mean, but an arbitrary polygon can be used to defined the ROI.

    Input:
        data - The multi-dimensional data to apply the ROI to (xarray)
        ROI_in - Dictionary of two lists, containing the vertices of the polygon for the ROI definition, in the form {'dim1': [pt1,pt2,pt3,...], 'dim2'=[pt1',pt2',pt3',...]}. As many points can be specified as required, but this should be given with the same number of points for each dimension.
        return_masked (optional, default=False) - call with return_masked=True to return the masked data, rather than taking the mean over the ROI dimensions

    Returns:
        masked_data - The original data with the ROI applied as a mask, and the mean taken over those remaining dimensions depending on return_masked flag'''

    # Check function has been fed with suitable dictionary for ROI generation
    err_str = 'ROI should be given via a dictionary containing two enteries for the relevant axes. Each of these should contain a list of the vertices of the polygon for the lavelled axis of the ROI. These must be of equal length for the two axes. Optionally call with return_masked=True to return masked rather than averaged data.'
    if type(ROI_in) != dict:
        raise Exception(err_str)
    elif len(ROI_in) != 2:
        raise Exception(err_str)
    else: #Seems correct format
        # Determine relevant dimensions for ROI
        dims = list(ROI_in)

        #Check lengths match
        if len(ROI_in[dims[0]]) != len(ROI_in[dims[1]]):
            raise Exception(err_str)

    # Define ROI and make a polygon path defining the ROI
    ROI = []
    for i in range(len(ROI_in[dims[0]])):
        ROI.append((ROI_in[dims[0]][i], ROI_in[dims[1]][i]))
    p = Path(ROI)  # make a polygon defining the ROI

    # Restrict the data cube down to the minimum possible size (making this as a copy of the relevant data to avoid overwriting problems)
    data_bounded = data.sel({dims[0]: slice(min(ROI_in[dims[0]]), max(ROI_in[dims[0]])),
                             dims[1]: slice(min(ROI_in[dims[1]]), max(ROI_in[dims[1]]))}).copy(deep=True)

    # Broadcast coordinate data
    b, dim0 = xr.broadcast(data_bounded, data_bounded[dims[0]])
    b, dim1 = xr.broadcast(data_bounded, data_bounded[dims[1]])

    # Convert coordinate data into a format for passing to matplotlib path function for identifying which points are in the relevant ROI
    points = np.vstack((dim0.data.flatten(), dim1.data.flatten())).T

    # Check which of these points fall within our ROI
    grid = p.contains_points(points, radius=0.01)
    # Reshape to make a data mask
    mask = grid.reshape(data_bounded.shape)

    if return_masked == False:  # Data should be averaged over the ROI dimensions
        masked_data = data_bounded.where(mask).mean(dims, keep_attrs=True)
        # Update history string
        hist = 'Data averaged over region of interest defined by polygon with vertices: ' + str(ROI_in)
    else:  # Masked data to be returned
        masked_data = data_bounded.where(mask)
        # Update history string
        hist = 'Data masked by region of interest defined by polygon with vertices: ' + str(ROI_in)

    # Update history
    update_hist(masked_data, hist)

    return masked_data


# Plot the ROI
def ROI_plot(ROI, **kwargs):
    '''This function plots an ROI.

    Input:
        ROI - Dictionary of two lists, containing the vertices of the polygon for the ROI definition, in the form {'dim1': [pt1,pt2,pt3,...], 'dim2'=[pt1',pt2',pt3',...]}. As many points can be specified as required, but this should be given with the same number of points for each dimension.
        **kwargs:
            - y = dim (str) to specify which dimension to plot in the y-direction
            - x = dim (str) to specify which dimension to plot in the y-direction (only one of these needs to be specified, the other is set automatically)
            - label (string), optional argument to pass to a legend to label the ROI
            - loc, standard matplotlib call to specify legend location
            - other standard matplotlib calls to pass to the plot, including ax= for specific axis call

    Returns:
        plot - Makes a new plot of the ROI, or appends it to the existing plot if called from the same cell.'''

    # Determine relevant dimensions for ROI
    dims = list(ROI)

    # Determine lists of vertices of ROI
    verts0 = ROI[dims[0]]
    verts1 = ROI[dims[1]]

    # Append initial value to end to close polygon
    verts0.append(verts0[0])
    verts1.append(verts1[0])

    # See if definition of y and x axis is given
    yax = 1  # default
    if 'y' in kwargs:
        yax = dims.index(kwargs['y'])
        kwargs.pop('y')
    if 'x' in kwargs:
        yax = 1 - dims.index(kwargs['x'])
        kwargs.pop('x')

    # Check if there is a label to feed to the legend
    leg_flag = 0
    if 'label' in kwargs:
        leg_flag = 1

    # Check if there is a legend location label supplied
    if 'loc' in kwargs:
        loc = kwargs['loc']
        kwargs.pop('loc')
    else:
        loc = 'best'

    # Plot the ROI
    ax_flag = 0
    if 'ax' in kwargs:  # If particular axis specified
        ax = kwargs['ax']
        ax_flag = 1
        kwargs.pop('ax')
        if yax == 0:
            ax.plot(verts1, verts0, **kwargs)
        else:
            ax.plot(verts0, verts1, **kwargs)
    else:  # Otherwise just call with plt
        if yax == 0:
            plt.plot(verts1, verts0, **kwargs)
        else:
            plt.plot(verts0, verts1, **kwargs)

    # If called with label, plot the legend
    if leg_flag == 1:
        if ax_flag == 1:
            ax.legend(loc=loc)
        else:
            plt.legend(loc=loc)