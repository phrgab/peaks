"""Static in-line plotting functions.

"""

# Phil King 03/05/2021
# Brendan Edwards 31/10/2023

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from cycler import cycler
from peaks.core.utils.misc import analysis_warning


def plot_grid(data, ncols=3, nrows=None, titles=None, sharex=False, sharey=False, figsize=None, **kwargs):
    """Plots an array of 2D DataArrays on a grid.

    Parameters
    ------------
    data : list
         List containing the 2D items (in xr.DataArray format) to be plotted.

    ncols : int (optional)
        Number of columns. Ignored if nrows is specified. Defaults to 3 (or lower if <3 plots).

    nrows : int (optional)
        Number of rows. Overwrites ncols. Defaults to None (i.e. ncols is used).

    titles : list (optional)
        List of subtitles to be supplied on the plots. Length must match length of the data list. Defaults to None.

    sharex : Boolean (optional)
        Whether to have the plots share an x-axis. Defaults to False.

    sharey : Boolean (optional)
        Whether to have the plots share a y-axis. Defaults to False.

    figsize : tuple (optional)
        Size of figure to be plotted. Defaults to None.

    **kwargs (optional)
        Additional standard matplotlib calls arguments to pass to the plot.

    Examples
    ------------
    from peaks import *

    disp1 = load('disp1.ibw')
    disp2 = load('disp2.ibw')
    disp3 = load('disp3.ibw')
    disp4 = load('disp4.ibw')

    disps_to_plot = [disp1, disp2, disp3, disp4]
    disp_titles = ['10 K', '20 K', '30 K', '40 K']

    plot_grid(disps_to_plot)  # Plots dispersions on a 2D grid

    plot_grid(disps_to_plot, titles=disp_titles, sharex=True, add_colorbar=False)  # Plots dispersions on a 2D grid,
                    where each dispersion has a title, all dispersions share an x-axis, and colour bars are not shown

    """

    # Number of plots
    nplots = len(data)

    # Check default columns is sensible
    if nplots < ncols:
        ncols = nplots

    # Number of rows required
    if nrows:  # If nrows specified in call
        ncols = int(np.ceil(nplots / nrows))  # Overwrite default columns call
    else:
        nrows = int(np.ceil(nplots / ncols))  # Set nrows based on ncols and nplots

    # If figsize not specified, set a default based on number of rows and columns
    if not figsize:
        figsize = (5 * ncols, 5 * nrows)

    # Make the figure layout
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize, sharex=sharex, sharey=sharey)

    # Clear any remaining subplot axes
    if nrows * ncols > nplots:
        for i in range(nplots, nrows * ncols):
            # Work out grid positions
            j0 = int(np.floor((i) / ncols))
            j1 = int(i - ncols * j0)
            axes[j0][j1].axis('off')

    # Check for whether plot titles are to be displayed
    plot_titles = False
    if titles:
        if len(titles) == nplots:  # Check the correct number are provided
            plot_titles = True
        else:  # If not, don't use them and give a warning
            analysis_warning(
                'Length of supplied titles list does not match number of plots. Supplied titles have not been used.',
                title='Plotting info', warn_type='danger')

    # Make the plots
    if nrows < 2 or ncols == 1:  # Figure out whether this is a 1D or 2D grid
        for count, value in enumerate(data):
            value.plot(ax=axes[count], **kwargs)  # Plot data
            if plot_titles:  # If plot titles to be displayed, update them here
                axes[count].set_title(titles[count])
    else:  # 2D grid to display on
        j0 = 0
        j1 = 0  # Some counters
        for count, value in enumerate(data):
            j0 = int(np.floor(count / ncols))  # Work out grid positions
            j1 = int(count - ncols * j0)
            value.plot(ax=axes[j0][j1], **kwargs)  # Plot data
            if plot_titles:  # If plot titles to be displayed, update them here
                axes[j0][j1].set_title(titles[count])

    # Tidy up the layout
    plt.tight_layout()


def plot_DCs(DCs, titles=None, cmap='coolwarm', color=None, offset=0, norm=False, stack_dim='auto', figsize=(6,6),
             linewidth=1, ax=None, **kwargs):
    """Plot a DC stack with the colours varying according to a colormap.

    Parameters
    ------------
    DCs : xr.DataArray, list
         DCs to plot, either as a list of single DC DataArrays, or a single xr.DataArray with the stack of DCs included.

    titles : list (optional)
        List of DC labels to be used in the legend. Length must match number of DCs to plot. Defaults to None.

    cmap : str (optional)
        Matplotlib cmap to use for line colors. Defaults to coolwarm.

    color : str (optional)
        Single matplotlib color to use for all lines. Takes precedence over cmap if defined. Defaults to None.

    offset : float (optional)
        Vertical offsets between subsequent DCs, represented as a fraction of the maximum peak height. Defaults to 0.

    norm : Boolean (optional)
        Whether to normalise the DCs to 1. Defaults to False.

    stack_dim : str (optional)
        Which dimension to stack DCs along. Ignored if a list of DCs is passed. Defaults to 'auto' which takes the
        smallest dimension.

    figsize : tuple (optional)
        Size of figure to be plotted. Defaults to None.

    linewidth : float (optional)
        Width of lines to be plotted. Defaults to 1.

    ax : numpy.ndarray (optional)
        Specific matplotlib axis call. Defaults to None.

    **kwargs (optional)
        Additional standard matplotlib calls arguments to pass to the plot.

    Examples
    ------------
    from peaks import *

    disp1 = load('disp1.ibw')

    EDCs_to_plot = disp.EDC(k=[0, 0.1, 0.2, 0.3, 0.4, 0.5], dk=0.01)

    plot_DCs(to_plot)  # Plots EDCs

    MDC_1 = disp.MDC(E=95.61, dE=0.01)
    MDC_2 = disp.MDC(E=95.62, dE=0.01)
    MDC_3 = disp.MDC(E=95.63, dE=0.01)

    plot_DCs([MDC_1, MDC_2, MDC_3], titles = ['EF', 'EF - 0.01 eV', 'EF -0.02 eV'], norm=True, offset=0.05)  # Plots
                                            normalised MDCs with a 5% (of maximum height) offset, and adds legend.

    """

    # If a list of DCs is supplied, combine them into a single DataArray
    if isinstance(DCs, list):
        DC_array = xr.concat(DCs, dim='DC_no')
        stack_dim = 'DC_no'

    # If a DataArray is instead supplied, finding the stacking dimension if not defined
    else:
        DC_array = DCs.copy(deep=True)  # Make a copy so we can safely modify
        if stack_dim == 'auto':
            # Assume the stacking dimension is the smallest dimension size
            if DC_array.shape[0] < DC_array.shape[1]:
                stack_dim = DC_array.dims[0]
            else:
                stack_dim = DC_array.dims[1]

    # Check correct dimension supplied
    if len(DC_array.shape) != 2:
        raise Exception(
            'Incorrect dimension of data supplied. Please supply a single 2D DataArray or list of 1D DataArrays.')

    # Get non-stacking dimension
    other_dim = [dim for dim in DC_array.dims if dim != stack_dim][0]

    # Normalise if required
    if norm:
        for coord in DC_array[stack_dim]:
            DC_array.loc[{stack_dim: coord}] = DC_array.loc[{stack_dim: coord}] / DC_array.loc[{stack_dim: coord}].max()

    # Get absolute offset from fractional offset
    absolute_offset = offset * float(DC_array.max())

    # Define linear offset wave in steps of the supplied or guessed offset
    offset_data = xr.DataArray(data=np.arange(len(DC_array[stack_dim])) * absolute_offset, dims=stack_dim,
                               coords={stack_dim: DC_array[stack_dim]})

    # Add offset to DC data
    DC_array += offset_data

    # If no tiles are provided, no legend will be displayed
    if not titles:
        kwargs['add_legend'] = False

    # If x and y are not defined, make a guess for which axes to plot the dimensions on.
    if 'y' not in kwargs and 'x' not in kwargs:
        # If it seems to be an EDC plot, set vertical by default
        if other_dim == 'eV':
            kwargs['y'] = 'eV'
        # For anything else, set horizontal
        else:
            kwargs['x'] = other_dim

    # Set up plot
    if ax:
        kwargs['ax'] = ax
    else:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    # If the user has requested to plot lines of a single color
    if color:
        DC_array.plot.line(hue=stack_dim, linewidth=linewidth, color=color, **kwargs)

    # Else use a colormap
    else:
        # Define the colour scheme
        cols = getattr(cm, cmap)(np.linspace(0, 1, len(DC_array[stack_dim])))
        cmap = colors.ListedColormap(cols)
        custom_cycler = cycler(color=cmap.colors)
        ax.set_prop_cycle(custom_cycler)
        DC_array.plot.line(hue=stack_dim, linewidth=linewidth, **kwargs)

    # Add axis titles
    if 'y' in kwargs:
        plt.xlabel("Intensity [arb. units]")
    else:
        plt.ylabel("Intensity [arb. units]")

    # Add legend if user has inputted titles
    if titles:
        if len(titles) == len(DC_array[stack_dim]):
            for ct, i in enumerate(ax.lines):
                i.set_label(titles[ct])
            ax.legend()
        else:
            analysis_warning(
                'Length of supplied titles list does not match number of DCs. Supplied titles have not been used.',
                title='Plotting info', warn_type='danger')

    # Tidy up the layout
    plt.tight_layout()


def plot_ROI(ROI, color='black', x=None, y=None, label=None, loc='best', ax=None, **kwargs):
    """This function plots a region of interest (ROI).

    Parameters
    ------------
    ROI : dict
        A dictionary of two lists which contains the vertices of the polygon for the ROI definition, in the form
        {'dim1': [pt1, pt2, pt3, ...], 'dim2'=[pt1', pt2', pt3', ...]}. As many points can be specified as required,
        but this should be given with the same number of points for each dimension.

    color : str (optional)
        Matplotlib color to use for ROI plot. Defaults to 'black'.

    x : str (optional)
        Specifies which coordinate to plot along the x-axis. Takes precedence over y (only one of x and y need to be
        specified, the other is set automatically). Defaults to the first coordinate listed in the ROI dictionary.

    y : str (optional)
        Specifies which coordinate to plot along the y-axis. Defaults to the second coordinate listed in the ROI
        dictionary.

    label : str (optional)
        ROI label to pass to a legend. Defaults to None (where no legend is plotted).

    loc : str, int (optional)
        Standard matplotlib call to specify legend location. Defaults to 'best'.

    ax : numpy.ndarray (optional)
        Specific matplotlib axis call. Defaults to None.

    **kwargs (optional)
        Additional standard matplotlib calls arguments to pass to the plot.

    Examples
    ------------
    from peaks import *

    disp = load('disp1.ibw')

    ROI1 = {'theta_par': [-8, -5.5, -3.1, -5.6], 'eV': [95.45, 95.45, 95.77, 95.77]}
    ROI2 = {'theta_par': [-9, -4, -2.5, -5.9], 'eV': [95.45, 95.45, 95.77, 95.77]}

    plot_ROI(ROI1)  # Plots ROI.
    plt.show()

    disp1.plot()
    plot_ROI(ROI1, x='eV')  # Plots ROI on top of a dispersion with eV plotted along the x-axis.
    plt.show()

    fig, axes= plt.subplots(ncols=2)
    disp.T.plot(ax=axes[0])
    plot_ROI(ROI1, label='ROI 1', ax=axes[0])  # Plot ROI on top of a dispersion on specific axis, axes[0], with a label
    disp.T.plot(ax=axes[1])
    plot_ROI(ROI2, label='ROI 2', ax=axes[1])  # Plot ROI on top of a dispersion on specific axis, axes[1], with a label

    """

    # Determine relevant dimensions for ROI
    dims = list(ROI)

    # Determine lists of vertices of ROI
    verts0 = ROI[dims[0]]
    verts1 = ROI[dims[1]]

    # Append initial value to end to close polygon
    verts0.append(verts0[0])
    verts1.append(verts1[0])

    # By default, plot the second dimension as the y-axis
    yax = 1

    # Check if definition of x- or y-axis is given
    if x:
        yax = 1 - dims.index(x)
    elif y:
        yax = dims.index(y)

    # Plot the ROI
    if ax:  # If particular axis specified
        if yax == 0:
            ax.plot(verts1, verts0, color=color, label=label, **kwargs)
        else:
            ax.plot(verts0, verts1, color=color, label=label, **kwargs)
    else:  # Otherwise just call with plt
        if yax == 0:
            plt.plot(verts1, verts0, color=color, label=label, **kwargs)
        else:
            plt.plot(verts0, verts1, color=color, label=label, **kwargs)

    # If called with label, plot the legend
    if label:
        if ax:  # If particular axis specified
            ax.legend(loc=loc)
        else:  # Otherwise just call with plt
            plt.legend(loc=loc)

    # Tidy up the layout
    plt.tight_layout()


def plot_nanofocus():
    pass
