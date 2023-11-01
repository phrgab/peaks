"""Static in-line plotting functions.

"""

# Phil King 03/05/2021
# Brendan Edwards 31/10/2023

import numpy as np
import matplotlib.pyplot as plt
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
    if figsize == None:
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
                'Length of supplied titles list does not match number of plots. Supplied titles have not been used',
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