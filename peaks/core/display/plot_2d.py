# Utility Functions for particular displays of 2D data
# Phil King 03/05/2021

import numpy as np
import matplotlib.pyplot as plt
import warnings
from peaks.utils.OOP_method import add_methods
import xarray as xr


#Function to do a quick grid plot of multiple 2D dataarrays
@add_methods(xr.DataArray)
def plot_grid(data, ncols=3, nrows=None, titles=None, titles_col=None, sharex=False, sharey=False, **kwargs):
    '''Plots an array of 2D dataarrays on a grid

    Parameters
    ------------
    data : list(xr.DataArray)
         List containing the dispersions to be plotted

    ncols : int, optional
        Number of columns <br>
        Defaults to 3, or lower if <3 plots <br>
        Ignored if nrows is specified

    nrows : int, optional
        Number of rows <br>
        Overwrites ncols <br>
        Defaults to None (ncols used)

    titles : list(str)
        List of subtitles to be supplied on the plots <br>
        Length must match number of dataarrays to plot <br>
        Defaults to None

    titles_colors : list(str)
        List of colours for the subtitles <br>
        Length must match number of dataarrays to plot <br>
        Defaults to None

    sharex : bool, optional
        Share the x-axis scales <br>
        Defaults to False

    sharey : bool, optional
        Share the y-axis scales <br>
        Defaults to False

    **kwargs : optional
        Additional arguments to pass to the plot <br>
        Standard matplotlib calls

    Returns
    ------------
    plot

    '''

    
    #Number of plots
    nplots = len(data)
    
    #Check default columns is sensible
    if nplots < ncols:
        ncols = nplots
        
    #Number of rows required
    if nrows: #If nrows specified in call
        ncols = int(np.ceil(nplots/nrows)) #Overwrite default columns call
    else: #Set nrows based on ncols and nplots
        nrows = int(np.ceil(nplots/ncols))
    
    #If figsize not specified in kwargs, set a default
    if 'figsize' in kwargs:
        figsize = kwargs['figsize'] #set specified value
        kwargs.pop('figsize') #Remove from kwargs to not duplicate the call
    else: #set a default based on number of rows and colums
        figsize = (5*ncols,5*nrows)

    #Make the figure layout
    fig, axes = plt.subplots(ncols = ncols, nrows = nrows, figsize=figsize, sharex=sharex, sharey=sharey)
    #Clear any remaining subplot axes
    if nrows*ncols > nplots:
        for i in range(nplots,nrows*ncols):
            j0 = np.int(np.floor((i)/ncols)) #Work out grid positions
            j1 = np.int(i - ncols*j0)
            axes[j0][j1].axis('off')
            
    #Check for whether subtitles are to be displayed
    title_flag = 0
    title_col_flag = 0
    if titles:
        if len(titles) == nplots: #Check the correct number are provided
            title_flag = 1
            if len(titles_col) != 0:
                title_col_flag = 1
        else: #If not, don't use them and give a warning
            warnings.warn('Length of supplied titles list does not match number of plots. Supplied titles have not been used.')
    
    #Make the plots
    if nrows < 2 or ncols == 1: #Figure out whether this is a 1D or 2D grid
        for count, value in enumerate(data):
            value.plot(ax=axes[count], **kwargs) #Plot
            if title_flag == 1: #If subtitles to be displayed, update them here
                if title_col_flag == 1:  # If we are showing colours
                    axes[count].set_title(titles[count], color=titles_col[count])
                else:
                    axes[count].set_title(titles[count])
    else: #2D grid to display on
        j0=0;j1=0 #Some counters
        for count, value in enumerate(data):
            j0 = np.int(np.floor((count)/ncols)) #Work out grid positions
            j1 = np.int(count - ncols*j0)
            value.plot(ax=axes[j0][j1], **kwargs) #Plot
            if title_flag == 1: #If subtitles to be displayed, update them here
                if title_col_flag == 1:  # If we are showing colours
                    axes[j0][j1].set_title(titles[count], color=titles_col[count])
                else:
                    axes[j0][j1].set_title(titles[count])
    
    #Tidy up the layout
    plt.tight_layout()
