#Function used to display matplotlib colourmap schemes
#Brendan Edwards 03/12/2021

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import xarray as xr
from IPython.display import display
from IPython.display import set_matplotlib_formats

def colours():
    ''' This function displays the colormap options avalible in matplotlib '''
    
    for colour in dir(pl.cm):
        if str(type(getattr(pl.cm, colour))) == "<class 'matplotlib.colors.LinearSegmentedColormap'>" or str(type(getattr(pl.cm, colour))) == "<class 'matplotlib.colors.ListedColormap'>":
            if not colour.endswith('_r'):
                display(getattr(pl.cm, colour))


def publish(mode=False, rendering='png', dpi=300, font='arial', fontsize =14, figsize=(4,3), cmap='Greys', cmap_divergent='seismic'):
    """
    easy way to change matplotlib settings for publishable figures

    mode: Boolean, activate (True) or not (False) the publish settings. If False the default matplotlib settings are restored disregarding all the other arguments.
    rendering: string, format of the inline image 'jpeg','pdf','svg','png','jpg','png2x','retina'
    dpi = the number pixel contained within one inch. 300 should be enough in most occasion. matplotlib default is only 100
    font: string, Arial as default
    fontsize = int, 14 as default
    figsize: touple as (horizontal size, vertical size), it doesn't work with plot_grid()
    cmap = non-divergent colormap
    cmap_divergent = divergent colormap
    """

    # change settings only if mode is True
    if mode:
        #set rendering format
        set_matplotlib_formats(str(rendering))
        #enhance DPI for better raster images, 300 should be enough
        plt.rcParams['figure.dpi'] = dpi
        # make text recognisable to Affinity for svg and pdf rendering
        plt.rcParams['svg.fonttype'] = 'none'
        plt.rcParams['pdf.fonttype'] = 'truetype'
        # change default figure size
        plt.rcParams["figure.figsize"] = figsize
        # change size font and fontsize
        plt.rcParams.update({'font.family':font})
        plt.rcParams.update({'font.size': fontsize})
        #set deafault colormap in xarray.
        xr.set_options(cmap_divergent=cmap_divergent, cmap_sequential=cmap)

    # if mode is False reset default setting and format = 'png'
    else:
        set_matplotlib_formats('png')
        matplotlib.rcdefaults()
        plt.rcParams['figure.dpi'] = 72
        plt.rcParams["figure.figsize"] = (6,4)
        plt.rcParams.update({'font.size': 10})
        xr.set_options(cmap_divergent='RdYlBl', cmap_sequential='viridis')

    # just a reminder
    print('heavy image needs to be rasterized with scan.plot(rasterized=True)')
