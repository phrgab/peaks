# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 14:08:59 2021

@author: lsh8
"""

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from peaks.utils.OOP_method import add_methods
import xarray as xr

@add_methods(xr.DataArray)
def LEED_3d_plot(xarray, sampling = 100):
    """
    Create a 3D image of LEED-iv data

    Parameters
    ----------
    xarray : Xr.DataArray
        LEED data.
    sampling : int, optional
        Higher values give better resolution, normal is ~100. The default is 100.

    Returns
    -------
    None.

    """
    
    #First create high energy LEED image
    
    #Find max and min of eV values
    eV_max = np.max(xarray.coords['eV'].values)
    eV_min = np.min(xarray.coords['eV'].values)

    # Make things more legible
    x = xarray.coords['q_x'].values
    y = xarray.coords['q_y'].values
    
    # Create axes
    xy, yx = np.meshgrid(x, y, indexing = 'ij')
    
    z = np.full((np.shape(xarray.sel(eV=eV_min).data)), eV_min)
    a = np.nan_to_num(xarray.data, 0)
    scale = np.max(a)
    
    # Make the plot
    fig = plt.figure(figsize=(8,8))
    ax  = fig.gca(projection='3d') 
    ax.plot_surface(xy, yx, z.T, shade=False, facecolors=cm.viridis((xarray.sel(eV=eV_min).data.T)/scale),
                        vmin=-1, vmax=1, antialiased=True, rcount=sampling, ccount=sampling, alpha = 1)
    
    #----------------------------------------------------------------------------------------------------------------------
    # Low energy LEED
    x_min = np.min(xarray.coords['q_x'].values)
    x = xarray.coords['q_x'].sel(q_x = slice(x_min, 0)).values
    y = xarray.coords['q_y'].values     
    
    # Convert the 4d-space's dimensions into grids
    xy, yx = np.meshgrid(x, y, indexing = 'ij')
    
    z = np.full((np.shape(xarray.sel(eV=eV_max).sel(q_x = slice(x_min, 0)).data)), eV_max)
    a = np.nan_to_num(xarray.data, 0)
    scale = np.max(a)
    
    ax.plot_surface(xy, yx, z.T, shade=False, facecolors=cm.viridis((xarray.sel(eV=eV_max).sel(q_x = slice(x_min, 0)).data.T)/scale),
                        vmin=-1, vmax=1, antialiased=True, rcount=sampling, ccount=sampling, alpha = 1)
    
    
    #------------------------------------------------------------------------------------------------------------------------
    
    #LEED iv data
    y = xarray.coords['q_y'].values
    z = xarray.coords['eV'].values    
    
    # Convert the 4d-space's dimensions into grids
    yz, zy = np.meshgrid(y, z, indexing = 'ij')
    
    x = np.full((np.shape(xarray.sel(q_x=0, method = 'nearest').data)), 0)
    
    ax.plot_surface(x, yz, zy, shade=False, facecolors=cm.viridis((xarray.sel(q_x = 0, method = 'nearest').data)/scale),
                        vmin=-1, vmax=1, antialiased=True, rcount=sampling, ccount=sampling, alpha =1)
    
    #--------------------------------------------------------------------------------------------------------------------------
    
    ax.azim = 0
    ax.elev = 35
    ax.set_xlabel('q_x')
    ax.set_ylabel('q_y')
    ax.set_zlabel('eV')
    plt.show()