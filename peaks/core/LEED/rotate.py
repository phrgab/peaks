# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 14:59:38 2021

@author: lsh8
"""

import numpy as np
import xarray as xr
from scipy.ndimage.interpolation import rotate
from peaks.core.utils.OOP_method import add_methods

@add_methods(xr.DataArray)
def LEED_rotate(file, angle):
    """
    A function to rotate a LEED image

    Parameters
    ----------
    file : Xarray DataArray
        The Xarray containing the LEED image.
    angle : Float
        The angle to rotate the image.

    Returns
    -------
    None.

    """
    
    #Rotate the data
    rotated = rotate(file.data, angle=angle)
    
    #If the image is black and white
    if np.ndim(rotated) == 2:   
        
        #Determine size of dimensions
        pix = np.shape(rotated)
        y_pix, x_pix = pix[0], pix[1]
        
        #Create dimensions and coordinate names 
        y_range = np.arange(0,y_pix,1)
        x_range = np.arange(0,x_pix,1)
        
        #Label dim names and coords
        dim_names = ('y_range', 'x_range')
        coords = {'y_range': y_range, 'x_range': x_range, }
        
    else:
        
        #Determine size of dimensions
        pix = np.shape(rotated)
        y_pix, x_pix, col  = pix[0], pix[1], pix[2]
        
        #Create dimensions and coordinate names 
        y_range = np.arange(0,y_pix,1)
        x_range = np.arange(0,x_pix,1)
        col_range = np.arange(0,col,1)
        
        #Label dim names and coords
        dim_names = ('y_range', 'x_range', 'col_range')
        coords = {'y_range': y_range, 'x_range': x_range, 'col_range': col_range}
    
    ds = xr.DataArray(
        data=rotated,
        dims=dim_names,
        coords = coords,
        attrs = file.attrs)
    ds.coords['x_range'].attrs = {'units':'pixels'}
    ds.coords['y_range'].attrs = {'units' : 'pixels'}
    
    return ds