# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 11:00:13 2021

@author: lsh8
"""

import xarray as xr
import numpy as np
from peaks.core.utils.OOP_method import add_methods

@add_methods(xr.DataArray)
def rotation_disorder(data, bragg, center):
    
    if 'q_x' in data.dims and center[0] > 5:
        bragg[0] = (bragg[0] - center[0])*data.camera_factor*np.sqrt(data.energy/data.calibration_energy)
        bragg[1] = (bragg[1] - center[1])*data.camera_factor*np.sqrt(data.energy/data.calibration_energy)
        center = [0,0]
        
    #Calculate the raadius of a circle of points
    r = np.sqrt((bragg[0]-center[0])**2 +(bragg[1]-center[1])**2)
    
    theta = np.arange(0,2*np.pi,0.01)
    
    #Create the points in a circle that intersect the Bragg peaks
    circle_x = [r*np.sin(x) + center[0] for x in theta]
    circle_y = [r*np.cos(x) + center[1] for x in theta]
    
    x = xr.DataArray(circle_x, dims="z")
    y = xr.DataArray(circle_y, dims="z")
    
    #Using interpolation determine intensity at each point of this circle
    if 'x_range' in data.dims:
        circle_coords = data.interp(x_range=x, y_range=y)
        
        if bragg[0] > center[0]:
            radial_x = np.linspace(center[0], bragg[0]*2, num=100)
        else:
            radial_x = np.linspace(center[0], bragg[0]*0.5, num=100)
            
        if bragg[0] > center[0]:
            radial_y = np.linspace(center[1], bragg[1]*2, num=100)
        else:
            radial_y = np.linspace(center[1], bragg[1]*0.5, num=100)
        
        coords = {'z': np.linspace(center[0], r*np.sqrt(8) + center[0], num=100)}
    else: 
        circle_coords = data.interp(q_x=x, q_y=y)
        
        radial_x = np.linspace(center[0], bragg[0]*2, num=100)
        radial_y = np.linspace(center[1], bragg[1]*2, num=100)
        
        coords = {'z': np.linspace(center[0], r*np.sqrt(8), num=100)}
    


    xa = xr.DataArray(radial_x, dims="z", coords = coords)
    ya = xr.DataArray(radial_y, dims="z", coords = coords)
    
    #Using interpolation determine intensity at each point of this circle
    if 'x_range' in data.dims:
        radial_coords = data.interp(x_range=xa, y_range=ya)
    else: 
        radial_coords = data.interp(q_x=xa, q_y=ya)
    
    return circle_coords, radial_coords
    
    
#if __name__ == '__main__':
    #circle_coords = rotation_disorder(data, bragg, center)