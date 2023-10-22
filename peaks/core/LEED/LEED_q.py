# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 11:01:43 2021

@author: lsh8
"""

import numpy as np
import xarray as xr
from math import sqrt
from peaks.core.utils.OOP_method import add_methods

@add_methods(xr.Dataset)
def LEED_q_convert(file, dqx = None, dqy = None):
    """
    Conversioon routine from pixel to q space

    Parameters
    ----------
    file : Xarray
        Input LEED data.

    Returns
    -------
    ds : Xarray dataset
        Converted LEED data.

    """
    
    dim_names = ('q_y', 'q_x')
    adict = {}
    
    if isinstance(file, xr.core.dataset.Dataset):
        for ii in file.data_vars:
            if file[ii].attrs['center'] != None and file[ii].attrs['energy'] != 0:
                if len(file[ii].dims) == 2:
                    q_x = (file[ii].coords['x_range'].values - file[ii].center[0])*file.camera_factor*sqrt(file[ii].energy/file.calibration_energy)
                    q_y = (file[ii].coords['y_range'].values - file[ii].center[1])*file.camera_factor*sqrt(file[ii].energy/file.calibration_energy)
                    coords = {'q_y': q_y, 'q_x': q_x}  
                    
                    q_x_min = np.min(q_x)
                    q_x_max = np.max(q_x)
                    q_y_min = np.min(q_y)
                    q_y_max = np.max(q_y)
                    
                    qx = np.linspace(q_x_min, q_x_max, len(file.x_range.data))
                    qy = np.linspace(q_y_min, q_y_max, len(file.y_range.data))
                    
                    test = xr.DataArray(file[ii].data, dims=dim_names, coords = coords, attrs=file[ii].attrs)
                    test = test.interp(q_x = qx, q_y = qy)
                    
                    test.coords['q_x'].attrs = {'units':r'$\AA^{-1}$'}
                    test.coords['q_y'].attrs = {'units' :r'$\AA^{-1}$'}
                    tail = test.attrs["scan_name"]
                    adict[tail]= test
                else:
                    #q is maximum at max E
                    E_max = np.max(file[ii].coords['eV'].values)
                    
                    #calculate q_x and q_y for maximum E
                    q_x = (file[ii].coords['x_range'].values - file[ii].center[0])*file.camera_factor*np.sqrt(E_max/file.calibration_energy)
                    q_y = (file[ii].coords['y_range'].values - file[ii].center[1])*file.camera_factor*np.sqrt(E_max/file.calibration_energy)
                    
                    #Limits on q
                    q_x_min = np.min(q_x)
                    q_x_max = np.max(q_x)
                    q_y_min = np.min(q_y)
                    q_y_max = np.max(q_y)
                    
                    if dqx is None:
                        dqx = len(file.x_range.data)
                        qx = np.linspace(q_x_min, q_x_max, dqx)
                    else:
                        qx = np.arange(q_x_min,q_x_max +dqx, dqx)
                    if dqy is None:
                        dqy = len(file.y_range.data)
                        qy = np.linspace(q_y_min, q_y_max, dqy)
                    else:
                        qy = np.arange(q_y_min,q_y_max +dqy, dqy)
                    
                    
                    #Create grid of q values
                    qxy, qyx = np.meshgrid(qx, qy, indexing = 'ij')
        
                    #Backward convert to pixels
                    xra = (np.sqrt(file.calibration_energy/file[ii].coords['eV'].values)*qxy[:,:,np.newaxis]/file.camera_factor) + file[ii].center[0]
                    yra = (np.sqrt(file.calibration_energy/file[ii].coords['eV'].values)*qyx[:,:,np.newaxis]/file.camera_factor) + file[ii].center[1]       
                    
                    #Create arrays for xpixel, ypixel and energy
                    x_pixel_from_q_xarray = xr.DataArray(xra, dims=['q_x', 'q_y', 'eV'], coords={'q_x': qx, 'q_y': qy, 'eV': file[ii].coords['eV'].data})
                    y_pixel_from_q_xarray = xr.DataArray(yra, dims=['q_x', 'q_y', 'eV'], coords={'q_x': qx, 'q_y': qy, 'eV': file[ii].coords['eV'].data})
                    eV_xarray = xr.DataArray(file[ii].coords['eV'].data, dims=['eV'], coords={'eV': file[ii].coords['eV'].data})
                    
                    # Do the interpolation
                    test = file[ii].interp({'y_range': y_pixel_from_q_xarray, 'x_range': x_pixel_from_q_xarray, 'eV': eV_xarray})
                    test = test.T
                    
                    #Assign units and metadata
                    test.coords['q_x'].attrs = {'units':r'$\AA^{-1}$'}
                    test.coords['q_y'].attrs = {'units' :r'$\AA^{-1}$'}
                    tail = test.attrs["scan_name"]
                    adict[tail]= test
                    
        #Create DataSet containing multiple DataArrays
        ds = xr.Dataset(data_vars=adict)
        ds.attrs=file.attrs
        
    else:
            if file.attrs['center'] != None and file.attrs['energy'] != 0:
                if len(file.dims) == 2:
                    q_x = (file.coords['x_range'].values - file.center[0])*file.camera_factor*sqrt(file.energy/file.calibration_energy)
                    q_y = (file.coords['y_range'].values - file.center[1])*file.camera_factor*sqrt(file.energy/file.calibration_energy)
                    coords = {'q_y': q_y, 'q_x': q_x}  
                    
                    q_x_min = np.min(q_x)
                    q_x_max = np.max(q_x)
                    q_y_min = np.min(q_y)
                    q_y_max = np.max(q_y)
                    
                    qx = np.linspace(q_x_min, q_x_max, len(file.x_range.data))
                    qy = np.linspace(q_y_min, q_y_max, len(file.y_range.data))
                    
                    test = xr.DataArray(file[ii].data, dims=dim_names, coords = coords, attrs=file.attrs)
                    test = test.interp(q_x = qx, q_y = qy)
                    
                    test.coords['q_x'].attrs = {'units':r'$\AA^{-1}$'}
                    test.coords['q_y'].attrs = {'units' :r'$\AA^{-1}$'}
                    tail = test.attrs["scan_name"]
                    adict[tail]= test
                else:
                    #q is maximum at max E
                    E_max = np.max(file.coords['eV'].values)
                    
                    #calculate q_x and q_y for maximum E
                    q_x = (file.coords['x_range'].values - file.center[0])*file.camera_factor*np.sqrt(E_max/file.calibration_energy)
                    q_y = (file.coords['y_range'].values - file.center[1])*file.camera_factor*np.sqrt(E_max/file.calibration_energy)
                    
                    #Limits on q
                    q_x_min = np.min(q_x)
                    q_x_max = np.max(q_x)
                    q_y_min = np.min(q_y)
                    q_y_max = np.max(q_y)
                    
                    if dqx is None:
                        dqx = len(file.x_range.data)
                        qx = np.linspace(q_x_min, q_x_max, dqx)
                    else:
                        qx = np.arange(q_x_min,q_x_max +dqx, dqx)
                    if dqy is None:
                        dqy = len(file.y_range.data)
                        qy = np.linspace(q_y_min, q_y_max, dqy)
                    else:
                        qy = np.arange(q_y_min,q_y_max +dqy, dqy)
                    
                    
                    #Create grid of q values
                    qxy, qyx = np.meshgrid(qx, qy, indexing = 'ij')
        
                    #Backward convert to pixels
                    xra = (np.sqrt(file.calibration_energy/file.coords['eV'].values)*qxy[:,:,np.newaxis]/file.camera_factor) + file.center[0]
                    yra = (np.sqrt(file.calibration_energy/file.coords['eV'].values)*qyx[:,:,np.newaxis]/file.camera_factor) + file.center[1]       
                    
                    #Create arrays for xpixel, ypixel and energy
                    x_pixel_from_q_xarray = xr.DataArray(xra, dims=['q_x', 'q_y', 'eV'], coords={'q_x': qx, 'q_y': qy, 'eV': file.coords['eV'].data})
                    y_pixel_from_q_xarray = xr.DataArray(yra, dims=['q_x', 'q_y', 'eV'], coords={'q_x': qx, 'q_y': qy, 'eV': file.coords['eV'].data})
                    eV_xarray = xr.DataArray(file.coords['eV'].data, dims=['eV'], coords={'eV': file.coords['eV'].data})
                    
                    # Do the interpolation
                    test = file.interp({'y_range': y_pixel_from_q_xarray, 'x_range': x_pixel_from_q_xarray, 'eV': eV_xarray})
                    test = test.T
                    
                    #Assign units and metadata
                    test.coords['q_x'].attrs = {'units':r'$\AA^{-1}$'}
                    test.coords['q_y'].attrs = {'units' :r'$\AA^{-1}$'}
                    ds = test
    
    return ds