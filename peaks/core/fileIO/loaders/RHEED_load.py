# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 10:31:07 2021

@author: lsh8
"""

import numpy as np
import cv2
import xarray as xr
import glob
import os
from peaks.core.utils.OOP_method import add_methods

@add_methods(xr)
def RHEED_load(filename, bit_depth = 28, dataset=None, color = False, sample_description = None, \
              start_time = None, time=None, date=None, substrate=None, sample=None, \
            space_group=None, temperature=None, growth_history=None, pressure=None, \
                azi=None, beam_direction=None, beam_x=None, beam_y=None, \
                rocking_angle=None, filament_current=None, energy=0, \
                focus=None, grid=None, z =None, calibration=None,center=None, \
                camera_factor=None, calibration_energy=None):
    """
    A function to load LEED images into an Xarray DataSet

    Parameters
    ----------
    filename : String
        Name of file.
    bit_depth : int
        Bit depth of the sinlge image. usually 28 (Default) or 12
    iv : BOOLEAN, optional
        True = LEED-iv measurement. The default is False.
    dataset : Xarray DataSet, optional
        Input name of a dataset to add too. The default is False.
    calibration : String, optional
        Name of file to use as calibration. The default is None.
    center : Tuple, optional
        Pixels coresponding to the zeroeth Bragg peak. The default is None.
    camera_factor : Tuple, optional
        Ratio of pixel to angle. The default is None.
    calibration_energy : Float, optional
        Electron energy of the calibration LEED image. The default is None.
    energy : Float, optional
        Energy of the LEED data. The default is 0.
    polar : Float, optional
        An angle (horiz axis). The default is 0.
    tilt : Float, optional
        An angle (verticle axis). The default is 0.
    azi : Float, optional
        Rotation angle. The default is 0.
    x1 : FLoat, optional
        Manipulator position. The default is 0.
    x2 : Float, optional
        Manipulator position. The default is 0.
    x3 : Float, optional
        Manipulator position. The default is 0.


    Returns
    -------
    ds : Xarray Dataset
        A dataset that acts like a dictionary of Xarray DataArrays.

    """
    
    #Create metadata for dataset
    dataset_attrs ={
    'calibration': calibration,
    'center' : center,
    'camera_factor': camera_factor,
    'scan_type': 'RHEED'}
    
    if filename[-4:] == '.txt':
        #Extract filename
        head, tail = os.path.split(filename)
        tail = tail.split(".", 1) 
        
        #Metadata for Xarray DataArray
        dataarray_attrs ={
            'scan_name': tail[0],
            'scan_type': "RHEED_timelapse" ,
            'sample_description': sample_description,
            'start_time': start_time,
            'time': time,
            'date': date,
            'substrate': substrate,
            'sample': sample,
            'space_group': space_group,
            'temperature': temperature,
            'growth_history': growth_history,
            'pressure': pressure,
            'azi': azi,
            'beam_direction': beam_direction,
            'focus': focus,
            'grid': grid,
            'z': z,
            'beam_x': beam_x,
            'beam_y': beam_y,
            'rocking_angle': rocking_angle,
            'filament_current': filament_current,
            'energy': energy,
            'calibration': calibration,
            'center' : center,
            'camera_factor': camera_factor,
            'calibration_energy' : calibration_energy}
        
        # Load file
        array = np.loadtxt(filename)
        
        #Locate maximum of array
        ind = np.where(array[:,0] == np.max(array[:,0]))
        
        #Calculate number of different spectra
        spectra = (len(array[:,0])/len(ind[0]))
        
        #Extract time axis
        time = array[:int(spectra),0]
        #time = time[:,np.newaxis]
        
        #Extract intensities
        loaded_data= array[:,1].reshape(-1,len(ind[0]), order = 'F')
        
        #Array counting different regions
        region = np.arange(1,len(ind[0])+1)
        
        dim_names = ('time', 'region')
        coords = {'time': time, 'region': region}
        
        ds = xr.Dataset(
            {tail[0]: (dim_names, loaded_data)}, 
            coords=coords,
            attrs=dataset_attrs)
        ds.coords['time'].attrs = {'units':'seconds'}
        
        #Assign metadata for DataArray
        ds[tail[0]].attrs = dataarray_attrs

    #If file is a single image and not for LEED iv
    elif (filename[-4:] == '.png' or filename[-4:] == '.jpg' or filename[-5:]=='.tiff' or filename[-4:]=='.bmp'):
        
        #Load data            
        loaded_data = cv2.imread(filename)
        
        #Determine size of dimensions
        pix = np.shape(loaded_data)
        y_pix, x_pix, col = pix[0], pix[1], pix[2]
        
        #Create dimensions and coordinate names 
        y_range = np.arange(0,y_pix,1)
        x_range = np.arange(0,x_pix,1)
        
        #The user wants an RGB
        if color == True:
            col = pix[2]
            col_range = np.arange(0,col,1)
            dim_names = ('y_range', 'x_range','col_range')
            coords = {'y_range': y_range, 'x_range': x_range, 'col_range': col_range}
            
        #The user wants greyscale
        elif color == False:
            loaded_data = cv2.cvtColor(loaded_data,cv2.COLOR_RGB2GRAY)
            dim_names = ('y_range', 'x_range')
            coords = {'y_range': y_range, 'x_range': x_range}
        
        #Use name of file as identifier to Xarray DatArray
        head, tail = os.path.split(filename)
        tail = tail.split(".", 1)            
        
        #Metadata for Xarray DataArray
        dataarray_attrs ={
            'scan_name': tail[0],
            'scan_type': "RHEED" ,
            'sample_description': sample_description,
            'start_time': start_time,
            'time': time,
            'date': date,
            'substrate': substrate,
            'sample': sample,
            'space_group': space_group,
            'temperature': temperature,
            'growth_history': growth_history,
            'pressure': pressure,
            'azi': azi,
            'beam_direction': beam_direction,
            'focus': focus,
            'grid': grid,
            'z': z,
            'beam_x': beam_x,
            'beam_y': beam_y,
            'rocking_angle': rocking_angle,
            'filament_current': filament_current,
            'energy': energy,
            'calibration': calibration,
            'center' : center,
            'camera_factor': camera_factor,
            'calibration_energy' : calibration_energy}
        
        #Create Dataset cotaining one DataArray
        ds = xr.Dataset(
            {tail[0]: (dim_names, loaded_data)}, 
            coords=coords,
            attrs=dataset_attrs)
        ds.coords['x_range'].attrs = {'units':'pixels'}
        ds.coords['y_range'].attrs = {'units' : 'pixels'}
        
        #Assign metadata for DataArray
        ds[tail[0]].attrs = dataarray_attrs
        
    #If file is a single RHEED image from KSA software
    elif (filename[-4:] == '.img'):
        
        #Load data accordingly with the bit depth
        if bit_depth == 28:
            npimg = np.fromfile(filename, dtype=np.uint32)
        if bit_depth == 12:
            npimg = np.fromfile(filename, dtype=np.uint16)
        # image size
        imageSize = (492,656)
        # line at the beginning of the file that to do cot contain the image data, but presumably the metadata
        attrs_nline=len(npimg)-(492*656)
        # reshape the image
        loaded_data = npimg[attrs_nline:].reshape(imageSize)
        
        #Determine size of dimensions
        pix = np.shape(loaded_data)
        y_pix, x_pix = pix[0], pix[1]
        
        #Create dimensions and coordinate names 
        y_range = np.arange(0,y_pix,1)
        x_range = np.arange(0,x_pix,1)
            
        #The user wants greyscale
        dim_names = ('y_range', 'x_range')
        coords = {'y_range': y_range, 'x_range': x_range}
        
        #Use name of file as identifier to Xarray DatArray
        head, tail = os.path.split(filename)
        tail = tail.split(".", 1)
        sample = head.split("/")[-1]
        
        #Metadata for Xarray DataArray
        dataarray_attrs ={
            'scan_name': tail[0],
            'scan_type': "RHEED" ,
            'sample_description': sample_description,
            'start_time': start_time,
            'time': time,
            'date': date,
            'substrate': substrate,
            'sample': sample,
            'space_group': space_group,
            'temperature': temperature,
            'growth_history': growth_history,
            'pressure': pressure,
            'azi': azi,
            'beam_direction': beam_direction,
            'focus': focus,
            'grid': grid,
            'z': z,
            'beam_x': beam_x,
            'beam_y': beam_y,
            'rocking_angle': rocking_angle,
            'filament_current': filament_current,
            'energy': energy,
            'calibration': calibration,
            'center' : center,
            'camera_factor': camera_factor,
            'calibration_energy' : calibration_energy}
        
        #Create Dataset containing one DataArray
        RHEEDimg = xr.DataArray(loaded_data, coords=coords)
        RHEEDimg.coords['x_range'].attrs = {'units':'pixels'}
        RHEEDimg.coords['y_range'].attrs = {'units' : 'pixels'}
        RHEEDimg.attrs=dataarray_attrs
        return RHEEDimg
        
    #If folder of multiple images
    else:
        adict = {}
        for file in glob.glob(os.path.join(filename, '*.bmp')):
            head, tail = os.path.split(file)
            tail = tail.split(".", 1)
        
        dataarray_attrs ={
            'scan_name': tail[0],
            'scan_type': "RHEED" ,
            'sample_description': sample_description,
            'start_time': start_time,
            'time': time,
            'date': date,
            'substrate': substrate,
            'sample': sample,
            'space_group': space_group,
            'temperature': temperature,
            'growth_history': growth_history,
            'pressure': pressure,
            'azi': azi,
            'beam_direction': beam_direction,
            'focus': focus,
            'grid': grid,
            'z': z,
            'beam_x': beam_x,
            'beam_y': beam_y,
            'rocking_angle': rocking_angle,
            'filament_current': filament_current,
            'energy': energy,
            'calibration': calibration,
            'center' : center,
            'camera_factor': camera_factor,
            'calibration_energy' : calibration_energy}
        
        #Loops through all filename in directory to open files and then
        #convert them to DataArray 
        #RGB
        if color == True:  
            dim_names = ('y_range', 'x_range','col_range')
            for file in glob.glob(os.path.join(filename, '*.bmp')):
                loaded_data = cv2.imread(file)
                head, tail = os.path.split(file)
                tail = tail.split(".", 1)
                test = xr.DataArray(loaded_data, dims=dim_names,attrs=dataarray_attrs)
                test.attrs["scan_name"] = tail[0]
                adict[tail[0]]=test
                
            #check size of dimensions
            pix = np.shape(loaded_data)
            y_pix, x_pix, col = pix[0], pix[1], pix[2]
            
            #Create dimensions and coordinates
            y_range = np.arange(0,y_pix,1)
            x_range = np.arange(0,x_pix,1)
            col_range = np.arange(0,col,1)
            coords = {'y_range': y_range, 'x_range': x_range, 'col_range': col_range}    
    
        #greyscale
        if color == False: 
            dim_names = ('y_range', 'x_range')
            for file in glob.glob(os.path.join(filename, '*.bmp')):
                loaded_data = cv2.imread(file)
                loaded_data = cv2.cvtColor(loaded_data,cv2.COLOR_RGB2GRAY)
                head, tail = os.path.split(file)
                tail = tail.split(".", 1)
                test = xr.DataArray(loaded_data, dims=dim_names,attrs=dataarray_attrs)
                test.attrs["scan_name"] = tail[0]
                adict[tail[0]]=test
                
            #check size of dimensions
            pix = np.shape(loaded_data)
            y_pix, x_pix = pix[0], pix[1]
            
            #Create dimensions and coordinates
            y_range = np.arange(0,y_pix,1)
            x_range = np.arange(0,x_pix,1)
            coords = {'y_range': y_range, 'x_range': x_range}        
        
        #Create DataSet containing multiple DataArrays
        ds = xr.Dataset(data_vars=adict,
                        coords=coords)
        ds.coords['x_range'].attrs = {'units':'pixels'}
        ds.coords['y_range'].attrs = {'units' : 'pixels'}
        ds.attrs=dataarray_attrs
        
    #If the user inputs a dataset then the new dataset will be 
    #merged with this dataset
    if dataset != None:
        ds_merged = xr.merge([dataset, ds])
        return ds_merged
    
    else:
        return ds

