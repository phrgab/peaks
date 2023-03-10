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
from peaks.utils.OOP_method import add_methods

@add_methods(xr)
def LEED_load(filename, iv=False, dataset=None, color = False, sample_description = None, \
              energy=0,polar=0,tilt=0,azi=0,\
              x1=0,x2=0,x3=0, temp_sample=None, temp_cryo = None,\
                  calibration=None,center=None,camera_factor=None,\
                 calibration_energy=None):
    """
    A function to load LEED images into an Xarray DataSet

    Parameters
    ----------
    filename : String
        Name of file.
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
    'calibration_energy' : calibration_energy}
    
    #If file is a single image and not for LEED iv
    if (filename[-4:] == '.png' or filename[-4:] == '.jpg' or filename[-5:]=='.tiff' or filename[-4:]=='.bmp') and iv==False:
        
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
            'scan_type': "LEED" ,
            'sample_description': sample_description,
            'energy': energy, 
            'polar': polar,
            'tilt': tilt,
            'azi': azi,
            'x1': x1,
            'x2': x2,
            'x3' :x3,
            'temp_sample': temp_sample,
            'temp_cryo': temp_cryo,
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
    
    #If file is a single image and the user wants to do LEED iv
    elif (filename[-4:] == '.png' or filename[-4:] == '.jpg' or filename[-5:]=='.tiff' or filename[-4:]=='.bmp') and iv==True:
        raise Exception("Cannot do LEED-iv on a single image")
        
    #If folder of multiple images but not for LEED-iv
    elif (filename[-4:] != '.png' or filename[-4:] != '.jpg' or filename[-5:]!='.tiff' or filename[-4:]!='.bmp') and iv==False:
        adict = {}
        
        dataarray_attrs ={
            'scan_name': filename,
            'scan_type': "LEED" ,
            'sample_description': sample_description,
            'energy': energy, 
            'polar': polar,
            'tilt': tilt,
            'azi': azi,
            'x1': x1,
            'x2': x2,
            'x3' :x3,
            'temp_sample': temp_sample,
            'temp_cryo': temp_cryo,
            'calibration': calibration,
            'center' : center,
            'camera_factor': camera_factor,
            'calibration_energy' : calibration_energy}
        
        #Loops through all filename in directory to open files and then
        #convert them to DataArray 
        #RGB
        if color == True:  
            dim_names = ('y_range', 'x_range','col_range')
            for file in glob.glob(os.path.join(filename, '*.tiff')):
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
            for file in glob.glob(os.path.join(filename, '*.tiff')):
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
    
    #A LEED-iv measurement
    else:
        alist = []
        i=0
        
        def do(filename): return(glob.glob(os.path.join(filename, '*.tiff')))
        def do2(filename): return(glob.glob(os.path.join(filename, '*.bmp')))
        def do3(filename): return(glob.glob(os.path.join(filename, '*.png')))
        def do4(filename): return(glob.glob(os.path.join(filename, '*.jpg')))
        
        try:
            do(filename)
        except:
            try:
                do2(filename)
            except:
                try:
                    do3(filename)
                except:
                    do4(filename)
        
        #Use name of file as identifier to Xarray DatArray
        head, tail = os.path.split(filename)
        tail = tail.split(".") 
        
        #Loads all LEED image and creates a multidimensional numpy array
        #load data in RGB
        if color == True:
            for file in glob.glob(os.path.join(filename, '*.tiff')):
                if i ==0:
                    start_E = file
                stop_E = file
                loaded_data = cv2.imread(file)
                imarray = np.array(loaded_data)
                alist.append(imarray)
                i =+ 1
            total = np.stack(alist)
        
            pix = np.shape(total)
            num_E, y_pix, x_pix, col = pix[0], pix[1], pix[2], pix[3]
        
        #Load in greyscale
        elif color == False:
            for file in glob.glob(os.path.join(filename, '*tiff')):
                if i ==0:
                    start_E = file
                stop_E = file
                loaded_data = cv2.imread(file)
                loaded_data = cv2.cvtColor(loaded_data,cv2.COLOR_RGB2GRAY)
                imarray = np.array(loaded_data)
                alist.append(imarray)
                i =+ 1
            total = np.stack(alist)
        
            pix = np.shape(total)
            num_E, y_pix, x_pix = pix[0], pix[1], pix[2]
            
                
        start_E=float(start_E.split('_')[-1].rstrip(".tiff"))
        stop_E=float(stop_E.split('_')[-1].rstrip(".tiff"))
        
        #Determine size of dimensions
        y_range = np.arange(0,y_pix,1)
        x_range = np.arange(0,x_pix,1)
        
        #Create dimensions and coordinates
        energy = np.linspace(start_E,stop_E,num_E)
        
        if color == True:
            col_range = np.arange(0,col,1)
            dim_names = ('eV','y_range', 'x_range','col_range')
            coords = {'eV' :energy, 'y_range': y_range, 'x_range': x_range, 'col_range': col_range}
            
        elif color == False:
            dim_names = ('eV','y_range', 'x_range')
            coords = {'eV' :energy, 'y_range': y_range, 'x_range': x_range}
            
        #Metadata showing range of electron energies
        energy = [start_E,stop_E]
        
        #Create DataArray metadata
        dataarray_attrs ={
            'scan_name': tail[0],
            'scan_type': "LEED" ,
            'sample_description': sample_description,
            'energy': energy, 
            'polar': polar,
            'tilt': tilt,
            'azi': azi,
            'x1': x1,
            'x2': x2,
            'x3' :x3,
            'temp_sample': temp_sample,
            'temp_cryo': temp_cryo,
            'calibration': calibration,
            'center' : center,
            'camera_factor': camera_factor,
            'calibration_energy' : calibration_energy}
        
        #Create DataSet containing the LEED-iv DataArray
        ds = xr.Dataset(
            {tail[0]: (dim_names, total)}, 
            coords=coords,
            attrs=dataset_attrs)
        ds.coords['x_range'].attrs = {'units':'pixels'}
        ds.coords['y_range'].attrs = {'units' : 'pixels'}
        ds[tail[0]].attrs = dataarray_attrs
    
    #If the user inputs a dataset then the new dataset will be 
    #merged with this dataset
    if dataset != None:
        ds_merged = xr.merge([dataset, ds])
        return ds_merged
    
    else:
        return ds