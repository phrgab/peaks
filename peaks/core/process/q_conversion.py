# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 11:48:07 2021

@author: lsh8
"""
#LH 27/04/2021


from scipy.interpolate import griddata
from math import sin, sqrt, pi
import sys
sys.path.insert(0, '../../..')
#import sys
#sys.path.insert(0, '../..')
import numpy as np
import xarray as xr
import warnings
from peaks.utils.metadata import update_hist
from peaks.core.process.fermi_level_corrections import apply_fermi
from peaks.utils.E_shift import Eshift_from_correction
#from peaks.utils.get_angles import get_angles, set_normals
#from peaks.utils.misc import warning_simple, warning_standard

def out_of_range_check(arr, angle):
    """
    A function to check that the scattering geometry is physical
    """
    if 135 + min(arr.coords['theta_par'].values) - angle > 90:
        print("Angle of reflection is greater than 90 degrees = not physical")
    elif angle > 90:
        print("Angle of incidence is greater than 90 degrees = not physical")
        
def q_conv(arr, polar = None, **kwargs):
    """
    A routine to convert from theta to q momentum space. Designed for EELS experiments in the
    King group lab. conversion by "backward" method; initally make a grid of converted data and
    then interpolate into the original data.
    Note: theta_par is the acceptance angle of the analyser
    
    ----------------------
    Parameters
    arr - Type: DataArray
    
    polar Type:Float - optional
    Allows the user to input the polar coords
    ------------------------
    Return
    
    DataArray
    """
    
    #Can set polar by function but if not done use metadata
    if polar == None and arr.attrs['polar'] != None: 
        polar = arr.attrs['polar']
    elif polar == None and arr.attrs['polar'] == None:
        raise Exception("Need to set polar attribute")
    
    #################################
    #####     ENERGY SCALES     #####
    #################################
    # determine angular-dependent Fermi energy correction based on the EF_correction attribute
    if 'EF_correction' in kwargs:  # If passed with call, write that to original dataarray (also updates original array outside of function)
        arr.attrs['EF_correction'] = kwargs['EF_correction']

    # Extract relevant energy shifts, correction type and raw correction parameters from dataarray
    E_shift, EF_correction_type = Eshift_from_correction(arr)
    EF_correction = arr.attrs['EF_correction']

    # Workfunction
    if EF_correction_type == 'fit' or EF_correction_type == 'array':
        wf = arr.attrs['hv'] - max(E_shift)
    elif EF_correction_type == 'value':
        wf = arr.attrs['hv'] - E_shift

    # Convert to binding energy if possible. If not possible, or if asked not to, keep in KE
    if arr.attrs['eV_type'] == 'kinetic':  # Data in KE
        if EF_correction_type == 'None' or kwargs['KE2BE']==False:  # No attributes to allow conversion to BE, or asked to keep in KE - in this case, we will stick with KE from the raw data
            data = arr.copy(deep=True)
            KE_values = arr.eV.data  # Get raw KE array

        elif 'EF_correction' not in kwargs:  # An EF correction has been applied, so we can use this to convert the data to BE - fails currently for EELS data
            data = arr.pipe(apply_fermi)  # Convert data to BE
            KE_values = data.attrs['hv'] - wf + data.eV.data# - (data.attrs['hv'] - 16.8)  # Get KE array, shifting from BE to KE for use in k-conv
            
        #A version using a different spectra to calibrate?
        else:
            data = arr.pipe(apply_fermi)  # Convert data to BE
            KE_values = data.attrs['hv'] - wf + data.eV.data# - (data.attrs['hv'] - 16.8)  # Get KE array, shifting from BE to KE for use in k-conv
        
    # If array already in BE
    else:
        data = arr.copy(deep=True)
        KE_values = data.attrs['hv'] - wf + data.eV.data
        #print(KE_values[0])
    
    #Total scattering angle is 135 degress and increasing the theta offset angle
    #(SMC manipulator angle) reduces the incident angle    
    angle = 135 - polar
    
    #Function to check user is trying something unphysical
    out_of_range_check(arr, angle)
    
    x_max=max(arr.coords['theta_par'].values)
    x_min=min(arr.coords['theta_par'].values)
    y_max=max(KE_values)
    y_min=min(KE_values)
   
    #Determine q_min and q_max
    #Note: q_min is greatest for smallest phi and vice versa
    q_min=(sin(np.radians(angle)) - sqrt(abs(1+ (-arr.attrs['hv']+y_min)/arr.attrs['hv']))*sin((np.radians(x_max+135-angle))))*sqrt(arr.attrs['hv'])*0.5123
    q_max=(sin(np.radians(angle)) - sqrt(abs(1+ (-arr.attrs['hv']+y_max)/arr.attrs['hv']))*sin((np.radians(x_min+135-angle))))*sqrt(arr.attrs['hv'])*0.5123

    #Create evenly spaced xtal momentum values    
    x = np.linspace(start = q_max, stop = q_min, num=len(arr['theta_par']))

    #Grid for new data
    xx2, yy2 = np.meshgrid(x, KE_values)

    #2 coloumn array for each data values in converted coords    
    points=np.ones((len(x)*len(KE_values),2))
   
    #backward convert xtal momentum to phi
    x2 = (180/pi)*np.arcsin(np.sqrt(abs(1/(1+(yy2-arr.attrs['hv'])/arr.attrs['hv'])))*(np.sin(np.radians(angle))-xx2/(np.sqrt(arr.attrs['hv'])*0.5123))) + angle-135
    
    #Creates the x,y coordinates for the intensity matrix (image matrix)
    x3 = x2.reshape((np.prod(x2.shape)),)
    y3 = yy2.reshape((np.prod(yy2.shape)),)
    points = np.column_stack((x3,y3))
    
    values = arr.data.T.flatten()
    
    #Meshgrid of evenly spaced theta_par and eV values to interpolate
    xx, yy = np.meshgrid(arr.coords['theta_par'].values, KE_values)

    #Interpolate and remove all nans
    grid_q = griddata(points, values, (xx, yy), method='linear', rescale ='true')
    grid_q = np.nan_to_num(grid_q)
    x = np.linspace(stop = q_min, start = q_max, num=len(arr['theta_par']))
    
    #Convert y axis back to BE if that is what was fed into fn
    #if arr.attrs['eV_type'] == 'kinetic':
    #KE_values = data.eV.data
        
    if 'KE2BE' in kwargs:
        if kwargs['KE2BE'] == True:
            if 'EF_correction' not in kwargs:
                KE_values = data.eV.data
            else:
                KE_values = data.eV.data - (data.attrs['hv'] - 16.8)
    
    #Create DataArray
    data = xr.DataArray(
    grid_q.T,
    dims = ("q", "eV"),
    coords={"eV": KE_values,
            "q": x},
    attrs=arr.attrs)
    data.coords['q'].attrs = {'units' : r'$\AA^{-1}$'}
    
    hist = "Converted into q space"
    update_hist(data, hist)
    
    return data

def binding_only(arr, **kwargs):
    if 'EF_correction' in kwargs:  # If passed with call, write that to original dataarray (also updates original array outside of function)
         arr.attrs['EF_correction'] = kwargs['EF_correction']

    # Extract relevant energy shifts, correction type and raw correction parameters from dataarray
    E_shift, EF_correction_type = Eshift_from_correction(arr)
    EF_correction = arr.attrs['EF_correction']

    # Convert to binding energy if possible. If not possible, or if asked not to, keep in KE
    if arr.attrs['eV_type'] == 'kinetic':  # Data in KE
        if 'EF_correction' not in kwargs:  # An EF correction has been applied, so we can use this to convert the data to BE - fails currently for EELS data
            BE_values = arr.pipe(apply_fermi)  # Convert data to BE
            
        #A version using a different spectra to calibrate?
        else:
            data = arr.pipe(apply_fermi)  # Convert data to BE
            BE_values = data.eV.data - (data.attrs['hv'] - 16.8)
            
    else:
        warning_str = 'Data already has a dimension that is binding energy'
        warnings.warn(warning_str)
        BE_values = arr.ev.data
        
    if 'q' in data.dims:
        #Create DataArray
        data = xr.DataArray(
        grid_q.T,
        dims = ("q", "eV"),
        coords={"eV": data,
                "q": x},
        attrs=arr.attrs)
        data.coords['q'].attrs = {'units' : r'$\AA^{-1}$'}
    
    else:
        #Create DataArray
        data = xr.DataArray(
        grid_q.T,
        dims = ("q", "eV"),
        coords={"eV": KE_values,
                "q": x},
        attrs=arr.attrs)
        data.coords['theta_par'].attrs = {'units' : r'$^{\circ}$'}

    
    hist = "Converted into binding energy"
    update_hist(data, hist)
            
    return data