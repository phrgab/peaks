#Functions used to smooth data
#Phil King 17/04/2021

import numpy as np 
from scipy.ndimage import gaussian_filter
import warnings
from peaks.utils.metadata import update_hist
from peaks.utils.OOP_method import add_methods
import xarray as xr

@add_methods(xr.DataArray)
def smooth(data, **kwargs):
    '''Smooth data contained in an xarray dataarray as per the specified parameters, and return the smoothed
    array. Uses scipy gaussian_filter function.
    
    Input:
        data - the data to smooth (xarray)
        **kwargs - axes to smooth over in the format ax=FWHM, where FWHM is the relevant FWHM of the Gaussian
          for convolution in this direction (coord=ax)
    
    Returns:
        smoothed_data - the smoothed data (xarray)
        
    Example:
        smooth(dispersion, theta_par=0.5, eV=0.2) to smooth dispersion by a Gaussian with FWHM 0.5 deg and 0.2 eV'''
    
    #copy the input xarray
    smoothed_data = data.copy(deep=True)
    
    #String for updating history
    hist = 'Data smoothed with following FWHM along given axes: '

    
    #Determine relevant standard deviations in pixels from xarray axis scaling
    sigma = np.zeros(len(data.dims)) #Make the sigma array as zeros to update from supplied definitions
    #Iterate through coordinates
    for count, value in enumerate(data.dims):
        if value not in kwargs: #No broadening for this dimension
            sigma[count] = 0
        else: #Determine broadening in pixels from axis scaling
            delta = abs(data[value].data[1]-data[value].data[0]) #pixel size in relevant units for axis
            sigma_px = np.round(kwargs[value]/delta)/2.35482005 #sigma for relevant axis in pixels (NB converting from given broadening in FWHM)
            sigma[count] = sigma_px #Update s.d. array
            hist += str(value)+': '+str(kwargs[value])+', ' #Update history
            kwargs.pop(value) #Remove this axis from kwargs for consistency check later
    
    #Extract the raw array
    array = smoothed_data.data
    
    #Apply gaussian convolution
    array_sm = gaussian_filter(array, sigma)
    
    #Return data into array
    smoothed_data.data = array_sm
    
    #Check that all supplied kwargs are used
    if len(kwargs) != 0:
        #Give a warning
        warning_str = "Not all supplied axes are coordinates of array: "+str(kwargs)+" have been ignored."
        warnings.warn(warning_str) 
    
    #Update the history
    update_hist(smoothed_data, hist)
    return smoothed_data
    




    
        
 