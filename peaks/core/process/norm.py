#Functions used to normalise data
#Phil King 17/04/2021

import numpy as np
import xarray as xr
import warnings
from peaks.utils.metadata import update_hist
from peaks.utils.misc import warning_simple, warning_standard
from peaks.utils.OOP_method import add_methods
from peaks.core.fit import Shirley


# Helper function for the bg subtraction or normaliastion
def norm_wave(a, b, bg_flag):
    if bg_flag == 0: #Perform normalisation of the wave a by b
        a /= b
        return a
    else: #Perform background subtraction of wave b from a
        a -= b
        return a
    
# Normalisation
@add_methods(xr.DataArray)
def norm(dispersion, *args, **kwargs): 
    ''' This function applies a simple normalisation by e.g. an MDC, EDC, or ROI
    
    Input:
        dispersion - the dispersion to be normalised (xarray). If supplied with no further arguments,
          then normalisation to the max value is performed (i.e. spectrum normalised to unity).
          Otherwise specify via:
        *args - optional arguments to normalise by a constant and/or the mean across a complete dimension. E.g.:
            int or float - normalise by that number
            'eV' - normalise by an integrated MDC
            'theta_par' or 'k_par' or 'k_perp' - normalise by an integrated EDC in that direction
            'all' - normalise by mean of the xarray
        **kwargs - optional arguments to define the slice to normalise by. E.g.:
            eV=slice(105,105.1)) -- normalized by an integrated MDC defined by the eV slice given, automatically
              broadcast over array
            eV=slice(105,105.1), theta=slice(-12, -8) -- for a 2D wave, normalizes by a constant defined by the
              mean of the eV and theta slices, but this should broadcast along a higher-D array (not tested
              carefully yet)
            bg_flag = 1 -- perform background subtraction instead of normalisation.
    
    Returns:
        norm_dispersion - the normalised (or background subtracted) dispersion (xarray) '''

    # Set up warnings display and formatting
    warnings.simplefilter('always', UserWarning)  # Give warnings every time, even on function re-run
    warnings.formatwarning = warning_simple  # Formatting of warnings for peaks user errors

    #Check if we are doing normalisation or background subtraction
    if 'bg_flag' in kwargs: #If bg_flag passed as a kwarg, then set this for background subtraction
        bg_flag = 1
        kwargs.pop('bg_flag')
    else:
        bg_flag = 0
    
    #copy the input xarray
    norm_dispersion = dispersion.copy(deep=True)
    
    #Data from which to determine the normalization wave
    ROI_norm = dispersion
    ROI_flag = 0 #Flag to indicate whether we should be normalising by ROI_norm at the end

    #Check if called with no arguments, in which case normalise to max val
    if len(args) == 0 and len(kwargs) == 0:
        args = ('max',)

    
    #Iterate first over normalisations defined in *args
    for i in args:
        if type(i) is float or type(i) is int: #If the supplied variable is an integer or float, normalise by that value
            norm_dispersion = norm_wave(norm_dispersion, float(i), bg_flag)
        elif type(i) is str: #String indicating axis to normalise over (integrated over full dimension)
            if i=='max': # normalise by max value of xarray
                norm_dispersion = norm_wave(norm_dispersion, np.max(norm_dispersion.data), bg_flag)
            elif i=='all': #normalise by mean of xarray
                norm_dispersion = norm_wave(norm_dispersion, dispersion.mean(), bg_flag)
            else:
                ROI_flag = 1
                ROI_norm = ROI_norm.mean(i) #normalise by integrated EDC/MDC etc.
        elif type(i) is list or type(i) is dict:
            raise Exception("Specify number (e.g. 1.5), string (e.g. 'eV') defining axis, slice (e.g. theta_par=slice(0,5)) to define normalisations, or pass directly an xarray with the relevant wave to normalise by. Multiple such parameters can be specified. ")
        else: #Assume that an xarray has been supplied which contains the data to use for normalisation
            #Extract the relevant dimension of the supplied normalisation wave
            if len(i.coords.dims) != 1: #Check supplied wave suitable format
                raise Exception("Multi-axis supplied normalisation waves are not supported.")
            else:
                if i.coords.dims[0] not in norm_dispersion.coords.dims: #Check dimensions consistent
                    raise Exception("Dimensions in normalisation wave do not match those in data.")
                else:
                    #Check to see if the co-ordinates in the supplied array match those of the array to be normalised
                    if norm_dispersion.coords[i.coords.dims[0]].equals(i.coords[i.coords.dims[0]]): #They are equal
                        norm_dispersion = norm_wave(norm_dispersion, i, bg_flag) #Normalise directly by that supplied wave
                    else:
                        #Warn that an interpolation will be performed
                        warnings.warn("Coordinates of normalisation wave do not match those in data. Interpolating for normalisation.") 
                        #Do the interpolation of supplied data onto matching named dimension of the original data
                        interp_norm_wave = i.interp({i.coords.dims[0]: norm_dispersion.coords[i.coords.dims[0]]}, kwargs={"fill_value": "extrapolate"})
                        #Normalise by this wave
                        norm_dispersion = norm_wave(norm_dispersion, interp_norm_wave, bg_flag)
    
    #Iterate over the supplied arguments in kwargs defining slices to make up the normalization wave
    for i in kwargs:
        ROI_flag = 1
        ROI_norm = ROI_norm.loc[{i: kwargs[i]}].mean(i) #Normalise by the EDC/MDC etc. integrated over the specified slice

    if ROI_flag == 1: #If something other than a single number or all was specified, normalise by relevant EDC/MDC/ROI
        norm_dispersion = norm_wave(norm_dispersion, ROI_norm, bg_flag)
        
    #Update the analysis history
    norm_wave_description = str(args)+str(kwargs) #String detailing normalisations applied
    if bg_flag == 0: #Normalisation applied
        norm_wave_description = 'Data normalised by: ' + norm_wave_description
    else: #bgs
        norm_wave_description = 'Background subtraction applied: ' + norm_wave_description
    update_hist(norm_dispersion, norm_wave_description)

    # Reset warnings display and formatting
    warnings.simplefilter('once', UserWarning)  # Give warnings first time only
    warnings.formatwarning = warning_standard  # Formatting of warnings for peaks user errors
        
    return norm_dispersion

# Background subtraction
@add_methods(xr.DataArray)
def bgs(dispersion, *args, **kwargs): 
    ''' This function applies a simple background subtraction by e.g. an MDC, EDC, or ROI. It works just by calling
    the norm function with a flag to indicate background subtraction
    
    Input:

        `dispersion` - the dispersion to be normalised (xarray)

        `*args` - optional arguments to normalise by a constant and/or the mean across a complete dimension. E.g.:

            - `int` or `float` - subtract that number

            - ``eV'` - subtract an integrated MDC

            - `'theta_par'` or `'k_par'` or `'k_perp'` - subtract an integrated EDC in that direction

            - `'Shirley'` - subtract a shirley background, assuming default parameters unless `shirley_options`
                also passed. NB if Shirley passed, all other arguments ignored and only a Shirley bgs performed

            - `'all'` - subtract the mean of the xarray

        `**kwargs` - optional arguments to define the slice to normalise by. E.g.:

            - `eV=slice(105,105.1))` -- subtract an integrated MDC defined by the eV slice given, automatically
              broadcast over array

            - `eV=slice(105,105.1), theta=slice(-12, -8)` -- for a 2D wave, subtracts a constant defined by the mean
              of the eV and theta slices, but I think this may broadcast along a higher-D array (not tested
              carefully yet)

            `Shirley_opts = dict` -- dictionary of options to pass to the Shirley fit function:

              - `average` : number of point to take in consideration to calulate the average value
                   of the y-start and y-end point .

              - `yl_offset` : offsets to subtract to the leftest  y value. Useful if the range does
                   not cover completely the left tail of the peak.

              - `yr_offset` : offsets to subtract to the rightest  y value. Useful if the range does
                    not cover completely the tail of the rightest peak.

    Returns:
        bgs_dispersion - the background subtracted dispersion (xarray) '''

    try:
        if 'Shirley' in args:
            bgs_data = dispersion.data
            bgs_dispersion = dispersion.copy(deep=True)

            if 'Shirley_opts' in kwargs:
                opts = kwargs.pop('Shirley_opts')
                bgs_dispersion.data = bgs_dispersion - Shirley(bgs_data, **opts)
            else:
                bgs_dispersion.data = bgs_dispersion - Shirley(bgs_data)

        else:
            bgs_dispersion = norm(dispersion, *args, **kwargs, bg_flag=1)
    except:
        bgs_dispersion = norm(dispersion, *args, **kwargs, bg_flag=1)
        
    return bgs_dispersion