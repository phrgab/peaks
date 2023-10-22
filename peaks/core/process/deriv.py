#Functions used for derivative operations on data
#Phil King 30/04/2021

import numpy as np
from scipy.ndimage import gaussian_gradient_magnitude
import warnings
from peaks.core.process.smooth import smooth
from peaks.core.utils.metadata import update_hist
from peaks.core.utils.OOP_method import add_methods
import xarray as xr

#Perform a double differential in the momentum direction, applying smoothing
@add_methods(xr.DataArray)
def d2k(dispersion, plot=False, **kwargs):
    '''Perform double differential in the momentum (or angle) direction of data contained in an xarray dataarray,
     with smoothing applied as per the specified parameters.
    
    Input:
        dispersion - the data to differentiate (xarray)
        plot (optional) - set to True to return a plot rather than an xarray
        **kwargs - optional arguments:
            - axes to smooth over in the format ax=FWHM, where FWHM is the relevant FWHM of the Gaussian for
              convolution in this direction (coord=ax)
            - can also supply optional parameters to pass to plotting function here (standard MatPlotLib calls)
    
    Returns:
        diff_data - the smoothed and differentiated data (xarray)
        
    Example:
        d2k(dispersion, theta_par=0.5, eV=0.2) to smooth dispersion by a Gaussian with FWHM 0.5 deg and 0.2 eV
         and perform d2/d(theta_par)^2'''
    
    #Check data is 2D
    if len(dispersion.dims) != 2:
        raise Exception("Function only acts on 2D data.")
    
    #Work out correct variable for dispersive direction (i.e. is data in angle or k-space)
    eV_dim = dispersion.dims.index('eV') #Work out index of energy dimension
    ang_dim = dispersion.dims[1-eV_dim] #Get the other dimension as the one which isn't energy dimension
        
    #Call function to do double derivative and possible smoothing
    diff_data = double_diff(dispersion, ang_dim, ang_dim, plot, **kwargs)
    
    return diff_data


#Perform a double differential in the momentum direction, applying smoothing
@add_methods(xr.DataArray)
def d2E(dispersion, plot=False, **kwargs):
    '''Perform double differential in the energy direction of data contained in an xarray dataarray, with
     smoothing applied as per the specified parameters.
    
    Input:
        dispersion - the data to differentiate (xarray)
        plot (optional) - set to True to return a plot rather than an xarray
        **kwargs - optional arguments:
            - axes to smooth over in the format ax=FWHM, where FWHM is the relevant FWHM of the Gaussian for
              convolution in this direction (coord=ax)
            - can also supply optional parameters to pass to plotting function here (standard MatPlotLib calls)
    
    Returns:
        diff_data - the smoothed and differentiated data (xarray)
        
    Example:
        d2E(dispersion, theta_par=0.5, eV=0.2) to smooth dispersion by a Gaussian with FWHM 0.5 deg and 0.2 eV and perform d2/d(E)^2'''
    
    #Check data is 2D
    if len(dispersion.dims) != 2:
        raise Exception("Function only acts on 2D data.")
        
    #Call function to do double derivative and possible smoothing
    diff_data = double_diff(dispersion, 'eV', 'eV', plot, **kwargs)
    
    return diff_data


#Perform d2/dEdk, applying smoothing
@add_methods(xr.DataArray)
def dEdk(dispersion, plot=False, **kwargs):
    '''Perform d2/dEdk of data contained in an xarray dataarray, with smoothing applied as per the specified
     parameters.
    
    Input:
        dispersion - the data to differentiate (xarray)
        plot (optional) - set to True to return a plot rather than an xarray
        **kwargs - optional arguments:
            - axes to smooth over in the format ax=FWHM, where FWHM is the relevant FWHM of the Gaussian for
              convolution in this direction (coord=ax)
            - can also supply optional parameters to pass to plotting function here (standard MatPlotLib calls)
    
    Returns:
        diff_data - the smoothed and differentiated data (xarray)
        
    Example:
        dEdk(dispersion, theta_par=0.5, eV=0.2) to smooth dispersion by a Gaussian with FWHM 0.5 deg and 0.2 eV and perform d2/dEdk'''
    
    #Check data is 2D
    if len(dispersion.dims) != 2:
        raise Exception("Function only acts on 2D data.")
      
    #Work out correct variable for dispersive direction (i.e. is data in angle or k-space)
    eV_dim = dispersion.dims.index('eV') #Work out index of energy dimension
    ang_dim = dispersion.dims[1-eV_dim] #Get the other dimension as the one which isn't energy dimension
    
    #Call function to do double derivative and possible smoothing
    diff_data = double_diff(dispersion, 'eV', ang_dim, plot, **kwargs)
    
    return diff_data



#Master function for applying a double derivative along two directions, including smoothing
@add_methods(xr.DataArray)
def double_diff(data, dim1, dim2, plot=False, **kwargs):
    '''Perform double differential along the specified dimensions of data contained in an xarray dataarray
    
    Input:
        data - the data to differentiate (xarray)
        dim1 - dimension for first derivative
        dim2 - dimension for second derivative
        plot (optional) - set to True to return a plot rather than an xarray
        **kwargs - optional arguments:
            - axes to smooth over in the format ax=FWHM, where FWHM is the relevant FWHM of the Gaussian for
              convolution in this direction (coord=ax)
            - can also supply optional parameters to pass to plotting function here (standard MatPlotLib calls)
    
    Returns:
        diff_data - differentiated data (xarray)
        
    Example:
        double_diff(dispersion, 'eV', 'eV') to perform d2/d(eV)^2'''
    
    #copy the input xarray
    diff_data = data.copy(deep=True)
    
    #Extract relevant kwargs for smoothing
    smooth_axis = {} #Make an empty dictionary
    for i in kwargs:
        if i in diff_data.dims: #If supplied kwarg is a dim name
            smooth_axis[i] = kwargs[i] #Add it to the smoothing list
        
    #smooth the data with any supplied parameters using smooth function
    if len(smooth_axis) != 0:
        #Remove these from kwargs list as this will be passed later to plotting function
        for i in smooth_axis:
            kwargs.pop(i) 

        #Smooth the data using peaks smooth function
        diff_data = smooth(diff_data, **smooth_axis)
    
    #Save the attributes as these get killed by the differentiate function, and keep_attrs doesn't seem to exist for that
    attributes = diff_data.attrs 

    #Perform differentiation
    diff_data = diff_data.differentiate(dim1).differentiate(dim2)
    
    #Rewrite attributes
    diff_data.attrs = attributes
    
    #Update history
    if dim1 == dim2:
        hist = 'Applied double derrivative in direction '+str(dim1) #New string
    else:
        hist = 'Applied derrivative in directions '+str(dim1)+', '+str(dim2) #New string
    update_hist(diff_data, hist)
    
    #Return plot if requested
    if plot != False:
        #Set some default plot settings, but overwrite them if same things given in kwargs call
        #Work out some sensible colour limits if not supplied
        if 'vmin' in kwargs:
            vmin = kwargs['vmin']
            kwargs.pop('vmin')
        elif 'clim' in kwargs:
            vmin = kwargs['clim'][0]
        else:
            vmin = diff_data.min()/2
            
        if 'vmax' in kwargs:
            vmax = kwargs['vmax']
            kwargs.pop('vmax')
        elif 'clim' in kwargs:
            vmax = kwargs['clim'][1]
            kwargs.pop('clim')
        else:
            vmax = 0
          
        #Set a default colour map
        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'viridis_r' #Reversed viridis
            
        #Set plot with nice orientation for dispersion
        if 'y' not in kwargs and 'eV' in diff_data.dims:
            kwargs['y'] = 'eV'
            
        
        diff_data.plot(vmin=vmin, vmax=vmax, **kwargs)
        
        return
    
    return diff_data
    
    

#2D curvature analysis [as described in Rev. Sci. Instrum.  82, 043712 (2011)]
@add_methods(xr.DataArray)
def curvature(data, C0, C1, plot=False, **kwargs):
    '''Perform 2D curvature analysis (see Rev. Sci. Instrum.  82, 043712 (2011)) to data contained in an xarray
       dataarray
    
    Input:
        data - the data to differentiate (xarray)
        C0 - free parameter for axis 0
        C1 - free parameter for axis 1
        plot (optional) - set to True to return a plot rather than an xarray
        **kwargs - optional arguments:
            - axes to smooth over in the format ax=FWHM, where FWHM is the relevant FWHM of the Gaussian for
              convolution in this direction (coord=ax)
            - can also supply optional parameters to pass to plotting function here (standard MatPlotLib calls)
    
    Returns:
        curve_data - curvature data (xarray)
        
    Example:
        curvature(dispersion, 10, 1, plot=1, theta_par=0.3, eV=0.01) to perform curvature with some smoothing'''
    
    #Check data is 2D
    if len(data.dims) != 2:
        raise Exception("Function only acts on 2D data.")
        
    #copy the input xarray
    curv_data = data.copy(deep=True)
    
    #Determine relevant axes
    dim0 = curv_data.dims[0]
    dim1 = curv_data.dims[1]
    
    print('C0 scaling for '+str(dim0)+', C1 scaling for '+str(dim1))
    
    #Extract relevant kwargs for smoothing
    smooth_axis = {} #Make an empty dictionary
    for i in kwargs:
        if i in curv_data.dims: #If supplied kwarg is a dim name
            smooth_axis[i] = kwargs[i] #Add it to the smoothing list
        
    #smooth the data with any supplied parameters using smooth function
    if len(smooth_axis) != 0:
        #Remove these from kwargs list as this will be passed later to plotting function
        for i in smooth_axis:
            kwargs.pop(i) 

        #Smooth the data using peaks smooth function
        curv_data = smooth(curv_data, **smooth_axis)
        
    #Save the attributes as these get killed by the differentiate function, and keep_attrs doesn't seem to exist for that
    attributes = curv_data.attrs 
        
    #Determine various derivatives 
    d0 = curv_data.copy(deep=True); d0 = d0.differentiate(dim0)
    d20 = curv_data.copy(deep=True); d20 = curv_data.differentiate(dim0).differentiate(dim0)
    d1 = curv_data.copy(deep=True); d1 = d1.differentiate(dim1)
    d21 = curv_data.copy(deep=True); d21 = curv_data.differentiate(dim1).differentiate(dim1)
    d0d1 = curv_data.copy(deep=True); d0d1 = curv_data.differentiate(dim0).differentiate(dim1)
    
    #Calculate 2D curvature
    curv_data = (((1+(C0*(d0**2)))*C1*d21) - (2*C0*C1*d0*d1*d0d1) + ((1+(C1*(d1**2)))*C0*d20))/(((C0*(d0**2))+(C1*(d1**2)))**1.5)
    
    #Rewrite attributes
    curv_data.attrs = attributes
    
    #Update history
    hist = '2D curvature with coefficients: '+str(C0)+' for '+str(dim0)+' and '+str(C1)+' for '+str(dim1)
    update_hist(curv_data, hist)
    
    #Return plot if requested
    if plot != False:
        #Set some default plot settings, but overwrite them if same things given in kwargs call
        #Work out some sensible colour limits if not supplied
        if 'vmin' in kwargs:
            vmin = kwargs['vmin']
            kwargs.pop('vmin')
        elif 'clim' in kwargs:
            vmin = kwargs['clim'][0]
        else:
            vmin = -1
            
        if 'vmax' in kwargs:
            vmax = kwargs['vmax']
            kwargs.pop('vmax')
        elif 'clim' in kwargs:
            vmax = kwargs['clim'][1]
            kwargs.pop('clim')
        else:
            vmax = 0
          
        #Set a default colour map
        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'viridis_r' #Reversed viridis
            
        #Set plot with nice orientation for dispersion
        if 'y' not in kwargs and 'eV' in curv_data.dims:
            kwargs['y'] = 'eV'
            
        #Return the plot
        curv_data.plot(vmin=vmin, vmax=vmax, **kwargs)
        
        return
    
    
    return curv_data
    

#1D curvature analysis [as described in Rev. Sci. Instrum.  82, 043712 (2011)]
@add_methods(xr.DataArray)
def curv1d(data, dim, C0, plot=False, **kwargs):
    '''Perform 1D curvature analysis (see Rev. Sci. Instrum.  82, 043712 (2011)) to data contained in an xarray
      dataarray
    
    Input:
        data - the data to differentiate (xarray)
        dim - name of the dimension to apply the curvature along
        C0 - free parameter
        plot (optional) - set to True to return a plot rather than an xarray
        **kwargs - optional arguments:
            - axes to smooth over in the format ax=FWHM, where FWHM is the relevant FWHM of the Gaussian for
              convolution in this direction (coord=ax)
            - can also supply optional parameters to pass to plotting function here (standard MatPlotLib calls)

    Returns:
        curve_data - curvature data (xarray)
        
    Example:
        curvature(dispersion, 10, 1, plot=1, theta_par=0.3, eV=0.01) to perform curvature with some smoothing'''
    
    #Check data is 2D
    if len(data.dims) != 2:
        raise Exception("Function only acts on 2D data.")
        
    #copy the input xarray
    curv_data = data.copy(deep=True)
    
    #Extract relevant kwargs for smoothing
    smooth_axis = {} #Make an empty dictionary
    for i in kwargs:
        if i in curv_data.dims: #If supplied kwarg is a dim name
            smooth_axis[i] = kwargs[i] #Add it to the smoothing list
        
    #smooth the data with any supplied parameters using smooth function
    if len(smooth_axis) != 0:
        #Remove these from kwargs list as this will be passed later to plotting function
        for i in smooth_axis:
            kwargs.pop(i) 

        #Smooth the data using peaks smooth function
        curv_data = smooth(curv_data, **smooth_axis)
        
    #Save the attributes as these get killed by the differentiate function, and keep_attrs doesn't seem to exist for that
    attributes = curv_data.attrs 
        
    #Determine various derivatives 
    d0 = curv_data.copy(deep=True); d0 = d0.differentiate(dim)
    d20 = curv_data.copy(deep=True); d20 = curv_data.differentiate(dim).differentiate(dim)
    
    #Calculate 1D curvature
    curv_data = d20/((C0+(d0**2))**1.5)
    
    #Rewrite attributes
    curv_data.attrs = attributes
    
    #Update history
    hist = '1D curvature with coefficients: '+str(C0)+' for '+str(dim)
    update_hist(curv_data, hist)
    
    #Return plot if requested
    if plot != False:
        #Set some default plot settings, but overwrite them if same things given in kwargs call
        #Work out some sensible colour limits if not supplied
        if 'vmin' in kwargs:
            vmin = kwargs['vmin']
            kwargs.pop('vmin')
        elif 'clim' in kwargs:
            vmin = kwargs['clim'][0]
        else:
            vmin = curv_data.min()/2
            
        if 'vmax' in kwargs:
            vmax = kwargs['vmax']
            kwargs.pop('vmax')
        elif 'clim' in kwargs:
            vmax = kwargs['clim'][1]
            kwargs.pop('clim')
        else:
            vmax = 0
          
        #Set a default colour map
        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'viridis_r' #Reversed viridis
            
        #Set plot with nice orientation for dispersion
        if 'y' not in kwargs and 'eV' in curv_data.dims:
            kwargs['y'] = 'eV'
            
        #Return the plot
        curv_data.plot(vmin=vmin, vmax=vmax, **kwargs)
        
        return
    
    
    return curv_data


#Minimum gradient analysis, based on model described in Rev. Sci. Instrum 88 (2017) 073903
@add_methods(xr.DataArray)
def min_grad(data, plot=False, **kwargs):
    '''Apply minimum gradient analysis, using Gaussian filtering. Method based on Rev. Sci. Instrum 88 (2017)
      073903, and uses the scipy.ndimage.gaussian_gradient_magnitude function.
    
    Input:
        data - the data to apply gradient magnetiude analysis to (xarray)
        plot (optional) - set to True to return a plot rather than an xarray
        **kwargs - optional arguments:
            - axes to smooth over in the format ax=FWHM, where FWHM is the relevant FWHM of the Gaussian for
              convolution in this direction (coord=ax)
            - can also supply optional parameters to pass to plotting function here (standard MatPlotLib calls)
    
    Returns:
        grad_mod - renormalised gradient modulus map (xarray)
        
    Example:
        min_grad(dispersion, theta_par=0.5, eV=0.2).'''
    
    #Check data is 2D
    if len(data.dims) != 2:
        raise Exception("Function only acts on 2D data.")
        
    #copy the input xarray
    grad_data = data.copy(deep=True)
    grad_mod = data.copy(deep=True)
    
    #String for updating history
    hist = 'Minimum gradient analysis applied, with Gaussian filter with the following FWHM along given axes: '

    
    #Determine relevant standard deviations in pixels from xarray axis scaling
    sigma = np.zeros(len(data.dims)) #Make the sigma array as zeros to update from supplied definitions
    #Iterate through coordinates
    for count, value in enumerate(data.dims):
        if value not in kwargs: #No broadening for this dimension
            sigma[count] = 1/2.35482005
            warning_str = "No broadening parameter supplied for "+str(value)+" dimension. Set to default of single pixel."
            warnings.warn(warning_str) 
        else: #Determine broadening in pixels from axis scaling
            delta = abs(data[value].data[1]-data[value].data[0]) #pixel size in relevant units for axis
            sigma_px = np.round(kwargs[value]/delta)/2.35482005 #sigma for relevant axis in pixels (NB converting from given broadening in FWHM)
            sigma[count] = sigma_px #Update s.d. array
            hist += str(value)+': '+str(kwargs[value])+', ' #Update history
            kwargs.pop(value) #Remove this axis from kwargs for consistency check later
    
    #Extract the raw array
    array = grad_data.data
    
    #Apply gradient magnitude 
    array_sm = gaussian_gradient_magnitude(array, sigma)
    
    #Return data into array
    grad_data.data = array_sm
    
    #Take the renormalised gradient modulus map
    grad_mod /= grad_data
    
    #Return plot if requested
    if plot != False:
        #Set some default plot settings, but overwrite them if same things given in kwargs call
        #Work out some sensible colour limits if not supplied
        if 'vmin' in kwargs:
            vmin = kwargs['vmin']
            kwargs.pop('vmin')
        elif 'clim' in kwargs:
            vmin = kwargs['clim'][0]
        else:
            vmin = 0
            
        if 'vmax' in kwargs:
            vmax = kwargs['vmax']
            kwargs.pop('vmax')
        elif 'clim' in kwargs:
            vmax = kwargs['clim'][1]
            kwargs.pop('clim')
        else:
            vmax = grad_mod.mean().data
          
        #Set a default colour map
        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'viridis' #Reversed
            
        #Set plot with nice orientation for dispersion
        if 'y' not in kwargs and 'eV' in grad_data.dims:
            kwargs['y'] = 'eV'
            
        #Return the plot
        grad_mod.plot(vmin=vmin, vmax=vmax, **kwargs)
        
        return
    
    #Update the history
    update_hist(grad_data, hist)
    return grad_mod