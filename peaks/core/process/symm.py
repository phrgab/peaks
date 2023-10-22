#Functions used to perform data symmetrisations
#Phil King 20/4/21
#Brendan Edwards 03/12/21

import xarray as xr
import numpy as np
from peaks.core.utils.metadata import update_hist
from peaks.core.utils.OOP_method import add_methods
from .rotation import rotation

#Rotation symmetrisations
@add_methods(xr.DataArray)
def rot_sym(xarray, order, expand=True, fillna=True):
    '''This function takes a 2D input xarray and symmetrises it around a centre coordinate, by a goiven rotation order
    
    Inputs:
        xarray - the data which will be symmetrised (xarray)
        order - rotation order (int)
        **kwargs
            expand - whether or not to expand the coord grid
            fillna - whether or not to plot data where some rotated xarrays have nans
            
    Returns:
        symmetrised_xarray - the symmetrised xarray
    '''
    
    #copy the input xarrays
    xarray_to_be_symmetrised = xarray.copy(deep=True)
    
    #determine the step between rotations
    rotation_step = 360/order
    
    #keep the same coord grid as the inputted xarray
    if expand == False:
        symmetrised_xarray = xarray_to_be_symmetrised
        
        #set up nan counter so that we know how to rescale certain data points
        if fillna != False:
            x_values = symmetrised_xarray.coords[symmetrised_xarray.dims[0]].data
            y_values = symmetrised_xarray.coords[symmetrised_xarray.dims[1]].data
            nan_counter = [[0 for i in range(len(y_values))] for j in range(len(x_values))]
        
        #combine the rotated xarrays
        for i in range(1, order):
            rotation_angle = i*rotation_step
            rotated_xarray = rotation(xarray_to_be_symmetrised, rotation_angle)
            #interpolate the xarray onto the original xarray coord grid
            interpolated_rotated_xarray = rotated_xarray.interp({xarray_to_be_symmetrised.dims[0]:xarray_to_be_symmetrised.coords[xarray_to_be_symmetrised.dims[0]], xarray_to_be_symmetrised.dims[1]:xarray_to_be_symmetrised.coords[xarray_to_be_symmetrised.dims[1]]})
            if fillna != False:
                for x in range(len(x_values)):
                    for y in range(len(y_values)):
                        if str(interpolated_rotated_xarray.data[x][y]) == 'nan':
                            #add 1 to the nan count at each x,y with a nan value
                            nan_counter[x][y] = nan_counter[x][y] + 1
                interpolated_rotated_xarray = interpolated_rotated_xarray.fillna(0)
            symmetrised_xarray = symmetrised_xarray + interpolated_rotated_xarray
        
        if fillna != False:
               for x in range(len(x_values)):
                    for y in range(len(y_values)):
                        #rescale data points where nans are present
                        if (order-nan_counter[x][y]) != 0:
                            symmetrised_xarray[x][y] = symmetrised_xarray[x][y]/(order-nan_counter[x][y])
    
    #change the coord grid to maximise the data being plotted
    else:
        #get the rotated xarrays and put them into a list
        rotated_xarrays = []
        min_x_values = []
        max_x_values = []
        min_y_values = []
        max_y_values = []
        for i in range(0, order):
            rotation_angle = i*rotation_step
            rotated_xarray = rotation(xarray_to_be_symmetrised, rotation_angle)
            rotated_xarrays.append(rotated_xarray)
            x_values = rotated_xarray.coords[rotated_xarray.dims[0]].data
            y_values = rotated_xarray.coords[rotated_xarray.dims[1]].data
            min_x_values.append(min(x_values))
            max_x_values.append(max(x_values))
            min_y_values.append(min(y_values))
            max_y_values.append(max(y_values))
        min_x = min(min_x_values)
        max_x = max(max_x_values)
        min_y = min(min_y_values)
        max_y = max(max_y_values)

        new_coords = {xarray_to_be_symmetrised.dims[0]:[min_x,max_x], xarray_to_be_symmetrised.dims[1]:[min_y,max_y]}

        #create new coordinate xarrays for us to interpolate onto
        coord_xarrays = []
        for coord in xarray_to_be_symmetrised.dims[0:2]:
            step = xarray_to_be_symmetrised.coords[coord].data[1] - xarray_to_be_symmetrised.coords[coord].data[0]
            coord_values = np.arange(new_coords[coord][0], new_coords[coord][1], step)
            coord_values_xarray = xr.DataArray(coord_values, dims=[coord], coords={coord: coord_values})
            coord_xarrays.append(coord_values_xarray)

        #interpolate the rotated xarrays onto the expanded coord grid
        interpolated_rotated_xarrays = []
        for rotated_xarray in rotated_xarrays:
            interpolated_rotated_xarrays.append(rotated_xarray.interp({rotated_xarray.dims[0]:coord_xarrays[0], rotated_xarray.dims[1]:coord_xarrays[1]}))

        #set up nan counter so that we know how to rescale certain data points
        if fillna != False:
            x_values = interpolated_rotated_xarrays[0].coords[interpolated_rotated_xarrays[0].dims[0]].data
            y_values = interpolated_rotated_xarrays[0].coords[interpolated_rotated_xarrays[0].dims[1]].data
            nan_counter = [[0 for i in range(len(y_values))] for j in range(len(x_values))]

        #combine the rotated xarrays
        nan_counter = xr.zeros_like(interpolated_rotated_xarrays[0])

        for interpolated_rotated_xarray in interpolated_rotated_xarrays:
            if fillna != False:
                temp = interpolated_rotated_xarray.fillna(-1)
                nan_counter -= temp.where(temp == -1).fillna(0)
                interpolated_rotated_xarray = interpolated_rotated_xarray.fillna(0)
            try:
                symmetrised_xarray = symmetrised_xarray + interpolated_rotated_xarray
            except:
                symmetrised_xarray = interpolated_rotated_xarray

        if fillna != False:
            symmetrised_xarray /= (order - nan_counter)

    symmetrised_xarray.attrs = xarray.attrs

    #if there are coords units, add them to the rotated xarray
    try:
        x_coord_units = xarray_to_be_symmetrised.coords[xarray_to_be_symmetrised.dims[0]].units
        symmetrised_xarray.coords[xarray_to_be_symmetrised.dims[0]].attrs = {'units' : x_coord_units}
    except:
        pass
    try:
        y_coord_units = xarray_to_be_symmetrised.coords[xarray_to_be_symmetrised.dims[1]].units
        symmetrised_xarray.coords[xarray_to_be_symmetrised.dims[1]].attrs = {'units' : y_coord_units}
    except:
        pass

    #update history attribute
    update_hist(symmetrised_xarray, 'Symmetrised data by rotation order ' + str(order))
    
    return symmetrised_xarray

#Energy symmetrisations
@add_methods(xr.DataArray)
def EF_sym(dispersion, En=0.):
    ''' This function applies a symmetrisation around a given energy
    
    Input:
        dispersion - the dispersion to be symmetrised (xarray)
        En - float; energy for symmetrising about. Default is 0.0 eV (suitable for symmetrising around E_F if
          data is in binding energy)
    
    Returns:
        sym_dispersion - the symmetrised dispersion (xarray) '''
    
    #copy the input xarray
    sym_dispersion = dispersion.copy(deep=True)
    
    #Check if specified energy in range
    if En < min(sym_dispersion.eV.data) or En > max(sym_dispersion.eV.data):
        raise Exception("Symmetrisation energy not in range of array")

    #Define flipped energy array
    eV_values = 2*En - sym_dispersion.eV.data
    eV_flipped_array = xr.DataArray(eV_values, dims=['eV'], coords={'eV': sym_dispersion.eV.data})
    
    #Interpolate onto new array and replace NaNs with 0
    flipped_dispersion = sym_dispersion.interp(eV=eV_flipped_array).fillna(0)
    
    #Sum arrays
    sym_dispersion += flipped_dispersion
    
    #Update the analysis history
    hist = 'Symmetrised about E='+str(En)+' eV'
    update_hist(sym_dispersion,hist)
    
    return sym_dispersion

#Generic symmetrisations
@add_methods(xr.DataArray)
def sym1d(dispersion, **kwargs):
    ''' This function applies a symmetrisation around a given single axis
    
    Input:
        dispersion - the dispersion to be symmetrised (xarray)
        **kwarg optional arguments
            - ax = value, where ax is the name of the relevant xarray coordinate and value is the
              value for symmetrising about.
            - return_flipped = True, to return the flipped dispersion rather than the sum of the
              original and flipped
    
    Returns:
        sym_dispersion - the symmetrised dispersion (xarray) '''
    
    #copy the input xarray
    sym_dispersion = dispersion.copy(deep=True)
    
    #Check function called properly with single kwarg which is an axis
    return_flipped = False  # Set flag
    if len(kwargs) != 1: #Check only called with single axis kwarg
        if 'return_flipped' not in kwargs:
            raise Exception("Incorrect or missing axis definition. Function can only be called with single axis. Call as sym_1d(a, ax=value).")
        else:
            return_flipped = kwargs['return_flipped']
            kwargs.pop('return_flipped')

    axis = next(iter(kwargs)) #Pull axis name from the supplied kwarg
    if axis not in sym_dispersion.coords.dims: #Check that axis is in array
        raise Exception("Supplied axis not co-ordiate of array. Call as sym_1d(a, ax=value).")
    
    #Extract value for symmetrisation about
    val_sym = next(iter(kwargs.values()))
    
    #Check if specified energy in range
    if val_sym < min(sym_dispersion[axis].data) or val_sym > max(sym_dispersion[axis].data):
        raise Exception("Symmetrisation value not in range of relevant array")

    #Define flipped axis array
    axis_values = 2*val_sym - sym_dispersion[axis].data
    axis_flipped_array = xr.DataArray(axis_values, dims=[axis], coords={axis: sym_dispersion[axis].data})
    
    #Interpolate onto new array and replace NaNs with 0
    flipped_dispersion = sym_dispersion.interp({axis: axis_flipped_array}).fillna(0)

    if return_flipped == True:  # Return flipped dispersion
        return flipped_dispersion
    else:
        #Sum arrays
        sym_dispersion += flipped_dispersion

        #Update the analysis history
        hist = 'Symmetrised about '+str(kwargs)
        update_hist(sym_dispersion,hist)

        return sym_dispersion