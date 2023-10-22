#Function used to rotate xarray data
#Brendan Edwards 02/12/2021

import xarray as xr
import numpy as np
from peaks.core.utils.metadata import update_hist
from peaks.core.utils.OOP_method import add_methods

@add_methods(xr.DataArray)
def rotation(xarray, rotation):
    '''This function takes a 2D input xarray and rotates it around (0,0)
    
    Inputs:
        xarray - the data which will be rotated (xarray)
        rotation - the rotation in degrees which will be applied (float/int)
    
    Returns:
        rotated_xarray - the rotated xarray
    '''
    
    #copy the input xarray
    xarray_to_be_expanded = xarray.copy(deep=True)
    
    #No need to do anything if the rotation is a multiple of 360
    if rotation % 360 == 0:
        return xarray_to_be_expanded
    
    #get the coordinates of the xarray coordinates
    x_min = float(min(xarray_to_be_expanded.coords[xarray_to_be_expanded.dims[1]]))
    x_max = float(max(xarray_to_be_expanded.coords[xarray_to_be_expanded.dims[1]]))
    y_min = float(min(xarray_to_be_expanded.coords[xarray_to_be_expanded.dims[0]]))
    y_max = float(max(xarray_to_be_expanded.coords[xarray_to_be_expanded.dims[0]]))
    corner_coords = [[x_min, y_max], [x_max, y_max], [x_min, y_min], [x_max, y_min]]
    
    #get the positions these corners will be rotated to (so we can resize the coordinates to not cut off data)
    rotated_corner_coords = []
    for coord in corner_coords:
        rotated_x = (np.cos(np.radians(rotation)) * coord[0]) - (np.sin(np.radians(rotation)) * coord[1])
        rotated_y = (np.sin(np.radians(rotation)) * coord[0]) + (np.cos(np.radians(rotation)) * coord[1])
        rotated_corner_coords.append([rotated_x, rotated_y])
    
    #get our new x and y limits
    rotated_min_x = min([coord[0] for coord in rotated_corner_coords])
    rotated_max_x = max([coord[0] for coord in rotated_corner_coords])
    rotated_min_y = min([coord[1] for coord in rotated_corner_coords])
    rotated_max_y = max([coord[1] for coord in rotated_corner_coords])
    
    #make sure we don't lose any data
    if rotated_min_x > x_min:
        new_min_x = x_min
    else:
        new_min_x = rotated_min_x
    if rotated_max_x < x_max:
        new_max_x = x_max
    else:
        new_max_x = rotated_max_x
    if rotated_min_y > y_min:
        new_min_y = y_min
    else:
        new_min_y = rotated_min_y
    if rotated_max_y < y_max:
        new_max_y = y_max
    else:
        new_max_y = rotated_max_y
    new_coords = {xarray_to_be_expanded.dims[0]:[new_min_y,new_max_y], xarray_to_be_expanded.dims[1]:[new_min_x,new_max_x]}
    
    #create new coordinate xarrays for us to interpolate onto
    coord_xarrays = []
    for coord in xarray_to_be_expanded.dims[0:2]:
        step = xarray_to_be_expanded.coords[coord].data[1] - xarray_to_be_expanded.coords[coord].data[0]
        coord_values = np.arange(new_coords[coord][0], new_coords[coord][1], step)
        coord_values_xarray = xr.DataArray(coord_values, dims=[coord], coords={coord: coord_values})
        coord_xarrays.append(coord_values_xarray)
    
    #interpolate the xarray onto the expanded coord grid
    xarray_to_be_rotated = xarray_to_be_expanded.interp({xarray_to_be_expanded.dims[0]:coord_xarrays[0], xarray_to_be_expanded.dims[1]:coord_xarrays[1]})

    #get coord information - note defintion of x and y flips here
    xarray_coords = xarray_to_be_rotated.dims
    x_coord = xarray_coords[0]
    y_coord = xarray_coords[1]
    
    #get data
    x_values = xarray_to_be_rotated.coords[x_coord].data
    y_values = xarray_to_be_rotated.coords[y_coord].data

    #work out mapping onto rotated grid
    orignal_coords = [[0 for i in range(len(y_values))] for j in range(len(x_values))]
    new_x_values = [[0 for i in range(len(y_values))] for j in range(len(x_values))]
    new_y_values = [[0 for i in range(len(y_values))] for j in range(len(x_values))]
    for i in range(len(x_values)):
        for j in range(len(y_values)):
            orignal_coords[i][j] = [x_values[i],y_values[j]]
            x1 = x_values[i]
            y1 = y_values[j]
            new_x_values[i][j] = (np.cos(np.radians(rotation)) * x1) - (np.sin(np.radians(rotation)) * y1)
            new_y_values[i][j] = (np.sin(np.radians(rotation)) * x1) + (np.cos(np.radians(rotation)) * y1)

    #x values to interpolate onto
    x_xarray = xr.DataArray(
        new_x_values,
        dims=[x_coord, y_coord],
        coords={x_coord: x_values, y_coord: y_values},
    )

    #y values to interpolate onto
    y_xarray = xr.DataArray(
        new_y_values,
        dims=[x_coord, y_coord],
        coords={x_coord: x_values, y_coord: y_values},
    )
    
    #perform interpolation, and cut off any coords we no longer need
    rotated_xarray = xarray_to_be_rotated.interp({x_coord:x_xarray, y_coord:y_xarray}).sel({y_coord:slice(rotated_min_x,rotated_max_x)}).sel({x_coord:slice(rotated_min_y,rotated_max_y)})

    #if there are coords units, add them to the rotated xarray
    try:
        x_coord_units = xarray_to_be_rotated.coords[x_coord].units
        rotated_xarray.coords[x_coord].attrs = {'units' : x_coord_units}
    except:
        pass
    try:
        y_coord_units = xarray_to_be_rotated.coords[y_coord].units
        rotated_xarray.coords[y_coord].attrs = {'units' : y_coord_units}
    except:
        pass
    
    #update history attribute
    update_hist(rotated_xarray, 'Rotated data by ' + str(rotation) + ' degrees')
    return rotated_xarray