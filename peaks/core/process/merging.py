#Functions used to merge data
#Brendan Edwards 08/06/2021

import numpy as np
import xarray as xr
import warnings
from peaks.core.utils.metadata import update_hist
from peaks.core.utils.misc import warning_simple

def merge_data(data_in, **kwargs):
    '''This function sets up the merging of the input xarrays
    
    Input:
        data_in - a list of xarrays to be merged (xarrays)
        **kwargs:
            coord - coordinate xarrays are being merged along (string)
            offsets - the offsets needed to be applied to each xarray prior to merging (list)
    
    Returns:
        merged_xarray - the merged xarray of the input xarrays (xarray)'''
    
    #get coord
    try:
        coord = kwargs['coord']
    except:
        coord='theta_par'
    
    #put the xarrays into a list
    xarrays = []
    for xarray in data_in:
        xarrays.append(xarray.copy(deep=True))
    number_of_xarrays = len(xarrays)
    
    #apply offsets to input xarrays if required
    if 'offsets' in kwargs and len(kwargs['offsets']) == len(xarrays):
        offsets = kwargs['offsets']
        for i in range(len(xarrays)):
            xarrays[i].coords[coord] = xarrays[i].coords[coord] - offsets[i]
            try:
                xarrays[i].coords[coord].attrs = {'units' : args[i].coords[coord].attrs['units']}
            except:
                pass
    else:
        warnings.simplefilter('always', UserWarning)  # Give warnings every time, even on function re-run
        warnings.formatwarning = warning_simple  # Formatting of warnings for peaks user errors
        warnings.warn("Assuming the xarrays have been shifted prior to the merge function being called")
    
    #sort xarrays into order of increasing theta_par
    min_values = []
    for xarray in xarrays:
        min_values.append([min(xarray.coords[coord].data),xarrays.index(xarray)])
    min_values.sort()
    xarrays_sorted = []
    for i in range(number_of_xarrays):
        xarray_index = (min_values[i][1])
        xarrays_sorted.append(xarrays[xarray_index])
    
    #merge xarrays
    xarrays_to_merge = list(xarrays_sorted)
    for i in range(number_of_xarrays-1):
        current_merge = merge_xarrays(xarrays_to_merge[0],xarrays_to_merge[1],coord)
        xarrays_to_merge.insert(0,current_merge)
        del xarrays_to_merge[1]
        del xarrays_to_merge[1]
    
    merged_xarray = xarrays_to_merge[0]
    
    #update metadata
    merged_xarray_name = xarrays_sorted[0].attrs['scan_name']
    for xarray in xarrays_sorted[1:]:
        merged_xarray_name = merged_xarray_name + ' & ' + xarray.attrs['scan_name']
    merged_xarray.attrs['scan_name'] = merged_xarray_name
    merged_xarray.name = merged_xarray_name
    
    #update the xarray analysis history
    update_hist(merged_xarray, 'Merged ' + merged_xarray_name + ' along ' + coord)
    
    return merged_xarray

def merge_xarrays(xarray_1, xarray_2, coord):
    '''This function merges the two input xarrays
    
    Input:
        xarray_1 - an xarray to be merged (xarray)
        xarray_2 - an xarray to be merged (xarray)
        coord - coordinate xarrays are being merged along (string)
    
    Returns:
        merged_xarray - the merged xarray of the two input xarrays (xarray)'''

    #copy the input xarrays
    xarray1 = xarray_1.copy(deep=True).transpose(...,'theta_par', 'eV')  # TODO NB hack to sort dimension order, need to fix properly
    xarray2 = xarray_2.copy(deep=True).transpose(...,'theta_par', 'eV')  # TODO NB hack to sort dimension order, need to fix properly

    xarray1_coords = []
    for coord_name in xarray1.coords:
        xarray1_coords.append(coord_name)

    xarray2_coords = []
    for coord_name in xarray2.coords:
        xarray2_coords.append(coord_name)
    
    if xarray1_coords != xarray2_coords:
        raise Exception("Both input xarrays must have the same coordinates")
    
    number_of_dims = len(xarray1_coords)
    
    #determine which xarray has the lowest coord value, and create array of the 2 xarrays in order of increasing coord value
    min_xarray1 = min(xarray1.coords[coord].data)
    min_xarray2 = min(xarray2.coords[coord].data)
    if min_xarray1 < min_xarray2:
        xarrays = [xarray1, xarray2]
    elif min_xarray2 < min_xarray1:
        xarrays = [xarray2, xarray1]
    else:
        raise Exception("Both xarrays have the same minimum " + coord + " value. Please shift the "+ coord + " axes")
    
    #get overlap region limits
    lower_limit = min(xarrays[1].coords[coord].data)
    upper_limit = max(xarrays[0].coords[coord].data)

    #get new axes for new merged xarray
    step1 = xarrays[0].coords[coord].data[1] - xarrays[0].coords[coord].data[0]
    step2 = xarrays[1].coords[coord].data[1] - xarrays[1].coords[coord].data[0]
    step = abs(min(step1,step2))
    min_coord = min(xarrays[0].coords[coord].data)
    max_coord = max(xarrays[1].coords[coord].data)
    number_of_coord_points = int((max_coord - min_coord) / step)
    coord_values = np.linspace(min_coord, max_coord, number_of_coord_points)

    #define new coord grid
    if coord != xarray1_coords[0]:
        coord_0_values = xarray1.coords[xarray1_coords[0]].data
        coord_0_xarray = xr.DataArray(coord_0_values, dims=[xarray1_coords[0]], coords={xarray1_coords[0]: coord_0_values})
    else:
        coord_0_xarray = xr.DataArray(coord_values, dims=[coord], coords={coord: coord_values})

    if coord != xarray1_coords[1]:
        coord_1_values = xarray1.coords[xarray1_coords[1]].data
        coord_1_xarray = xr.DataArray(coord_1_values, dims=[xarray1_coords[1]], coords={xarray1_coords[1]: coord_1_values})
    else:
        coord_1_xarray = xr.DataArray(coord_values, dims=[coord], coords={coord: coord_values})

    if number_of_dims == 3:
        if coord != xarray1_coords[2]:
            coord_2_values = xarray1.coords[xarray1_coords[2]].data
            coord_2_xarray = xr.DataArray(coord_2_values, dims=[xarray1_coords[2]], coords={xarray1_coords[2]: coord_2_values})
        else:
            coord_2_xarray = xr.DataArray(coord_values, dims=[coord], coords={coord: coord_values})
            
    #interpolate original xarrays onto new grids
    if number_of_dims == 2:
        new_xarray_left = xarrays[0].interp({xarray1_coords[0]:coord_0_xarray, xarray1_coords[1]:coord_1_xarray})
        new_xarray_right = xarrays[1].interp({xarray1_coords[0]:coord_0_xarray, xarray1_coords[1]:coord_1_xarray})
        
    elif number_of_dims == 3:
        new_xarray_left = xarrays[0].interp({xarray1_coords[0]:coord_0_xarray, xarray1_coords[1]:coord_1_xarray, xarray1_coords[2]:coord_2_xarray})
        new_xarray_right = xarrays[1].interp({xarray1_coords[0]:coord_0_xarray, xarray1_coords[1]:coord_1_xarray, xarray1_coords[2]:coord_2_xarray})
    else:
        raise Exception("Merging only works for 2D or 3D xarrays")
  
    #overlap region now slightly different due to new coord system - find new overlap region
    lower_limit_index = (np.abs(coord_values - lower_limit)).argmin()
    upper_limit_index = (np.abs(coord_values - upper_limit)).argmin()
    
    #normalise the weaker array such that the two xarrays have the same total number of counts in the overlap region
    if number_of_dims == 2:
        new_xarray_left_overlap_intensity = float(new_xarray_left.sel({coord:slice(coord_values[lower_limit_index],coord_values[upper_limit_index])}).sum(xarray1_coords[0]).sum(xarray1_coords[1]))
        new_xarray_right_overlap_intensity = float(new_xarray_right.sel({coord:slice(coord_values[lower_limit_index],coord_values[upper_limit_index])}).sum(xarray1_coords[0]).sum(xarray1_coords[1]))
    
    elif number_of_dims == 3:
        new_xarray_left_overlap_intensity = float(new_xarray_left.sel({coord:slice(coord_values[lower_limit_index],coord_values[upper_limit_index])}).sum(xarray1_coords[0]).sum(xarray1_coords[1]).sum(xarray1_coords[2]))
        new_xarray_right_overlap_intensity = float(new_xarray_right.sel({coord:slice(coord_values[lower_limit_index],coord_values[upper_limit_index])}).sum(xarray1_coords[0]).sum(xarray1_coords[1]).sum(xarray1_coords[2]))

    ratio = new_xarray_right_overlap_intensity / new_xarray_left_overlap_intensity
    new_xarray_left.data = new_xarray_left.data * ratio
    
    #get arrays for the intensity reduction to apply to either xarray in the overlap region
    linear_intensity_reduction_left = np.linspace(1,0,(upper_limit_index-lower_limit_index+1))
    linear_intensity_reduction_right = np.linspace(0,1,(upper_limit_index-lower_limit_index+1))
    
    #apply linear intensity reduction in overlap region
    i_range = len(new_xarray_left.coords[xarray1_coords[0]].data)
    j_range = len(new_xarray_left.coords[xarray1_coords[1]].data)

    if number_of_dims == 2:
        if coord == xarray1_coords[0]:
            for i in range(lower_limit_index, upper_limit_index+1):
                    new_xarray_left.data[i] = new_xarray_left.data[i] * linear_intensity_reduction_left[i-lower_limit_index] 
                    new_xarray_right.data[i] = new_xarray_right.data[i] * linear_intensity_reduction_right[i-lower_limit_index]
        elif coord == xarray1_coords[1]:
            for i in range(i_range):
                for j in range(lower_limit_index, upper_limit_index+1):
                    new_xarray_left.data[i][j] = new_xarray_left.data[i][j] * linear_intensity_reduction_left[j-lower_limit_index] 
                    new_xarray_right.data[i][j] = new_xarray_right.data[i][j] * linear_intensity_reduction_right[j-lower_limit_index]
  
    elif number_of_dims == 3:
        if coord == xarray1_coords[0]:
            for i in range(lower_limit_index, upper_limit_index+1):
                for j in range(j_range):
                    new_xarray_left.data[i][j] = new_xarray_left.data[i][j] * linear_intensity_reduction_left[i-lower_limit_index] 
                    new_xarray_right.data[i][j] = new_xarray_right.data[i][j] * linear_intensity_reduction_right[i-lower_limit_index]
        elif coord == xarray1_coords[1]:
            for i in range(i_range):
                for j in range(lower_limit_index, upper_limit_index+1):
                    new_xarray_left.data[i][j] = new_xarray_left.data[i][j] * linear_intensity_reduction_left[j-lower_limit_index] 
                    new_xarray_right.data[i][j] = new_xarray_right.data[i][j] * linear_intensity_reduction_right[j-lower_limit_index]

        elif coord == xarray1_coords[2]:
            for i in range(i_range):
                for j in range(j_range):
                    for k in range(lower_limit_index, upper_limit_index+1):
                        new_xarray_left.data[i][j][k] = new_xarray_left.data[i][j][k] * linear_intensity_reduction_left[k-lower_limit_index] 
                        new_xarray_right.data[i][j][k] = new_xarray_right.data[i][j][k] * linear_intensity_reduction_right[k-lower_limit_index]
    
    #replace nan values with 0's to allow the two xarrays to be summed
    new_xarray_left = new_xarray_left.fillna(0)
    new_xarray_right = new_xarray_right.fillna(0)
    
    #create merged xarray
    merged_xarray = new_xarray_left
    merged_xarray.data = new_xarray_left.data + new_xarray_right.data
    for coord_name in merged_xarray.coords:
        try:
            merged_xarray.coords[coord_name].attrs = {'units' : xarray1.coords[coord_name].attrs['units']}
        except:
            pass
    return merged_xarray.transpose(...,'eV','theta_par') # TODO NB hack to undo sort dimension order hack, need to fix properly

    