#Functions to load data from netCDF files into an xarray
#Brendan Edwards 16/02/2021

import xarray as xr

def load_netCDF_data(file, **kwargs):
    '''This function loads xarrays from netCDF files
    
    Input:
        file - Path to the file to be loaded (string)
        **kwargs - Additional slicing options for partial loading of data or for binning on load (strings)
    
    Returns:
        data - DataArray or DataSet with loaded data (xarray)'''

    #load netCDF file as an xarray
    data = xr.open_dataarray(file)
    
    #netCDF attrs cannot be saved as None type
    for attr in data.attrs:
        if data.attrs[attr] == '':
            data.attrs[attr] = None
    
    #deals with how analysis_history is saved
    try:
        data.attrs['analysis_history'] = data.attrs['analysis_history'].split(',')[1:]
    except:
        data.attrs['analysis_history'] = []
    
    return data