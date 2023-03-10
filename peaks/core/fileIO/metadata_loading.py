#Functions used for metadata loading from a google sheet logbook into a peaks xarray
#Brendan Edwards 24/05/2021

import xarray as xr
import random
import time
from IPython.display import clear_output
from peaks.logbook.logbook_fns import authentication
from peaks.utils.OOP_method import add_methods

@add_methods(xr.DataArray)
def load_metadata(*args, **kwargs):
    '''This function retrieves the metadata saved in a spreadsheet for the input scans, converts it to peaks conventions and updates the input xarrays attrs accordingly
    
    Input:
        *args:
            data for which metadata will be loaded from the logbook (DataArray, DataSet, or list of DataArrays and/or DataSets)
        **kwargs:
            sample_name - the name of the sample (string)
            spreadsheet identifier (one of the following):
                key_ID - Key ID of the project spreadsheet (string)
                project_name - name of the project (string)
                proposal_ID - beamtime proposal ID (string)
    '''
    
    #make sure there are some args loaded
    if len(args) == 0:
        raise Exception("Please enter some args (these are the xarrays you are getting metadata for)")
        
    #list of xarrays which we will get metadata for
    xarrays_to_load_metadata = []
    
    #go through arguments to get all the DataArrays for which we will get metadata
    for i in range(len(args)):
        #if argument is a dataset, add individual DataArrays to xarrays_to_load_metadata
        if type(args[i]) == xr.core.dataset.Dataset:
            for xarray_name in args[i].data_vars:
                xarrays_to_load_metadata.append(args[i][xarray_name])
        
        #if argument is a DataArray, add it to xarrays_to_load_metadata
        elif type(args[i]) == xr.core.dataarray.DataArray:
            xarrays_to_load_metadata.append(args[i])

        #if argument is a list of DataArrays/DataSets, add individual DataArrays to xarrays_to_load_metadata
        elif type(args[i]) == list:
            for item in args[i]:
                #if list item is a dataset, add individual DataArrays to xarrays_to_load_metadata
                if type(item) == xr.core.dataset.Dataset:
                    for xarray_name in item.data_vars:
                        xarrays_to_load_metadata.append(item[xarray_name])
                #if list item is a DataArray, add it to xarrays_to_load_metadata
                elif type(item) == xr.core.dataarray.DataArray:
                    xarrays_to_load_metadata.append(item)
    
    #make sure there are only two kwargs, one being sample_name and one being key_ID, project_name or proposal_ID
    if len(kwargs) != 2 or 'sample_name' not in kwargs:
        raise Exception("Please enter two kwargs (sample_name, and either key_ID, project_name or proposal_ID)")
    if 'key_ID' not in kwargs and 'project_name' not in kwargs and 'proposal_ID' not in kwargs:
        raise Exception("Please enter two kwargs (sample_name, and either key_ID, project_name or proposal_ID)")
    
    #get logbook metadata from the get_logbook_metadata function
    logbook_metadata = get_logbook_metadata(xarrays_to_load_metadata, **kwargs)
    
    i=0
    for xarray in xarrays_to_load_metadata:
        # get the current scan logbook metadata dictionary 
        current_scan_logbook_metadata = logbook_metadata[i]
        
        #get xarray identifier so we know what metadata to extract (BL, or if it is a LEED image)
        try:
            identifier = xarray.attrs['beamline']
        except:
            if xarray.attrs['scan_type'] == 'LEED':
                identifier = 'LEED'
            else:
                raise Exception("Cannot identify data type")

        #update xarray metadata
        if identifier == 'St Andrews - Phoibos':
            xarray.attrs['sample_description'] = current_scan_logbook_metadata['Sample description:']
            xarray.attrs['hv'] = current_scan_logbook_metadata['hv\n(eV)']
            xarray.attrs['pol'] = current_scan_logbook_metadata['Polarisation']
            xarray.attrs['x1'] = current_scan_logbook_metadata['y\n(mm)']
            xarray.attrs['x2'] = current_scan_logbook_metadata['z\n(mm)']
            xarray.attrs['x3'] = current_scan_logbook_metadata['x\n(mm)']
            xarray.attrs['polar'] = current_scan_logbook_metadata['Theta\n(deg)']
            xarray.attrs['tilt'] = current_scan_logbook_metadata['Phi\n(deg)']
            xarray.attrs['azi'] = current_scan_logbook_metadata['Azi\n(deg)']
            xarray.attrs['temp_sample'] = current_scan_logbook_metadata['Sample\n(K)']
            xarray.attrs['temp_cryo'] = current_scan_logbook_metadata['Cryostat\n(K)']

        elif identifier == 'St Andrews - MBS':
            xarray.attrs['sample_description'] = current_scan_logbook_metadata['Sample description:']
            xarray.attrs['hv'] = current_scan_logbook_metadata['hv\n(eV)']
            xarray.attrs['pol'] = current_scan_logbook_metadata['Polarisation']
            xarray.attrs['ana_slit'] = current_scan_logbook_metadata['Slit']
            xarray.attrs['x1'] = current_scan_logbook_metadata['y\n(mm)']
            xarray.attrs['x2'] = current_scan_logbook_metadata['z\n(mm)']
            xarray.attrs['x3'] = current_scan_logbook_metadata['x\n(mm)']
            xarray.attrs['polar'] = current_scan_logbook_metadata['Theta\n(deg)']
            xarray.attrs['tilt'] = current_scan_logbook_metadata['Phi\n(deg)']
            xarray.attrs['azi'] = current_scan_logbook_metadata['Azi\n(deg)']
            xarray.attrs['temp_sample'] = current_scan_logbook_metadata['Sample\n(K)']
            xarray.attrs['temp_cryo'] = current_scan_logbook_metadata['Cryostat\n(K)']

        elif identifier == 'Diamond I05-HR':
            xarray.attrs['sample_description'] = current_scan_logbook_metadata['Sample description:']

        elif identifier == 'Diamond I05-nano':
            xarray.attrs['sample_description'] = current_scan_logbook_metadata['Sample description:']

        elif identifier == 'MAX IV Bloch':
            xarray.attrs['sample_description'] = current_scan_logbook_metadata['Sample description:']
            xarray.attrs['pol'] = current_scan_logbook_metadata['Polarisation']
            xarray.attrs['ana_slit'] = current_scan_logbook_metadata['Slit']
            xarray.attrs['exit_slit'] = current_scan_logbook_metadata['Exit slit\n(um)']
            xarray.attrs['temp_sample'] = current_scan_logbook_metadata['Sample\n(K)']
            xarray.attrs['temp_cryo'] = current_scan_logbook_metadata['Cryostat\n(K)']
            xarray.attrs['defl_par'] = current_scan_logbook_metadata['Theta_x\n(deg)']
            if xarray.attrs['scan_type'] != 'FS map':
                xarray.attrs['defl_perp'] = current_scan_logbook_metadata['Theta_y\n(deg)']

        elif identifier == 'Elettra APE':
            xarray.attrs['sample_description'] = current_scan_logbook_metadata['Sample description:']
            xarray.attrs['pol'] = current_scan_logbook_metadata['Polarisation']
            xarray.attrs['ana_slit'] = current_scan_logbook_metadata['Slit']
            xarray.attrs['exit_slit'] = current_scan_logbook_metadata['Exit slit\n(um)']
            xarray.attrs['x1'] = current_scan_logbook_metadata['y\n(mm)']
            xarray.attrs['x2'] = current_scan_logbook_metadata['z\n(mm)']
            xarray.attrs['x3'] = current_scan_logbook_metadata['x\n(mm)']
            xarray.attrs['polar'] = current_scan_logbook_metadata['Polar\n(deg)']
            xarray.attrs['tilt'] = current_scan_logbook_metadata['Tilt\n(deg)']
            xarray.attrs['azi'] = current_scan_logbook_metadata['Azi\n(deg)']
            xarray.attrs['temp_sample'] = current_scan_logbook_metadata['Sample\n(K)']
            xarray.attrs['temp_cryo'] = current_scan_logbook_metadata['Cryostat\n(K)']
            xarray.attrs['defl_par'] = current_scan_logbook_metadata['Theta_x\n(deg)']
            if xarray.attrs['scan_type'] != 'FS map':
                xarray.attrs['defl_perp'] = current_scan_logbook_metadata['Theta_y\n(deg)']    

        elif identifier == 'SOLEIL CASSIOPEE':
            xarray.attrs['sample_description'] = current_scan_logbook_metadata['Sample description:']
            xarray.attrs['ana_slit'] = current_scan_logbook_metadata['Slit']
            xarray.attrs['exit_slit'] = current_scan_logbook_metadata['Exit slit\n(um)']
            xarray.attrs['temp_sample'] = current_scan_logbook_metadata['Sample\n(K)']
            xarray.attrs['temp_cryo'] = current_scan_logbook_metadata['Cryostat\n(K)']

            if xarray.attrs['scan_type'] != 'FS map':
                xarray.attrs['pol'] = current_scan_logbook_metadata['Polarisation']
                xarray.attrs['x1'] = current_scan_logbook_metadata['x\n(mm)']
                xarray.attrs['x2'] = current_scan_logbook_metadata['z\n(mm)']
                xarray.attrs['x3'] = current_scan_logbook_metadata['y\n(mm)']
                xarray.attrs['polar'] = current_scan_logbook_metadata['Polar\n(deg)']
                xarray.attrs['tilt'] = current_scan_logbook_metadata['Tilt\n(deg)']
                xarray.attrs['azi'] = current_scan_logbook_metadata['Azi\n(deg)']

        elif identifier == 'LEED':
            #assume LEED is done on StA Phoibos system
            xarray.attrs['sample_description'] = current_scan_logbook_metadata['Sample description:']
            xarray.attrs['energy'] = current_scan_logbook_metadata['hv\n(eV)']
            xarray.attrs['x1'] = current_scan_logbook_metadata['y\n(mm)']
            xarray.attrs['x2'] = current_scan_logbook_metadata['z\n(mm)']
            xarray.attrs['x3'] = current_scan_logbook_metadata['x\n(mm)']
            xarray.attrs['polar'] = current_scan_logbook_metadata['Theta\n(deg)']
            xarray.attrs['tilt'] = current_scan_logbook_metadata['Phi\n(deg)']
            xarray.attrs['azi'] = current_scan_logbook_metadata['Azi\n(deg)']
            xarray.attrs['temp_sample'] = current_scan_logbook_metadata['Sample\n(K)']
            xarray.attrs['temp_cryo'] = current_scan_logbook_metadata['Cryostat\n(K)']

        for attr in xarray.attrs:
            #if a logbook metadata entry is blank, make the attr None type,
            if xarray.attrs[attr] == '':
                xarray.attrs[attr] = None
            #if an attr can be made a float, do it
            try:
                xarray.attrs[attr] = float(xarray.attrs[attr])
            except:
                pass
        
        i = i + 1
        
def get_logbook_metadata(xarrays_to_load_metadata, **kwargs):
    '''This function retrieves the metadata saved in a spreadsheet for the inputted scans
    
    Input:
        xarrays_to_load_metadata - xarrays for which metadata will be loaded from the logbook (list of xarrays)
        **kwargs:
            sample_name - the name of the sample (string)
            spreadsheet identifier (one of the following):
                key_ID - Key ID of the project spreadsheet (string)
                project_name - name of the project (string)
                proposal_ID - beamtime proposal ID (string)
    
    Returns:
        logbook_metadata - list of all the metadata saved in the spreadsheet for the input scans (list of dictionaries)
    '''
    
    #exponential backoff
    n_start = 2
    n_stop = 10
    for n in range(n_start, n_stop):
        try:
            #get key_ID
            if 'project_name' in kwargs:
                #open usage spreadsheet
                sh = authentication('1b-xjFcCrhU66wUcTHAzUfpES92qkDnq7ueIXvLkdR2U')
                active_sheet = sh.worksheet('Projects')
                #get project names
                spreadsheet_project_names = active_sheet.col_values(2)
                #get project key IDs
                spreadsheet_key_IDs = active_sheet.col_values(4)
                #get key ID for project with project_name
                key_ID = spreadsheet_key_IDs[spreadsheet_project_names.index(kwargs['project_name'])] 
            elif 'proposal_ID' in kwargs:
                #open usage spreadsheet
                sh = authentication('1b-xjFcCrhU66wUcTHAzUfpES92qkDnq7ueIXvLkdR2U')
                active_sheet = sh.worksheet('Beamtimes')
                #get beamtime proposal IDs
                spreadsheet_proposal_IDs = active_sheet.col_values(6)
                #get beamtime key IDs
                spreadsheet_key_IDs = active_sheet.col_values(7)
                #get key ID for beamtime with proposal_ID
                key_ID = spreadsheet_key_IDs[spreadsheet_proposal_IDs.index(kwargs['proposal_ID'])] 
            elif 'key_ID' in kwargs:
                key_ID = kwargs['key_ID']

            #open project/beamtime spreadsheet
            sh = authentication(key_ID)
            active_sheet = sh.worksheet(kwargs['sample_name'])
            #get the scans listed in the sample spreadsheet
            spreadsheet_scan_names = active_sheet.col_values(1)
            #get metadata labels
            metadata_labels = active_sheet.row_values(2)
            break
        except Exception as e:
            #If the error is an API error
            if str(e).split(",")[0] == "{'code': 429":
                if n < n_stop-1:
                    random_number_milliseconds = random.random()
                    print("Request limit reached. Waiting " + str((2 ** n)) + " seconds before retrying action")
                    time.sleep(((2 ** n)) + random_number_milliseconds)
                    clear_output(wait=True)
                else:
                    #If no success after the 256 second wait
                    raise Exception("There has been an error. The request never succeeded")
            #If the error is not an API error
            else:
                raise Exception(e)           
            
    logbook_metadata = []
    for xarray in xarrays_to_load_metadata:
        #exponential backoff
        n_start = 2
        n_stop = 10
        for n in range(n_start, n_stop):
            try:
                scan_name = xarray.attrs['scan_name']
                #get scan_name row in spreadsheet
                index = spreadsheet_scan_names.index(scan_name) + 1
                #get scan metadata
                scan_metadata = active_sheet.row_values(index)
                sample_des = active_sheet.acell('B4').value
                #produce and return current_scan_logbook_metadata dictionary
                current_scan_logbook_metadata = {}
                for i in range (len(scan_metadata)):
                    current_scan_logbook_metadata[metadata_labels[i]] = scan_metadata[i]
                current_scan_logbook_metadata['Sample description:'] = sample_des
                #append current_scan_logbook_metadata dictionary to logbook_metadata
                logbook_metadata.append(current_scan_logbook_metadata)
                break
            except Exception as e:
                #If the error is an API error
                if str(e).split(",")[0] == "{'code': 429":
                    if n < n_stop-1:
                        random_number_milliseconds = random.random()
                        print("Request limit reached. Waiting " + str((2 ** n)) + " seconds before retrying action")
                        time.sleep(((2 ** n)) + random_number_milliseconds)
                        clear_output(wait=True)
                    else:
                        #If no success after the 256 second wait
                        raise Exception("There has been an error. The request never succeeded")
                #If the error is not an API error
                else:
                    raise Exception(e)  
    return logbook_metadata