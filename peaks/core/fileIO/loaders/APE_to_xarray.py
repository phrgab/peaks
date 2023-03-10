#Functions to load data from the Elettra APE beamline into an xarray, or for the logbook maker
#Liam Trzaska 15/04/2020
#Brendan Edwards 15/06/2021
#PK Converting to use general SES and ibw loader functions 24/7/22

import os
import numpy as np

from peaks.core.fileIO.loaders.SES_loader import SES_load, my_find_SES

def load_APE_data(file, logbook):
    '''Loads ARPES data from APE beamline @ Elettra

    Parameters
    ------------
    file : str
        Path to file to load

    logbook : bool
        Whether the data should be returned for use by the logbook maker (True) or data load (False)

    Returns
    ------------
    if logbook == False:
        data : chunked(xr.DataArray)
            xarray DataArray or DataSet with loaded data <br>
            Returned with arrays as dask objects or list of these
    else:
        sample_upload : list
            List of relevant metadata
        scan_timestamp : str
            Timestamp of the scan
    '''

    #determine the scan type
    filename, file_extension = os.path.splitext(file)

    if file_extension == '.txt':
        scan_type = 'dispersion'
    elif file_extension == '.zip':
        scan_type = 'FS'
    elif file_extension == '.ibw':
        scan_type = 'dispersion'

    # get logbook metadata
    if logbook:
        meta_lines = SES_load(file, logbook, file_extension)
        sample_upload, scan_timestamp = get_APE_metadata(file, meta_lines, scan_type, logbook)
        return sample_upload, scan_timestamp

    # get data and xarray metadata
    else:
        data, meta_lines = SES_load(file, logbook, file_extension)
        meta_list = get_APE_metadata(file, meta_lines, scan_type, logbook)

        # Check if an add dimension scan
        if 'scan_no' in data.dims:
            scan_type = 'dispersion - multiple scans'

        # Check for BE
        try:
            if data.attrs['eV_type'] == 'binding':
                be_flag = 1
        except:
            be_flag = 0

        # attach metadata
        for i in meta_list:
            data.attrs[i] = meta_list[i]
        data.name = meta_list['scan_name']

        if be_flag == 1:
            data.attrs['eV_type'] = 'binding'

    return data
        

def get_APE_metadata(file,lines,scan_type,logbook):
    '''This function will extract the relevant metadata from the data files obtained at the the Elettra APE beamline
    
    Input:
        file - Path to the file being loaded (string)
        lines - A list where each entry is a line from the text file (list)
        scan_type - The type of scan, e.g. FS map (string)
        logbook - Whether the data should be returned for use by the logbook maker (True) or xarray (False) (Boolean)
        
    Retuns:
        if logbook == True:
            meta_list - List of relevant metadata (dictionary)
        else:
            sample_upload - List of relevant metadata (list)
            scan_timestamp - Timestamp of the scan (string)
    '''
    
    #split file full path to seperate file name
    fname = file.split('/')
    fname_noext = fname[len(fname)-1].split('.')
    
    #assign metadata
    meta_list = {}
    meta_list['scan_name'] = fname_noext[0]
    meta_list['scan_type'] = scan_type
    meta_list['sample_description'] = None
    meta_list['eV_type'] = 'kinetic'
    meta_list['beamline'] = 'Elettra APE'
    meta_list['analysis_history'] = []
    meta_list['EF_correction'] = None
    meta_list['PE'] = float(my_find_SES(lines, 'Pass Energy'))
    hv = my_find_SES(lines, 'Excitation Energy')
    try:
        meta_list['hv'] = np.round(float(hv),2)
    except:
        hv=hv.replace(',', '.')
        meta_list['hv'] = np.round(float(hv),2)
    meta_list['pol'] = None
    meta_list['sweeps'] = int(my_find_SES(lines, 'Number of Sweeps'))
    meta_list['dwell'] = float(my_find_SES(lines, 'Step Time'))/1000.0
    meta_list['ana_mode'] = my_find_SES(lines, 'Lens Mode')
    meta_list['ana_slit'] = None
    meta_list['ana_slit_angle'] = 90
    meta_list['exit_slit'] = None
    meta_list['defl_par'] = None
    try:
        defl_Y_high = my_find_SES(lines, 'Thetay_High')
        defl_Y_low = my_find_SES(lines, 'Thetay_Low')
        defl_delta = my_find_SES(lines, 'Thetay_StepSize')
        meta_list['defl_perp'] =  defl_Y_low + ':' + defl_Y_high + ' (' + defl_delta + ')'
    except:
        meta_list['defl_perp'] = None
    meta_list['x1'] = None
    meta_list['x2'] = None
    meta_list['x3'] = None
    meta_list['polar'] = None
    meta_list['tilt'] = None
    meta_list['azi'] = None
    meta_list['norm_polar'] = None
    meta_list['norm_tilt'] = None
    meta_list['norm_azi'] = None
    meta_list['temp_sample'] = None
    meta_list['temp_cryo'] = None
    
    #if the data is being used for the logbook maker, we need additional data
    if logbook == True:
        scan_ext = fname_noext[1]
        scan_starttime = my_find_SES(lines, 'Time')

        #kinetic Energy
        if my_find_SES(lines, 'Acquisition Mode') == 'Swept':
            KE_st = str(my_find_SES(lines, 'Low Energy'))
            KE_end = str(my_find_SES(lines, 'High Energy'))
            analyser_KE = str(KE_st)+':'+str(KE_end)
            analyser_step = str(1000*float(my_find_SES(lines, 'Energy Step')))
        else:
            analyser_KE = str(my_find_SES(lines, 'Center Energy'))
            analyser_step = 'Fixed'
        
        #deflector
        if meta_list['defl_perp'] != None:
            analyser_thy = meta_list['defl_perp']
        else:
            analyser_thy = ''
        
        #data to return
        scan_timestamp = str(os.path.getmtime(file))
        sample_upload = [scan_ext,meta_list['scan_name'],'',scan_type,scan_starttime,analyser_KE,analyser_step,str(meta_list['PE']),str(meta_list['sweeps']),str(meta_list['dwell']),meta_list['ana_mode'],'','',str(analyser_thy),'','','','','','','','',str(meta_list['hv'])]
        return sample_upload,scan_timestamp
        
    #if the data is being used for an xarray
    else:
        return meta_list