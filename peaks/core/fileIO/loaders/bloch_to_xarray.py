#Functions to load data from the MAX IV Bloch beamline into an xarray, or for the logbook maker
#Liam Trzaska 15/04/2020
#Brendan Edwards 15/06/2021
#PK Converting to use general SES and ibw loader functions 24/7/22

import os
import numpy as np

from peaks.core.fileIO.loaders.SES_loader import SES_load, my_find_SES, manip_from_SES_metadata
from peaks.core.utils.misc import ana_warn

def load_bloch_data(file, logbook):
    '''Loads ARPES data from BLOCH (A-branch) beamline @ Max-IV

    Parameters
    ------------
    file : str
        Path to file to load

    logbook : bool
        Whether the data should be returned for use by the logbook maker (True) or data load (False)

    Returns
    ------------
    if logbook == False:
        data : xr.DataArray
            xarray DataArray or DataSet with loaded data <br>
            Returned with arrays as dask objects for spatial mapping data
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
        scan_type = 'dispersion'  # For now, can update later

    # get logbook metadata
    if logbook:
        meta_lines = SES_load(file, logbook, file_extension)
        sample_upload, scan_timestamp = get_APE_metadata(file, meta_lines, scan_type, logbook)
        return sample_upload, scan_timestamp

    # get data and xarray metadata
    else:
        data, meta_lines = SES_load(file, logbook, file_extension)
        meta_list = get_bloch_metadata(file, meta_lines, scan_type, logbook)

        # Check if an add dimension scan
        if 'scan_no' in data.dims:
            scan_type = 'dispersion - multiple scans'

        # Check for BE
        try:
            if data.attrs['eV_type'] == 'binding':
                be_flag = 1
        except:
            be_flag = 0

        # For manipulator-based scans, need to read additional metadata and re-define dims
        # Try to read the data
        try:
            axis = manip_from_SES_metadata(meta_lines)  # Extract the manipulator scan data
        except:
            axis = None


        if 'dim1' in data.dims:  # 4D scan with; almost certainly a spatial map
            scan_type = "spatial map"

            # Try and get manipulator data, sometimes this seems missing
            try:
                mapping_short = []
                mapping_dim = []
                for i in axis:
                    if abs(axis[i].max() - axis[i].min()) > 0.01:  # This is probably a scan direction
                        if i == 'X':
                            mapping_short.append(i)
                            mapping_dim.append('x1')
                        elif i == 'Y':
                            mapping_short.append(i)
                            mapping_dim.append('x2')
                        elif i == 'Z':
                            mapping_short.append(i)
                            mapping_dim.append('x3')


                # Work out which is the fast axis
                if abs(axis[mapping_short[0]][1] - axis[mapping_short[0]][0]) > 0.01:  # this one is the fast axis
                    mapping_short_fast = mapping_short[0]
                    mapping_dim_fast = mapping_dim[0]
                    mapping_short_slow = mapping_short[1]
                    mapping_dim_slow = mapping_dim[1]
                else:
                    mapping_short_fast = mapping_short[1]
                    mapping_dim_fast = mapping_dim[1]
                    mapping_short_slow = mapping_short[0]
                    mapping_dim_slow = mapping_dim[0]

                mapping_fast0 = axis[mapping_short_fast][0]
                mapping_fast1 = axis[mapping_short_fast][-1]
                N_fast = len(data.dim0)

                mapping_slow0 = axis[mapping_short_slow][0]
                mapping_slow1 = axis[mapping_short_slow][-1]
                N_slow = len(data.dim1)

                mapping_dim_fast_values = np.linspace(mapping_fast0, mapping_fast1, N_fast)
                mapping_dim_slow_values = np.linspace(mapping_slow0, mapping_slow1, N_slow)

                # redefine the dimensions
                data['dim0'] = mapping_dim_fast_values
                data = data.rename({'dim0': mapping_dim_fast})

                data['dim1'] = mapping_dim_slow_values
                data = data.rename({'dim1': mapping_dim_slow})

            except:
                # Give a warning
                warning_str = 'Manipulator data seems to be missing - assumed a spatial map, please check this.'
                ana_warn(warning_str, warn_type='warn')

        elif 'dim0' in data.dims:  # 3D scan with a manipulator dimension; need to try and figure out what from the data
            try:  # Not everything seems to come with a full manipulator list
                if abs(axis['P'][1] - axis['P'][0]) > 0.02 or abs(
                        axis['T'][1] - axis['T'][0]) > 0.02:  # Looks like a manipulator map
                    scan_type = 'FS map'

                    # Extract relevant manipulator coordinates
                    if abs(axis['P'][1] - axis['P'][0]) > 0.02:  # Looks like a polar map
                        mapping_dim = 'polar'
                        mapping_short = 'P'
                    elif abs(axis['T'][1] - axis['T'][0]) > 0.02:  # Looks like a tilt map
                        mapping_dim = 'tilt'
                        mapping_short = 'T'

                    # redefine the dimensions
                    data['dim0'] = axis[mapping_short]
                    data = data.rename({'dim0': mapping_dim})

                else:
                    scan_type = 'spatial map'
                    for i in axis:
                        if abs(axis[i].max() - axis[i].min()) > 0.01:  # This is probably the scan direction
                            mapping_short = i

                            if i == 'X':
                                mapping_dim = 'x1'
                            elif i == 'Y':
                                mapping_dim = 'x2'
                            elif i == 'Z':
                                mapping_dim = 'x3'
                                scan_type = 'focus scan'

                    # redefine the dimensions
                    data['dim0'] = axis[mapping_short]
                    data = data.rename({'dim0': mapping_dim})

            except:  # Best guess is a spatial map, as polar maps seems to generally come with the table
                scan_type = 'line scan'
                # Give a warning
                warning_str = 'Manipulator data seems to be missing - assumed a spatial scan, please check this.'
                ana_warn(warning_str, warn_type='warn')

        # attach metadata
        meta_list['scan_type'] = scan_type
        for i in meta_list:
            data.attrs[i] = meta_list[i]
        data.name = meta_list['scan_name']

        if be_flag == 1:
            data.attrs['eV_type'] = 'binding'

        # If a spatial map, chunk the data and return as a dask-backed array
        if scan_type == 'spatial map':
            try:
                data = data.chunk({'x1': 'auto', 'x2': 'auto'})
            except:  # Catch for when the dims are not properly defined
                try:
                    data = data.chunk({'dim0': 'auto', 'dim1': 'auto'})
                except:
                    pass
    return data
        

def get_bloch_metadata(file,lines,scan_type,logbook):
    '''This function will extract the relevant metadata from the data files obtained at the the MAX IV Bloch beamline
    
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
    meta_list['beamline'] = 'MAX IV Bloch'
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

    try:
        axis = manip_from_SES_metadata(lines)  # Extract the wavenote manipulator scan data
        if abs(axis['X'][1] - axis['X'][0]) > 0.01:
            meta_list['x1'] = str(axis['X'][0]) + ' : ' + str(axis['X'][-1])
        else:
            meta_list['x1'] = float(axis['X'][0])

        if abs(axis['Y'][1] - axis['Y'][0]) > 0.01:
            meta_list['x2'] = str(axis['Y'][0]) + ' : ' + str(axis['Y'][-1])
        else:
            meta_list['x2'] = float(axis['Y'][0])

        if abs(axis['Z'][1] - axis['Z'][0]) > 0.01:
            meta_list['x3'] = str(axis['Z'][0]) + ' : ' + str(axis['Z'][-1])
        else:
            meta_list['x3'] = float(axis['Z'][0])

        if abs(axis['P'][1] - axis['P'][0]) > 0.02:
            meta_list['polar'] = str(axis['P'][0]) + ' : ' + str(axis['P'][-1])
        else:
            meta_list['polar'] = float(axis['P'][0])

        if abs(axis['T'][1] - axis['T'][0]) > 0.02:
            meta_list['tilt'] = str(axis['T'][0]) + ' : ' + str(axis['T'][-1])
        else:
            meta_list['tilt'] = float(axis['T'][0])

        if abs(axis['A'][1] - axis['A'][0]) > 0.02:
            meta_list['azi'] = str(axis['A'][0]) + ' : ' + str(axis['A'][-1])
        else:
            meta_list['azi'] = float(axis['A'][0])
    except:
        try:
            meta_list['x1'] = float(my_find_SES(lines, 'X='))
        except:
            meta_list['x1'] = None
        try:
            meta_list['x2'] = float(my_find_SES(lines, 'Y='))
        except:
            meta_list['x2'] = None
        try:
            meta_list['x3'] = float(my_find_SES(lines, 'Z='))
        except:
            meta_list['x3'] = None
        try:
            meta_list['polar'] = float(my_find_SES(lines, 'P='))
        except:
            meta_list['polar'] = None
        try:
            meta_list['tilt'] = float(my_find_SES(lines, 'T='))
        except:
            meta_list['tilt'] = None
        try:
            meta_list['azi'] = float(my_find_SES(lines, 'A='))
        except:
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
        sample_upload = [scan_ext,meta_list['scan_name'],'',scan_type,scan_starttime,analyser_KE,analyser_step,str(meta_list['PE']),str(meta_list['sweeps']),str(meta_list['dwell']),str(meta_list['ana_mode']),'','',analyser_thy,str(meta_list['polar']),str(meta_list['tilt']),str(meta_list['azi']),str(meta_list['x1']),str(meta_list['x2']),str(meta_list['x3']),'','',str(meta_list['hv'])]
        return sample_upload,scan_timestamp
    
    #if the data is being used for an xarray
    else:
        return meta_list