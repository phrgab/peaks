#Functions to load data from the SOLEIL CASSIOPEE beamline into an xarray, or for the logbook maker
#Liam Trzaska 15/04/2020
#Brendan Edwards 15/06/2021
#PK Converting to use general SES and ibw loader functions and some speed up for maps 27/7/22

import os
from os.path import isfile, join
import natsort
import itertools
import numpy as np
import xarray as xr
import gc

from peaks.core.fileIO.loaders.SES_loader import SES_load, my_find_SES

def load_cassiopee_data(file, logbook):

    '''Loads ARPES data from CASSIOPEE beamline @ SOLEIL

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
    else:
        sample_upload : list
            List of relevant metadata
        scan_timestamp : str
            Timestamp of the scan
    '''

    #determine the scan type
    filename, file_extension = os.path.splitext(file)

    if file_extension == '':
        # rip all the relevant filenames into two lists
        file_list_ROI = natsort.natsorted(
            [item for item in os.listdir(file) if 'ROI1' in item and isfile(join(file, item))])
        file_list_i = natsort.natsorted(
            [item for item in os.listdir(file) if '_i' in item and isfile(join(file, item))])
        with open(file + '/' + file_list_ROI[0]) as f:
            meta_lines = [line for line in itertools.islice(f, 0, 45)]
        scan_type = 'FS map'

        # get logbook metadata
        if logbook == True:
            sample_upload, scan_timestamp = get_cassiopee_metadata(file, meta_lines, scan_type, logbook)
            return sample_upload, scan_timestamp

        # load energies and angles
        eK_axis = my_find_SES(meta_lines, 'Dimension 1 scale=')
        theta_par_values = my_find_SES(meta_lines, 'Dimension 2 scale=')
        eK_axis = eK_axis.split(' ')  # convert to array
        for i in range(len(eK_axis)):
            eK_axis[i] = float(eK_axis[i])  # convert values to floats
        theta_par_values = theta_par_values.split(' ')  # convert to array
        for i in range(len(theta_par_values)):
            theta_par_values[i] = float(theta_par_values[i])  # convert values to floats

        # Extract waves for changing values
        theta_perp_values = []
        hv = []

        for file_i in file_list_i:
            with open(file + '/' + file_i) as f:
                lines = f.readlines()
                t = float(my_find_SES(lines, 'theta (deg)'))
                h = float(my_find_SES(lines, 'hv (eV)'))
            theta_perp_values.append(t)
            hv.append(h)

        if abs(hv[-1] - hv[0]) > 1:  # Then should be an hv scan
            spectrum = np.zeros((len(theta_par_values), len(eK_axis), len(hv)))
            scan_type = 'hv scan'
        else:  # Should be a polar scan
            spectrum = np.zeros((len(theta_par_values), len(eK_axis), len(theta_perp_values)))

        # Extract the analyser data
        def extract_data(fname):
            with open(file + '/' + fname) as f:
                lines = f.readlines()
                for i in range(len(lines)):
                    if lines[i].startswith('inputA='):
                        break

                # Make array to hold measured dispersion
                disp = np.zeros((len(theta_par_values), len(eK_axis)))
                for j in range(len(lines) - (i + 2)):
                    disp[:, j] = np.fromstring(lines[i + 2 + j], sep='\t')[1:]

                return disp

        for i, item in enumerate(file_list_ROI):
            spectrum[:, :, i] = extract_data(item)

        gc.collect()

        # create xarray
        if scan_type == 'hv scan':
            data = xr.DataArray(spectrum, dims=("theta_par", "eV", "hv"),
                                coords={"hv": hv, "theta_par": theta_par_values, "eV": eK_axis})
            data.coords['hv'].attrs = {'units': 'eV'}
            data.coords['theta_par'].attrs = {'units': 'deg'}
            # Add KE_delta coordinate (shift with photon energy) - for an hv scan taken in this way, this should be just the hv shift
            KE_delta = hv - hv[0]  # Change in KE value of detector as a function of hv
            data.coords['KE_delta'] = (i, KE_delta)
            data.coords['KE_delta'].attrs = {'units': 'eV'}  # Set units
        else:
            data = xr.DataArray(spectrum, dims=("theta_par", "eV", "polar"),
                                coords={"polar": theta_perp_values, "theta_par": theta_par_values, "eV": eK_axis})
            data.coords['polar'].attrs = {'units': 'deg'}
            data.coords['theta_par'].attrs = {'units': 'deg'}

        # # Convert to a faster dtype ordering for future processing
        # data = data.astype('<f8', order='C', casting='equiv')

        # get metadata
        meta_list = get_cassiopee_metadata(file, meta_lines, scan_type, logbook, theta_perp_values)

    else:
        if file_extension == '.txt':
            scan_type = 'dispersion'
        elif file_extension == '.ibw':
            scan_type = 'dispersion'

        # get logbook metadata
        if logbook:
            meta_lines = SES_load(file, logbook, file_extension)
            sample_upload, scan_timestamp = get_cassiopee_metadata(file, meta_lines, scan_type, logbook)
            return sample_upload, scan_timestamp

        # get data and xarray metadata
        else:
            data, meta_lines = SES_load(file, logbook, file_extension)
            meta_list = get_cassiopee_metadata(file, meta_lines, scan_type, logbook)

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

    try:
        if be_flag == 1:
            data.attrs['eV_type'] = 'binding'
    except:
        pass

    return data

def get_cassiopee_metadata(file,lines,scan_type,logbook,*args):
    '''This function will extract the relevant metadata from the data files obtained at the the SOLEIL CASSIOPEE beamline

    Input:
        file - Path to the file being loaded (string)
        lines - A list where each entry is a line from the text file (list)
        scan_type - The type of scan, e.g. FS map (string)
        logbook - Whether the data should be returned for use by the logbook maker (True) or xarray (False) (Boolean)
        *args - Used to pass tilt values for Fermi maps (list)

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
    meta_list['beamline'] = 'SOLEIL CASSIOPEE'
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

    #additional metadata within FS maps
    if scan_type == 'FS map':
        file_list_i = natsort.natsorted([item for item in os.listdir(file) if '_i' in item and isfile(join(file, item))])
        with open(join(file, file_list_i[0])) as f_i:
            lines_alt = f_i.readlines()
        pol_list = ['LV', 'LH', 'AV', 'AH', 'CR']
        pol_index = int(my_find_SES(lines_alt, 'Polarisation [0:LV, 1:LH, 2:AV, 3:AH, 4:CR]'))
        meta_list['pol'] = pol_list[pol_index]
        meta_list['x1'] = float(my_find_SES(lines_alt, 'x (mm)'))
        meta_list['x2'] = float(my_find_SES(lines_alt, 'z (mm)'))
        meta_list['x3'] = float(my_find_SES(lines_alt, 'y (mm)'))
        #already have polar values if using the data loader
        if logbook == False:
            polar_values = args[0]
        #still need to get the polar values if using the logbook maker
        else:
            file_list_i = natsort.natsorted([item for item in os.listdir(file) if '_i' in item and isfile(join(file, item))])
            polar_values = []
            for file_i in file_list_i:
                with open(file+'/'+file_i) as f:
                    lines_to_polar = f.readlines()
                    t = float(my_find_SES(lines_to_polar, 'theta (deg)'))
                polar_values.append(t)
        num_polar_values = len(polar_values)
        first_polar = polar_values[0]
        last_polar = polar_values[num_polar_values-1]
        step = (last_polar - first_polar) / (num_polar_values-1)
        meta_list['polar'] = str(first_polar) + ':' + str(last_polar) + ' (' + str(step) + ')'
        # Tommaso 2022 Feb: change in the metadata maybe due to change in the manipulator at cassiopee
        try:
            meta_list['tilt'] = float(my_find_SES(lines_alt, 'tilt (deg)'))
            meta_list['azi'] = float(my_find_SES(lines_alt, 'phi (deg)'))
        except:
            print('tilt and azi NOT in metadata!')
            pass

    #if the data is being used for the logbook maker, we need additional data
    if logbook == True:
        try:
            scan_ext = fname_noext[1]
        except:
            scan_ext = 'folder'
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

        #data to return
        scan_timestamp = str(os.path.getmtime(file))
        if scan_type == 'FS map':
            beamline_resolution = str(my_find_SES(lines_alt, 'Resolution (meV)'))
            sample_upload = [scan_ext,meta_list['scan_name'],'',scan_type,scan_starttime,analyser_KE,analyser_step,str(meta_list['PE']),str(meta_list['sweeps']),str(meta_list['dwell']),str(meta_list['ana_mode']),'',str(meta_list['polar']),str(meta_list['tilt']),str(meta_list['azi']),str(meta_list['x1']),str(meta_list['x3']),str(meta_list['x2']),'','',str(meta_list['hv']),'',beamline_resolution,str(meta_list['pol'])]
        else:
            sample_upload = [scan_ext,meta_list['scan_name'],'',scan_type,scan_starttime,analyser_KE,analyser_step,str(meta_list['PE']),str(meta_list['sweeps']),str(meta_list['dwell']),str(meta_list['ana_mode']),'','','','','','','','','',str(meta_list['hv']),'','','']
        return sample_upload,scan_timestamp
        
    #if the data is being used for an xarray
    else:
        return meta_list
        
