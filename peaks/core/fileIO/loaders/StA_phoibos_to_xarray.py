#Functions to load data from the St Andrews Phoibos ARPES system into an xarray, or for the logbook maker
#Lewis Hart 11/01/2021
#Brendan Edwards 01/04/2021

import os
from datetime import datetime, timedelta
import numpy as np
from decimal import Decimal
import xarray as xr
import dask.array as da

def load_StA_phoibos_data(file, logbook, **kwargs):
    '''This function loads ARPES data from the St Andrews Phoibos ARPES system
    
    Input:
        file - Path to the file to be loaded (string)
        logbook - Whether the data should be returned for use by the logbook maker (True) or xarray (False) (Boolean)
        **kwargs - Additional slicing options for partial loading of data or for binning on load (strings)
    
    Returns:
        if logbook == False:
            data - DataArray or DataSet with loaded data (xarray)
        else:
            sample_upload - List of relevant metadata (list)
            scan_timestamp - Timestamp of the scan (string)'''

    #determine if the file is a dispersion, Fermi map or XPS measurement
    with open(file) as f:
        lines = f.readlines()
    for line in lines:
        #if the analysis method is XPS, it is an XPS measurment
        if line.startswith('# Analysis Method:   XPS'):
            scan_type = 'XPS'
            break     
        #if the file contains azi values, it is a Fermi map
        elif line.startswith('# Parameter: "Phi [deg]" = '):
            scan_type = 'FS map'
            break
        #only option left is a standard dispersion
        elif line.startswith('# Number of Scans:'):
            scan_type = 'dispersion'
            break

    #get xarray metadata
    if logbook == False:
        meta_list, phi_values = get_StA_phoibos_metadata(file,lines,scan_type,logbook)
    
    #get logbook metadata
    else:
        sample_upload, scan_timestamp = get_StA_phoibos_metadata(file,lines,scan_type,logbook)
        return sample_upload, scan_timestamp

    if scan_type == 'dispersion':
        #load data
        data_to_be_loaded = np.loadtxt(file)
        energies = data_to_be_loaded[:, 0]
        counts = data_to_be_loaded[:, 1]
        
        #get non-repeating KE values
        start_KEs = np.where(energies == energies[0])
        number_of_energies = start_KEs[0][1]       
        energy_values = energies[0:number_of_energies]
        
        #get theta_par values
        theta_par_values = []
        for line in lines:
            if line.startswith('# NonEnergyOrdinate: '):
                theta_par_values.append(float(line.replace('# NonEnergyOrdinate: ', "").strip()))
        
        #put measured spectra into a 2D array
        spectrum = [[0 for i in range(len(energy_values))] for j in range(len(theta_par_values))]
        for i in range(len(theta_par_values)):
            for j in range(len(energy_values)):
                spectrum[i][j] = counts[(i*len(energy_values))+j]

        spectrum = da.from_array(spectrum)
        
        #create xarray
        data = xr.DataArray(spectrum, dims=("theta_par","eV"), coords={"theta_par": theta_par_values,"eV": energy_values})
        data.coords['theta_par'].attrs = {'units' : 'deg'}
    
    elif scan_type == 'FS map':
        #load data - takes 1 min 42 seconds to do a map that Igor takes 3 mins 18 seconds
        data_to_be_loaded = np.loadtxt(file)
        energies = data_to_be_loaded[:, 0]
        counts = data_to_be_loaded[:, 1]
        
        #get non-repeating KE values
        start_KEs = np.where(energies == energies[0])
        number_of_energies = start_KEs[0][1]       
        energy_values = energies[0:number_of_energies]
        
        #get theta_par values
        theta_par_values = []
        for line in lines:
            if line.startswith('# NonEnergyOrdinate: '):
                theta_par_values.append(float(line.replace('# NonEnergyOrdinate: ', "").strip()))
            elif line.startswith('# Cycle: 1'):
                break
        
        #get theta_perp values
        theta_perp_values = phi_values

        #put measured spectra into a 3D array
        spectrum = [[[0 for i in range(len(energy_values))] for j in range(len(theta_par_values))] for k in range(len(theta_perp_values))]
        for i in range(len(theta_perp_values)):
            for j in range(len(theta_par_values)):
                for k in range(len(energy_values)):
                    spectrum[i][j][k] = counts[(i*len(theta_par_values)*len(energy_values))+(j*len(energy_values))+k]

        # Convert to a faster dtype for future processing
        spectrum = np.asarray(spectrum, dtype=np.float64)
        spectrum = spectrum.newbyteorder().newbyteorder()

        spectrum = da.from_array(spectrum, chunks='auto')

        #create xarray
        data = xr.DataArray(spectrum, dims=("tilt","theta_par","eV"), coords={"tilt": theta_perp_values, "theta_par": theta_par_values,"eV": energy_values})
        data.coords['tilt'].attrs = {'units' : 'deg'}
        data.coords['theta_par'].attrs = {'units' : 'deg'}

        # Convert to a faster dtype ordering for future processing
        data = data.astype('<f8', casting='equiv')
    
    elif scan_type == 'XPS':
        #load data
        data_to_be_loaded = np.loadtxt(file)
        energies = data_to_be_loaded[:, 0]
        counts = data_to_be_loaded[:, 1]
        
        #get non-repeating KE values
        start_KEs = np.where(energies == energies[0])
        number_of_energies = start_KEs[0][1]
        energy_values = energies[0:number_of_energies]
        
        #combine different spatial counts into a 1D array
        number_of_repeats = int((len(energies)/number_of_energies))
        spectrum = []
        for i in range(number_of_energies):
            total_count = 0.0
            for j in range(number_of_repeats):
                total_count = total_count + float(counts[i+(j*number_of_energies)])
            spectrum.append(total_count)

        spectrum = da.from_array(spectrum)
        
        #create xarray
        data = xr.DataArray(spectrum, dims=("eV"), coords={"eV": energy_values})
            
    #attach metadata
    for i in meta_list:
        data.attrs[i] = meta_list[i]
    data.name = meta_list['scan_name']
    return data


def my_find_StA_phoibos(lines, item):
    '''This function will loop over the lines in the file and then pick out the line
    starting with the desired keyword. Works specifically for Phoibos output files. 
    
    Input:
        lines - A list where each entry is a line from the text file (list)
        item - The word or group of characters you are searching for (string)
        
    Returns:
        The line starting with item (string)
    '''
    
    for line in lines:
        if line.startswith('# ' + item):
            return line.replace('# ' + item + ': ', "").strip()

def get_StA_phoibos_metadata(file,lines,scan_type,logbook):
    '''This function will extract the relevant metadata from the data files obtained using the phoibos analyser
    
    Input:
        file - Path to the file being loaded (string)
        lines - A list where each entry is a line from the text file (list)
        scan_type - The type of scan, e.g. FS map (string)
        logbook - Whether the data should be returned for use by the logbook maker (True) or xarray (False) (Boolean)
        
    Retuns:
        if logbook == True:
            meta_list - List of relevant metadata (dictionary)
            phi_values - List of phi values measured (list)
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
    meta_list['beamline'] = 'St Andrews - Phoibos'
    meta_list['analysis_history'] = []
    meta_list['EF_correction'] = None
    meta_list['PE'] = float(my_find_StA_phoibos(lines, 'Pass Energy'))
    meta_list['hv'] = float(my_find_StA_phoibos(lines, 'Excitation Energy'))
    meta_list['pol'] = None
    meta_list['sweeps'] = int(my_find_StA_phoibos(lines, 'Number of Scans'))
    meta_list['dwell'] = float(my_find_StA_phoibos(lines, 'Dwell Time'))
    meta_list['ana_mode'] = my_find_StA_phoibos(lines, 'Analyzer Lens')
    meta_list['ana_slit'] = my_find_StA_phoibos(lines, 'Analyzer Slit')
    meta_list['ana_slit_angle'] = 0
    meta_list['exit_slit'] = None
    meta_list['x1'] = None
    meta_list['x2'] = None
    meta_list['x3'] = None
    meta_list['polar'] = None
    
    #phi and azi values are only present in Fermi maps
    phi_values = []
    azi_values = []
    if scan_type == 'FS map':
        for line in lines:
            if line.startswith('# Parameter: "Azi [deg]" = '):
                azi_values.append(float(line.replace('# Parameter: "Azi [deg]" = ', "").strip()))
            elif line.startswith('# Parameter: "Phi [deg]" = '):
                phi_values.append(float(line.replace('# Parameter: "Phi [deg]" = ', "").strip()))

        #get the max minimum and maximum phi values
        phi_last = Decimal(phi_values[0])
        phi_first = Decimal(phi_values[len(phi_values)-1])
        phi_step = (phi_last - phi_first) / (len(phi_values)-1)
        #maximum phi_last, phi_first and phi_step decimal place is 3
        if Decimal(phi_last).as_tuple().exponent < -2:
           phi_last = round(Decimal(phi_last), 3)
        if Decimal(phi_first).as_tuple().exponent < -2:
           phi_first = round(Decimal(phi_first), 3)
        if Decimal(phi_step).as_tuple().exponent < -2:
           phi_step = round(Decimal(phi_step), 3)
        meta_list['tilt'] = str(phi_first) + ':' + str(phi_last) + ' (' + str(phi_step) + ')'
        
        #get the max minimum and maximum azi values
        azi_last = Decimal(azi_values[0])
        azi_first = Decimal(azi_values[len(azi_values)-1])
        azi_step = (azi_last - azi_first) / (len(azi_values)-1)
        #maximum azi_last, azi_first and azi_step decimal place is 3
        if Decimal(azi_last).as_tuple().exponent < -2:
           azi_last = round(Decimal(azi_last), 3)
        if Decimal(azi_first).as_tuple().exponent < -2:
           azi_first = round(Decimal(azi_first), 3)
        if Decimal(azi_step).as_tuple().exponent < -2:
           azi_step = round(Decimal(azi_step), 3)
        meta_list['azi'] = str(azi_first) + ':' + str(azi_last) + ' (' + str(azi_step) + ')'
    else:
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
        
        #scan time
        #meed to remove the date to get just the time
        date_and_time_UTC = my_find_StA_phoibos(lines, 'Acquisition Date')
        #check to see if we need to account for British summer time
        date_and_time = date_and_time_UTC.split(' ')[0] + ' ' + date_and_time_UTC.split(' ')[1]
        date_and_time = datetime.strptime(date_and_time, "%m/%d/%y %H:%M:%S")
        time_zone = date_and_time.astimezone().tzinfo 
        if str(time_zone) == 'GMT Daylight Time':
            date_and_time = date_and_time + timedelta(hours=1)
            scan_time = datetime.strftime(date_and_time,'%H:%M:%S')
        else:
            scan_time = date_and_time_UTC.split(' ')[1]
    
        #kinetic energy
        #line of first data entry with minimum KE
        KE_start_index = lines.index("# Cycle: 0, Curve: 0\n") + 5
        #extract minimum KE
        KE_start = lines[KE_start_index].split('  ')[0]
        #line of first data etry with maximum KE
        KE_end_index = lines.index("# Cycle: 0, Curve: 1\n") - 2
        #extract maximum KE
        KE_end = lines[KE_end_index].split('  ')[0]
        #get scan mode
        scan_mode = my_find_StA_phoibos(lines, 'Scan Mode')
        if scan_mode == 'SnapshotFAT':
            #calculate center energy
            analyser_KE = str(round(((Decimal(KE_start) + Decimal(KE_end)) / 2),3))
            analyser_step = 'Fixed'
        else:
            #calculate energy step
            analyser_step = (Decimal(KE_end) - Decimal(KE_start)) / (KE_end_index - KE_start_index)
            analyser_step = str(int(analyser_step*1000))
            #maximum KE_start and KE_end decimal place is 3
            if Decimal(KE_start).as_tuple().exponent < -2:
                KE_start = round(Decimal(KE_start), 3)
            if Decimal(KE_end).as_tuple().exponent < -2:
                KE_end = round(Decimal(KE_end), 3)
            #define KE range
            analyser_KE = str(KE_start) + ':' + str(KE_end)
        
        analyser_bias = my_find_StA_phoibos(lines, 'Bias Voltage')
        
        #excitation source
        if str(meta_list['hv']) == '21.2182':
            excitation_source = 'He I'
        elif str(meta_list['hv']) == '1486.71':
            excitation_source = 'Al Mono'
        else:
            excitation_source = ''
        
        #angles only present for FS map
        if meta_list['tilt'] == None:
            phi = ''
        else:
            phi = meta_list['tilt']
        if meta_list['azi'] == None:
            azi = ''
        else:
            azi = meta_list['azi']
        
        #data to return
        scan_timestamp = str(os.path.getmtime(file))
        sample_upload = [scan_ext,meta_list['scan_name'],'',scan_type,scan_time,analyser_KE,analyser_step,str(meta_list['PE']),str(meta_list['sweeps']),str(meta_list['dwell']),analyser_bias,'','',meta_list['ana_mode'],meta_list['ana_slit'],'',phi,azi,'','','','','',excitation_source,str(meta_list['hv']),'']
        return sample_upload, scan_timestamp
    
    #if the data is being used for an xarray
    else:
        return meta_list, phi_values