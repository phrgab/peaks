#Functions to load data from the I05-HR beamline into an xarray, or for the logbook maker
#Phil King 11/01/2021
#Brendan Edwards 22/09/2021

import os
from datetime import datetime
import numpy as np
import math
import xarray as xr
import h5py
import dask.array as da

from peaks.core.fileIO.data_loading import h5py_str
from peaks.core.utils.misc import ana_warn

def load_i05HR_data(file, logbook, **kwargs):
    '''Loads ARPES data from I05-HR beamline

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

    #open the file (read only)
    f = h5py.File(file, 'r')
    
    #list the analyser_keys: this is a useful entry in the Diamond .nxs files which shows what is a variable co-ordinate in the scan vs. static metadata
    analyser_keys = list(f['entry1/analyser/'].keys())

    #determine scan type
    if 'energy' in analyser_keys: #should be a photon energy scan
        scan_type = "hv scan"
    else: #for all other scan types, the KE measured on the detector shouldn't be varying across the scan files
        if 'sapolar' in analyser_keys: #should be a FS manipulator polar map
            scan_type = "FS map"
        elif 'deflector_x' in analyser_keys:  #should be a FS deflector map
            scan_type = "defl map"
        elif 'salong' in analyser_keys: #should be a focus scan
            scan_type = "focus scan"
        else:
            if 'saz' in analyser_keys and 'sax' in analyser_keys: #should be a spatial map
                # #check which is the heirarchy of the spatial scan co-ordinates (fast and slow axis), extract relevant 1D waves for x and z, and build DataArray
                # if f['entry1/instrument/manipulator/sax'][[1][0]] > f['entry1/instrument/manipulator/sax'][[0][0]]:
                #     scan_type = "spatial map"
                # else:
                scan_type = "spatial map"
            elif 'saz' in analyser_keys: #should be a 1D spatial scan
                scan_type = "line scan"
            elif 'sax' in analyser_keys: #should be a 1D spatial scan
                scan_type = "line scan"
            elif 'say' in analyser_keys: #should be a form of focus scan
                scan_type = "focus scan"
            elif 'temperature' in analyser_keys: #should be an automated temp-dependence
                scan_type = "temp dep"
            else: #should be a simple dispersion
                scan_type = "dispersion"

    
    #get xarray metadata
    if logbook == False:
        meta_list = get_i05HR_metadata(f,scan_type,logbook)
    
    #get logbook metadata
    else:
        sample_upload, scan_timestamp = get_i05HR_metadata(f,scan_type,logbook)
        return sample_upload, scan_timestamp

    #read some core data from file
    # For a spatial map, return as a dask array chunked along the spatial directions
    if scan_type == 'spatial map':
        spectrum = da.from_array(f['entry1/instrument/analyser/data'],chunks={0: 'auto', 1: 'auto', 2: -1, 3: -1})  # the actual (cube) of measured spectra, as a dask array
    else:
        spectrum = f['entry1/instrument/analyser/data'] #the actual (cube) of measured spectra as np array
    theta = f['entry1/instrument/analyser/angles'][()] #the analyser angle scale readout

    #determine type of scan from heirarchy of driven/secondary axes and load data accordingly
    if 'energy' in analyser_keys: #should be a photon energy scan
        hv = np.round(f['entry1/instrument/monochromator/energy'][()],2) #Extract the photon energy, with rounding
        
        #Extract kinetic energy of detector
        KE_values = f['entry1/instrument/analyser/energies'][()] #Array of KE values
        KE0 = KE_values[:,0] #Determine max of kinetic energy scan as a function of photon energy
        KE_delta = KE0 - KE0[0] #Change in KE value of detector as a function of hv
        
        #create a DataArray with the measured hv map
            #eV coordinate kept as that of first scan, with then the relative shift in KE from scan to scan recorded as a separate (non-dimenison) coordinate
        data = xr.DataArray(spectrum, dims=("hv","theta_par","eV"), coords={"hv": hv, "theta_par": theta, "eV": KE_values[0]})
        data.coords['theta_par'].attrs = {'units' : 'deg'}
        data.coords['hv'].attrs = {'units': 'eV'}  # Set units
        #Add KE_delta coordinate
        data.coords['KE_delta'] = ('hv', KE_delta)
        data.coords['KE_delta'].attrs = {'units': 'eV'}  # Set units
        
        #Give a notification that this is how the data is stored
        warning_str = "Note: single xarray dataarray returned. The kinetic energy coordinate is that of the first scan; corresponding offsets for successive scans are included in the KE_delta coordinate. Run .disp_from_hv(hv) where hv is the relevant photon energy to extract a dispersion with the proper KE scaling for that photon energy."
        ana_warn(warning_str)
        
    else: #for all other scan types, the KE measured on the detector shouldn't be varying across the scan files
        # Get KE values
        KE_values_pointer = f['entry1/instrument/analyser/energies']
        # Ensure KE_values are a 1D array of energies
        if len(KE_values_pointer.shape) == 1:
            KE_values = KE_values_pointer[()]
        if len(KE_values_pointer.shape) == 2:
            KE_values = KE_values_pointer[0]
        elif len(KE_values_pointer.shape) == 3:
            KE_values = KE_values_pointer[0][0]

        if 'sapolar' in analyser_keys: #should be a FS manipulator polar map
            sapolar = f['entry1/instrument/manipulator/sapolar'][()] #extract the polar mapping dimension
            #create a DataArray with the measured map 
            data = xr.DataArray(spectrum, dims=("polar","theta_par","eV"), coords={"polar": sapolar, "theta_par": theta, "eV": KE_values})
            data.coords['polar'].attrs = {'units' : 'deg'}
            data.coords['theta_par'].attrs = {'units' : 'deg'}
        elif 'deflector_x' in analyser_keys:  #should be a FS deflector map
            deflector_x = f['entry1/instrument/deflector_x/deflector_x'][()] #extract the deflector mapping dimension
            #create a DataArray with the measured map 
            data = xr.DataArray(spectrum, dims=("defl_perp","theta_par","eV"), coords={"defl_perp": deflector_x, "theta_par": theta, "eV": KE_values})
            data.coords['defl_perp'].attrs = {'units' : 'deg'}
            data.coords['theta_par'].attrs = {'units' : 'deg'}
        elif 'salong' in analyser_keys: #should be a focus scan
            salong = f['entry1/instrument/manipulator/salong'][()] #extract the focus co-ordinate (along the beam direction)
            #create a DataArray with the measured map 
            data = xr.DataArray(spectrum, dims=("focus","theta_par","eV"), coords={"focus": salong, "theta_par": theta, "eV": KE_values})
            data.coords['focus'].attrs = {'units' : 'mm'}
            data.coords['theta_par'].attrs = {'units' : 'deg'}
    
        else:
            if 'saz' in analyser_keys and 'sax' in analyser_keys: #should be a spatial map
                #check which is the heirarchy of the spatial scan co-ordinates (fast and slow axis), extract relevant 1D waves for x and z, and build DataArray
                if f['entry1/instrument/manipulator/sax'][1][0] > f['entry1/instrument/manipulator/sax'][0][0]:
                    sax = f['entry1/instrument/manipulator/sax'][:,0]
                    saz = f['entry1/instrument/manipulator/saz'][0,:]
                    #create a DataArray with the measured map, NB, using x/y as per peaks geometry conventions
                    data = xr.DataArray(spectrum, dims=("x1","x2","theta_par","eV"), coords={"x1": sax, "x2": saz, "theta_par": theta, "eV": KE_values})
                    data.coords['x1'].attrs = {'units' : 'mm'}
                    data.coords['x2'].attrs = {'units' : 'mm'}
                    data.coords['theta_par'].attrs = {'units' : 'deg'}
                    
                else:
                    sax = f['entry1/instrument/manipulator/sax'][0,:]
                    saz = f['entry1/instrument/manipulator/saz'][:,0]
                    #create a DataArray with the measured map, NB, using x/y as per peaks geometry conventions
                    data = xr.DataArray(spectrum, dims=("x2","x1","theta_par","eV"), coords={"x2": saz, "x1": sax, "theta_par": theta, "eV": KE_values})
                    data.coords['x1'].attrs = {'units' : 'mm'}
                    data.coords['x2'].attrs = {'units' : 'mm'}
                    data.coords['theta_par'].attrs = {'units' : 'deg'}
            
            elif 'saz' in analyser_keys: #should be a 1D spatial scan
                saz = f['entry1/instrument/manipulator/saz'][()] #extract the saz co-ordinate
                data = xr.DataArray(spectrum, dims=("x2","theta_par","eV"), coords={"x2": saz, "theta_par": theta, "eV": KE_values}) #corresponding DataArray
                data.coords['x2'].attrs = {'units' : 'mm'}
                data.coords['theta_par'].attrs = {'units' : 'deg'}
            
            elif 'sax' in analyser_keys: #should be a 1D spatial scan
                sax = f['entry1/instrument/manipulator/sax'][()] #extract the sax co-ordinate
                data = xr.DataArray(spectrum, dims=("x1","theta_par","eV"), coords={"x1": sax, "theta_par": theta, "eV": KE_values}) #corresponding DataArray
                data.coords['x1'].attrs = {'units' : 'mm'}
                data.coords['theta_par'].attrs = {'units' : 'deg'}
            
            elif 'say' in analyser_keys: #should be a form of focus scan
                say = f['entry1/instrument/manipulator/say'][()] #extract the say co-ordinate
                data = xr.DataArray(spectrum, dims=("x3","theta_par","eV"), coords={"x3": say, "theta_par": theta, "eV": KE_values}) #corresponding DataArray
                data.coords['x3'].attrs = {'units' : 'mm'}
                data.coords['theta_par'].attrs = {'units' : 'deg'}
            
            elif 'temperature' in analyser_keys: #should be an automated temp-dependence
                temperature = f['entry1/instrument/sample/temperature'][()] #extract sample temperature
                data = xr.DataArray(spectrum, dims=("temp_sample","theta_par","eV"), coords={"temp_sample": temperature, "theta_par": theta, "eV": KE_values}) #corresponding DataArray
                data.coords['temp_sample'].attrs = {'units' : 'K'}
                data.coords['theta_par'].attrs = {'units' : 'deg'}
            
            else: #should be a simple dispersion
                #check data format looks right for a single dispersion
                if len(spectrum.shape) == 3 and spectrum.shape[0] == 1:
                    #data array
                    data = xr.DataArray(spectrum[0], dims=("theta_par","eV"), coords={"theta_par": theta, "eV": KE_values})
                    data.coords['theta_par'].attrs = {'units' : 'deg'}
                
                else:
                    raise Exception("Scan type does not seem to be supported.")
    
    #attach metadata
    for i in meta_list:
        data.attrs[i] = meta_list[i]
    data.name = meta_list['scan_name']
    return data

def get_i05HR_metadata(f,scan_type,logbook):
    '''This function will extract the relevant metadata from the data tree of a Nexus file otained at the Diamond i05 HR branch.
    
    Input:
        f - Opened file handle to file in h5py (h5py._hl.files.File)
        scan_type - The type of scan, e.g. FS map (string)
        logbook - Whether the data should be returned for use by the logbook maker (True) or xarray (False) (Boolean)
        
    Retuns:
        if logbook == True:
            meta_list - List of relevant metadata (dictionary)
        else:
            sample_upload - List of relevant metadata (list)
            scan_timestamp - Timestamp of the scan (string)
    '''
    
    #assign metadata
    meta_list = {}
    meta_list['scan_name'] = 'i05-'+h5py_str(f['entry1/entry_identifier'])
    meta_list['scan_type'] = scan_type
    meta_list['sample_description'] = None
    meta_list['eV_type'] = 'kinetic'
    meta_list['beamline'] = 'Diamond I05-HR'
    meta_list['analysis_history'] = []
    meta_list['EF_correction'] = None

    # PE
    try:
        meta_list['PE'] = float(f['entry1/instrument/analyser/pass_energy'][0])
    except:
        meta_list['PE'] = None
    
    #photon energy
    try:
        f_hv = f['entry1/instrument/monochromator/energy']
        N_hv = f_hv.len()  #number of recorded photon energy values
        if N_hv == 1:
            meta_list['hv'] = float(np.round(f_hv[0],1))
        else:
            hv_first = np.round(f_hv[0],1)
            hv_last = np.round(f_hv[-1],1)
            hv_step = (hv_last - hv_first)/(N_hv-1)
            meta_list['hv'] = str(hv_first) + ':' + str(hv_last) + ' (' + str(hv_step) + ')'
    except:
        meta_list['hv'] = None

    # polarisation
    try:
        f_pol = f['entry1/instrument/insertion_device/beam/final_polarisation_label']
        if f_pol.len() == 1:  # Only single value recorded
            meta_list['pol'] = h5py_str(f_pol)
        else:
            meta_list['pol'] = 'var'
    except:
        # odd way they sometimes save polarisation
        try:
            f_pol = f['entry1/instrument/insertion_device/beam/final_polarisation_label\n\t\t\t\t\t\t\t\t']
            if f_pol.len() == 1:  # Only single value recorded
                meta_list['pol'] = h5py_str(f_pol)
            else:
                meta_list['pol'] = 'var'

        except:
            meta_list['pol'] = None

    # number of sweeps
    try:
        meta_list['sweeps'] = int(f['entry1/instrument/analyser/number_of_iterations'][0])
    except:
        try:
            meta_list['sweeps'] = int(f['entry1/instrument/analyser/number_of_cycles'][0])
        except:
            meta_list['sweeps'] = None

    # Dwell
    try:
        meta_list['dwell'] = float(np.round(f['entry1/instrument/analyser/time_for_frames'][0],2))
    except:
        meta_list['dwell'] = None
    
    # analyser mode
    try:
        ana_mode = h5py_str(f['entry1/instrument/analyser/lens_mode'])
        if 'Angular30' in ana_mode:
            meta_list['ana_mode'] = 'Ang30'
        elif 'Angular14' in ana_mode:
            meta_list['ana_mode'] = 'Ang14'
        elif 'Transmission' in ana_mode:
            meta_list['ana_mode'] = 'Trans'
        else:
            meta_list['ana_mode'] = ana_mode
    except:
        meta_list['ana_mode'] = None

    #analyser entrance slit
    try:
        slit_no = h5py_str(f['entry1/instrument/analyser/entrance_slit_setting'])
        slit_size = f['entry1/instrument/analyser/entrance_slit_size'][0]
        if 'straight' in h5py_str(f['entry1/instrument/analyser/entrance_slit_shape']):
            slit_shape = 's'
        elif 'curved' in h5py_str(f['entry1/instrument/analyser/entrance_slit_shape']):
            slit_shape = 'c'
        else:
            slit_shape = h5py_str(f['entry1/instrument/analyser/entrance_slit_shape'])
        meta_list['ana_slit'] = str(slit_size)+slit_shape+' (#'+str(slit_no)+')'
    except:
        meta_list['ana_slit'] = None

    meta_list['ana_slit_angle'] = 90
    
    #exit slits
    try:
        f_ex_sl = f['entry1/instrument/monochromator/exit_slit_size']
        N_slits = f_ex_sl.len()  #number of recorded exit slit values
        if N_slits == 1:
            meta_list['exit_slit'] = float(np.round(f_ex_sl[0],3)*1000)
        else:
            meta_list['exit_slit'] = 'var'
    except:
        meta_list['exit_slit'] = None
    
    #deflector
    meta_list['defl_par'] = None
    try:
        if scan_type == 'defl map':
            deflector_x = f['entry1/instrument/deflector_x/deflector_x'][()] #extract the deflector mapping dimension
            defl_high = max(deflector_x)
            defl_low = min(deflector_x)
            defl_delta = (defl_high-defl_low)/(len(deflector_x)-1)
            meta_list['defl_perp'] =  str(defl_low) + ':' + str(defl_high) + ' (' + str(defl_delta) + ')'
        else:
            try:
                meta_list['defl_perp'] = float(f['entry1/instrument/deflector_x/deflector_x'][0])
            except:
                meta_list['defl_perp'] = None
    except:
        meta_list['defl_perp'] = None
    
    #x position
    try:
        Dim_sax = len(f['entry1/instrument/manipulator/sax'].shape) #dimensions of scan
        if Dim_sax == 1:  #either single value or single array
            N_x = f['entry1/instrument/manipulator/sax'].len()  #number of x values
            if N_x == 1:  #if a single value 
                x = float(np.round(f['entry1/instrument/manipulator/sax'][0],3))
            else:  #if varying
                x_start = np.round(f['entry1/instrument/manipulator/sax'][0],3)
                x_end = np.round(f['entry1/instrument/manipulator/sax'][-1],3)
                x_step = np.round((x_end - x_start)/(N_x-1),2)
                x = str(x_start) + ':' + str(x_end) + ' (' + str(x_step) + ')'
        elif Dim_sax == 2:  #2D array
            if f['entry1/instrument/manipulator/sax'][0,0] == f['entry1/instrument/manipulator/sax'][1,0]:  #then scan direction is along dim 1
                N_x = f['entry1/instrument/manipulator/sax'].shape[1]
                x_start = np.round(f['entry1/instrument/manipulator/sax'][0,0],3)
                x_end = np.round(f['entry1/instrument/manipulator/sax'][0,-1],3)
            else:  #scan direction along dim 0    
                N_x = f['entry1/instrument/manipulator/sax'].shape[0]
                x_start = np.round(f['entry1/instrument/manipulator/sax'][0,0],3)
                x_end = np.round(f['entry1/instrument/manipulator/sax'][-1,0],3)
            x_step = np.round((x_end - x_start)/(N_x-1),2)
            x = str(x_start) + ':' + str(x_end) + ' (' + str(x_step) + ')'
        else:
            x = 'var'
    except:
        x = None
    

    #y position
    try:
        Dim_say = len(f['entry1/instrument/manipulator/say'].shape) #dimensions of scan
        if Dim_say == 1:  #either single value or single array
            N_y = f['entry1/instrument/manipulator/say'].len()  #number of y values
            if N_y == 1:  #if a single value
                y = float(np.round(f['entry1/instrument/manipulator/say'][0],3))
            else:  #if varying
                y_start = np.round(f['entry1/instrument/manipulator/say'][0],3)
                y_end = np.round(f['entry1/instrument/manipulator/say'][-1],3)
                y_step = np.round((y_end - y_start)/(N_y-1),2)
                y = str(y_start) + ':' + str(y_end) + ' (' + str(y_step) + ')'
        elif Dim_say == 2:  #2D array
            if f['entry1/instrument/manipulator/say'][0,0] == f['entry1/instrument/manipulator/say'][1,0]:  #then scan direction is along dim 1
                N_y = f['entry1/instrument/manipulator/say'].shape[1]
                y_start = np.round(f['entry1/instrument/manipulator/say'][0,0],3)
                y_end = np.round(f['entry1/instrument/manipulator/say'][0,-1],3)
            else:  #scan direction along dim 0
                N_y = f['entry1/instrument/manipulator/say'].shape[0]
                y_start = np.round(f['entry1/instrument/manipulator/say'][0,0],3)
                y_end = np.round(f['entry1/instrument/manipulator/say'][-1,0],3)
            y_step = np.round((y_end - y_start)/(N_y-1),2)
            y = str(y_start) + ':' + str(y_end) + ' (' + str(y_step) + ')'
        else:
            y = 'var'
    except:
        y = None

    #z position
    try:
        #z-axis
        Dim_saz = len(f['entry1/instrument/manipulator/saz'].shape) #dimensions of scan
        if Dim_saz == 1:  #either single value or single array
            N_z = f['entry1/instrument/manipulator/saz'].len()  #number of z values
            if N_z == 1:  #if a single value 
                z = float(np.round(f['entry1/instrument/manipulator/saz'][0],3))
            else:  #if varying
                z_start = np.round(f['entry1/instrument/manipulator/saz'][0],3)
                z_end = np.round(f['entry1/instrument/manipulator/saz'][-1],3)
                z_step = np.round((z_end - z_start)/(N_z-1),2)
                z = str(z_start) + ':' + str(z_end) + ' (' + str(z_step) + ')'
        elif Dim_saz == 2:  #2D array
            if f['entry1/instrument/manipulator/saz'][0,0] == f['entry1/instrument/manipulator/saz'][1,0]:  #then scan direction is along dim 1
                N_z = f['entry1/instrument/manipulator/saz'].shape[1]
                z_start = np.round(f['entry1/instrument/manipulator/saz'][0,0],3)
                z_end = np.round(f['entry1/instrument/manipulator/saz'][0,N_z-1],3)
            else:  #scan direction along dim 0    
                N_z = f['entry1/instrument/manipulator/saz'].shape[0]
                z_start = np.round(f['entry1/instrument/manipulator/saz'][0,0],3)
                z_end = np.round(f['entry1/instrument/manipulator/saz'][-1,0],3)
            z_step = np.round((z_end - z_start)/(N_z-1),2)
            z = str(z_start) + ':' + str(z_end) + ' (' + str(z_step) + ')'
        else:
            z = 'var'  
    except:
        z = None
    
    #set to pyPhoto conventions
    meta_list['x1'] = x
    meta_list['x2'] = z
    meta_list['x3'] = y

    #polar
    try:
        f_polar = f['entry1/instrument/manipulator/sapolar']
        N_polar = f_polar.len()  #number of polar values
        if N_polar == 1:  #if a single value
            meta_list['polar'] = float(np.round(f['entry1/instrument/manipulator/sapolar'][0],1))
        else:  #if a scan
            polar_first = np.round(f_polar[0],1)
            polar_last = np.round(f_polar[-1],1)
            polar_step = np.round((polar_last - polar_first)/(N_polar-1),3)
            meta_list['polar'] = str(polar_first) + ':' + str(polar_last) + ' (' + str(polar_step) + ')'
    except:
        meta_list['polar'] = None

    #tilt
    try:
        meta_list['tilt'] = float(np.round(f['entry1/instrument/manipulator/satilt'][0],1))
    except:
        try:
            meta_list['tilt'] = str(np.round(f['entry1/instrument/manipulator/satilt'][0],1))
        except:
            meta_list['tilt'] = None

    #azi
    try:
        meta_list['azi'] = float(np.round(f['entry1/instrument/manipulator/saazimuth'][0],1))
    except:
        try:
            meta_list['azi'] = str(np.round(f['entry1/instrument/manipulator/saazimuth'][0],1))
        except:
            meta_list['azi'] = None

    meta_list['norm_polar'] = None
    meta_list['norm_tilt'] = None
    meta_list['norm_azi'] = None

    # temperatures
    try:
        f_Temp = f['entry1/sample/temperature']
        f_TempC = f['entry1/sample/cryostat_temperature']
        if f_Temp.len() == 1:  #if a single value
            meta_list['temp_sample'] = float(np.round(f_Temp[0],1))
            meta_list['temp_cryo'] = float(np.round(f_TempC[0],1))
        else:  #if varying
            T_sample_first = np.round(f_Temp[0],1)
            T_sample_last = np.round(f_Temp[-1],1)
            T_cryo_first = np.round(f_TempC[0],1)
            T_cryo_last = np.round(f_TempC[-1],1)
            meta_list['temp_sample'] = str(T_sample_first)+' : '+str(T_sample_last)
            meta_list['temp_cryo'] = str(T_cryo_first)+' : '+str(T_cryo_last)
    except:
        meta_list['temp_sample'] = None
        meta_list['temp_cryo'] = None

    # if the data is being used for the logbook maker, we need additional data
    if logbook == True:
        # start time
        try:
            t1 = h5py_str(f['entry1/start_time'])
            #two different time formats, perhaps a summer time issue, or an updated format
            try:
                starttime = datetime.strptime(t1,'%Y-%m-%dT%H:%M:%S.%fZ')
                scan_starttime = starttime.strftime("%H:%M:%S")
            except:
                try:
                    starttime = datetime.strptime(t1,'%Y-%m-%dT%H:%M:%S.%f+01:00')
                    scan_starttime = starttime.strftime("%H:%M:%S")
                except:
                    starttime = datetime.strptime(t1,'%Y-%m-%dT%H:%M:%S.%f+01')
                    scan_starttime = starttime.strftime("%H:%M:%S")
        except:
            scan_starttime = ''
    
        # duration
        try:
            t2 = h5py_str(f['entry1/end_time'])
            try:
                endtime = datetime.strptime(t2,'%Y-%m-%dT%H:%M:%S.%fZ')
            except:
                try:
                    endtime = datetime.strptime(t2,'%Y-%m-%dT%H:%M:%S.%f+01:00')
                except:
                    endtime = datetime.strptime(t2,'%Y-%m-%dT%H:%M:%S.%f+01')
            tdelta = endtime-starttime
            if tdelta.total_seconds() > 3600:
                td_h = math.floor(tdelta.total_seconds()/3600)
                td_m = math.floor((tdelta.total_seconds()-(3600*td_h))/60)
                scan_duration = str(td_h)+'h:'+str(td_m)+'m'
            elif tdelta.total_seconds() > 60:
                td_m = math.floor(tdelta.total_seconds()/60)
                td_s = math.floor((tdelta.total_seconds()-(60*td_m)))
                scan_duration = str(td_m)+'m:'+str(td_s)+'s'
            else:
                td_s = math.floor(tdelta.total_seconds())
                scan_duration = str(td_s)+'s'
        except:
            scan_duration = ''
        
        #kinetic Energy
        try:
            if 'Swept' in h5py_str(f['entry1/instrument/analyser/acquisition_mode']):
                KE_st = f['entry1/instrument/analyser/kinetic_energy_start'][0]
                try:
                    KE_end = f['entry1/instrument/analyser/kinetic_energy_end'][0]
                except:
                    KE_centre = f['entry1/instrument/analyser/kinetic_energy_center'][0]
                    KE_end = float(KE_centre) + (float(KE_centre)- float(KE_st))
                analyser_KE = str(KE_st)+':'+str(KE_end)
                try:
                    analyser_step = str(float(f['entry1/instrument/analyser/kinetic_energy_step'][0])*1000)
                except:
                    step_size = np.round((KE_end-KE_st)*1000/(f['entry1/instrument/analyser/binding_energies'].len()),1)
                    analyser_step = str(step_size[0])
            else:
                analyser_KE = str(f['entry1/instrument/analyser/kinetic_energy_center'][0])
                analyser_step = 'Fixed'
        except:
            analyser_KE = ''
            analyser_step = ''
        
        #Deflector
        if meta_list['defl_perp'] == None:
            defl_X = ''
        else:
            defl_X = meta_list['defl_perp']
        
        #data to return
        scan_timestamp = str(os.path.getmtime(file))
        sample_upload = [meta_list['scan_name'] ,'',scan_type,scan_starttime,scan_duration,analyser_KE,analyser_step,str(meta_list['PE']),str(meta_list['sweeps']),str(meta_list['dwell']),str(meta_list['ana_mode']),str(meta_list['ana_slit']),defl_X,str(meta_list['polar']),str(meta_list['tilt']),str(meta_list['azi']),str(x),str(y),str(z),str(meta_list['temp_sample']),str(meta_list['temp_cryo']),str(meta_list['hv']),str(meta_list['exit_slit']),str(meta_list['pol'])]
        return sample_upload,scan_timestamp
        
    #if the data is being used for an xarray
    else:
        # Add scan command to the attributes
        meta_list['scan_command'] = h5py_str(f['entry1/scan_command/'])

        # Return the relevant data needed for the xarray attributes
        return meta_list
