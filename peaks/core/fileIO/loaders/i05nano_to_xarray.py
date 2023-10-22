#Functions to load data from the I05-nano beamline into an xarray, or for the logbook maker
#Phil King 11/01/2021
#Brendan Edwards 01/03/2021
#PK, added capability for loading SES .zip deflector maps and corrected some axis ordering questions
#PK 21/2/22 added support for analyser_total scans from I05 nano

import os
from datetime import datetime
import numpy as np
import math
import xarray as xr
import dask.array as da
import h5py
import gc

from peaks.core.fileIO.loaders.SES_loader import SES_zip_load, my_find_SES
from peaks.core.fileIO.data_loading import h5py_str
from peaks.core.utils.misc import ana_warn

def load_i05nano_data(file, logbook):
    '''Loads ARPES data from I05-nano beamline

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

    # determine the scan type
    filename, file_extension = os.path.splitext(file)
    if file_extension == '.nxs':
        #open the file (read only)
        f = h5py.File(file, 'r')

        #list the analyser_keys: this is a useful entry in the Diamond .nxs files which shows what is a variable co-ordinate in the scan vs. static metadata
        try:
            analyser_keys = list(f['entry1/analyser/'].keys())
        except:
            analyser_keys = list(f['entry1/analyser_total/'].keys())

        #determine scan type
        if 'energy' in analyser_keys:  # should be a photon energy scan
            scan_type = "hv scan"
        else:  # for all other scan types, the KE measured on the detector shouldn't be varying across the scan files
            if 'analyser_polar_angle' in analyser_keys and 'angles' in analyser_keys:
                scan_type = 'FS map'
            elif 'anapolar' in analyser_keys and 'angles' in analyser_keys:
                scan_type = 'FS map'
            elif 'anapolar' in analyser_keys and 'angles' in analyser_keys:
                scan_type = 'FS map'
            elif 'da30_z' in analyser_keys:
                scan_type = 'focus scan'
            elif 'smdefocus' in analyser_keys and 'smx' in analyser_keys:
                scan_type = '1D focus scan'
            elif 'smdefocus' in analyser_keys and 'smy' in analyser_keys:
                scan_type = '1D focus scan'
            elif 'smx' in analyser_keys and 'smy' in analyser_keys:
                scan_type = 'spatial map'
            elif 'smx' in analyser_keys:
                scan_type = 'line scan'
            elif 'smy' in analyser_keys:
                scan_type = 'line scan'
            elif 'smdefocus' in analyser_keys:
                scan_type = 'line scan'
            else:
                scan_type = 'dispersion'

        #get xarray metadata
        if logbook == False:
            meta_list = get_i05nano_metadata(f,scan_type,logbook)

        #get logbook metadata
        else:
            sample_upload, scan_timestamp = get_i05nano_metadata(file,scan_type,logbook)
            return sample_upload, scan_timestamp

        #read some core data from file
        if 'analyser_total' in analyser_keys:
            KE_values = f['entry1/instrument/analyser_total/kinetic_energy_center'][()]
        else:
            try:
                angles = f['entry1/instrument/analyser/angles'][()]
            except:
                pass
            KE_values = f['entry1/instrument/analyser/energies'][()]

        #Now read in the spectrum

        if 'analyser_total' in analyser_keys:
            data_loc = f['entry1/instrument/analyser_total/analyser_total'][()] # pointer to the actual (cube) of data

            # Data as regular np array
            spectrum = data_loc
            # Add dummy angle and energy axes to make compatible with our analysis scripts
            spectrum = spectrum[..., np.newaxis, np.newaxis]
            angles = [0]
        elif scan_type == 'spatial map':  # For a spatial map, return as a dask array chunked along the spatial directions
            spectrum = da.from_array(f['entry1/instrument/analyser/data'], chunks={0: 'auto', 1: 'auto', 2: -1, 3: -1})
        else:
            spectrum = f['entry1/instrument/analyser/data'] #numpy array

        if 'energy' in analyser_keys:
            # read relevant data from file
            hv = np.round(f['entry1/instrument/monochromator/energy'][()], 2)

            # Need to treat the KE values differently here
            KE0 = KE_values[:, 0]  # Determine max of kinetic energy scan as a function of photon energy
            KE_delta = KE0 - KE0[0]  # Change in KE value of detector as a function of hv

            # create xarray
            data = xr.DataArray(spectrum, dims=("hv", "theta_par", "eV"),
                                coords={"hv": hv, "theta_par": angles, "eV": KE_values[0]})

            data.coords['theta_par'].attrs = {'units': 'deg'}
            data.coords['hv'].attrs = {'units': 'eV'}  # Set units
            # Add KE_delta coordinate
            data.coords['KE_delta'] = ('hv', KE_delta)
            data.coords['KE_delta'].attrs = {'units': 'eV'}  # Set units

            warn_str = "Note: single xarray dataarray returned. The kinetic energy coordinate is that of the first scan; corresponding offsets for successive scans are included in the KE_delta coordinate. Run .pipe(disp_from_hv, hv) where hv is the relevant photon energy to extract a dispersion with the proper KE scaling for that photon energy."
            ana_warn(warn_str)

        elif 'analyser_polar_angle' in analyser_keys and 'angles' in analyser_keys:
            #read relevant data from file
            analyser_polar_angles = f['entry1/instrument/analyser/analyser_polar_angle'][()]
            #create xarray
            data = xr.DataArray(spectrum, dims=("ana_polar","theta_par","eV"), coords={"ana_polar": analyser_polar_angles, "theta_par": angles, "eV": KE_values})
            data.coords['ana_polar'].attrs = {'units' : 'deg'}
            data.coords['ana_polar'].attrs = {'units' : 'deg'}

        elif 'anapolar' in analyser_keys and 'angles' in analyser_keys:
            #read relevant data from file
            analyser_polar_angles = f['entry1/instrument/scan_group/anapolar'][()]
            #create xarray
            data = xr.DataArray(spectrum, dims=("ana_polar","theta_par","eV"), coords={"ana_polar": analyser_polar_angles, "theta_par": angles, "eV": KE_values})
            data.coords['ana_polar'].attrs = {'units' : 'deg'}
            data.coords['ana_polar'].attrs = {'units' : 'deg'}

        elif 'da30_z' in analyser_keys and 'location' in analyser_keys:
            #read relevant data from file
            da30_z = f['entry1/instrument/da30_z/da30_z'][()]
            location = f['entry1/instrument/analyser/location'][()]
            #create xarray
            data = xr.DataArray(spectrum, dims=("focus","location","eV"), coords={"location": location, "focus": da30_z, "eV": KE_values})
            data.coords['location'].attrs = {'units' : 'mm'}

        elif 'da30_z' in analyser_keys and 'analyser_total' in analyser_keys:
            #read relevant data from file
            da30_z = f['entry1/instrument/da30_z/da30_z'][()]
            #create xarray
            data = xr.DataArray(spectrum, dims=("focus", "theta_par", "eV"), coords={"focus": da30_z, "theta_par": angles, "eV": KE_values})

        #not convinced this is correct
        elif 'smdefocus' in analyser_keys and 'smx' in analyser_keys:
            #read relevant data from file
            smdefocus = f['entry1/instrument/manipulator/smdefocus'][()]
            smx = f['entry1/instrument/manipulator/smx'][()]
            if smdefocus[0][1] == smdefocus[0][0]:
                smdefocus_reduced = [i[0] for i in smdefocus]
                #create xarray
                data = xr.DataArray(spectrum, dims=("defocus","x1","theta_par","eV"), coords={"defocus": smdefocus_reduced, "x1": smx[0], "theta_par": angles, "eV": KE_values})
            else:
                smx_reduced = [i[0] for i in smx]
                # create xarray
                data = xr.DataArray(spectrum, dims=("x1", "defocus", "theta_par", "eV"), coords={"defocus": smdefocus[0], "x1": smx_reduced, "theta_par": angles, "eV": KE_values})
            data.coords['x1'].attrs = {'units' : 'um'}
            data.coords['theta_par'].attrs = {'units' : 'deg'}

        elif 'smdefocus' in analyser_keys and 'smy' in analyser_keys:
            #read relevant data from file
            smdefocus = f['entry1/instrument/manipulator/smdefocus'][()]
            smy = f['entry1/instrument/manipulator/smy'][()]
            if smdefocus[0][1] == smdefocus[0][0]:
                smdefocus_reduced = [i[0] for i in smdefocus]
                #create xarray
                data = xr.DataArray(spectrum, dims=("defocus","x2","theta_par","eV"), coords={"defocus": smdefocus_reduced, "x2": smy[0], "theta_par": angles, "eV": KE_values})
            else:
                smy_reduced = [i[0] for i in smy]
                # create xarray
                data = xr.DataArray(spectrum, dims=("x2", "defocus", "theta_par", "eV"), coords={"defocus": smdefocus[0], "x2": smy_reduced, "theta_par": angles, "eV": KE_values})
            data.coords['x2'].attrs = {'units' : 'um'}
            data.coords['theta_par'].attrs = {'units' : 'deg'}

        elif 'smx' in analyser_keys and 'smy' in analyser_keys:
            #read relevant data from file
            smx = f['entry1/instrument/manipulator/smx'][()]
            smy = f['entry1/instrument/manipulator/smy'][()]
            if smx[0][1] != smx[0][0]:
                smy_reduced = [i[0] for i in smy]
                #create xarray
                data = xr.DataArray(spectrum, dims=("x2","x1","theta_par","eV"), coords={"x2": smy_reduced, "x1": smx[0], "theta_par": angles, "eV": KE_values})
            else:
                smx_reduced = [i[0] for i in smx]
                # create xarray
                data = xr.DataArray(spectrum, dims=("x1", "x2", "theta_par", "eV"), coords={"x1": smx_reduced, "x2": smy[0], "theta_par": angles, "eV": KE_values})
            data.coords['x2'].attrs = {'units' : 'um'}
            data.coords['x1'].attrs = {'units' : 'um'}
            data.coords['theta_par'].attrs = {'units' : 'deg'}

        elif 'smx' in analyser_keys:
            #read relevant data from file
            smx = f['entry1/instrument/manipulator/smx'][()]
            #create xarray
            data = xr.DataArray(spectrum, dims=("x1","theta_par","eV"), coords={"x1": smx, "theta_par": angles, "eV": KE_values})
            data.coords['x1'].attrs = {'units' : 'um'}
            data.coords['theta_par'].attrs = {'units' : 'deg'}

        elif 'smy' in analyser_keys:
            #read relevant data from file
            smy = f['entry1/instrument/manipulator/smy'][()]
            #create xarray
            data = xr.DataArray(spectrum, dims=("x2","theta_par","eV"), coords={"x2": smy, "theta_par": angles, "eV": KE_values})
            data.coords['x2'].attrs = {'units' : 'um'}
            data.coords['theta_par'].attrs = {'units' : 'deg'}

        elif 'smdefocus' in analyser_keys:
            #read relevant data from file
            smdefocus = f['entry1/instrument/manipulator/smdefocus'][()]
            #create xarray
            data = xr.DataArray(spectrum, dims=("defocus","theta_par","eV"), coords={"defocus": smdefocus, "theta_par": angles, "eV": KE_values})
            data.coords['theta_par'].attrs = {'units' : 'deg'}

        else:
            #create xarray
            data = xr.DataArray(spectrum[0], dims=("theta_par","eV"), coords={"theta_par": angles, "eV": KE_values})
            data.coords['theta_par'].attrs = {'units' : 'deg'}


        #attach metadata
        for i in meta_list:
            data.attrs[i] = meta_list[i]
        data.name = meta_list['scan_name']

    elif file_extension == '.zip':  # SES deflector map format
        scan_type = 'FS map'

        # get logbook metadata
        if logbook:
            meta_lines = SES_zip_load(file, logbook)
            sample_upload, scan_timestamp = get_I05_SES_metadata(file, meta_lines, scan_type, logbook)
            return sample_upload, scan_timestamp

        # get data and xarray metadata
        else:
            data, meta_lines = SES_zip_load(file, logbook)
            meta_list = get_I05_SES_metadata(file, meta_lines, scan_type, logbook)

        # attach metadata
        for i in meta_list:
            data.attrs[i] = meta_list[i]
        data.name = meta_list['scan_name']


    gc.collect() #Clear the trash again to free up some memory
    return data

def get_i05nano_metadata(f,scan_type,logbook):
    '''This function will extract the relevant metadata from the data tree of a Nexus file otained at the Diamond i05 nano branch.
    
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
    meta_list['scan_name'] = 'i05-1-'+h5py_str(f['entry1/entry_identifier'])
    meta_list['scan_type'] = scan_type
    meta_list['sample_description'] = None
    meta_list['eV_type'] = 'kinetic'
    meta_list['beamline'] = 'Diamond I05-nano'
    meta_list['analysis_history'] = []
    meta_list['EF_correction'] = None
    try:
        meta_list['PE'] = float(f['entry1/instrument/analyser/pass_energy'][0])
    except:
        try:
            meta_list['PE'] = float(f['entry1/instrument/analyser_total/pass_energy'][0])
        except:
            meta_list['PE'] = None

    # photon energy
    try:
        f_hv = f['entry1/instrument/monochromator/energy']
        N_hv = f_hv.len()  # number of recorded photon energy values
        if N_hv == 1:
            meta_list['hv'] = float(np.round(f_hv[0], 1))
        else:
            hv_first = np.round(f_hv[0], 1)
            hv_last = np.round(f_hv[-1], 1)
            hv_step = (hv_last - hv_first) / (N_hv - 1)
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
            try:
                meta_list['sweeps'] = int(f['entry1/instrument/analyser_total/number_of_frames'][0])
            except:
                meta_list['sweeps'] = None

    # Dwell
    try:
        meta_list['dwell'] = float(np.round(f['entry1/instrument/analyser/time_for_frames'][0], 2))
    except:
        try:
            meta_list['dwell'] = float(np.round(f['entry1/instrument/analyser_total/time_for_frames'][0], 2))
        except:
            meta_list['dwell'] = None
    
    #analyser mode
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
        try:
            if 'analyser_total' in list(f['entry1/'].keys()):
                meta_list['ana_mode'] = 'Total'
            else:
                meta_list['ana_mode'] = None
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
        meta_list['ana_slit'] = str(slit_size) + slit_shape + ' (#' + str(slit_no) + ')'
    except:
        try:
            slit_no = h5py_str(f['entry1/instrument/analyser_total/entrance_slit_setting'])
            slit_size = f['entry1/instrument/analyser_total/entrance_slit_size'][0]
            if 'straight' in h5py_str(f['entry1/instrument/analyser_total/entrance_slit_shape']):
                slit_shape = 's'
            elif 'curved' in a.entry1.instrument.analyser.entrance_slit_shape:
                slit_shape = 'c'
            else:
                slit_shape = h5py_str(f['entry1/instrument/analyser_total/entrance_slit_shape'])
            meta_list['ana_slit'] = str(slit_size) + slit_shape + ' (#' + str(slit_no) + ')'
        except:
            meta_list['ana_slit'] = None

    meta_list['ana_slit_angle'] = 90
    
    #exit slits
    try:
        f_ex_sl = f['entry1/instrument/monochromator/exit_slit_size']
        N_slits = f_ex_sl.len()  # number of recorded exit slit values
        if N_slits == 1:
            meta_list['exit_slit'] = float(np.round(f_ex_sl[0], 3) * 1000)
        else:
            meta_list['exit_slit'] = 'var'
    except:
        meta_list['exit_slit'] = None

    # x position
    try:
        Dim_sax = len(f['entry1/instrument/manipulator/smx'].shape)  # dimensions of scan
        if Dim_sax == 1:  # either single value or single array
            N_x = f['entry1/instrument/manipulator/smx'].len()  # number of x values
            if N_x == 1:  # if a single value
                x = float(np.round(f['entry1/instrument/manipulator/smx'][0], 3))
            else:  # if varying
                x_start = np.round(f['entry1/instrument/manipulator/smx'][0], 3)
                x_end = np.round(f['entry1/instrument/manipulator/smx'][-1], 3)
                x_step = np.round((x_end - x_start) / (N_x - 1), 2)
                x = str(x_start) + ':' + str(x_end) + ' (' + str(x_step) + ')'
        elif Dim_sax == 2:  # 2D array
            if f['entry1/instrument/manipulator/smx'][0, 0] == f['entry1/instrument/manipulator/smx'][1, 0]:  # then scan direction is along dim 1
                N_x = f['entry1/instrument/manipulator/smx'].shape[1]
                x_start = np.round(f['entry1/instrument/manipulator/smx'][0, 0], 3)
                x_end = np.round(f['entry1/instrument/manipulator/smx'][0, -1], 3)
            else:  # scan direction along dim 0
                N_x = f['entry1/instrument/manipulator/smx'].shape[0]
                x_start = np.round(f['entry1/instrument/manipulator/smx'][0, 0], 3)
                x_end = np.round(f['entry1/instrument/manipulator/smx'][-1, 0], 3)
            x_step = np.round((x_end - x_start) / (N_x - 1), 2)
            x = str(x_start) + ':' + str(x_end) + ' (' + str(x_step) + ')'
        else:
            x = 'var'
    except:
        x = None

    # y position
    try:
        Dim_say = len(f['entry1/instrument/manipulator/smy'].shape)  # dimensions of scan
        if Dim_say == 1:  # either single value or single array
            N_y = f['entry1/instrument/manipulator/smy'].len()  # number of y values
            if N_y == 1:  # if a single value
                y = float(np.round(f['entry1/instrument/manipulator/smy'][0], 3))
            else:  # if varying
                y_start = np.round(f['entry1/instrument/manipulator/smy'][0], 3)
                y_end = np.round(f['entry1/instrument/manipulator/smy'][-1], 3)
                y_step = np.round((y_end - y_start) / (N_y - 1), 2)
                y = str(y_start) + ':' + str(y_end) + ' (' + str(y_step) + ')'
        elif Dim_say == 2:  # 2D array
            if f['entry1/instrument/manipulator/smy'][0, 0] == f['entry1/instrument/manipulator/smy'][1, 0]:  # then scan direction is along dim 1
                N_y = f['entry1/instrument/manipulator/smy'].shape[1]
                y_start = np.round(f['entry1/instrument/manipulator/smy'][0, 0], 3)
                y_end = np.round(f['entry1/instrument/manipulator/smy'][0, -1], 3)
            else:  # scan direction along dim 0
                N_y = f['entry1/instrument/manipulator/smy'].shape[0]
                y_start = np.round(f['entry1/instrument/manipulator/smy'][0, 0], 3)
                y_end = np.round(f['entry1/instrument/manipulator/smy'][-1, 0], 3)
            y_step = np.round((y_end - y_start) / (N_y - 1), 2)
            y = str(y_start) + ':' + str(y_end) + ' (' + str(y_step) + ')'
        else:
            y = 'var'
    except:
        y = None

    # z position
    try:
        # z-axis
        Dim_saz = len(f['entry1/instrument/manipulator/smz'].shape)  # dimensions of scan
        if Dim_saz == 1:  # either single value or single array
            N_z = f['entry1/instrument/manipulator/smz'].len()  # number of z values
            if N_z == 1:  # if a single value
                z = float(np.round(f['entry1/instrument/manipulator/smz'][0], 3))
            else:  # if varying
                z_start = np.round(f['entry1/instrument/manipulator/smz'][0], 3)
                z_end = np.round(f['entry1/instrument/manipulator/smz'][-1], 3)
                z_step = np.round((z_end - z_start) / (N_z - 1), 2)
                z = str(z_start) + ':' + str(z_end) + ' (' + str(z_step) + ')'
        elif Dim_saz == 2:  # 2D array
            if f['entry1/instrument/manipulator/saz'][0, 0] == f['entry1/instrument/manipulator/smz'][1, 0]:  # then scan direction is along dim 1
                N_z = f['entry1/instrument/manipulator/smz'].shape[1]
                z_start = np.round(f['entry1/instrument/manipulator/smz'][0, 0], 3)
                z_end = np.round(f['entry1/instrument/manipulator/smz'][0, N_z - 1], 3)
            else:  # scan direction along dim 0
                N_z = f['entry1/instrument/manipulator/smz'].shape[0]
                z_start = np.round(f['entry1/instrument/manipulator/smz'][0, 0], 3)
                z_end = np.round(f['entry1/instrument/manipulator/smz'][-1, 0], 3)
            z_step = np.round((z_end - z_start) / (N_z - 1), 2)
            z = str(z_start) + ':' + str(z_end) + ' (' + str(z_step) + ')'
        else:
            z = 'var'
    except:
        z = None
    
    #set to pyPhoto conventions
    meta_list['x1'] = x
    meta_list['x2'] = y
    meta_list['x3'] = z

    #de-focus
    try:
        f_defocus = f['entry1/instrument/manipulator/smdefocus']
        if f_defocus.len() == 1:  #If a single value
            meta_list['defocus'] = float(np.round(f_defocus[0],3))
        else:  #If varying
            meta_list['defocus'] = 'var'
    except:
        meta_list['defocus'] = None

    #Analyser polar angle
    try:
        try:
            f_anapolar = f['entry1/instrument/scan_group/anapolar']
        except:
            f_anapolar = f['entry1/instrument/analyser/analyser_polar_angle']
        N_anapolar = f_anapolar.len()  # Number of analyser polar values
        if N_anapolar == 1:
            meta_list['ana_polar'] = np.round(f_anapolar[0],2)
        else:
            anapolar_first = np.round(f_anapolar[0], 1)
            anapolar_last = np.round(f_anapolar[-1], 1)
            anapolar_step = (anapolar_last - anapolar_first) / (N_anapolar - 1)
            meta_list['ana_polar'] = str(anapolar_first) + ':' + str(anapolar_last) + ' (' + str(anapolar_step) + ')'
    except:
        meta_list['ana_polar'] = None

    #Polar
    try:
        f_polar = f['entry1/instrument/manipulator/smpolar']
        N_polar = f_polar.len()  # number of polar values
        if N_polar == 1:  # if a single value
            meta_list['polar'] = float(np.round(f['entry1/instrument/manipulator/smpolar'][0], 1))
        else:  # if a scan
            polar_first = np.round(f_polar[0], 1)
            polar_last = np.round(f_polar[-1], 1)
            polar_step = np.round((polar_last - polar_first) / (N_polar - 1), 3)
            meta_list['polar'] = str(polar_first) + ':' + str(polar_last) + ' (' + str(polar_step) + ')'
    except:
        meta_list['polar'] = None

    #Azimuth
    # azi
    try:
        meta_list['azi'] = float(np.round(f['entry1/instrument/manipulator/smazimuth'][0], 1))
    except:
        meta_list['azi'] = None

    #Tilt - no tilt, 5-axis manipulator
    meta_list['tilt'] = 0
    
    #meta_list['norm_ana_polar'] = None
    meta_list['norm_polar'] = None
    meta_list['norm_azi'] = None
    meta_list['norm_tilt'] = None


    #OSA
    try:
        f_OSAx = f['entry1/instrument/order_sorting_aperture/osax']
        f_OSAy = f['entry1/instrument/order_sorting_aperture/osay']
        f_OSAz = f['entry1/instrument/order_sorting_aperture/osaz']

        if f_OSAx.len() == 1:
            meta_list['OSAx'] = float(np.round(f_OSAx[0],2))
        else:
            meta_list['OSAx'] = str('var')

        if f_OSAy.len() == 1:
            meta_list['OSAy'] = float(np.round(f_OSAy[0], 2))
        else:
            meta_list['OSAy'] = str('var')

        if f_OSAz.len() == 1:
            meta_list['OSAz'] = float(np.round(f_OSAz[0],2))
        else:
            meta_list['OSAz'] = str('var')

    except:
        meta_list['OSAx'] = None
        meta_list['OSAy'] = None
        meta_list['OSAz'] = None
    
    # Zone Plate
    try:
        f_zpx = f['entry1/instrument/zone_plate/zpx']
        f_zpy = f['entry1/instrument/zone_plate/zpy']
        f_zpz = f['entry1/instrument/zone_plate/zpz']

        if f_zpx.len() == 1:
            meta_list['ZPx'] = float(np.round(f_zpx[0],2))
        else:
            meta_list['ZPx'] = str('var')

        if f_zpy.len() == 1:
            meta_list['ZPy'] = float(np.round(f_zpy[0], 2))
        else:
            meta_list['ZPy'] = str('var')

        if f_zpz.len() == 1:
            meta_list['ZPz'] = float(np.round(f_zpz[0], 2))
        else:
            meta_list['ZPz'] = str('var')
    except:
        meta_list['ZPx'] = None
        meta_list['ZPy'] = None
        meta_list['ZPz'] = None

    # temperatures
    try:
        f_Temp = f['entry1/sample/temperature']
        f_TempC = f['entry1/sample/cryostat_temperature']
        if f_Temp.len() == 1:  # if a single value
            meta_list['temp_sample'] = float(np.round(f_Temp[0], 1))
            meta_list['temp_cryo'] = float(np.round(f_TempC[0], 1))
        else:  # if varying
            T_sample_first = np.round(f_Temp[0], 1)
            T_sample_last = np.round(f_Temp[-1], 1)
            T_cryo_first = np.round(f_TempC[0], 1)
            T_cryo_last = np.round(f_TempC[-1], 1)
            meta_list['temp_sample'] = str(T_sample_first) + ' : ' + str(T_sample_last)
            meta_list['temp_cryo'] = str(T_cryo_first) + ' : ' + str(T_cryo_last)
    except:
        meta_list['temp_sample'] = None
        meta_list['temp_cryo'] = None
    
    #if the data is being used for the logbook maker, we need additional data
    if logbook == True:
        # start time
        try:
            t1 = h5py_str(f['entry1/start_time'])
            # two different time formats, perhaps a summer time issue, or an updated format
            try:
                starttime = datetime.strptime(t1, '%Y-%m-%dT%H:%M:%S.%fZ')
                scan_starttime = starttime.strftime("%H:%M:%S")
            except:
                try:
                    starttime = datetime.strptime(t1, '%Y-%m-%dT%H:%M:%S.%f+01:00')
                    scan_starttime = starttime.strftime("%H:%M:%S")
                except:
                    starttime = datetime.strptime(t1, '%Y-%m-%dT%H:%M:%S.%f+01')
                    scan_starttime = starttime.strftime("%H:%M:%S")
        except:
            scan_starttime = ''

        # duration
        try:
            t2 = h5py_str(f['entry1/end_time'])
            try:
                endtime = datetime.strptime(t2, '%Y-%m-%dT%H:%M:%S.%fZ')
            except:
                try:
                    endtime = datetime.strptime(t2, '%Y-%m-%dT%H:%M:%S.%f+01:00')
                except:
                    endtime = datetime.strptime(t2, '%Y-%m-%dT%H:%M:%S.%f+01')
            tdelta = endtime - starttime
            if tdelta.total_seconds() > 3600:
                td_h = math.floor(tdelta.total_seconds() / 3600)
                td_m = math.floor((tdelta.total_seconds() - (3600 * td_h)) / 60)
                scan_duration = str(td_h) + 'h:' + str(td_m) + 'm'
            elif tdelta.total_seconds() > 60:
                td_m = math.floor(tdelta.total_seconds() / 60)
                td_s = math.floor((tdelta.total_seconds() - (60 * td_m)))
                scan_duration = str(td_m) + 'm:' + str(td_s) + 's'
            else:
                td_s = math.floor(tdelta.total_seconds())
                scan_duration = str(td_s) + 's'
        except:
            scan_duration = ''
        
        #kinetic Energy
        try:
            try:
                if 'Swept' in h5py_str(f['entry1/instrument/analyser/acquisition_mode']):
                    KE_st = f['entry1/instrument/analyser/kinetic_energy_start'][0]
                    if 'kinetic_energy_end' in f['entry1/instrument/analyser'].keys():
                        KE_end = f['entry1/instrument/analyser/kinetic_energy_end'][0]
                    else:
                        KE_end = KE_st + (f['entry1/instrument/analyser/kinetic_energy_step'][0] * (f['entry1/instrument/analyser/binding_energies'].len()-1))
                    analyser_KE = str(KE_st)+':'+str(KE_end)
                    step_size = np.round((KE_end-KE_st)*1000/(f['entry1/instrument/analyser/binding_energies'].len()),1)
                    analyser_step = str(step_size[0])

                else:
                    analyser_KE = str(f['entry1/instrument/analyser/kinetic_energy_center'][0])
                    analyser_step = 'Fixed'

            except:
                analyser_KE = h5py_str(f['entry1/instrument/analyser_total/kinetic_energy_center'])
                analyser_step = 'Fixed'
        except:
            analyser_KE = ''
            analyser_step = ''

        #data to return
        scan_timestamp = str(os.path.getmtime(file))
        sample_upload = [meta_list['scan_name'],'',scan_type,scan_starttime,scan_duration,analyser_KE,analyser_step,str(meta_list['PE']),str(meta_list['sweeps']),str(meta_list['dwell']),str(meta_list['ana_mode']),str(meta_list['ana_slit']),str(meta_list['ana_polar']),str(meta_list['polar']),str(meta_list['azi']),str(x),str(y),str(z),str(meta_list['defocus']),str(meta_list['temp_sample']),str(meta_list['temp_cryo']),str(meta_list['hv']),str(meta_list['exit_slit']),str(meta_list['pol']),str(meta_list['ZPx']),str(meta_list['ZPy']),str(meta_list['ZPz']),str(meta_list['OSAx']),str(meta_list['OSAy']),str(meta_list['OSAz'])]
        return sample_upload,scan_timestamp
        
    #if the data is being used for an xarray
    else:
        # Add scan command to the attributes
        meta_list['scan_command'] = h5py_str(f['entry1/scan_command/'])

        # Return the relevant data needed for the xarray attributes
        return meta_list





def get_I05_SES_metadata(file, lines, scan_type, logbook):
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

    # split file full path to seperate file name
    fname = file.split('/')
    fname_noext = fname[len(fname) - 1].split('.')

    # assign metadata
    meta_list = {}
    meta_list['scan_name'] = fname_noext[0]
    meta_list['scan_type'] = scan_type
    meta_list['sample_description'] = None
    meta_list['eV_type'] = 'kinetic'
    meta_list['beamline'] = 'Diamond I05-nano'
    meta_list['analysis_history'] = []
    meta_list['EF_correction'] = None
    meta_list['PE'] = float(my_find_SES(lines, 'Pass Energy'))
    hv = my_find_SES(lines, 'Excitation Energy')
    try:
        meta_list['hv'] = np.round(float(hv), 2)
    except:
        hv = hv.replace(',', '.')
        meta_list['hv'] = np.round(float(hv), 2)
    meta_list['pol'] = None
    meta_list['sweeps'] = int(my_find_SES(lines, 'Number of Sweeps'))
    meta_list['dwell'] = float(my_find_SES(lines, 'Step Time')) / 1000.0
    meta_list['ana_mode'] = my_find_SES(lines, 'Lens Mode')
    meta_list['ana_slit'] = None
    meta_list['ana_slit_angle'] = 90
    meta_list['exit_slit'] = None
    meta_list['defl_par'] = None
    if scan_type == 'FS map':
        defl_Y_high = my_find_SES(lines, 'Thetay_High')
        defl_Y_low = my_find_SES(lines, 'Thetay_Low')
        defl_delta = my_find_SES(lines, 'Thetay_StepSize')
        meta_list['defl_perp'] = defl_Y_low + ':' + defl_Y_high + ' (' + defl_delta + ')'
    else:
        meta_list['defl_perp'] = None
    meta_list['x1'] = None
    meta_list['x2'] = None
    meta_list['x3'] = None
    meta_list['polar'] = None
    meta_list['tilt'] = None
    meta_list['azi'] = None
    try:
        meta_list['ana_polar'] = my_find_SES(lines, 'Comments=Anapolar')
    except:
        meta_list['ana_polar'] = None
    meta_list['norm_polar'] = None
    meta_list['norm_tilt'] = None
    meta_list['norm_azi'] = None
    meta_list['temp_sample'] = None
    meta_list['temp_cryo'] = None

    # if the data is being used for the logbook maker, we need additional data
    if logbook == True:
        scan_ext = fname_noext[1]
        scan_starttime = my_find_SES(lines, 'Time')

        # kinetic Energy
        if my_find_SES(lines, 'Acquisition Mode') == 'Swept':
            KE_st = str(my_find_SES(lines, 'Low Energy'))
            KE_end = str(my_find_SES(lines, 'High Energy'))
            analyser_KE = str(KE_st) + ':' + str(KE_end)
            analyser_step = str(1000 * float(my_find_SES(lines, 'Energy Step')))
        else:
            analyser_KE = str(my_find_SES(lines, 'Center Energy'))
            analyser_step = 'Fixed'

        # deflector
        if meta_list['defl_perp'] != None:
            analyser_thy = meta_list['defl_perp']
        else:
            analyser_thy = ''

        # data to return
        scan_timestamp = str(os.path.getmtime(file))
        sample_upload = [scan_ext, meta_list['scan_name'], '', scan_type, scan_starttime, analyser_KE, analyser_step,
                         str(meta_list['PE']), str(meta_list['sweeps']), str(meta_list['dwell']),
                         str(meta_list['ana_mode']), '', '', analyser_thy, str(meta_list['polar']),
                         str(meta_list['tilt']), str(meta_list['azi']), str(meta_list['x1']), str(meta_list['x2']),
                         str(meta_list['x3']), '', '', str(meta_list['hv'])]
        return sample_upload, scan_timestamp

    # if the data is being used for an xarray
    else:
        return meta_list
