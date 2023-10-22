#Functions to load data from the St Andrews MBS spin-ARPES system into an xarray, or for the logbook maker
#Brendan Edwards 01/03/2021
#Phil King 04/06/2021 - added first version of .krx loader

import os
import numpy as np
from decimal import Decimal
import xarray as xr

def load_StA_MBS_data(file, logbook, **kwargs):
    '''This function loads ARPES data from the St Andrews MBS spin-ARPES system
    
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

    #determine the file format
    filename, file_extension = os.path.splitext(file)
    if file_extension == '.txt':  # .txt loader
        with open(file) as f:
            lines = f.readlines()
        try:
            float(my_find_StA_MBS(lines, 'MapStartX'))
            err_str = 'Loader not configured to load maps from .txt data for MBS. Please load from a .krx file instead.'
        except:
            scan_type = 'dispersion'

        # get xarray metadata
        if logbook == False:
            meta_list = get_StA_MBS_txt_metadata(file, lines, scan_type, logbook)

        # get logbook metadata
        else:
            sample_upload, scan_timestamp = get_StA_MBS_txt_metadata(file, lines, scan_type, logbook)
            return sample_upload, scan_timestamp

        if scan_type == 'dispersion':
            # load data
            lines_to_skip = lines.index('DATA:\t\n') + 1
            data_to_be_loaded = np.loadtxt(file, skiprows=lines_to_skip)
            energy_values = data_to_be_loaded[:, 0]
            data_cube = data_to_be_loaded[:, 1:]
            spectrum = np.transpose(data_cube)
            spectrum = da.from_array(spectrum)

            # get theta_par values
            ScaleMult = float(my_find_StA_MBS(lines, 'ScaleMult'))
            ScaleMin = float(my_find_StA_MBS(lines, 'ScaleMin'))
            ScaleMax = float(my_find_StA_MBS(lines, 'ScaleMax'))
            num_of_theta_par_values = int(np.round(((ScaleMax - ScaleMin) / ScaleMult) + 1, 0))
            theta_par_values = np.linspace(ScaleMin, ScaleMax, num_of_theta_par_values)

            # create xarray
            data = xr.DataArray(spectrum, dims=("theta_par", "eV"), coords={"theta_par": theta_par_values, "eV": energy_values})
            data.coords['theta_par'].attrs = {'units': 'deg'}


    elif file_extension == '.krx':  # PK - first version of a loader for loading single dispersion, simple maps or spin files, 4/6/21
        with open(file, 'rb') as f:  # Open the binary file in read mode
            # Determine whether the file is 32-bit or 64-bit
            bit1_2 = np.fromfile(f, dtype='<i4', count=2)  # Data type is little endian, read as 32 bit, if either of the first 2 32-bit words are 0, the file is 64-bit
            if 0 in bit1_2:  # File is 64 bit
                dtype = '<i8'  # 8-byte signed integer (little endian)
            else:  # File is 32 bit
                dtype = '<i4'  # 4-byte signed integer (little endian)

            # Read pointer array size (first word of array)
            f.seek(0)  # Set back to start of file
            PAS = np.fromfile(f, dtype=dtype, count=1)  # First entry of the file is the pointer array size

            N_im = int(PAS[0] / 3)  # Number of images

            # Read image positions and sizes
            Im_pos = []
            Y_size = []
            X_size = []
            for i in range(N_im):
                Im_pos.append(np.fromfile(f, dtype=dtype, count=1)[0])  # Position in array of the images in 32-bit integers. Position in bytes is x4
                Y_size.append(np.fromfile(f, dtype=dtype, count=1)[0])  # Y size of array (angular direction)
                X_size.append(np.fromfile(f, dtype=dtype, count=1)[0])  # X size of array (energy direction)

            Dim_size = np.fromfile(f, dtype=dtype, count=1)[0]  # 5 for spin, 4 for ARPES
            L = np.fromfile(f, dtype=dtype, count=1)[0]  # 5 for spin, 4 for ARPES

            MSA = np.fromfile(f, dtype=dtype, count=L)

            # Determine scan type
            if L == 5:
                scan_type = 'spin'
            elif N_im == 1:
                scan_type = 'dispersion'
            else:
                scan_type = 'FS map'

            array_size = X_size[0] * Y_size[0]

            # Read the first header and extract relevant scaling information and metadata from there
            f.seek((Im_pos[0] + array_size + 1) * 4)  # Set file position to first header
            header = f.read(1200).decode(
                'ascii')  # Read the header (allowing up to 1200 bytes) and convert into ascii format

            # Shorten header to only required length
            header = header.split('\r\nDATA:')[0]

            # Convert header to metadata dictionary
            meta = dict(i.split('\t', 1) for i in header.split('\r\n'))

        # get xarray metadata in correct form
        if logbook == False:
            meta_list = get_StA_MBS_krx_metadata(file, meta, scan_type, logbook)

        # or get logbook metadata
        else:
            sample_upload, scan_timestamp = get_StA_MBS_krx_metadata(file, meta, scan_type, logbook)
            return sample_upload, scan_timestamp

        # If not in logbook mode, go on to read the data
        with open(file, 'rb') as f:  # Open the binary file in read mode
            # Load the data
            if N_im == 1:  # Load a single slice
                f.seek(Im_pos[0] * 4)  # Set position in file to image location
                data_temp = np.fromfile(f, dtype='<i4', count=array_size)  # Read the image (images written as 32-bit words even in 64-bit format .krx file)
                spectrum = np.reshape(data_temp, [Y_size[0], X_size[0]])  # Put into desired data structure
            else:  # Load into an array
                # Make the master array in order [mapping_dim,theta_par,eV]
                spectrum = np.zeros((N_im, Y_size[0], X_size[
                    0]))  # Assumes all images will be the same shape, and we want to load into a cube

                # Loop through each data set
                for count, i in enumerate(Im_pos):
                    f.seek(i * 4)  # Set position in file to image location
                    data_temp = np.fromfile(f, dtype='<i4', count=array_size)  # Read the image (images written as 32-bit words even in 64-bit format .krx file)
                    spectrum[count, :, :] = np.reshape(data_temp, [Y_size[0], X_size[0]])  # Put into desired data structure

        # Energy scaling
        eV_values = np.linspace(float(meta['Start K.E.']), float(meta['End K.E.']), X_size[0], endpoint=True)

        # Angle scaling
        if L == 4:  # ARPES scan, get analyser MCP angular scale
            theta_par_values = np.linspace(float(meta['ScaleMin']), float(meta['ScaleMax']), Y_size[0], endpoint=True)
        elif L == 5:  # Spin scan, get spin MCP angular scale
            theta_par_values = np.linspace(float(meta['S0ScaleMin']), float(meta['S0ScaleMax']), Y_size[0], endpoint=True)

        # Mapping variable
        if N_im > 1:
            # Deflector angular scaling
            if L == 4:  # ARPES Deflector scan
                mapping_values = np.linspace(float(meta['MapStartX']), float(meta['MapEndX']), N_im, endpoint=True)
                mapping_dim = 'defl_perp'
            elif L == 5:  # Spin scan
                mapping_values = []
                for i in range(N_im):
                    spin_temp = float(meta['SpinComp#' + str(i)].split(',', 1)[1].split('>', 1)[0])
                    mapping_values.append(spin_temp)
                mapping_dim = 'spin_rot_angle'

            # Convert to a faster dtype for future processing
            spectrum = spectrum.newbyteorder().newbyteorder()

            #spectrum = da.from_array(spectrum, chunks='auto')

            # Make the xarray
            data = xr.DataArray(spectrum, dims=(mapping_dim, "theta_par", "eV"), coords={mapping_dim: mapping_values, "theta_par": theta_par_values, "eV": eV_values})
            data.coords[mapping_dim].attrs = {'units': 'deg'}
            data.coords['theta_par'].attrs = {'units': 'deg'}

            # Convert to a faster dtype ordering for future processing
            data = data.astype('<f8', casting='equiv')
        else:
            #spectrum = da.from_array(spectrum, chunks='auto')

            # Make the xarray
            data = xr.DataArray(spectrum, dims=("theta_par", "eV"),
                                coords={"theta_par": theta_par_values, "eV": eV_values})
            data.coords['theta_par'].attrs = {'units': 'deg'}


    #attach metadata
    for i in meta_list:
        data.attrs[i] = meta_list[i]
    data.name = meta_list['scan_name']
    return data

def my_find_StA_MBS(lines, item):
    '''This function will loop over the lines in the file and then pick out the line
    starting with the desired keyword. Works specifically for MBS output files. 
    
    Input:
        lines - A list where each entry is a line from the text file (list)
        item - The word or group of characters you are searching for (string)
        
    Returns:
        The line starting with item (string)
    '''
    
    for line in lines:
        if line.startswith(item):
            return line.replace(item, "").strip()

def get_StA_MBS_txt_metadata(file,lines,scan_type,logbook):
    '''This function will extract the relevant metadata from the .txt data files obtained using the MBS analyser
    
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
    meta_list['beamline'] = 'St Andrews - MBS'
    meta_list['analysis_history'] = []
    meta_list['EF_correction'] = None
    meta_list['PE'] = float(my_find_StA_MBS(lines, 'Pass Energy').replace('PE0', ""))
    meta_list['hv'] = None
    meta_list['pol'] = None
    meta_list['frames'] = int(my_find_StA_MBS(lines, 'Frames Per Step'))
    meta_list['sweeps'] = int(my_find_StA_MBS(lines, 'No Scans'))
    meta_list['steps'] = int(my_find_StA_MBS(lines, 'No. Steps'))
    meta_list['ana_mode'] = my_find_StA_MBS(lines, 'Lens Mode')
    meta_list['ana_slit'] = None
    meta_list['ana_slit_angle'] = 0
    meta_list['exit_slit'] = None
    
    #defl_par
    meta_list['defl_par'] = my_find_StA_MBS(lines, 'DeflY')
    if Decimal(meta_list['defl_par']).as_tuple().exponent < -1:
        meta_list['defl_par'] = float(round(Decimal(meta_list['defl_par']), 2))
        
    #defl_perp
    #Fermi map
    try:
        MapStartX = my_find_StA_MBS(lines, 'MapStartX')
        MapEndX = my_find_StA_MBS(lines, 'MapEndX')
        MapNoXSteps = my_find_StA_MBS(lines, 'MapNoXSteps')
        MapXStepSize = (Decimal(MapEndX)-Decimal(MapStartX))/(Decimal(MapNoXSteps)-1)
        if Decimal(MapStartX).as_tuple().exponent < -1:
            MapStartX = round(Decimal(MapStartX), 2)
        if Decimal(MapEndX).as_tuple().exponent < -1:
            MapEndX = round(Decimal(MapEndX), 2)
        if Decimal(MapXStepSize).as_tuple().exponent < -1:
            MapXStepSize = round(Decimal(MapXStepSize), 2)
        meta_list['defl_perp'] = str(MapStartX) + ':' + str(MapEndX) + ' (' + str(MapXStepSize) + ')'
    #2D dispersion
    except:
        meta_list['defl_perp'] = my_find_StA_MBS(lines, 'DeflX')
        if Decimal(meta_list['defl_perp']).as_tuple().exponent < -1:
            meta_list['defl_perp'] = float(round(Decimal(meta_list['defl_perp']), 2))

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
        
        #region name
        scan_region_name = my_find_StA_MBS(lines, 'RegName')
        date_and_time = my_find_StA_MBS(lines, 'TIMESTAMP:')
        scan_time = date_and_time.split('   ')[1]

        #kinetic energy
        Acq_mode = my_find_StA_MBS(lines, 'AcqMode')
        if Acq_mode == 'Fixed':
            analyser_KE = my_find_StA_MBS(lines, 'Center K.E.')
            if Decimal(analyser_KE).as_tuple().exponent < -1:
                analyser_KE = str(round(Decimal(analyser_KE), 2))
            analyser_step = 'Fixed'
        else:
            KE_start = my_find_StA_MBS(lines, 'Start K.E.')
            KE_end = my_find_StA_MBS(lines, 'End K.E.')
            analyser_step = str(1000*float(my_find_StA_MBS(lines, 'Step Size')))
            #maximum KE_start and KE_end decimal place is 3
            if Decimal(KE_start).as_tuple().exponent < -2:
                KE_start = round(Decimal(KE_start), 3)
            if Decimal(KE_end).as_tuple().exponent < -2:
                KE_end = round(Decimal(KE_end), 3)
            analyser_KE = str(KE_start) + ':' + str(KE_end)

        #data to return
        scan_timestamp = str(os.path.getmtime(file))
        sample_upload = [scan_ext,meta_list['scan_name'],'',scan_region_name,scan_type,scan_time,analyser_KE,analyser_step,str(meta_list['PE']),str(meta_list['frames']),str(meta_list['sweeps']),str(meta_list['steps']),meta_list['ana_mode'],'',str(meta_list['defl_perp']),str(meta_list['defl_par']),'','','','','','','','','','','','', '']
        return sample_upload,scan_timestamp
        
    #if the data is being used for an xarray
    else:
        return meta_list

def get_StA_MBS_krx_metadata(file, meta_dict, scan_type, logbook):
    '''This function will extract the relevant metadata from the .krx data files obtained using the MBS analyser

    Input:
        file - Path to the file being loaded (string)
        meta_dict - A dictionary containing the metadata extracted from the headers of the .krx files (dict)
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
    meta_list['beamline'] = 'St Andrews - MBS'
    meta_list['analysis_history'] = []
    meta_list['EF_correction'] = None
    meta_list['PE'] = float(meta_dict['Pass Energy'].replace('PE0', ""))
    meta_list['hv'] = None
    meta_list['pol'] = None
    meta_list['frames'] = int(meta_dict['Frames Per Step'])
    meta_list['sweeps'] = int(meta_dict['No Scans'])
    meta_list['steps'] = int(meta_dict['No. Steps'])
    meta_list['ana_mode'] = meta_dict['Lens Mode']
    meta_list['ana_slit'] = None
    meta_list['ana_slit_angle'] = 0
    meta_list['exit_slit'] = None

    # defl_par
    meta_list['defl_par'] = meta_dict['DeflY']
    if Decimal(meta_list['defl_par']).as_tuple().exponent < -1:
        meta_list['defl_par'] = float(round(Decimal(meta_list['defl_par']), 2))

    # defl_perp
    # Fermi map
    try:
        MapStartX = meta_dict['MapStartX']
        MapEndX = meta_dict['MapEndX']
        MapNoXSteps = meta_dict['MapNoXSteps']
        MapXStepSize = (Decimal(MapEndX) - Decimal(MapStartX)) / (Decimal(MapNoXSteps) - 1)
        if Decimal(MapStartX).as_tuple().exponent < -1:
            MapStartX = round(Decimal(MapStartX), 2)
        if Decimal(MapEndX).as_tuple().exponent < -1:
            MapEndX = round(Decimal(MapEndX), 2)
        if Decimal(MapXStepSize).as_tuple().exponent < -1:
            MapXStepSize = round(Decimal(MapXStepSize), 2)
        meta_list['defl_perp'] = str(MapStartX) + ':' + str(MapEndX) + ' (' + str(MapXStepSize) + ')'
    # 2D dispersion
    except:
        meta_list['defl_perp'] = meta_dict['DeflX']
        if Decimal(meta_list['defl_perp']).as_tuple().exponent < -1:
            meta_list['defl_perp'] = float(round(Decimal(meta_list['defl_perp']), 2))

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

    # if the data is being used for the logbook maker, we need additional data
    if logbook == True:
        scan_ext = fname_noext[1]

        # region name
        scan_region_name = meta_dict['RegName']
        date_and_time = meta_dict['TIMESTAMP:']
        scan_time = date_and_time.split('   ')[1]

        # kinetic energy
        Acq_mode = meta_dict['AcqMode']
        if Acq_mode == 'Fixed':
            analyser_KE = meta_dict['Center K.E.']
            if Decimal(analyser_KE).as_tuple().exponent < -1:
                analyser_KE = str(round(Decimal(analyser_KE), 2))
            analyser_step = 'Fixed'
        else:
            KE_start = meta_dict['Start K.E.']
            KE_end = meta_dict['End K.E.']
            analyser_step = str(1000 * float(meta_dict['Step Size']))
            # maximum KE_start and KE_end decimal place is 3
            if Decimal(KE_start).as_tuple().exponent < -2:
                KE_start = round(Decimal(KE_start), 3)
            if Decimal(KE_end).as_tuple().exponent < -2:
                KE_end = round(Decimal(KE_end), 3)
            analyser_KE = str(KE_start) + ':' + str(KE_end)

        # data to return
        scan_timestamp = str(os.path.getmtime(file))
        sample_upload = [scan_ext, meta_list['scan_name'], '', scan_region_name, scan_type, scan_time, analyser_KE,
                         analyser_step, str(meta_list['PE']), str(meta_list['frames']), str(meta_list['sweeps']),
                         str(meta_list['steps']), meta_list['ana_mode'], '', str(meta_list['defl_perp']),
                         str(meta_list['defl_par']), '', '', '', '', '', '', '', '', '', '', '', '', '']
        return sample_upload, scan_timestamp

    # if the data is being used for an xarray
    else:
        return meta_list