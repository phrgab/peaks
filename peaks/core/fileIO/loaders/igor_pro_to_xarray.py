#Functions to load data from Igor Pro files into an xarray
#Brendan Edwards 15/06/2021
#Phil King 5/6/21 - added .ibw loading based on SES format (works for Bloch @ Max-IV, CASSIOPEE @ Soleil, and APE-LE @ Elettra, possibly others but some metadata may need tweaking)
#PK Rewrite to move BL-specific part to BL loaders and make ibw loader more general 24/7/22
#PK Added function to read only ibw wavenote not entire file (useful for file identification etc.) 24/7/22

import os
import numpy as np
import xarray as xr
from .igor import binarywave
import warnings


def read_ibw_Wnote(file):
    '''Read wavenote from Igor binary wave (.ibw) file

    Parameters
    ----------
    fname : str
        File path

    Returns
    ------------
    wNote : str
        String of wavenotes
    '''

    maxdims = 4

    # readIBWbinheader - read BinHeader segment of Igor IBW file IBW version 2,5 only
    with open(file, 'rb') as f:
        version = np.fromfile(f,dtype=np.dtype('int16'),count=1)[0]
        if version == 2:  # typedef struct BinHeader2 {
            wfmSize = np.fromfile(f,dtype=np.dtype('uint32'),count=1)[0]  # long wfmSize; The size of the WaveHeader2 data structure plus the wave data plus 16 bytes of padding.
            noteSize = np.fromfile(f,dtype=np.dtype('uint32'),count=1)[0]  # long noteSize; The size of the note text.
            pictSize = np.fromfile(f,dtype=np.dtype('uint32'),count=1)[0]  # long pictSize; Reserved. Write zero. Ignore on read.
            checksum = np.fromfile(f,dtype=np.dtype('int16'),count=1)[0]  # short checksum; Checksum over this header and the wave header.
        elif version == 5:
            checksum = np.fromfile(f,dtype=np.dtype('short'),count=1)[0]
            wfmSize =  np.fromfile(f,dtype=np.dtype('int32'),count=1)[0]  # The size of the WaveHeader5 data structure plus the wave data.
            formulaSize = np.fromfile(f,dtype=np.dtype('int32'),count=1)[0]  # The size of the dependency formula, if any.
            noteSize = np.fromfile(f,dtype=np.dtype('int32'),count=1)[0]  # The size of the note text.
            dataEUnitsSize = np.fromfile(f,dtype=np.dtype('int32'),count=1)[0]  # The size of optional extended data units.
            dimEUnitsSize =  np.fromfile(f,dtype=np.dtype('int32'),count=maxdims)  # The size of optional extended dimension units.
            dimLabelsSize =  np.fromfile(f,dtype=np.dtype('int32'),count=4)  # The size of optional dimension labels.
            sIndicesSize =  np.fromfile(f,dtype=np.dtype('int32'),count=1)[0]  # The size of string indicies if this is a text wave.
            optionsSize1 =  np.fromfile(f,dtype=np.dtype('int32'),count=1)[0]  # Reserved. Write zero. Ignore on read.
            optionsSize2 =  np.fromfile(f,dtype=np.dtype('int32'),count=1)[0]  # Reserved. Write zero. Ignore on read.

    # Read the wavenote
    with open(file, 'rb') as f:
        # Move the cursor to the end of the file
        f.seek(0, os.SEEK_END)
        # Get the current position of pointer i.e eof
        pointer_location = f.tell()

        if version == 2:
            offset = noteSize

        elif version == 5:
            # Work out file location of wavenote
            offset = noteSize + dataEUnitsSize.sum() + dimEUnitsSize.sum() + dimLabelsSize.sum() + sIndicesSize.sum() + optionsSize1.sum() + optionsSize2.sum()

        # Move the file pointer to the location pointed by pointer_location
        f.seek(pointer_location - offset)
        # Shift pointer location by -1

        # read that byte / character
        WNote = f.read(offset).decode()

    return WNote

def load_ibw(file):
    '''Generic file loader of data in an Igor binary wave

        Parameters
        ------------
        file : str
            Path to file to load

        Returns
        ------------
        data : xr.DataArray
            xarray DataArray or DataSet with loaded data <br>

        meta_lines : list
            Raw metadata from file
    '''

    file_contents = binarywave.load(file)

    spectrum = file_contents['wave']['wData']

    # print(spectrum.dtype)
    # # # Convert to a faster dtype ordering for future processing
    # # data = data.astype('<f8', order='C', casting='equiv')
    #
    # spectrum = spectrum.newbyteorder().newbyteorder()
    # spectrum = spectrum.astype('<f8', order='C', casting='equiv')
    #
    # print(spectrum.dtype)


    # Get relevant number of dimensions
    ndim = spectrum.ndim

    # Read relevant data from header of file
    dim_size = file_contents['wave']['bin_header']['dimEUnitsSize']
    dim_units = file_contents['wave']['dimension_units'].decode()
    # Wave scaling
    dim0 = file_contents['wave']['wave_header']['sfB']  # Initial value
    ddim = file_contents['wave']['wave_header']['sfA']  # step
    N_dim = file_contents['wave']['wave_header']['nDim']  # Number of points

    dims = []
    coords = {}
    ct = 0

    # Determine the relevant dimension waves and names
    for i in range(ndim):
        # Determine dimension units/name as labelled in file
        dims.append(dim_units[ct:ct + dim_size[i]])
        coords[dims[i]] = np.linspace(dim0[i], dim0[i] + ddim[i] * N_dim[i], N_dim[i], endpoint=False)
        ct += dim_size[i]

    # Make an xarray from this
    data = xr.DataArray(spectrum, dims=(tuple(dims)), coords=coords)

    # Get the metadata from the wavenote
    wnote = file_contents['wave']['note'].decode('ascii')

    return data, wnote


def load_igor_pro_data(file, logbook, **kwargs):
    #TODO Do we need this? Can we depreciate and use the general load_ibw one?
    '''This function loads ARPES data from Igor Pro files
    
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

    #get user defined scan type if entered (useful for XPS not performed in the StA ARPES lab at hv = 1486.71 eV)
    try:
        scan_type = kwargs['scan_type']
    #set to dispersion as default if no scan type entered (should be able to work out the scan type later if it is different)
    except:
        scan_type = 'dispersion'

    filename, file_extension = os.path.splitext(file)

    if file_extension == '.txt':  # works for Igor Pro txt files
        #get relevant metadata lines
        with open(file) as f:
            lines = f.readlines()
        wave_info = lines[1]
        start_of_data_index = lines.index('BEGIN\n')
        end_of_data_index = lines.index('END\n')
        #for waves with more than one dimension
        try:
            dimensions = wave_info.split('(')[1].split(')')[0]
            number_of_dimensions = len(dimensions.split(','))
            wave_name = wave_info.split(')	')[1].split()[0]
        #for 1D waves e.g. EDC/MDC
        except:
            dimensions = str(end_of_data_index - start_of_data_index - 1)
            number_of_dimensions = 1
            wave_name = wave_info.split('	')[1].split()[0]
        metadata_to_extract = lines[end_of_data_index+1:]
        metadata = []
        for line in metadata_to_extract:
            if line.startswith('X '):
                if line.startswith('X SetScale/P'):
                    line_split = line.split('X ')[1].split(', ' + str(wave_name) + '; ')
                    metadata.append(line_split[0])
                    metadata.append(line_split[1])
                elif line.startswith('X Note ' + str(wave_name)):
                    line_split = line.split('X Note ' + str(wave_name) + ', "')[1].split('\\r')
                    for line in line_split:
                        metadata.append(line)
                elif line.startswith('X Note/NOCR ' + str(wave_name)):
                    line_split = line.split('X Note/NOCR ' + str(wave_name) + ', "')[1].split('\\r')
                    for line in line_split:
                        metadata.append(line)
            else:
                #more data is saved in the file - therefore the data contains multiple slices
                raise Exception("Scan type is not supported")

        #try to determine scan type
        if my_find_igor_pro(metadata,'PhotonEnergy=') == '1486.71':
            scan_type = 'XPS'
        elif number_of_dimensions == 1:
            if metadata[2] == 'Energy Distribution Curve':
                scan_type = 'EDC'
            elif metadata[2] == 'Momentum Distribution Curve':
                scan_type = 'MDC'

        #extract metadata
        meta_list = get_igor_pro_metadata(file,metadata,scan_type)

        #load x axis data
        x_dimensions = int(dimensions.split(',')[0])
        x_axis = my_find_igor_pro(metadata,'SetScale/P x').split(',')
        x_min = float(x_axis[0])
        x_step = float(x_axis[1])
        x_max = x_min + x_step*(x_dimensions-1)
        x_range = np.linspace(x_min,x_max,x_dimensions)

        #load y axis data
        if number_of_dimensions == 2:
            y_dimensions = int(dimensions.split(',')[1])
            y_axis = my_find_igor_pro(metadata,'SetScale/P y').split(',')
            y_min = float(y_axis[0])
            y_step = float(y_axis[1])
            y_max = y_min + y_step*(y_dimensions-1)
            y_range = np.linspace(y_min,y_max,y_dimensions)

        #determine if the dipsersion is in angle or k space
        k_conversion = False
        try:
            if my_find_igor_pro(metadata,'AngleMapping=').startswith('EXACT'):
                k_conversion = True
        except:
            #make best guess if there is no AngleMapping parameter in the metadata
            if abs(x_min) < 2:
                k_conversion = True

            # Convert loaded data into xarray format
            if scan_type == 'dispersion':
                # load data
                spectrum = np.loadtxt(file, skiprows=start_of_data_index + 1,
                                      max_rows=end_of_data_index - start_of_data_index - 1)

                # create xarray
                if k_conversion == True:
                    data = xr.DataArray(spectrum, dims=("k_par", "eV"), coords={"k_par": x_range, "eV": y_range})
                    data.coords['k_par'].attrs = {'units': 'inv. ang.'}
                elif k_conversion == False:
                    data = xr.DataArray(spectrum, dims=("theta_par", "eV"),
                                        coords={"theta_par": x_range, "eV": y_range})
                    data.coords['theta_par'].attrs = {'units': 'deg'}

            elif scan_type == 'XPS':
                # load data
                data_to_be_loaded = np.loadtxt(file, skiprows=start_of_data_index + 1,
                                               max_rows=end_of_data_index - start_of_data_index - 1)

                # combine different spatial counts into a 1D array
                spectrum = []
                for i in range(y_dimensions):
                    total_count = 0.0
                    for j in range(x_dimensions):
                        total_count = total_count + data_to_be_loaded[j][i]
                    spectrum.append(total_count)

                # create xarray
                data = xr.DataArray(spectrum, dims=("eV"), coords={"eV": y_range})

            elif scan_type == 'EDC':
                # load data
                spectrum = np.loadtxt(file, skiprows=start_of_data_index + 1,
                                      max_rows=end_of_data_index - start_of_data_index - 1)

                # create xarray
                data = xr.DataArray(spectrum, dims=("eV"), coords={"eV": x_range})

            elif scan_type == 'MDC':
                # load data
                spectrum = np.loadtxt(file, skiprows=start_of_data_index + 1,
                                      max_rows=end_of_data_index - start_of_data_index - 1)

                # create xarray
                if k_conversion == True:
                    data = xr.DataArray(spectrum, dims=("k_par"), coords={"k_par": x_range})
                    data.coords['k_par'].attrs = {'units': 'inv. ang.'}
                elif k_conversion == False:
                    data = xr.DataArray(spectrum, dims=("theta_par"), coords={"theta_par": x_range})
                    data.coords['theta_par'].attrs = {'units': 'deg'}

    #loader for SES .ibw output
    elif file_extension == '.ibw':  # works for Igor Binary waves saved by Scienta SES software
        file_contents = binarywave.load(file)

        # Figure out scan type
        dimension_units = file_contents['wave']['dimension_units'].decode('ascii')
        dims_in_scan = list(filter(None, dimension_units.split(']')))
        number_of_dimensions = len(dims_in_scan)

        # Try to determine the scan type (this has been made based on files from Max-IV, so need to check with other places)
        if 'Photon Energy' in dimension_units:  # Should be a hv map
            scan_type = 'hv scan'

        elif 'Point' in dimension_units:  # Should be some form of manipulator map
            if number_of_dimensions >3:  # Almost certainly a spatial map
                scan_type = "spatial map"
            else:  # Need to try and figure out from the data
                axis = manip_from_wavenotes(file_contents['wave']['note'].decode('ascii'))  # Extract the wavenote manipulator scan data

                try:  # Not everything seems to come with a full manipulator list
                    if abs(axis['P'][1] - axis['P'][0]) > 0.02 or abs(axis['T'][1] - axis['T'][0]) > 0.02:  # Looks like a manipulator map
                        scan_type = 'FS map'
                    else:
                        scan_type = 'spatial map'
                except:  # Best guess is a spatial map, as polar maps seems to generally come with the table
                    scan_type = 'line scan'
                    # Give a warning
                    warning_str = 'Manipulator data seems to be missing - assumed a spatial scan, please check this.'
                    warnings.warn(warning_str)

        elif 'Seq. Iteration' in dimension_units or 'Region Iteration' in dimension_units:  # Dispersion measured in 'add dimension' mode (each sweep as a new slice in the file)
            scan_type = 'dispersion - multiple scans'

        else:  # Otherwise, it is already defaulted to dispersion unless specifically called with something else as a kwarg (see above)
            pass

        # Get the metadata from the wavenote
        wnote = file_contents['wave']['note'].decode('ascii')
        
        #get xarray metadata
        if logbook == False:
            meta_list = get_metadata_from_wnote(file, wnote, scan_type, logbook)

        #get logbook metadata
        else:
            sample_upload, scan_timestamp = get_metadata_from_wnote(file, wnote, scan_type, logbook)
            return sample_upload, scan_timestamp


        # Get the data
        spectrum = file_contents['wave']['wData']

        # Get relevant wave scalings
        # Energy scale
        if 'Kinetic' in dims_in_scan[0]:  # First index should be energy, check if wave scaling in KE
            E0 = file_contents['wave']['wave_header']['sfB'][0]  # Initial energy
            dE = file_contents['wave']['wave_header']['sfA'][0]  # Energy step
            N_E = file_contents['wave']['wave_header']['nDim'][0]  # Number of pixels in energy axis
            E_values = np.linspace(E0, E0 + dE*N_E, N_E)
        elif 'Binding' in dims_in_scan[0]:  # First index should be energy, check if wave scaling in KE
            # Try and get this from the wavenotes
            if file_contents['wave']['note'].decode('ascii').split('Energy Unit=')[1].split('\r')[0] == 'Kinetic': # Wave note has this in KE
                E0 = float(file_contents['wave']['note'].decode('ascii').split('Low Energy=')[1].split('\r')[0])  # Initial KE
                E1 = float(file_contents['wave']['note'].decode('ascii').split('High Energy=')[1].split('\r')[0])  # Final KE
                N_E = int(file_contents['wave']['wave_header']['nDim'][0])  # Number of pixels in energy axis
                E_values = np.linspace(E0, E1, N_E, endpoint=True)
            else:  # Leave in binding energy
                E0 = file_contents['wave']['wave_header']['sfB'][0]  # Initial energy
                dE = file_contents['wave']['wave_header']['sfA'][0]  # Energy step
                N_E = file_contents['wave']['wave_header']['nDim'][0]  # Number of pixels in energy axis
                E_values = np.linspace(E0, E0 + dE * N_E, N_E)
                meta_list['eV_type'] = 'binding'
                warning_str = 'Data loaded in binding energy.'
                warnings.warn(warning_str)

        # Angle scale
        th0 = file_contents['wave']['wave_header']['sfB'][1]  # Initial angle
        dth = file_contents['wave']['wave_header']['sfA'][1]  # angle step
        N_th = file_contents['wave']['wave_header']['nDim'][1]  # Number of pixels in angle axis
        theta_par_values = np.linspace(th0, th0 + dth * N_th, N_th, endpoint=False)

        if scan_type == 'dispersion':
            data = xr.DataArray(spectrum, dims=("eV", "theta_par"), coords={"eV": E_values, "theta_par": theta_par_values})  # Make xarray
            data = data.transpose('theta_par', 'eV')  # Reorder into our conventional order
            data.coords['theta_par'].attrs = {'units': 'deg'}  # Set units

        elif scan_type == 'dispersion - multiple scans':
            N_scans = file_contents['wave']['wave_header']['nDim'][2]  # Number of scans
            data = xr.DataArray(spectrum, dims=("eV", "theta_par", "scan_no"), coords={"eV": E_values, "theta_par": theta_par_values, "scan_no": np.arange(N_scans)})  # Make xarray
            data = data.transpose('theta_par', 'eV', 'scan_no')  # Reorder into our conventional order for theta_par and eV, but leave scans as a final index
            data.coords['theta_par'].attrs = {'units': 'deg'}  # Set units

        elif scan_type == 'hv scan':
            hv0 = file_contents['wave']['wave_header']['sfB'][2]  # Initial photon energy
            dhv = file_contents['wave']['wave_header']['sfA'][2]  # hv step
            N_hv = file_contents['wave']['wave_header']['nDim'][2]  # Number of pixels in hv axis
            hv_values = np.linspace(hv0, hv0 + dhv * N_hv, N_hv, endpoint=False)

            # eV coordinate kept as that of first scan, with then the relative shift in KE from scan to scan recorded as a separate (non-dimenison) coordinate
            data = xr.DataArray(spectrum, dims=("eV", "theta_par", "hv"), coords={"eV": E_values, "theta_par": theta_par_values, "hv": hv_values})  # create a DataArray with the measured hv map
            data.coords['theta_par'].attrs = {'units': 'deg'}  # Set units
            data.coords['hv'].attrs = {'units': 'eV'}  # Set units

            data = data.transpose('hv', 'theta_par', 'eV')  # Reorder into our conventional order

            # Add KE_delta coordinate (shift with photon energy) - for an hv scan taken in this way, this should be just the hv shift
            KE_delta = hv_values - hv_values[0]  # Change in KE value of detector as a function of hv
            data.coords['KE_delta'] = ('hv', KE_delta)
            data.coords['KE_delta'].attrs = {'units': 'eV'}  # Set units

            # Give a notification that this is how the data is stored
            warn = "Note: single xarray dataarray returned. The kinetic energy coordinate is that of the first scan; corresponding offsets for successive scans are included in the KE_delta coordinate. Run .pipe(disp_from_hv, hv) where hv is the relevant photon energy to extract a dispersion with the proper KE scaling for that photon energy."
            warnings.warn(warn)

        elif scan_type == 'FS map':
            # Extract relevant manipulator coordinates
            if abs(axis['P'][1] - axis['P'][0]) > 0.02:  # Looks like a polar map
                mapping_dim = 'polar'
                mapping_short = 'P'
            elif abs(axis['T'][1] - axis['T'][0]) > 0.02:  # Looks like a tilt map
                mapping_dim = 'tilt'
                mapping_short = 'T'

            data = xr.DataArray(spectrum, dims=("eV", "theta_par", mapping_dim), coords={"eV": E_values, "theta_par": theta_par_values, mapping_dim: axis[mapping_short]})  # create a DataArray
            data.coords['theta_par'].attrs = {'units': 'deg'}  # Set units
            data.coords[mapping_dim].attrs = {'units': 'deg'}  # Set units

            data = data.transpose(mapping_dim, 'theta_par', 'eV')  # Reorder into our conventional order

        elif scan_type == 'line scan':
            # Try and get manipulator data, sometimes this seems missing
            try:
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

                data = xr.DataArray(spectrum, dims=("eV", "theta_par", mapping_dim), coords={"eV": E_values, "theta_par": theta_par_values, mapping_dim: axis[mapping_short]})  # create a DataArray
                data.coords['theta_par'].attrs = {'units': 'deg'}  # Set units
                data.coords[mapping_dim].attrs = {'units': 'mm'}  # Set units

                data = data.transpose(mapping_dim, 'theta_par', 'eV')  # Reorder into our conventional order

            except:  # If manipulator info missing
                N_pos = file_contents['wave']['wave_header']['nDim'][2]  # Number of pixels in scan axis
                data = xr.DataArray(spectrum, dims=("eV", "theta_par", 'spatial_dim'), coords={"eV": E_values, "theta_par": theta_par_values, 'spatial_dim': np.arange(N_pos)})  # create a DataArray
                data.coords['theta_par'].attrs = {'units': 'deg'}  # Set units

                data = data.transpose('spatial_dim' 'theta_par', 'eV')  # Reorder into our conventional order

        elif scan_type == 'spatial map':
            axis = manip_from_wavenotes(file_contents['wave']['note'].decode('ascii'))  # Extract the wavenote manipulator scan data

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
                if abs( axis[mapping_short[0]][1] - axis[mapping_short[0]][0] ) > 0.01:  # this one is the fast axis
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
                N_fast = file_contents['wave']['wave_header']['nDim'][2]

                mapping_slow0 = axis[mapping_short_slow][0]
                mapping_slow1 = axis[mapping_short_slow][-1]
                N_slow = file_contents['wave']['wave_header']['nDim'][3]

                mapping_dim_fast_values = np.linspace(mapping_fast0, mapping_fast1, N_fast)
                mapping_dim_slow_values = np.linspace(mapping_slow0, mapping_slow1, N_slow)

                data = xr.DataArray(spectrum, dims=("eV", "theta_par", mapping_dim_fast, mapping_dim_slow),
                                    coords={"eV": E_values, "theta_par": theta_par_values,
                                            mapping_dim_fast: mapping_dim_fast_values,
                                            mapping_dim_slow: mapping_dim_slow_values})  # create a DataArray
                data.coords['theta_par'].attrs = {'units': 'deg'}  # Set units
                data.coords[mapping_dim_fast].attrs = {'units': 'mm'}  # Set units
                data.coords[mapping_dim_slow].attrs = {'units': 'mm'}  # Set units

                data = data.transpose(mapping_dim_fast, mapping_dim_slow, 'theta_par', 'eV')  # Reorder into our conventional order

            except:  # If manipulator info missing
                N_pos0 = file_contents['wave']['wave_header']['nDim'][2]  # Number of pixels in scan axis 0
                N_pos1 = file_contents['wave']['wave_header']['nDim'][3]  # Number of pixels in scan axis 1

                data = xr.DataArray(spectrum, dims=("eV", "theta_par", 'spatial_dim0', 'spatial_dim1'),
                                    coords={"eV": E_values, "theta_par": theta_par_values,
                                            'spatial_dim0': np.arange(N_pos0),
                                            'spatial_dim1': np.arange(N_pos1)})  # create a DataArray
                data.coords['theta_par'].attrs = {'units': 'deg'}  # Set units

                data = data.transpose('spatial_dim0', 'spatial_dim1', 'theta_par', 'eV')  # Reorder into our conventional order

        # Check if the data was in transmission mode and update dimension if so
        if 'mm' in dims_in_scan[1]:
            data = data.rename({'theta_par': 'y_scale'})
            data.coords['y_scale'].attrs = {'units': 'mm'}  # Set units

    #attach metadata
    for i in meta_list:
        data.attrs[i] = meta_list[i]
    data.name = meta_list['scan_name']

    return data

def manip_from_wavenotes(wnote):
    #TODO depreciate - make everything use similar function in SES loader
    '''This function will parse the table of manipulator positions generated in a SES manipultor scan.

        Input:
            wnote - Corresponding wavenote where the info is stored (string, i.e. decode from binary to ascii first)

        Returns:
            axis - dictionary of axis positions, with each entry a np array of axis positions thrugh the scan (dict)
        '''

    # Extract the relevant manipulator data
    axis = {}  # Dictionary to hold axis data

    # Extract the header info of the manipulator settings
    wnote_manip = wnote.split('[Run Mode Information]\r')[1].split('\r')
    i = 0
    if 'Name' in wnote_manip[0]:
        i += 1

    for j in wnote_manip[i].split('\x0b'):
        axis[j] = []  # Make the dictionary

    i += 1
    while True:  # Iterate through and fill in the rest of the data
        try:
            axis_val = wnote_manip[i].split('\x0b')
            for j, v in enumerate(axis):
                axis[v].append(float(axis_val[j]))
            i += 1
        except:
            break

    # Covnert from lists to np arrays
    for i in axis:
        axis[i] = np.array(axis[i])

    return axis

def my_find_igor_pro(lines, item):
    '''This function will loop over the lines in the file and then pick out the line
    starting with the desired keyword. Works specifically for igor pro files. 
    
    Input:
        lines - A list where each entry is a line from the text file (list)
        item - The word or group of characters you are searching for (string)
        
    Returns:
        The line starting with item (string)
    '''
    
    for line in lines:
        if line.startswith(item + '       '):
            return line.replace(item + '       ', '').strip()
        elif line.startswith(item + ' '):
            return line.replace(item + ' ', '').strip()
        elif line.startswith(item + ''):
            return line.replace(item, '').strip()
        
def get_igor_pro_metadata(file,lines,scan_type):
    '''This function will extract the relevant metadata from the Igor Pro files
    
    Input:
        file - Path to the file being loaded (string)
        lines - A list where each entry is a line from the text file (list)
        scan_type - The type of scan, e.g. FS map (string)
        
    Retuns:
        meta_list - List of relevant metadata (dictionary)
    '''

    #determine BL
    if my_find_igor_pro(lines,'AnalyzerType=') == 'Phoibos':
        BL = 'St Andrews - Phoibos'
    elif my_find_igor_pro(lines,'Location=') != None:
        if my_find_igor_pro(lines,'Location=') == 'Bloch' or my_find_igor_pro(lines,'Location=') == 'MAXIV' or my_find_igor_pro(lines,'Location=') == 'MaxIV':
            BL = 'MAX IV Bloch'
        elif my_find_igor_pro(lines,'Location=') == 'APE':
            BL = 'Elettra APE'
        elif my_find_igor_pro(lines,'Location=') == 'CASSIOPEE':
            BL = 'SOLEIL CASSIOPEE'
    elif my_find_igor_pro(lines,'Instrument=') != None:
        if my_find_igor_pro(lines,'Instrument=') == 'i05':
            BL = 'Diamond I05-HR'
        elif my_find_igor_pro(lines,'Instrument=') == 'i05-1':
            BL = 'Diamond I05-nano'
    else:
        BL = None

    #assign metadata
    meta_list = {}
    fname = file.split('/')
    fname_noext = fname[len(fname)-1].split('.')
    meta_list['scan_name'] = fname_noext[0]
    meta_list['scan_type'] = scan_type
    meta_list['sample_description'] = None
    meta_list['eV_type'] = None
    meta_list['beamline'] = BL
    meta_list['analysis_history'] = []

    #EF correction
    meta_list['EF_correction'] = my_find_igor_pro(lines,'DispersionCorrection=')
    if meta_list['EF_correction'] == None:
        meta_list['EF_correction'] = 'unknown'
    elif meta_list['EF_correction'] == '':
        meta_list['EF_correction'] = None
        meta_list['eV_type'] = 'kinetic'
    else:
        meta_list['eV_type'] = 'binding'

    #PE
    try:
        meta_list['PE'] = float(my_find_igor_pro(lines,'PassEnergy='))
    except:
        try:
            meta_list['PE'] = float(my_find_igor_pro(lines,'Pass Energy='))
        except:
            meta_list['PE'] = None
    
    #hv
    try:       
        meta_list['hv'] = float(my_find_igor_pro(lines,'PhotonEnergy='))
    except:
        try:
            meta_list['hv'] = float(my_find_igor_pro(lines,'Excitation Energy='))
        except:
            meta_list['hv'] = None
    
    meta_list['pol'] = None
    
    #sweeps
    try:
        meta_list['sweeps'] = int(my_find_igor_pro(lines,'NumberOfSweeps='))
    except:
        try:
            meta_list['sweeps'] = int(my_find_igor_pro(lines,'Number of Sweeps='))
        except:
            meta_list['sweeps'] = None
    
    #dwell
    try:
        meta_list['dwell'] = float(my_find_igor_pro(lines,'DwellTime='))
    except:
        try:
            meta_list['dwell'] = float(my_find_igor_pro(lines,'Dwell Time='))
        except:
            meta_list['dwell'] = None
    
    #ana_mode
    meta_list['ana_mode'] = my_find_igor_pro(lines,'LensMode=')
    if meta_list['ana_mode'] == None:
        meta_list['ana_mode'] = my_find_igor_pro(lines,'Lens Mode=')
    
    #ana_slit
    meta_list['ana_slit'] = my_find_igor_pro(lines,'EntranceSlit=')
    
    #ana_slit_angle
    if BL == 'St Andrews - Phoibos':
        meta_list['ana_slit_angle'] = 0
    elif BL == None:
        meta_list['ana_slit_angle'] = None
    else:
        meta_list['ana_slit_angle'] = 90
    
    meta_list['exit_slit'] = None
    meta_list['defl_par'] = None
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
    
    #temp_sample
    try:
        meta_list['temp_sample'] = float(my_find_igor_pro(lines,'SampleTemperature='))
    except:
        meta_list['temp_sample'] = None
    meta_list['temp_cryo'] = None
    
    return meta_list


def get_metadata_from_wnote(file, wnote, scan_type, logbook):
    '''This function will extract the relevant metadata from the wavenote saved by Scienta SES in an Igor Pro .ibw file

    Input:
        file - Path to the file being loaded (string)
        wnote - The wavenote, in ascii format (string)
        scan_type - The type of scan, e.g. FS map (string)
        logbook - Whether the data should be returned for use by the logbook maker (True) or xarray (False) (Boolean)

    Retuns:
        if logbook == True:
            meta_list - List of relevant metadata (dictionary)
        else:
            sample_upload - List of relevant metadata (list)
            scan_timestamp - Timestamp of the scan (string)
    '''

    def equals_in(wnote_line):  # For cleaning up the meta list
        if '=' in wnote_line:
            return True
        else:
            return False

    wnote_list = wnote.split('\r')  # List of the wavenote
    wnote_list = list(filter(equals_in, wnote_list))  # Filter out all entries that don't have an '=' in
    meta_dict = dict(
        i.split('=', 1) for i in wnote_list)  # Turn this into a dictionary of {field_name: meta_value} pairs

    # determine BL
    try:
        if meta_dict['Location'] == 'Bloch' or meta_dict['Location'] == 'MAXIV' or meta_dict['Location'] == 'MaxIV':
            BL = 'MAX IV Bloch'
        elif meta_dict['Location'] == 'APE':
            BL = 'Elettra APE'
        elif meta_dict['Location'] == 'CASSIOPEE':
            BL = 'SOLEIL CASSIOPEE'
        else:
            BL = None
    except:
        BL = None

    # assign metadata
    meta_list = {}
    fname = file.split('/')
    fname_noext = fname[len(fname) - 1].split('.')
    meta_list['scan_name'] = fname_noext[0]
    meta_list['scan_type'] = scan_type
    meta_list['sample_description'] = None
    meta_list['eV_type'] = 'kinetic'  # Assume kinetic by default; changed to binding later in the .ibw loader script if required
    meta_list['beamline'] = BL
    meta_list['analysis_history'] = []

    # EF correction
    meta_list['EF_correction'] = None

    # PE
    try:
        meta_list['PE'] = float(meta_dict['Pass Energy'])
    except:
        meta_list['PE'] = None

    # hv
    try:
        meta_list['hv'] = float(meta_dict['Excitation Energy'])
    except:
        meta_list['hv'] = None

    meta_list['pol'] = None

    # sweeps
    try:
        meta_list['sweeps'] = int(meta_dict['Number of Sweeps'])
    except:
        meta_list['sweeps'] = None

    # dwell
    try:
        meta_list['dwell'] = float(meta_dict['Step Time'])/1000
    except:
        meta_list['dwell'] = None

    # ana_mode
    try:
        meta_list['ana_mode'] = meta_dict['Lens Mode']
    except:
        meta_list['ana_mode'] = None

    # ana_slit
    try:
        meta_list['ana_slit'] = None
    except:
        meta_list['ana_slit'] = None

    # ana_slit_angle
    if BL == 'St Andrews - Phoibos':
        meta_list['ana_slit_angle'] = 0
    elif BL == None:
        meta_list['ana_slit_angle'] = None
    else:
        meta_list['ana_slit_angle'] = 90

    # Beamline res
    meta_list['exit_slit'] = None

    # Deflectors
    if scan_type == 'FS map':
        try:
            meta_list['defl_par'] = float(meta_dict['ThetaX'])
        except:
            meta_list['defl_par'] = None
        try:
            meta_list['defl_perp'] = float(meta_dict['ThetaY'])
        except:
            meta_list['defl_perp'] = None
    else:
        meta_list['defl_par'] = None
        meta_list['defl_perp'] = None

    # Manipulator positions
    # .ibw manipulator angle reading is currently for BLOCH; APE and CASSIOPEE do not include this in the file, but may need to update in future
    if scan_type == 'FS map' or scan_type == 'spatial map' or scan_type == 'line scan' or scan_type == 'focus scan':  # This is a manipulator-type scan where the meta should be written in a table in the wavenote
        axis = manip_from_wavenotes(wnote)  # Extract the wavenote manipulator scan data
        try:
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
            meta_list['x1'] = None
            meta_list['x2'] = None
            meta_list['x3'] = None
            meta_list['polar'] = None
            meta_list['tilt'] = None
            meta_list['azi'] = None

    else:  # Just look for single manipulator coordinates written directly in wavenotes
        try:
            meta_list['x1'] = float(meta_dict['X'])
            meta_list['x2'] = float(meta_dict['Y'])
            meta_list['x3'] = float(meta_dict['Z'])
            meta_list['polar'] = float(meta_dict['P'])
            meta_list['tilt'] = float(meta_dict['T'])
            meta_list['azi'] = float(meta_dict['A'])
        except:
            meta_list['x1'] = None
            meta_list['x2'] = None
            meta_list['x3'] = None
            meta_list['polar'] = None
            meta_list['tilt'] = None
            meta_list['azi'] = None

    # Set norm_emission placeholders
    meta_list['norm_polar'] = None
    meta_list['norm_tilt'] = None
    meta_list['norm_azi'] = None

    # Sample temperature
    meta_list['temp_sample'] = None
    meta_list['temp_cryo'] = None
    
    #if the data is being used for the logbook maker, we need additional data
    if logbook == True:
        scan_ext = fname_noext[1]
        scan_starttime = (meta_dict['Time'])
    
        #kinetic Energy
        if (meta_dict['Acquisition Mode']) == 'Swept':
            KE_st = str((meta_dict['Low Energy']))
            KE_end = str((meta_dict['High Energy']))
            analyser_KE = str(KE_st)+':'+str(KE_end)
            analyser_step = str(1000*float(str((meta_dict['Energy Step']))))
        else:
            analyser_KE = str((meta_dict['Center Energy']))
            analyser_step = 'Fixed'
        
        #deflector
        if meta_list['defl_perp'] != None:
            analyser_thy = meta_list['defl_perp']
        else:
            analyser_thy = ''
        
        #data to return
        scan_timestamp = str(os.path.getmtime(file))
        if BL == 'MAX IV Bloch':
            sample_upload = [scan_ext,meta_list['scan_name'],'',scan_type,scan_starttime,analyser_KE,analyser_step,str(meta_list['PE']),str(meta_list['sweeps']),str(meta_list['dwell']),str(meta_list['ana_mode']),'','',analyser_thy,str(meta_list['polar']),str(meta_list['tilt']),str(meta_list['azi']),str(meta_list['x1']),str(meta_list['x2']),str(meta_list['x3']),'','',str(meta_list['hv'])]
        elif BL == 'Elettra APE':
            sample_upload = [scan_ext,meta_list['scan_name'],'',scan_type,scan_starttime,analyser_KE,analyser_step,str(meta_list['PE']),str(meta_list['sweeps']),str(meta_list['dwell']),str(meta_list['ana_mode']),'','',analyser_thy,'','','','','','','','',str(meta_list['hv'])]
        elif BL == 'SOLEIL CASSIOPEE':
            sample_upload = [scan_ext,meta_list['scan_name'],'',scan_type,scan_starttime,analyser_KE,analyser_step,str(meta_list['PE']),str(meta_list['sweeps']),str(meta_list['dwell']),str(meta_list['ana_mode']),'','','','','','','','','',str(meta_list['hv']),'','',''] 
        for i in range(len(sample_upload)):
            if sample_upload[i] == 'None':
                sample_upload[i] = ''
        return sample_upload,scan_timestamp
    
    return meta_list