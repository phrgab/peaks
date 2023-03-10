#Generic functions to load data from the Scienta SES software
#Phil King 21/07/22

import numpy as np
import xarray as xr
import dask.array as da
import zipfile

from peaks.core.fileIO.loaders.igor_pro_to_xarray import load_ibw, read_ibw_Wnote
from peaks.utils.misc import ana_warn

def SES_load(file, logbook, ext):
    '''Load data from Scienta SES file formats<br>

    Parameters
    ------------
    file : str
        File path

    logbook : bool
        Whether the data should be returned for use by the logbook maker (True) or data load (False)

    ext : str
        File extension, to enable determining which loader should be called

    Returns
    ------------
    if logbook == False:
        data : chunked(xr.DataArray)
            xarray DataArray or DataSet with loaded data <br>
            Returned with arrays as dask objects or list of these

        meta_lines : list
            Raw metadata from file

    else:
        meta_lines : list
            Raw metadata from file
    '''

    if ext == '.txt':
        return SES_txt_load(file, logbook)
    elif ext == '.zip':
        return SES_zip_load(file, logbook)
    elif ext == '.ibw':
        return SES_ibw_load(file, logbook)
    else:
        raise Exception("Scan type is not supported")


def SES_txt_load(file, logbook):
    '''Load data from SES .txt file format <br>
    NB currently only supports single region scans

    Parameters
    ------------
    file : str
        file path

    logbook : bool
        Whether the data should be returned for use by the logbook maker (True) or data load (False)

    Returns
    ------------
    if logbook == False:
        data : chunked(xr.DataArray)
            xarray DataArray or DataSet with loaded data <br>
            Returned with arrays as dask objects or list of these

        meta_lines : list
            Raw metadata from file

    else:
        meta_lines : list
            Raw metadata from file
        '''

    # Get metadata
    with open(file) as f:
        # Read the metadata
        meta_lines = []
        line = ''
        while('[Data' not in line):
            line = f.readline()
            meta_lines.append(line)

        # If for logbooking, this is all that is required
        if logbook:
            return meta_lines

   # load data
    data_to_be_loaded = np.loadtxt(file, skiprows=len(meta_lines))
    data_cube = data_to_be_loaded[:, 1:]
    spectrum = da.from_array(data_cube)
    eK_axis = data_to_be_loaded[:, 0]
    analyzer_axis = np.fromstring(my_find_SES(meta_lines,'Dimension 2 scale='), dtype=np.float64, sep=" ")  # convert to array

    # create xarray
    data = xr.DataArray(spectrum, dims=("eV", "theta_par"), coords={"theta_par": analyzer_axis, "eV": eK_axis})
    data.coords['theta_par'].attrs = {'units': 'deg'}

    return data, meta_lines


def SES_ibw_load(file, logbook):
    '''Load data from SES .ibw file format <br>

    Parameters
    ------------
    file : str
        file path

    logbook : bool
        Whether the data should be returned for use by the logbook maker (True) or data load (False)

    Returns
    ------------
    if logbook == False:
        data : chunked(xr.DataArray)
            xarray DataArray or DataSet with loaded data <br>
            Returned with arrays as dask objects or list of these

        meta_lines : list
            Raw metadata from file

    else:
        meta_lines : list
            Raw metadata from file
        '''

    # If for logbooking, only read wavenotes from the file and return them with mild formatting
    if logbook:
        return read_ibw_Wnote(file).split('\r')

    else:
        # Read data and wavenotes using our generic ibw file loader
        data, meta_lines = load_ibw(file)

        # Format meta_lines to be like standard SES format
        meta_lines = meta_lines.split('\r')

        # Try to parse the wave axis names into peaks format
        for i in data.dims:
            if 'kinetic' in i.lower():  # KE scale
                data = data.rename({i: 'eV'})
            elif 'binding' in i.lower():  # Data loaded in BE, try to parse to KE instead
                if my_find_SES(meta_lines, 'Energy Unit') == 'Kinetic':  # Wave note has this in KE; extract from there
                    E0 = my_find_SES(meta_lines, 'Low Energy')
                    data[i] = data[i] + float(E0) - data[i][0]
                    data = data.rename({i: 'eV'})
                else:
                    warning_str = 'Data loaded in binding energy.'
                    ana_warn(warning_str, 'warning')
                    data.attrs['eV_type'] = 'binding'
            elif 'photon' in i.lower() or 'hv' in i.lower():  # Should be an hv scan
                # Add KE_delta coordinate (shift with photon energy) - for an hv scan taken in this way, this should be just the hv shift
                KE_delta = data[i] - data[i][0]  # Change in KE value of detector as a function of hv
                data.coords['KE_delta'] = (i, KE_delta)
                data.coords['KE_delta'].attrs = {'units': 'eV'}  # Set units
                # Give a notification that this is how the data is stored
                warn = "Note: single xarray dataarray returned. The kinetic energy coordinate is that of the first scan; corresponding offsets for successive scans are included in the KE_delta coordinate. Run .pipe(disp_from_hv, hv) where hv is the relevant photon energy to extract a dispersion with the proper KE scaling for that photon energy."
                ana_warn(warn)
                data = data.rename({i: 'hv'})
            elif 'deg' in i.lower():
                data = data.rename({i: 'theta_par'})
                data.coords['theta_par'].attrs = {'units': 'deg'}  # Set units
            elif 'scale' in i.lower():
                data = data.rename({i: 'y_scale'})
                data.coords['y_scale'].attrs = {'units': 'mm'}  # Set units
            elif 'iteration' in i.lower():
                data = data.rename({i: 'scan_no'})

            # The rest needs beamline-specific data to parse
            # Rename the final ones as generic dimensions
            elif 'point' in i.lower():
                data = data.rename({i: 'dim0'})

            elif 'position' in i.lower():
                data = data.rename({i: 'dim1'})

    return data, meta_lines


def SES_zip_load(file, logbook):
    '''Load data from SES .zip file format, used for deflector maps <br>
    Adapted from the PESTO file loader of Craig Polley (BLOCH) with some speed ups
    and updated for returning dask and xarray data

    Parameters
    ------------
    file : str
        file path

    logbook : bool
        Whether the data should be returned for use by the logbook maker (True) or data load (False)

    Returns
    ------------
    if logbook == False:
        data : chunked(xr.DataArray)
            xarray DataArray or DataSet with loaded data <br>
            Returned with arrays as dask objects or list of these

        meta_lines : list
            Raw metadata from file

    else:
        meta_lines : list
            Raw metadata from file
    '''

    # Find metadata file
    with zipfile.ZipFile(file) as z:
        files = z.namelist()
        file_ini = [file for file in files if 'Spectrum_' not in file and '.ini' in file]
        filename = file_ini[0]
        with z.open(filename) as f:
            lines_to_decode = f.readlines()
            meta_lines = [line.decode() for line in lines_to_decode]

    # If for logbooking, this is all that is required
    if logbook:
        return meta_lines

    # load data
    with zipfile.ZipFile(file) as z:
        files = z.namelist()
        file_bin = [file for file in files if '.bin' in file]
        file_ini = [file for file in files if 'Spectrum_' in file and '.ini' in file]
        filename = file_bin[0]
        filename_ini = file_ini[0]

        with z.open(filename_ini) as f:
            lines = f.readlines()
            lineText = [line.decode() for line in lines]

            # energy axis
            numEnergyPoints = int(my_find_SES(lineText, "width="))
            eK_axis = np.zeros([numEnergyPoints])
            eK_axis[0] = float(my_find_SES(lineText, "widthoffset="))
            energyStepSize = float(my_find_SES(lineText, 'widthdelta='))
            eK_axis = [eK_axis[0] + i * energyStepSize for i in range(numEnergyPoints)]

            # analyzer slit axis
            numAnalyzerPoints = int(my_find_SES(lineText, 'height='))
            analyzer_axis = np.zeros([numAnalyzerPoints])
            analyzer_axis[0] = float(my_find_SES(lineText, "heightoffset="))
            analyzerStepSize = float(my_find_SES(lineText, "heightdelta="))
            analyzer_axis = [analyzer_axis[0] + i * analyzerStepSize for i in range(numAnalyzerPoints)]

            # deflector slit axis
            numDeflectorPoints = int(my_find_SES(lineText, "depth="))
            deflector_axis = np.zeros([numDeflectorPoints])
            deflector_axis[0] = float(my_find_SES(lineText, "depthoffset="))
            deflectorStepSize = float(my_find_SES(lineText, "depthdelta="))
            deflector_axis = [deflector_axis[0] + i * deflectorStepSize for i in range(numDeflectorPoints)]

        with z.open(filename, 'r') as f:
            data_cube = np.frombuffer(f.read(), dtype=np.dtype(np.float32))
            data_cube = da.from_array(data_cube.reshape((len(eK_axis), len(analyzer_axis), len(deflector_axis)), order='F'), chunks='auto')

    # create xarray
    data = xr.DataArray(data_cube, dims=("eV", "theta_par", "defl_perp"),
                        coords={"defl_perp": deflector_axis, "theta_par": analyzer_axis, "eV": eK_axis})
    data.coords['defl_perp'].attrs = {'units': 'deg'}
    data.coords['theta_par'].attrs = {'units': 'deg'}

    return data, meta_lines



def my_find_SES(lines, item):
    ''' Loop over the lines in the file and then pick out the line
    starting with the desired keyword, based on SES metadata format.

    Parameters
    ------------
    lines : list
        A list where each entry is a line from the text file

    item : str
        The word or group of characters you are searching for

    Returns
    ------------
    The line starting with item
    '''


    for line in lines:
        if line.startswith(item) and '=' in line:
            return (line.split('='))[1].strip()
        elif line.startswith(item) and ':' in line:
            i = len(line.split(':'))-1
            return line.split(':')[i].strip()

def manip_from_SES_metadata(meta):
    ''' Function to parse the table of manipulator positions generated in an SES manipultor scan

    Parameters
    ------------
    meta : list
        A list containing the output from a SES metadata file

    Returns
    ------------
    axis : dict(np.array)
        Dictionary of axis positions, with each entry a np.array of axis positions through the scan
    '''

    # Extract the relevant manipulator data
    axis = {}  # Dictionary to hold axis data

    # Find location of manipulator info
    i = meta.index('[Run Mode Information]') + 1
    if 'Name' in meta[i]:
        i += 1

    for j in meta[i].split('\x0b'):
        axis[j] = []  # Make the dictionary

    i += 1
    while True:  # Iterate through and fill in the rest of the data
        try:
            axis_val = meta[i].split('\x0b')
            for j, v in enumerate(axis):
                axis[v].append(float(axis_val[j]))
            i += 1
        except:
            break

    # Covnert from lists to np arrays
    for i in axis:
        axis[i] = np.array(axis[i])

    return axis