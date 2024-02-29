"""Functions to load data from the CASSIOPEE beamline at SOLEIL.

"""

# Liam Trzaska 15/04/2020
# Brendan Edwards 15/06/2021
# Phil King 27/07/2022
# Brendan Edwards 28/02/2024

import os
import natsort
import itertools
import numpy as np
from os.path import isfile, join
from peaks.core.fileIO.loaders.SES import _load_SES_data, _load_SES_metalines, _SES_find
from peaks.core.fileIO.data_loading import _extract_mapping_metadata



def _load_CASSIOPEE_data(fname):
    """This function loads data that was obtained at the CASSIOPEE beamline at SOLEIL.

    Parameters
    ------------
    fname : str
        Path to the file to be loaded.

    Returns
    ------------
    data : dict
        Dictionary containing the file scan type, spectrum, and coordinates.

    Examples
    ------------
    Example usage is as follows::

        from peaks.core.fileIO.loaders.CASSIOPEE import _load_CASSIOPEE_data

        fname = 'C:/User/Documents/Research/disp1.ibw'

        # Extract data from file obtained at the CASSIOPEE beamline at SOLEIL
        data = _load_CASSIOPEE_data(fname)

    """

    # Get file_extension to determine the file type
    filename, file_extension = os.path.splitext(fname)

    # If the file is a folder, it is a mapping measurements (either a Fermi map or hv scan)
    if file_extension == '':
        # Extract the relevant filenames into two lists
        file_list_ROI = natsort.natsorted(
            [item for item in os.listdir(fname) if 'ROI1' in item and isfile(join(fname, item))])
        file_list_i = natsort.natsorted(
            [item for item in os.listdir(fname) if '_i' in item and isfile(join(fname, item))])

        # Open the first file and extract the lines containing metadata
        with open(fname + '/' + file_list_ROI[0]) as f:
            metadata_lines = [line for line in itertools.islice(f, 0, 45)]

        # Extract kinetic energy and theta_par values
        KE_values = _SES_find(metadata_lines, 'Dimension 1 scale=')
        theta_par_values = _SES_find(metadata_lines, 'Dimension 2 scale=')

        # Convert kinetic energy and theta_par values to lists, and ensure each item is a float
        KE_values = KE_values.split(' ')
        theta_par_values = theta_par_values.split(' ')
        for i, KE in enumerate(KE_values):
            KE_values[i] = float(KE)
        for i, theta_par in enumerate(theta_par_values):
            theta_par_values[i] = float(theta_par)

        # Loop through images and extract potential mapping coordinates polar and hv (one of polar or hv will vary, one
        # will be static)
        polar_values = []
        hv_values = []
        for file_i in file_list_i:
            with open(fname + '/' + file_i) as f:
                lines = f.readlines()
                polar = float(_SES_find(lines, 'theta (deg)'))
                hv = float(_SES_find(lines, 'hv (eV)'))
            polar_values.append(polar)
            hv_values.append(hv)

        # Determine the scan type by seeing which out of theta_perp and hv varies, and define spectrum
        if abs(hv_values[-1] - hv_values[0]) > 1:  # Should be an hv scan
            scan_type = 'hv scan'
            spectrum = np.zeros((len(hv_values), len(theta_par_values), len(KE_values)))
        else:  # Should be an FS map
            scan_type = 'FS map'
            spectrum = np.zeros((len(polar_values), len(theta_par_values), len(KE_values)))

        # Define the shape of the individual 2D data slices of the 3D data to be extracted
        slice_shape = (len(theta_par_values), len(KE_values))

        # Loop through the individual 2D data slices of the 3D data and extract the spectrum
        for i, file in enumerate(file_list_ROI):
            spectrum[i, :, :] = _load_CASSIOPEE_slice(fname, file, slice_shape)

        # If the scan type is an hv scan
        if scan_type == 'hv scan':
            # Get the change in KE value of detector as a function of hv
            KE_delta = hv_values - hv_values[0]

            # Define data
            data = {'scan_type': scan_type, 'spectrum': spectrum, 'hv': hv_values, 'theta_par': theta_par_values,
                    'eV': KE_values, 'KE_delta': KE_delta}

        # If the scan type is a Fermi map
        elif scan_type == 'FS map':
            # Define data
            data = {'scan_type': scan_type, 'spectrum': spectrum, 'polar': polar_values, 'theta_par': theta_par_values,
                    'eV': KE_values}

    # All other SOLEIL CASSIOPEE files will be either txt, zip or ibw in the SES format, so use the _load_SES_data
    # function to load the data
    else:
        data = _load_SES_data(fname)

        # If the data is an ibw file, the scan type will not yet be identified
        if data['scan_type'] is None:
            # If scan_no is a dimension in data, the scan was run in add dimension mode. Thus, update scan type to a
            # multiple scan dispersion
            if 'scan_no' in list(data):
                data['scan_type'] = 'dispersion (multiple scans)'

            # If not set the scan type to a dispersion by default
            else:
                data['scan_type'] = 'dispersion'

    return data


def _load_CASSIOPEE_metadata(fname, scan_type):
    """This function loads metadata from data that was obtained at the CASSIOPEE beamline at SOLEIL.

    Parameters
    ------------
    fname : str
        Path to the file to be loaded.

    scan_type : str
        The scan type of the data.

    Returns
    ------------
    metadata : dict
        Dictionary containing the relevant metadata.

    Examples
    ------------
    Example usage is as follows::

        from peaks.core.fileIO.loaders.CASSIOPEE import _load_CASSIOPEE_metadata

        fname = 'C:/User/Documents/Research/disp1.ibw'

        # Extract metadata from file obtained at the CASSIOPEE beamline at SOLEIL
        metadata = _load_CASSIOPEE_metadata(fname)

    """

    # Get file_extension to determine the file type
    filename, file_extension = os.path.splitext(fname)

    # If the file is a folder, it is a mapping measurements (either a Fermi map or hv scan)
    if file_extension == '':
        # Extract the relevant filenames into two lists
        file_list_ROI = natsort.natsorted(
            [item for item in os.listdir(fname) if 'ROI1' in item and isfile(join(fname, item))])
        file_list_i = natsort.natsorted(
            [item for item in os.listdir(fname) if '_i' in item and isfile(join(fname, item))])

        # Open the first file and extract the lines containing metadata
        with open(fname + '/' + file_list_ROI[0]) as f:
            metalines = [line for line in itertools.islice(f, 0, 45)]

    # All other SOLEIL CASSIOPEE files will be either txt, zip or ibw in the SES format
    else:
        # Extract the lines containing metadata
        metalines = _load_SES_metalines(fname)

    # Define dictionary to store metadata
    metadata = {}

    # Extract the scan name from the file full path
    fname_split = fname.split('/')
    metadata['scan_name'] = fname_split[len(fname_split)-1].split('.')[0]

    # Define initial attributes
    metadata['scan_type'] = scan_type
    metadata['sample_description'] = None
    metadata['eV_type'] = 'kinetic'
    metadata['beamline'] = 'SOLEIL CASSIOPEE'
    metadata['analysis_history'] = []
    metadata['EF_correction'] = None

    # Define attributes, using the _SES_find function to obtain metadata where possible
    metadata['PE'] = float(_SES_find(metalines, 'Pass Energy'))
    hv = _SES_find(metalines, 'Excitation Energy')
    hv = hv.replace(',', '.')
    metadata['hv'] = round(float(hv), 2)
    metadata['pol'] = None
    metadata['sweeps'] = int(_SES_find(metalines, 'Number of Sweeps'))
    metadata['dwell'] = float(_SES_find(metalines, 'Step Time'))/1000
    metadata['ana_mode'] = _SES_find(metalines, 'Lens Mode')

    # Define other attributes that are not typically saved in the metadata (see following code for the exception of hv
    # scans and Fermi maps)
    metadata['ana_slit'] = None
    metadata['ana_slit_angle'] = 90
    metadata['exit_slit'] = None
    metadata['x1'] = None
    metadata['x2'] = None
    metadata['x3'] = None
    metadata['polar'] = None
    metadata['tilt'] = None
    metadata['azi'] = None
    metadata['norm_polar'] = None
    metadata['norm_tilt'] = None
    metadata['norm_azi'] = None
    metadata['temp_sample'] = None
    metadata['temp_cryo'] = None

    # Extract additional metadata that is stored within hv scans and Fermi maps
    if scan_type == 'hv scan' or scan_type == 'FS map':
        # Open the first file and extract additional lines containing metadata
        with open(join(fname, file_list_i[0])) as f_i:
            additional_metalines = f_i.readlines()

        # Extract polarisation
        pol_list = ['LV', 'LH', 'AV', 'AH', 'CR']
        pol_index = int(_SES_find(additional_metalines, 'Polarisation [0:LV, 1:LH, 2:AV, 3:AH, 4:CR]'))
        metadata['pol'] = pol_list[pol_index]

        # Extract spatial positions
        metadata['x1'] = round(float(_SES_find(additional_metalines, 'x (mm)')), 3)
        metadata['x2'] = round(float(_SES_find(additional_metalines, 'z (mm)')), 3)
        metadata['x3'] = round(float(_SES_find(additional_metalines, 'y (mm)')), 3)

        # Extract polar and hv values
        if scan_type == 'hv scan':  # Polar constant, hv varies
            metadata['polar'] = float(_SES_find(additional_metalines, 'theta (deg)'))
            hv_values = []
            for file_i in file_list_i:
                with open(fname + '/' + file_i) as f:
                    lines = f.readlines()
                    hv = float(_SES_find(lines, 'hv (eV)'))
                hv_values.append(hv)
            # Represent the hv metadata using a 'min:max (step)' format (rounding to 3 decimal places)
            metadata['hv'] = _extract_mapping_metadata(hv_values, num_dp=3)
        elif scan_type == 'FS map':  # hv constant, polar varies
            polar_values = []
            for file_i in file_list_i:
                with open(fname + '/' + file_i) as f:
                    lines = f.readlines()
                    polar = float(_SES_find(lines, 'theta (deg)'))
                polar_values.append(polar)
            # Represent the polar metadata using a 'min:max (step)' format (rounding to 3 decimal places)
            metadata['polar'] = _extract_mapping_metadata(polar_values, num_dp=3)

        # Extract tilt and azi
        try:
            metadata['tilt'] = round(float(_SES_find(additional_metalines, 'tilt (deg)')), 3)
            metadata['azi'] = round(float(_SES_find(additional_metalines, 'phi (deg)')), 3)
        except (UnboundLocalError, ValueError):
            metadata['tilt'] = None
            metadata['azi'] = None

    return metadata


def _load_CASSIOPEE_slice(folder, file, slice_shape):
    """This function loads a single 2D slice of 3D data (either a Fermi map or hv scan) that was obtained at the
    CASSIOPEE beamline at SOLEIL.

    Parameters
    ------------
    folder : str
        Path to the folder of the 3D data that a data slice will be loaded from.

    file : str
        Name of the file within the folder of the 3D data corresponding to the data slice to be loaded.

    slice_shape : tuple
        Shape of the numpy.ndarray data slice to be extracted.

    Returns
    ------------
    slice_data : numpy.ndarray
        A single 2D data slice of the 3D data stored in folder.

    Examples
    ------------
    Example usage is as follows::

        from peaks.core.fileIO.loaders.CASSIOPEE import _load_CASSIOPEE_slice

        folder = 'C:/User/Documents/Research/FS1'

        # Extract the sixth slice of data from the FS1 file obtained at the CASSIOPEE beamline at SOLEIL
        data_slice = _load_CASSIOPEE_slice(folder, 'FS1_1_ROI6_.txt')

    """

    # Open the file within the folder of the 3D data corresponding to the data slice to be loaded, loop through the
    # lines, and stop when the scan data is reached
    with open(folder + '/' + file) as f:
        lines = f.readlines()
        for counter, line in enumerate(lines):
            if line.startswith('inputA='):
                break

        # Define numpy.ndarray to store data slice
        slice_data = np.zeros(slice_shape)

        # Extract the data slice
        for i in range(len(lines) - (counter + 2)):
            slice_data[:, i] = np.fromstring(lines[counter + 2 + i], sep='\t')[1:]

        return slice_data
