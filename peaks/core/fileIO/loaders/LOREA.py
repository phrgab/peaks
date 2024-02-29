"""Functions to load data from the LOREA beamline at ALBA.

"""

# Phil King 04/01/2023
# Brendan Edwards 28/02/2024

import h5py
import numpy as np
from peaks.core.fileIO.data_loading import _h5py_find_attr, _h5py_find_coord


def _load_LOREA_data(fname):
    """This function loads data that was obtained at the LOREA beamline at ALBA.

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

        from peaks.core.fileIO.loaders.LOREA import _load_LOREA_data

        fname = 'C:/User/Documents/Research/disp1.nxs'

        # Extract data from file obtained at the LOREA beamline at ALBA
        data = _load_LOREA_data(fname)

    """

    # Open the file (read only)
    f = h5py.File(fname, 'r')

    # Obtain the analyser keys to identify scan types. These keys are useful since they tell us what coordinates are
    # variable in the scan (and not simply static metadata).
    analyser_keys = list(f['entry1/data/'].keys())

    # Extract the spectrum, kinetic energies and theta_par_values (if present) from the file
    spectrum = f['entry1/data/data']
    KE_values = f['entry1/instrument/analyser/energies'][()]
    try:
        theta_par_values = f['entry1/data/angles'][()]
    except KeyError:
        pass

    # Determine the scan type and extract the data
    # Photon energy scan (photon energy varies)
    if 'photon_energies' in analyser_keys:
        # Assign scan type
        scan_type = 'hv scan'

        # Extract hv coordinates and round to 2 d.p
        hv_values = np.round(f['entry1/data/photon_energies'][()], 2)

        # We want to save the kinetic energy coordinates of the first scan, and also the corresponding offsets for
        # successive scans (KE_delta)
        KE0 = KE_values[:, 0]  # Get  the maximum kinetic energy of the scan as a function of photon energy
        KE_delta = KE0 - KE0[0]  # Get the change in KE value of detector as a function of hv

        # Define data
        data = {'scan_type': scan_type, 'spectrum': spectrum, 'hv': hv_values, 'theta_par': theta_par_values,
                'eV': KE_values[0], 'KE_delta': KE_delta}

    # Fermi map (polar/delf varies)
    elif (('sapolar' in analyser_keys and 'angles' in analyser_keys) or
          ('defl_angles' in analyser_keys and 'angles' in analyser_keys)):
        # Assign scan type
        scan_type = 'FS map'

        # Extract Fermi map mapping coordinates and assign a mapping label (e.g. polar, defl_perp)
        if 'sapolar' in analyser_keys:  # polar varies
            mapping_coords = f['entry1/instrument/manipulator/sapolar'][()]
            mapping_label = 'polar'
        elif 'defl_angles' in analyser_keys:  # defl_perp varies
            mapping_coords = f['entry1/data/defl_angles'][()]
            mapping_label = 'defl_perp'

        # Define data
        data = {'scan_type': scan_type, 'spectrum': spectrum, mapping_label: mapping_coords,
                'theta_par': theta_par_values, 'eV': KE_values}

    # Spatial map (two spatial coords vary)
    elif 'saz' in analyser_keys and 'sax' in analyser_keys:
        # Assign scan type
        scan_type = 'spatial map'

        # Extract spatial coordinates
        x1_values = f['entry1/instrument/manipulator/sax'][()]
        x2_values = f['entry1/instrument/manipulator/saz'][()]

        # Define data
        data = {'scan_type': scan_type, 'spectrum': spectrum, 'x1': x1_values, 'x2': x2_values,
                'theta_par': theta_par_values, 'eV': KE_values}

    # Line scan (x or z varies)
    elif ('sax' in analyser_keys or
          'saz' in analyser_keys):
        # Assign scan type
        scan_type = 'line scan'

        # Extract mapping coordinates and assign a mapping label (e.g. x1, x2)
        if 'sax' in analyser_keys:  # x varies
            mapping_coords = f['entry1/instrument/manipulator/sax'][()]
            mapping_label = 'x1'
        elif 'saz' in analyser_keys:  # z varies
            mapping_coords = f['entry1/instrument/manipulator/saz'][()]
            mapping_label = 'x2'

        # Define data
        data = {'scan_type': scan_type, 'spectrum': spectrum, mapping_label: mapping_coords,
                'theta_par': theta_par_values, 'eV': KE_values}

    # 1D Focus scan (salong or y varies)
    elif (('salong' in analyser_keys) or
          ('say' in analyser_keys)):
        # Assign scan type
        scan_type = '1D focus scan'

        # Extract focussing coordinates and assign a focussing_label label (e.g. focus, defocus)
        if 'salong' in analyser_keys:  # focus varies
            focussing_coords = f['entry1/instrument/manipulator/salong'][()]
            focussing_label = 'focus'
        elif 'say' in analyser_keys:  # y varies
            focussing_coords = f['entry1/instrument/manipulator/say'][()]
            focussing_label = 'x3'

        # Define data
        data = {'scan_type': scan_type, 'spectrum': spectrum, focussing_label: focussing_coords,
                'theta_par': theta_par_values, 'eV': KE_values}

    # Temperature-dependent scan (temperature varies)
    elif 'temperature' in analyser_keys:
        # Assign scan type
        scan_type = 'temp-dependent scan'

        # Extract temperature coordinates
        temp_values = f['entry1/sample/temperature'][()]

        # Define data
        data = {'scan_type': scan_type, 'spectrum': spectrum, 'temp_sample': temp_values,
                'theta_par': theta_par_values, 'eV': KE_values}

    # Dispersion (static scan)
    elif 'angles' in analyser_keys:
        # Assign scan type
        scan_type = 'dispersion'

        # Define data
        data = {'scan_type': scan_type, 'spectrum': spectrum, 'theta_par': theta_par_values, 'eV': KE_values}

    # Unable to identify scan, so raise error
    else:
        raise Exception("Scan type not supported.")

    return data


def _load_LOREA_metadata(fname, scan_type):
    """This function loads metadata from data that was obtained at the LOREA beamline at ALBA.

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

        from peaks.core.fileIO.loaders.LOREA import _load_LOREA_metadata

        fname = 'C:/User/Documents/Research/disp1.nxs'

        # Extract metadata from file obtained at the LOREA beamline at ALBA
        metadata = _load_LOREA_metadata(fname)

    """

    # Define dictionary to store metadata
    metadata = {}

    # Open the file (read only)
    f = h5py.File(fname, 'r')

    # Extract the scan name from the file full path
    fname_split = fname.split('/')
    metadata['scan_name'] = fname_split[len(fname_split)-1].split('.')[0]

    # Define initial attributes
    metadata['scan_type'] = scan_type
    metadata['sample_description'] = None
    metadata['eV_type'] = 'kinetic'
    metadata['beamline'] = 'ALBA LOREA'
    metadata['analysis_history'] = []
    metadata['EF_correction'] = None

    # Define other attributes, using the _h5py_find_attr _h5py_find_coord functions to obtain metadata where possible

    # Extract pass energy
    PE_handles = ['entry1/instrument/analyser/pass_energy']
    metadata['PE'] = _h5py_find_attr(f, PE_handles, 'PE', float)

    # Extract photon energy
    hv_handles = ['entry1/instrument/monochromator/energy']
    metadata['hv'] = _h5py_find_coord(f, hv_handles, 'hv', num_dp=2)

    # Extract polarisation
    pol_handles = ['entry1/instrument/insertion_device/beam/final_polarisation']
    metadata['pol'] = _h5py_find_attr(f, pol_handles, 'pol', str)
    if metadata['pol'] == 'linear_horizontal':
        metadata['pol'] = 'LH'
    elif metadata['pol'] == 'linear_vertical':
        metadata['pol'] = 'LV'

    # Extract number of sweeps
    sweep_handles = ['entry1/instrument/analyser/number_of_scans',
                     'entry1/instrument/analyser/number_of_cycles']
    metadata['sweeps'] = _h5py_find_attr(f, sweep_handles, 'sweeps', int)

    # Extract dwell time
    dwell_handles = ['entry1/instrument/analyser/time_per_scan']
    metadata['dwell'] = _h5py_find_attr(f, dwell_handles, 'dwell', float)/1000

    # Extract analyser mode (method depends on if analyser_total measurement mode was used)
    ana_mode_handles = ['entry1/instrument/analyser/lens_mode']
    metadata['ana_mode'] = _h5py_find_attr(f, ana_mode_handles, 'ana_mode', str)

    # Extract analyser slit (this requires the slit setting, size and shape to be found)
    slit_setting_handles = ['entry1/instrument/analyser/entrance_slit_setting']
    slit_setting = _h5py_find_attr(f, slit_setting_handles, 'slit setting', str)
    slit_size_handles = ['entry1/instrument/analyser/entrance_slit_size']
    slit_size = _h5py_find_attr(f, slit_size_handles, 'slit size', str)
    slit_shape_handles = ['entry1/instrument/analyser/entrance_slit_shape']
    slit_shape = _h5py_find_attr(f, slit_shape_handles, 'slit shape', str)
    # Reduce slit shape name size
    if 'straight' in slit_shape:
        slit_shape = 's'
    elif 'curved' in slit_shape:
        slit_shape = 'c'
    metadata['ana_slit'] = str(slit_size) + slit_shape + ' (#' + str(slit_setting) + ')'

    # Define analyser slit angle
    metadata['ana_slit_angle'] = 0

    # Extract exit slit
    exit_handles = ['entry1/instrument/monochromator/exit_slit_size']
    metadata['exit_slit'] = _h5py_find_attr(f, exit_handles, 'exit_slit', float)
    if isinstance(metadata['exit_slit'], float):
        metadata['exit_slit'] = round(metadata['exit_slit'], 2)

    # Extract deflector values
    metadata['defl_par'] = None
    defl_perp_handles = ['entry1/instrument/analyser/defl_angles']
    metadata['defl_perp'] = _h5py_find_coord(f, defl_perp_handles, 'defl_perp', num_dp=2)

    # Extract x1
    x1_handles = ['entry1/instrument/manipulator/sax']
    metadata['x1'] = _h5py_find_coord(f, x1_handles, 'x1', num_dp=2)

    # Extract x2
    x2_handles = ['entry1/instrument/manipulator/saz']
    metadata['x2'] = _h5py_find_coord(f, x2_handles, 'x2', num_dp=2)

    # Extract x3
    x3_handles = ['entry1/instrument/manipulator/say']
    metadata['x3'] = _h5py_find_coord(f, x3_handles, 'x3', num_dp=2)

    # Extract polar
    polar_handles = ['entry1/instrument/manipulator/sapolar']
    metadata['polar'] = _h5py_find_coord(f, polar_handles, 'polar', num_dp=2)

    # Extract tilt (absent at nano branch due to 5-axis manipulator)
    tilt_handles = ['entry1/instrument/manipulator/satilt']
    metadata['tilt'] = _h5py_find_coord(f, tilt_handles, 'tilt', num_dp=2)

    # Extract azi
    azi_handles = ['entry1/instrument/manipulator/saazimuth']
    metadata['azi'] = _h5py_find_coord(f, azi_handles, 'azi', num_dp=2)

    # Define normal emission attributes
    metadata['norm_polar'] = None
    metadata['norm_azi'] = None
    metadata['norm_tilt'] = None

    # Extract sample temperature (only saved in metadata if it varies)
    temp_sample_handles = ['entry1/sample/temperature']
    try:
        metadata['temp_sample'] = _h5py_find_coord(f, temp_sample_handles, 'temp_sample', num_dp=2)
    except AttributeError:
        metadata['temp_sample'] = None

    # Define cryostat temperature
    metadata['temp_cryo'] = None

    return metadata
