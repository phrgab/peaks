"""Functions to load data from the I05-HR and i05-nano beamlines at Diamond Light Source.

"""

# Phil King 11/01/2021
# Brendan Edwards 01/03/2021
# Phil King 21/02/2022
# Brendan Edwards 14/02/2024

import os
import h5py
from peaks.core.utils.misc import analysis_warning
from peaks.core.fileIO.data_loading import _h5py_str, _extract_mapping_metadata


def _load_I05_data(fname):
    """This function loads data that was obtained at the I05 beamline at Diamond Light Source.

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
    from peaks.core.fileIO.loaders.I05 import _load_I05_data

    fname = 'C:/User/Documents/Research/i05-12345.nxs'

    # Extract data from file obtained at the I05 beamline
    data = _load_I05_data(fname)

    """

    # Get file_extension to determine the file type
    filename, file_extension = os.path.splitext(fname)

    # Nexus files are the standard Diamond Light Source format
    if file_extension == '.nxs':
        # Open the file (read only)
        f = h5py.File(fname, 'r')

        # Obtain the analyser keys to identify scan types. These keys are useful since they tell us what coordinates are
        # variable in the scan (and not simply static metadata).
        try:
            analyser_keys = list(f['entry1/analyser/'].keys())
        except KeyError:
            analyser_keys = list(f['entry1/analyser_total/'].keys())

        # Extract the spectrum, kinetic energies and angles (if present) from the file
        if 'analyser_total' in analyser_keys:
            spectrum = f['entry1/instrument/analyser_total/analyser_total']
            KE_values = f['entry1/instrument/analyser_total/kinetic_energy_center'][()]
        else:
            spectrum = f['entry1/instrument/analyser/data']
            KE_values = f['entry1/instrument/analyser/energies'][()]
            try:
                angles = f['entry1/instrument/analyser/angles'][()]
            except KeyError:
                pass

        # Determine the scan type and extract the data
        # Photon energy scan (kinetic energy varies)
        if 'energy' in analyser_keys:
            # Assign scan type
            scan_type = 'hv scan'

            # Extract hv coordinates and round to 2 d.p
            hv_values = round(f['entry1/instrument/monochromator/energy'][()], 2)

            # We want to save the kinetic energy coordinates of the first scan, and also the corresponding offsets for
            # successive scans (KE_delta)
            KE0 = KE_values[:, 0]  # Get  the maximum kinetic energy of the scan as a function of photon energy
            KE_delta = KE0 - KE0[0]  # Get the change in KE value of detector as a function of hv

            # Define data
            data = {'scan_type': scan_type, 'spectrum': spectrum, 'hv': hv_values, 'theta_par': angles,
                    'eV': KE_values[0], 'KE_delta': KE_delta}

            # Display warning explaining how kinetic energy values are saved
            warn_str = ("The kinetic energy coordinates saved are that of the first scan. The corresponding offsets "
                        "for successive scans are included in the KE_delta coordinate. Run DataArray.disp_from_hv(hv) "
                        "where DataArray is the loaded hv scan xr.DataArray and hv is the relevant photon energy to "
                        "extract a dispersion at using the proper kinetic energy scaling for that photon energy.")
            analysis_warning(warn_str, title='Loading info', warn_type='info')

        # Fermi map (polar/delf varies)
        elif (('analyser_polar_angle' in analyser_keys and 'angles' in analyser_keys) or
                ('anapolar' in analyser_keys and 'angles' in analyser_keys) or
                ('sapolar' in analyser_keys and 'angles' in analyser_keys) or
                ('deflector_x' in analyser_keys and 'angles' in analyser_keys)):
            # Assign scan type
            scan_type = 'FS map'

            # Extract Fermi map mapping coordinates and assign a mapping label (e.g. ana_polar, polar, delf_perp)
            if 'analyser_polar_angle' in analyser_keys:
                mapping_coords = f['entry1/instrument/analyser/analyser_polar_angle'][()]
                mapping_label = 'ana_polar'
            elif 'anapolar' in analyser_keys:
                mapping_coords = f['entry1/instrument/scan_group/anapolar'][()]
                mapping_label = 'ana_polar'
            elif 'sapolar' in analyser_keys:
                mapping_coords = f['entry1/instrument/manipulator/sapolar'][()]
                mapping_label = 'polar'
            elif 'deflector_x' in analyser_keys:
                mapping_coords = f['entry1/instrument/deflector_x/deflector_x'][()]
                mapping_label = 'delf_perp'

            # Define data
            data = {'scan_type': scan_type, 'spectrum': spectrum, mapping_label: mapping_coords,
                    'theta_par': angles, 'eV': KE_values}

        # da30_z scan (da30_z and location vary)
        elif 'da30_z' in analyser_keys and 'location' in analyser_keys:
            # Assign scan type
            scan_type = 'da30_z scan'

            # Extract da30_z and location coordinates
            da30_z = f['entry1/instrument/da30_z/da30_z'][()]
            location = f['entry1/instrument/analyser/location'][()]

            # Define data
            data = {'scan_type': scan_type, 'spectrum': spectrum, 'da30_z': da30_z,  'location': location,
                    'eV': KE_values}

        # Focus scan (smdefocus and x/y vary)
        elif (('smdefocus' in analyser_keys and 'smx' in analyser_keys) or
              ('smdefocus' in analyser_keys and 'smy' in analyser_keys)):
            # Assign scan type
            scan_type = 'Focus scan'

            # Extract smdefocus and x/y coordinates
            smdefocus = f['entry1/instrument/manipulator/smdefocus'][()]
            if 'smx' in analyser_keys:  # If scan is along x
                smx_or_y = f['entry1/instrument/manipulator/smx'][()]
            else:  # If scan is along y
                smx_or_y = f['entry1/instrument/manipulator/smy'][()]

            # Since smdefocus and x/y are defined on a 2D grid, we must extract the 1D axes coordinates
            if smdefocus[0][1] == smdefocus[0][0]:
                smdefocus = [i[0] for i in smdefocus]
                smx_or_y = smx_or_y[0]
            else:
                smdefocus = smdefocus[0]
                smx_or_y = [i[0] for i in smx_or_y]

            # Define data
            if 'smx' in analyser_keys:  # If scan is along x, label smx_or_y axis as x1
                data = {'scan_type': scan_type, 'spectrum': spectrum, 'defocus': smdefocus, 'x1': smx_or_y,
                        'theta_par': angles, "eV": KE_values}

            else:  # If scan is along x, label smx_or_y axis as x2
                data = {'scan_type': scan_type, 'spectrum': spectrum, 'defocus': smdefocus, 'x2': smx_or_y,
                        'theta_par': angles, "eV": KE_values}

        # 1D Focus scan (da30_z, smdefocus or salong vary)
        elif (('da30_z' in analyser_keys) or
              ('smdefocus' in analyser_keys) or
              ('salong' in analyser_keys)):
            # Assign scan type
            scan_type = '1D focus scan'

            # Extract focussing coordinates and assign a focussing_label label (e.g. focus, defocus)
            if 'da30_z' in analyser_keys:
                focussing_coords = f['entry1/instrument/da30_z/da30_z'][()]
                focussing_label = 'da30_z'
            elif 'smdefocus' in analyser_keys:
                focussing_coords = f['entry1/instrument/manipulator/smdefocus'][()]
                focussing_label = 'defocus'
            elif 'salong' in analyser_keys:
                focussing_coords = f['entry1/instrument/manipulator/salong'][()]
                focussing_label = 'focus'

            # Define data
            data = {'scan_type': scan_type, 'spectrum': spectrum, focussing_label: focussing_coords,
                    'theta_par': angles, 'eV': KE_values}

        # Spatial map (two spatial coords vary)
        elif (('smx' in analyser_keys and 'smy' in analyser_keys) or
              ('saz' in analyser_keys and 'sax' in analyser_keys)):
            # Assign scan type
            scan_type = 'spatial map'

            # Extract spatial coordinates
            if 'smx' in analyser_keys and 'smy' in analyser_keys:
                x1_values = f['entry1/instrument/manipulator/smx'][()]
                x2_values = f['entry1/instrument/manipulator/smy'][()]
            elif 'saz' in analyser_keys and 'sax' in analyser_keys:
                x1_values = f['entry1/instrument/manipulator/sax'][()]
                x2_values = f['entry1/instrument/manipulator/saz'][()]

            # Since x1 and x2 are defined on a 2D grid, we must extract the 1D axes coordinates
            if x1_values[0][1] != x1_values[0][0]:
                x1_values = x1_values[0]
                x2_values = [i[0] for i in x2_values]
            else:
                x1_values = [i[0] for i in x1_values]
                x2_values = x2_values[0]

            # Define data
            data = {'scan_type': scan_type, 'spectrum': spectrum, 'x1': x1_values, 'x2': x2_values,
                    'theta_par': angles, 'eV': KE_values}

        # Line scan (x, y or z varies)
        elif ('smx' in analyser_keys or
              'smy' in analyser_keys or
              'smz' in analyser_keys or
              'sax' in analyser_keys or
              'say' in analyser_keys or
              'saz' in analyser_keys):
            # Assign scan type
            scan_type = 'line scan'

            # Extract mapping coordinates and assign a mapping label (e.g. ana_polar, polar, delf_perp)
            if 'smx' in analyser_keys:  # x varies
                mapping_coords = f['entry1/instrument/manipulator/smx'][()]
                mapping_label = 'x1'
            elif 'smy' in analyser_keys:  # y varies
                mapping_coords = f['entry1/instrument/manipulator/smy'][()]
                mapping_label = 'x2'
            elif 'smz' in analyser_keys:  # defocus varies
                mapping_coords = f['entry1/instrument/manipulator/smz'][()]
                mapping_label = 'x3'
            elif 'sax' in analyser_keys:  # y varies
                mapping_coords = f['entry1/instrument/manipulator/sax'][()]
                mapping_label = 'x1'
            elif 'saz' in analyser_keys:  # y varies
                mapping_coords = f['entry1/instrument/manipulator/saz'][()]
                mapping_label = 'x2'
            elif 'say' in analyser_keys:  # y varies
                mapping_coords = f['entry1/instrument/manipulator/say'][()]
                mapping_label = 'x3'

            # Define data
            data = {'scan_type': scan_type, 'spectrum': spectrum, mapping_label: mapping_coords,
                    'theta_par': angles, 'eV': KE_values}

        # Dispersion (static scan)
        elif 'angles' in analyser_keys:
            # Assign scan type
            scan_type = 'dispersion'

            # Define data
            data = {'scan_type': scan_type, 'spectrum': spectrum[0], 'theta_par': angles, 'eV': KE_values}

        # Unable to identify scan, so raise error
        else:
            raise Exception("Scan type not supported.")

    # Zip files are the format used for SES deflector maps
    elif file_extension == '.zip':
        raise Exception("Code here to be completed.")

    return data


def _load_I05_metadata(fname, scan_type, loc):
    """This function loads metadata from data that was obtained at the I05 beamline at Diamond Light Source.

    Parameters
    ------------
    fname : str
        Path to the file to be loaded.

    scan_type : str
        The scan type of the data.

    loc : str
        The name of the location (typically a beamline). Here, it must be either 'Diamond I05-HR' or 'Diamond I05-nano'.

    Returns
    ------------
    metadata : dict
        Dictionary containing the relevant metadata.

    Examples
    ------------
    from peaks.core.fileIO.loaders.I05 import _load_I05_metadata

    fname = 'C:/User/Documents/Research/i05-12345.nxs'

    # Extract metadata from file obtained at the I05 beamline at Diamond Light Source
    metadata = _load_I05_metadata(fname)

    """

    # Define dictionary to store  metadata
    metadata = {}

    # Extract the scan name from the file full path
    fname_split = fname.split('/')
    metadata['scan_name'] = fname_split[len(fname_split)-1].split('.')[0]

    # Define initial attributes
    metadata['scan_type'] = scan_type
    metadata['sample_description'] = None
    metadata['eV_type'] = 'kinetic'
    metadata['beamline'] = loc
    metadata['analysis_history'] = []
    metadata['EF_correction'] = None

    # Get file_extension to determine the file type, to allow other metadata to be determined
    filename, file_extension = os.path.splitext(fname)

    # Nexus files are the standard Diamond Light Source format
    if file_extension == '.nxs':
        # Open the file (read only)
        f = h5py.File(fname, 'r')

        # Extract pass energy
        PE_handles = ['entry1/instrument/analyser/pass_energy',
                      'entry1/instrument/analyser_total/pass_energy']
        metadata['PE'] = _I05_find_attr(f, PE_handles, 'PE', float)

        # Extract photon energy
        hv_handles = ['entry1/instrument/monochromator/energy']
        metadata['hv'] = _I05_find_coord(f, hv_handles, 'hv', num_dp=2)

        # Extract polarisation
        pol_handles = ['entry1/instrument/insertion_device/beam/final_polarisation_label',
                       'entry1/instrument/insertion_device/beam/final_polarisation_label\n\t\t\t\t\t\t\t\t']
        metadata['pol'] = _I05_find_attr(f, pol_handles, 'pol', str)

        # Extract number of sweeps
        sweep_handles = ['entry1/instrument/analyser/number_of_iterations',
                         'entry1/instrument/analyser/number_of_cycles',
                         'entry1/instrument/analyser_total/number_of_frames']
        metadata['sweeps'] = _I05_find_attr(f, sweep_handles, 'sweeps', int)

        # Extract dwell time
        dwell_handles = ['entry1/instrument/analyser/time_for_frames',
                         'entry1/instrument/analyser_total/time_for_frames']
        metadata['dwell'] = _I05_find_attr(f, dwell_handles, 'dwell', float)

        # Extract analyser mode (method depends on if analyser_total measurement mode was used)
        if 'analyser_total' in list(f['entry1/'].keys()):
            metadata['ana_mode'] = 'Total'
        else:
            ana_mode_handles = ['entry1/instrument/analyser/lens_mode']
            ana_mode = _I05_find_attr(f, ana_mode_handles, 'ana_mode', str)
            # Reduce ana_mode name size
            if 'Angular30' in ana_mode or 'A30' in ana_mode:
                metadata['ana_mode'] = 'Ang30'
            elif 'Angular14' in ana_mode or 'A14' in ana_mode:
                metadata['ana_mode'] = 'Ang14'
            elif 'Transmission' in ana_mode:
                metadata['ana_mode'] = 'Trans'
            else:
                metadata['ana_mode'] = ana_mode

        # Extract analyser slit (this requires the slit setting, size and shape to be found)
        slit_setting_handles = ['entry1/instrument/analyser/entrance_slit_setting',
                                'entry1/instrument/analyser_total/entrance_slit_setting']
        slit_setting = _I05_find_attr(f, slit_setting_handles, 'slit setting', str)
        slit_size_handles = ['entry1/instrument/analyser/entrance_slit_size',
                             'entry1/instrument/analyser_total/entrance_slit_size']
        slit_size = _I05_find_attr(f, slit_size_handles, 'slit size', str)
        slit_shape_handles = ['entry1/instrument/analyser/entrance_slit_shape',
                              'entry1/instrument/analyser_total/entrance_slit_shape']
        slit_shape = _I05_find_attr(f, slit_shape_handles, 'slit shape', str)
        # Reduce slit shape name size
        if 'straight' in slit_shape:
            slit_shape = 's'
        elif 'curved' in slit_shape:
            slit_shape = 'c'
        metadata['ana_slit'] = str(slit_size) + slit_shape + ' (#' + str(slit_setting) + ')'

        # Define analyser slit angle
        metadata['ana_slit_angle'] = 90

        # Extract exit slit
        exit_handles = ['entry1/instrument/monochromator/exit_slit_size']
        metadata['exit_slit'] = _I05_find_attr(f, exit_handles, 'exit_slit', float)
        if isinstance(metadata['exit_slit'], float):
            metadata['exit_slit'] = round(metadata['exit_slit'], 3) * 1000

        # Extract deflector values (absent at nano branch)
        if loc == 'Diamond I05-HR':
            metadata['defl_par'] = None
            defl_perp_handles = ['entry1/instrument/deflector_x/deflector_x']
            metadata['defl_perp'] = _I05_find_coord(f, defl_perp_handles, 'defl_perp', num_dp=2)

        # Extract x1
        x1_handles = ['entry1/instrument/manipulator/smx',
                      'entry1/instrument/manipulator/sax']
        metadata['x1'] = _I05_find_coord(f, x1_handles, 'x1', num_dp=2)

        # Extract x2
        x2_handles = ['entry1/instrument/manipulator/smy',
                      'entry1/instrument/manipulator/saz']
        metadata['x2'] = _I05_find_coord(f, x2_handles, 'x2', num_dp=2)

        # Extract x3
        x3_handles = ['entry1/instrument/manipulator/smz',
                      'entry1/instrument/manipulator/say']
        metadata['x3'] = _I05_find_coord(f, x3_handles, 'x3', num_dp=2)

        # Extract metadata specific to the nano branch
        if loc == 'Diamond I05-nano':
            # Extract defocus
            defocus_handles = ['entry1/instrument/manipulator/smdefocus']
            metadata['defocus'] = _I05_find_coord(f, defocus_handles, 'defocus', num_dp=2)

            # Extract analyser polar angle
            ana_polar_handles = ['entry1/instrument/scan_group/anapolar',
                                 'entry1/instrument/analyser/analyser_polar_angle']
            metadata['ana_polar'] = _I05_find_coord(f, ana_polar_handles, 'ana_polar', num_dp=2)

        # Extract polar
        polar_handles = ['entry1/instrument/manipulator/smpolar',
                         'entry1/instrument/manipulator/sapolar']
        metadata['polar'] = _I05_find_coord(f, polar_handles, 'polar', num_dp=2)

        # Extract tilt (absent at nano branch due to 5-axis manipulator)
        if loc == 'Diamond I05-HR':
            tilt_handles = ['entry1/instrument/manipulator/satilt']
            metadata['tilt'] = _I05_find_coord(f, tilt_handles, 'tilt', num_dp=2)
        else:
            metadata['tilt'] = None

        # Extract azi
        azi_handles = ['entry1/instrument/manipulator/smazimuth',
                       'entry1/instrument/manipulator/saazimuth']
        metadata['azi'] = _I05_find_coord(f, azi_handles, 'azi', num_dp=2)

        # Define normal emission attributes
        metadata['norm_polar'] = None
        metadata['norm_azi'] = None
        metadata['norm_tilt'] = None

        # Extract optics metadata specific to the nano branch
        if loc == 'Diamond I05-nano':
            # Extract order sorting aperture x
            OSAx_handles = ['entry1/instrument/order_sorting_aperture/osax']
            metadata['OSAx'] = _I05_find_coord(f, OSAx_handles, 'OSAx', num_dp=2)

            # Extract order sorting aperture y
            OSAy_handles = ['entry1/instrument/order_sorting_aperture/osay']
            metadata['OSAy'] = _I05_find_coord(f, OSAy_handles, 'OSAy', num_dp=2)

            # Extract order sorting aperture z
            OSAz_handles = ['entry1/instrument/order_sorting_aperture/osaz']
            metadata['OSAz'] = _I05_find_coord(f, OSAz_handles, 'OSAz', num_dp=2)

            # Extract zone plate x
            ZPx_handles = ['entry1/instrument/zone_plate/zpx']
            metadata['ZPx'] = _I05_find_coord(f, ZPx_handles, 'ZPx', num_dp=2)

            # Extract zone plate y
            ZPy_handles = ['entry1/instrument/zone_plate/zpy']
            metadata['ZPy'] = _I05_find_coord(f, ZPy_handles, 'ZPy', num_dp=2)

            # Extract zone plate z
            ZPz_handles = ['entry1/instrument/zone_plate/zpz']
            metadata['ZPz'] = _I05_find_coord(f, ZPz_handles, 'ZPz', num_dp=2)

        # Extract sample temperature
        temp_sample_handles = ['entry1/sample/temperature']
        metadata['temp_sample'] = _I05_find_coord(f, temp_sample_handles, 'temp_sample', num_dp=2)

        # Extract cryostat temperature
        temp_cryo_handles = ['entry1/sample/cryostat_temperature']
        metadata['temp_cryo'] = _I05_find_coord(f, temp_cryo_handles, 'temp_cryo', num_dp=2)

        # Extract the command to run the scan
        scan_command_handles = ['entry1/scan_command/']
        metadata['scan_command'] = _I05_find_attr(f, scan_command_handles, 'scan_command', str)

    # Zip files are the format used for SES deflector maps
    elif file_extension == '.zip':
        raise Exception("Code here to be completed.")

    return metadata


def _I05_find_attr(file, file_handles, attr, attr_type):
    """Helper function which checks possible h5py file handles, and returns an attribute for a valid file handle.

    Parameters
    ------------
    file : h5py._hl.files.File
        An open h5py format file.

    file_handles : list
        A list of strings describing the file handle of h5py file object.

    attr : str
        The name of the metadata attribute that is being extracted.

    attr_type : type
        The expected type of the attr, e.g. int or float.

    Returns
    ------------
    extracted_attr : str, attr_type, None
        The extracted attribute.

    Examples
    ------------
    from peaks.core.fileIO.loaders.I05 import _I05_find_attr

    fname = 'C:/User/Documents/Research/i05-12345.nxs'

    # Open the file (read only)
    f = h5py.File(fname, 'r')

    # Extract the pass energy attribute from a file obtained at the I05 beamline at Diamond Light Source
    PE_handles = ['entry1/instrument/analyser/pass_energy', 'entry1/instrument/analyser_total/pass_energy']
    PE = _I05_find_attr(f, PE_handles, 'P.E', float)

    """

    # Set default value
    extracted_attr = None

    # Loop through file_handles to determine if any are valid
    for handle in file_handles:
        try:  # Attempt to extract value as type defined by attr_type
            extracted_attr = attr_type(_h5py_str(file, handle))
            break
        except TypeError:  # Attempt to extract value as type str
            extracted_attr = str(_h5py_str(file, handle))
            break
        except KeyError:  # Invalid file handle
            pass

    # Inform user if the metadata cannot be extracted
    if extracted_attr is None:
        analysis_warning(
            'Unable to extract {attr} metadata. Update peaks.core.fileIO.loaders.I05._load_I05_metadata to account for '
            'new file format.'.format(attr=attr), title='Loading info', warn_type='danger')

    return extracted_attr


def _I05_find_coord(file, file_handles, coord, num_dp):
    """Helper function which checks possible h5py file handles, and returns coordinate metadata for a valid file handle.

    Parameters
    ------------
    file : h5py._hl.files.File
        An open h5py format file.

    file_handles : list
        A list of strings describing the file handle of h5py file object.

    coord : str
        The name of the coordinate that is being extracted.

    num_dp : int
        The number of decimal places round to.

    Returns
    ------------
    extracted_coord : float, np.array, list, None
        The extracted coordinate(s).

    Examples
    ------------
    from peaks.core.fileIO.loaders.I05 import _I05_find_coord

    fname = 'C:/User/Documents/Research/i05-12345.nxs'

    # Open the file (read only)
    f = h5py.File(fname, 'r')

    # Extract the x coordinate metadata from a file obtained at the I05 beamline at Diamond Light Source
    x_handles = ['entry1/instrument/manipulator/smx', 'entry1/instrument/manipulator/sax']
    x = _I05_find_coord(f, x_handles, 'x1', num_dp=2)

    """

    # Set default value
    extracted_coord = None

    # Loop through file_handles to determine if any are valid, and extract coordinate information
    for handle in file_handles:
        try:  # Attempt to extract coordinate information
            # Extract coordinate values
            coord_values = file[handle][()]
            if len(coord_values) == 1:  # Simple case of a single coordinate value (rounding to num_dp decimal places)
                extracted_coord = round(coord_values[0], num_dp)
            elif len(coord_values) != 1 and len(coord_values.shape) == 1:  # 1D array of coordinate values (line scan)
                # Extract coordinate metadata in 'min:max (step)' format (rounding to num_dp decimal places)
                extracted_coord = _extract_mapping_metadata(coord_values, num_dp=num_dp)
            elif len(coord_values.shape) == 2:  # 2D array of coordinate values (spatial map)
                if coord_values[0][1] != coord_values[0][0]:
                    coord_values = coord_values[0]
                else:
                    coord_values = [i[0] for i in coord_values]
                # Extract coordinate metadata in 'min:max (step)' format (rounding to num_dp decimal places)
                extracted_coord = _extract_mapping_metadata(coord_values, num_dp=num_dp)
            else:  # Unable to identify coordinate
                extracted_coord = None
            break

        except KeyError:  # Invalid file handle
            pass

    # Inform user if the coordinate metadata cannot be extracted
    if extracted_coord is None:
        analysis_warning(
            'Unable to extract {coord} metadata. Update peaks.core.fileIO.loaders.I05._load_I05_metadata to account '
            'for new file format.'.format(coord=coord), title='Loading info', warn_type='danger')

    return extracted_coord
