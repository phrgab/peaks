"""Functions to load data from the I05-HR and i05-nano beamlines at Diamond Light Source.

"""

# Phil King 11/01/2021
# Brendan Edwards 01/03/2021
# Phil King 21/02/2022
# Brendan Edwards 08/02/2024

import os
import h5py
from peaks.core.utils.misc import analysis_warning


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
            data = {'scan_type': scan_type, 'spectrum': spectrum, 'theta_par': angles, 'eV': KE_values}

        # Unable to identify scan, so raise error
        else:
            raise Exception("Scan type not supported.")

    # Zip files are the format used for SES deflector maps
    elif file_extension == '.zip':
        raise Exception("Code here to be completed.")

    return data
