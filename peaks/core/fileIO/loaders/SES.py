"""Functions to load data from the Scienta SES software.

"""

# Phil King 21/07/22
# Brendan Edwards 15/02/2024

import os
import zipfile
import numpy as np
from peaks.core.fileIO.loaders.ibw import _load_ibw_wavenote


def _load_SES_data(fname):
    """Master function to load data stored in SES format files.

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

        from peaks.core.fileIO.loaders.SES import _load_SES_data

        fname = 'C:/User/Documents/Research/FM1.zip'

        # Extract data from an SES format file
        data = _load_SES_data(fname)

    """

    # Get file_extension to determine the file type
    filename, file_extension = os.path.splitext(fname)

    # Use the relevant function to load the SES format data
    if file_extension == '.txt':
        data = _load_SES_txt_data(fname)
    elif file_extension == '.zip':
        data = _load_SES_zip_data(fname)
    elif file_extension == '.ibw':
        data = _load_SES_ibw_data(fname)
    else:
        raise Exception("File type is not supported")

    return data


def _load_SES_txt_data(fname):
    """This function loads data stored in SES format txt files.

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

        from peaks.core.fileIO.loaders.SES import _load_SES_txt_data

        fname = 'C:/User/Documents/Research/disp1.txt'

        # Extract data from an SES format file
        data = _load_SES_txt_data(fname)

    """

    # SES format text files only support dispersions
    scan_type = 'dispersion'

    # Open the file and extract the lines containing metadata (used to extract theta_par values, and determine when scan
    # data starts)
    metadata_lines = _load_SES_metalines(fname)

    # Load file data
    file_data = np.loadtxt(fname, skiprows=len(metadata_lines))
    spectrum = file_data[:, 1:].T
    KE_values = file_data[:, 0]
    theta_par_values = np.fromstring(_SES_find(metadata_lines, 'Dimension 2 scale='), sep=" ")

    # Define data
    data = {'scan_type': scan_type, 'spectrum': spectrum, 'theta_par': theta_par_values,
            'eV': KE_values}

    return data


def _load_SES_zip_data(fname):
    """This function loads data stored in SES format zip files (adapted from the PESTO file loader by Craig Polley).

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

        from peaks.core.fileIO.loaders.SES import _load_SES_zip_data

        fname = 'C:/User/Documents/Research/FM1.zip'

        # Extract data from an SES format file
        data = _load_SES_zip_data(fname)

    """

    # SES format text files only support deflector Fermi maps
    scan_type = 'FS map'

    # Open the file and load the data
    with zipfile.ZipFile(fname) as z:
        files = z.namelist()
        file_bin = [file for file in files if '.bin' in file]
        file_ini = [file for file in files if 'Spectrum_' in file and '.ini' in file]
        filename = file_bin[0]
        filename_ini = file_ini[0]

        # Extract coordinate information
        with z.open(filename_ini) as f:
            # Read and decode lines in file
            lines = f.readlines()
            lineText = [line.decode() for line in lines]

            # Extract kinetic energy axis
            num_KE = int(_SES_find(lineText, "width="))
            KE_start = float(_SES_find(lineText, "widthoffset="))
            KE_step = float(_SES_find(lineText, 'widthdelta='))
            KE_end = KE_start + (KE_step * (num_KE - 1))
            KE_values = np.linspace(KE_start, KE_end, num_KE)

            # Extract theta_par axis
            num_theta_par = int(_SES_find(lineText, 'height='))
            theta_par_start = float(_SES_find(lineText, "heightoffset="))
            theta_par_step = float(_SES_find(lineText, 'heightdelta='))
            theta_par_end = theta_par_start + (theta_par_step * (num_theta_par - 1))
            theta_par_values = np.linspace(theta_par_start, theta_par_end, num_theta_par)

            # Extract deflector axis
            num_defl_perp = int(_SES_find(lineText, 'depth='))
            defl_perp_start = float(_SES_find(lineText, "depthoffset="))
            defl_perp_step = float(_SES_find(lineText, 'depthdelta='))
            defl_perp_end = defl_perp_start + (defl_perp_step * (num_defl_perp - 1))
            defl_perp_values = np.linspace(defl_perp_start, defl_perp_end, num_defl_perp)

        # Extract spectrum and reshape into a data cube to be consistent with loading
        with z.open(filename, 'r') as f:
            spectrum = np.frombuffer(f.read(), dtype=np.dtype(np.float32))
            spectrum = spectrum.reshape(num_KE, num_theta_par, num_defl_perp, order='F').T

    # Define data
    data = {'scan_type': scan_type, 'spectrum': spectrum, 'defl_perp': defl_perp_values, 'theta_par': theta_par_values,
            'eV': KE_values}

    return data


def _load_SES_ibw_data(fname):
    """This function loads data stored in SES format ibw files.

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

        from peaks.core.fileIO.loaders.SES import _load_SES_ibw_data

        fname = 'C:/User/Documents/Research/disp1.ibw'

        # Extract data from an SES format file
        data = _load_SES_ibw_data(fname)

    """

    raise Exception('SES ibw file loading is not currently supported')


def _load_SES_metalines(fname):
    """This function will extract the lines containing metadata in an SES format .txt, .zip or .ibw file.

    Parameters
    ------------
    fname : str
        Path to the file to be loaded.

    Returns
    ------------
    metadata_lines : list
        Lines extracted from the file containing the metadata.

    Examples
    ------------
    Example usage is as follows::

        from peaks.core.fileIO.loaders.SES import _extract_SES_metalines

        fname = 'C:/User/Documents/Research/disp1.zip'

        # Extract the lines containing metadata
        loc = _extract_SES_metalines(fname)

    """
    # Get file_extension to determine the file type
    filename, file_extension = os.path.splitext(fname)

    # If the file is .txt format
    if file_extension == '.txt':
        # Open the file and extract the lines containing metadata
        with open(fname) as f:
            metadata_lines = []
            while True:
                line = f.readline()
                metadata_lines.append(line)
                # When the line starting with '[Data' is encountered, metadata has ended and scan data has begun
                if '[Data' in line:
                    break

    # If the file is .zip format
    elif file_extension == '.zip':
        # Open the file and extract the lines containing metadata
        with zipfile.ZipFile(fname) as z:
            files = z.namelist()
            file_ini = [file for file in files if 'Spectrum_' not in file and '.ini' in file]
            filename = file_ini[0]
            with z.open(filename) as f:
                lines_to_decode = f.readlines()
                metadata_lines = [line.decode() for line in lines_to_decode]

    # If the file is .ibw format
    elif file_extension == '.ibw':
        # Extract the lines containing metadata from the wavenote of the ibw file using the _load_ibw_wavenote function
        metadata_lines = _load_ibw_wavenote(fname)

    return metadata_lines


def _SES_find(lines, item):
    """This function will loop over the lines in an SES format file, and then pick out the line starting with the
    desired keyword.

    Parameters
    ------------
    lines : list
        A list where each entry is a line from the SES format file.

    item : str
        The keyword that is being searched for.

    Returns
    ------------
    line_contents : str
        The line contents following the desired keyword.

    Examples
    ------------
    Example usage is as follows::

        from peaks.core.fileIO.loaders.SES import _SES_find

        fname = 'C:/User/Documents/Research/disp1.txt'

        # Open the file and load lines
        with open(fname) as f:
            lines = f.readlines()

        # Extract the analyser mode
        ana_mode = _SES_find(lines, 'Analyzer Lens')

    """

    # Loop over lines to extract the line starting with the desired keyword.
    for line in lines:
        if line.startswith(item) and '=' in line:
            line_contents = line.split('=')[-1].strip()
            break
        elif line.startswith(item) and ':' in line:
            line_contents = line.split(':')[-1].strip()
            break

    return line_contents
