"""Functions to load data from the Scienta SES software.

"""

# Phil King 21/07/22
# Brendan Edwards 12/02/2024

import zipfile
from peaks.core.fileIO.loaders.ibw import _load_ibw_wavenote


def _load_SES_data(fname):
    pass


def _load_SES_txt_data(fname):
    pass


def _load_SES_zip_data(fname):
    pass


def _load_SES_ibw_data(fname):
    pass


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
            line = ''
            while '[Data' not in line:
                line = f.readline()
                metadata_lines.append(line)

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
    """This function will loop over the lines in an SES file, and then pick out the line starting with the desired
    keyword.

    Parameters
    ------------
    lines : list
        A list where each entry is a line from the .xy text file.

    item : str
        The keyword that is being searched for.

    Returns
    ------------
    line_contents : str
        The line contents following '# ' + the desired keyword.

    Examples
    ------------
    from peaks.core.fileIO.loaders.SES import _SES_find

    fname = 'C:/User/Documents/Research/disp1.txt'

    # Open the file and load lines
    with open(fname) as f:
        lines = f.readlines()

    # Extract the location
    loc = _SES_find(lines, 'location')

    """

    # Loop over lines to extract the line starting with the desired keyword.
    for line in lines:
        if line.startswith(item) and '=' in line:
            line_contents = (line.split('='))[1].strip()
            break
        elif line.startswith(item) and ':' in line:
            i = len(line.split(':'))-1
            line_contents = line.split(':')[i].strip()
            break

    return line_contents
