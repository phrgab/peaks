"""Functions to load data from the Scienta SES software.

"""

# Phil King 21/07/22
# Brendan Edwards 28/02/2024

import os
import zipfile
import numpy as np
from peaks.core.fileIO.loaders.ibw import _load_ibw_data, _load_ibw_wavenote
from peaks.utils import analysis_warning


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
    if file_extension == ".txt":
        data = _load_SES_txt_data(fname)
    elif file_extension == ".zip":
        data = _load_SES_zip_data(fname)
    elif file_extension == ".ibw":
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
    scan_type = "dispersion"

    # Open the file and extract the lines containing metadata (used to extract theta_par values, and determine when scan
    # data starts)
    metadata_lines = _load_SES_metalines(fname)

    # Load file data
    file_data = np.loadtxt(fname, skiprows=len(metadata_lines))
    spectrum = file_data[:, 1:].T
    KE_values = file_data[:, 0]
    theta_par_values = np.fromstring(
        _SES_find(metadata_lines, "Dimension 2 scale="), sep=" "
    )

    # Define data
    data = {
        "scan_type": scan_type,
        "spectrum": spectrum,
        "theta_par": theta_par_values,
        "eV": KE_values,
    }

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
    scan_type = "FS map"

    # Open the file and load the data
    with zipfile.ZipFile(fname) as z:
        files = z.namelist()
        file_bin = [file for file in files if ".bin" in file]
        file_ini = [file for file in files if "Spectrum_" in file and ".ini" in file]
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
            KE_step = float(_SES_find(lineText, "widthdelta="))
            KE_end = KE_start + (KE_step * (num_KE - 1))
            KE_values = np.linspace(KE_start, KE_end, num_KE)

            # Extract theta_par axis
            num_theta_par = int(_SES_find(lineText, "height="))
            theta_par_start = float(_SES_find(lineText, "heightoffset="))
            theta_par_step = float(_SES_find(lineText, "heightdelta="))
            theta_par_end = theta_par_start + (theta_par_step * (num_theta_par - 1))
            theta_par_values = np.linspace(
                theta_par_start, theta_par_end, num_theta_par
            )

            # Extract deflector axis
            num_defl_perp = int(_SES_find(lineText, "depth="))
            defl_perp_start = float(_SES_find(lineText, "depthoffset="))
            defl_perp_step = float(_SES_find(lineText, "depthdelta="))
            defl_perp_end = defl_perp_start + (defl_perp_step * (num_defl_perp - 1))
            defl_perp_values = np.linspace(
                defl_perp_start, defl_perp_end, num_defl_perp
            )

        # Extract spectrum and reshape into a data cube to be consistent with loading
        with z.open(filename, "r") as f:
            spectrum = np.frombuffer(f.read(), dtype=np.dtype(np.float32))
            spectrum = spectrum.reshape(
                num_KE, num_theta_par, num_defl_perp, order="F"
            ).T

    # Define data
    data = {
        "scan_type": scan_type,
        "spectrum": spectrum,
        "defl_perp": defl_perp_values,
        "theta_par": theta_par_values,
        "eV": KE_values,
    }

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
        Dictionary containing the file scan type, spectrum, and coordinates. At this stage, the scan type
        will not be identified, and the coordinate names may not all be in peaks format.

    Examples
    ------------
    Example usage is as follows::

        from peaks.core.fileIO.loaders.SES import _load_SES_ibw_data

        fname = 'C:/User/Documents/Research/disp1.ibw'

        # Extract data from an SES format file
        data = _load_SES_ibw_data(fname)

    """

    # Use the _load_ibw_data function to load the data (at this stage, the scan type will not be identified, and the
    # coordinate names will not be in peaks format.)and metalines
    data, metalines = _load_ibw_data(fname)

    # Loop through and update the contents of data so that the dimension names are in peaks format
    for dim in list(data):
        # Replace kinetic energy with eV
        if "kinetic" in dim.lower():
            # Rename dimension to eV
            data["eV"] = data.pop(dim)

            # Ensure eV is arranged in increasing eV
            if data["eV"][0] > data["eV"][1]:
                data["eV"] = np.flip(data["eV"])

        # Attempt to replace binding energy with kinetic energy
        elif "binding" in dim.lower():
            # If the metadata has this in KE; extract from there

            if _SES_find(metalines, "Energy Unit") == "Kinetic":
                E0 = _SES_find(metalines, "Low Energy")
                data[dim] = data[dim] + float(E0) - data[dim][0]

            # If not, give a warning that the data energy axis has been loaded as binding energy
            else:
                analysis_warning(
                    "Data energy axis has been loaded as binding energy.",
                    title="Loading info",
                    warn_type="danger",
                )

            # Rename dimension to eV
            data["eV"] = data.pop(dim)

            # Ensure eV is arranged in increasing eV
            if data["eV"][0] > data["eV"][1]:
                data["eV"] = np.flip(data["eV"])

        # Should be an hv scan, rename photon energy axis to hv and define a KE_delta coordinate
        elif "photon" in dim.lower() or "hv" in dim.lower():
            # Get the change in KE value of detector as a function of hv
            data["KE_delta"] = data[dim] - data[dim][0]

            # Rename dimension to hv
            data["hv"] = data.pop(dim)

        # Replace deg with theta_par
        elif "deg" in dim.lower():
            # Rename dimension to theta_par
            data["theta_par"] = data.pop(dim)

        # Replace scale with y_scale
        elif "scale" in dim.lower():
            # Rename dimension to y_scale
            data["y_scale"] = data.pop(dim)

        # Replace iteration with scan_no
        elif "iteration" in dim.lower():
            # Redefine the iteration coordinates to ensure they are 1, 2, 3, ... (shockingly this is not a guarantee)
            data[dim] = np.array(range(len(data[dim]))) + 1

            # Rename dimension to scan_no
            data["scan_no"] = data.pop(dim)

        # Any other dimensions require beamline-specific loaders to identify, so rename any remaining dimensions to
        # generic dimension names
        elif "point" in dim.lower():
            # Rename dimension to dim0
            data["dim0"] = data.pop(dim)

        elif "position" in dim.lower():
            # Rename dimension to dim1
            data["dim1"] = data.pop(dim)

    return data


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

        from peaks.core.fileIO.loaders.SES import _load_SES_metalines

        fname = 'C:/User/Documents/Research/disp1.zip'

        # Extract the lines containing metadata
        lines = _load_SES_metalines(fname)

    """
    # Get file_extension to determine the file type
    filename, file_extension = os.path.splitext(fname)

    # If the file is .txt format
    if file_extension == ".txt":
        # Open the file and extract the lines containing metadata
        with open(fname) as f:
            metadata_lines = []
            while True:
                line = f.readline()
                metadata_lines.append(line)
                # When the line starting with '[Data' is encountered, metadata has ended and scan data has begun
                if "[Data" in line:
                    break

    # If the file is .zip format
    elif file_extension == ".zip":
        # Open the file and extract the lines containing metadata
        with zipfile.ZipFile(fname) as z:
            files = z.namelist()
            file_ini = [
                file for file in files if "Spectrum_" not in file and ".ini" in file
            ]
            filename = file_ini[0]
            with z.open(filename) as f:
                lines_to_decode = f.readlines()
                metadata_lines = [line.decode() for line in lines_to_decode]

    # If the file is .ibw format
    elif file_extension == ".ibw":
        # Extract the lines containing metadata from the wavenote of the ibw file using the _load_ibw_wavenote function
        metadata_lines = _load_ibw_wavenote(fname).split("\r")

    return metadata_lines


def _load_SES_manipulator_coords(metadata_lines, unique=True):
    """This function will extract manipulator coordinates in an SES format manipulator scan.

    Parameters
    ------------
    metadata_lines : list
        Lines extracted from the file containing the metadata.

    unique : bool, optional
        Whether to only return unique coordinate values, or all coordinate values during a scan.

    Returns
    ------------
    manipulator_coords : dict
        Dictionary containing the manipulator coordinates, where each key is an axis dimension and each entry is a
        numpy.ndarray.

    Examples
    ------------
    Example usage is as follows::

        from peaks.core.fileIO.loaders.SES import _load_SES_metalines, _load_SES_manipulator_coords

        fname = 'C:/User/Documents/Research/SM1.ibw'

        # Extract the lines containing metadata
        metadata_lines = _load_SES_metalines(fname)

        # Extract the unique manipulator coordinates
        manipulator_coords = _load_SES_manipulator_coords(metadata_lines)

        # Extract the all manipulator coordinates during a scan
        manipulator_coords = _load_SES_manipulator_coords(metadata_lines, unique=False)

    """

    # Define a dictionary to store the manipulator coordinates
    manipulator_coords = {}

    # Determine the location of the manipulator coordinates information in the metadata_lines
    manip_index = metadata_lines.index("[Run Mode Information]") + 1
    if "Name" in metadata_lines[manip_index]:
        manip_index += 1

    # Extract the axes dimensions
    for dim in metadata_lines[manip_index].split("\x0b"):
        manipulator_coords[dim] = []

    # Loop through the rest of the data and extract the manipulator coordinates
    manip_index += 1
    while True:
        try:
            current_coordinates = metadata_lines[manip_index].split("\x0b")
            for i, dim in enumerate(manipulator_coords):
                manipulator_coords[dim].append(float(current_coordinates[i]))
            manip_index += 1
        except ValueError:
            break

    # Convert from lists to np arrays (extracting only unique values if requested)
    for dim in manipulator_coords:
        if unique:
            manipulator_coords[dim] = np.unique(np.array(manipulator_coords[dim]))
        else:
            manipulator_coords[dim] = np.array(manipulator_coords[dim])

    return manipulator_coords


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
        metalines = _load_SES_metalines(fname)

        # Extract the analyser mode
        ana_mode = _SES_find(metalines, 'Analyzer Lens')

    """

    # Loop over lines to extract the line starting with the desired keyword.
    for line in lines:
        if line.startswith(item) and "=" in line:
            line_contents = line.split("=")[-1].strip()
            break
        elif line.startswith(item) and ":" in line:
            line_contents = line.split(":")[-1].strip()
            break

    return line_contents
