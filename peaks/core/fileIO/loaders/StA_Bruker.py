"""Functions to load data from the XRD system (Bruker diffractometer) at St Andrews.

"""

# Phil King 08/06/2021
# Brendan Edwards 22/02/2024

import numpy as np
from peaks.utils.consts import consts


def _load_StA_Bruker_data(fname):
    """This function loads data that was obtained using the XRD system (Bruker diffractometer) at St Andrews.

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

        from peaks.core.fileIO.loaders.StA_Bruker import _load_StA_Bruker_data

        fname = 'C:/User/Documents/Research/scan1.xy'

        # Extract data from file obtained using the XRD system (Bruker diffractometer) at St Andrews
        data = _load_StA_Bruker_data(fname)

    """

    # Load file data (we expect a 1D dataset)
    file_data = np.loadtxt(fname, skiprows=1)
    mapping_coords = file_data[:, 0]
    spectrum = file_data[:, 1]

    # Open the file and read the first line (only one line of metadata)
    with open(fname) as f:
        metaline = f.readline()

    # Determine the scan type
    scan_type = metaline.split('Scantype: "')[1].split('"')[0].split()[0]

    # Depending on scan_type, define mapping_label and rename scan_type to be consistent with peaks (either 'phi scan',
    # 'reflectivity', or '2Theta-Omega')
    if scan_type == "Phi-H":
        mapping_label = "phi"
        scan_type = "phi scan"
    else:
        mapping_label = "two_theta"
        if scan_type == "2Theta-Omega":
            if mapping_coords.max() < 10:
                # If maximum angle is small, reflectivity seems a good guess for scan type
                scan_type = "reflectivity scan"
            else:
                scan_type = "2Theta-Omega scan"
        elif scan_type == "TwoTheta":
            if "Integrate frame" in metaline:
                # This should be a 2Th-Om scan integrated from the detector
                scan_type = "2Theta-Omega scan"
        else:
            raise Exception("The scan type could not be identified.")

    # Define data
    data = {"scan_type": scan_type, "spectrum": spectrum, mapping_label: mapping_coords}

    return data


def _load_StA_Bruker_metadata(fname, scan_type):
    """This function loads metadata from data that was obtained using the XRD system (Bruker diffractometer) at
    St Andrews.

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

        from peaks.core.fileIO.loaders.StA_Bruker import _load_StA_Bruker_metadata

        fname = 'C:/User/Documents/Research/scan1.xy'

        # Extract metadata from file obtained using the XRD system (Bruker diffractometer) at St Andrews
        metadata = _load_StA_Bruker_metadata(fname)

    """

    # Open the file and read the first line (only one line of metadata)
    with open(fname) as f:
        metaline = f.readline()

    # Define dictionary to store metadata
    metadata = {}

    # Extract the scan name from the file full path
    fname_split = fname.split("/")
    metadata["scan_name"] = fname_split[len(fname_split) - 1].split(".")[0]

    # Define attributes
    metadata["scan_type"] = scan_type
    metadata["sample_description"] = None
    metadata["beamline"] = "StA-Bruker"
    metadata["analysis_history"] = []
    metadata["anode"] = metaline.split('Anode: "')[1].split('"')[0]
    if metadata["anode"] == "Cu":
        # If the copper anode is used, we know the wavelength of X-ray
        metadata["wavelength"] = (
            consts.Cu_Ka_lambda
        )  # Cu K-alpha wavelength in units of Angstroms
    else:
        metadata["wavelength"] = None
    metadata["dwell"] = float(metaline.split('TimePerStep: "')[1].split('"')[0])
    metadata["temp_sample"] = "Room temperature"

    return metadata
