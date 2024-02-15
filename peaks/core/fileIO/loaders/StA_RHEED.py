"""Functions to load data from the RHEED system at St Andrews.

"""

# Lewis Hart 28/01/2021
# Brendan Edwards 08/02/2024


def _load_StA_RHEED_data(fname):
    """This function loads data that was obtained using the RHEED system at St Andrews.

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
    from peaks.core.fileIO.loaders.StA_RHEED import _load_StA_RHEED_data

    fname = 'C:/User/Documents/Research/RHEED_VSe2.iso'

    # Extract data from file obtained using the RHEED system at St Andrews
    data = _load_StA_RHEED_data(fname)

    """

    raise Exception('StA_RHEED is not currently supported')


def _load_StA_RHEED_metadata(fname, scan_type):
    """This function loads metadata from data that was obtained using the RHEED system at St Andrews.

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
    from peaks.core.fileIO.loaders.StA_RHEED import _load_StA_RHEED_metadata

    fname = 'C:/User/Documents/Research/RHEED_VSe2.iso'

    # Extract metadata from file obtained using the RHEED system at St Andrews
    metadata = _load_StA_RHEED_metadata(fname)

    """

    raise Exception('StA_RHEED is not currently supported')
