"""Functions to load data from the spin-ARPES system (MBS analyser) at St Andrews.

"""

# Brendan Edwards 01/03/2021
# Phil King 04/06/2021
# Brendan Edwards 09/02/2024


def _load_StA_MBS_data(fname):
    """This function loads data that was obtained using the spin-ARPES system (MBS analyser) at St Andrews.

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
    from peaks.core.fileIO.loaders.StA_MBS import _load_StA_MBS_data

    fname = 'C:/User/Documents/Research/disp1.krx'

    # Extract data from file obtained using the spin-ARPES system (MBS analyser) at St Andrews
    data = _load_StA_MBS_data(fname)

    """

    raise Exception('StA_MBS is not currently supported')


def _load_StA_MBS_metadata(fname, scan_type):
    """This function loads metadata from data that was obtained using the spin-ARPES system (MBS analyser) at
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
    from peaks.core.fileIO.loaders.StA_MBS import _load_StA_MBS_metadata

    fname = 'C:/User/Documents/Research/disp1.krx'

    # Extract metadata from file obtained using the spin-ARPES system (MBS analyser) at St Andrews
    metadata = _load_StA_MBS_metadata(fname)

    """

    raise Exception('StA_MBS is not currently supported')
