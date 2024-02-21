"""Functions to load data from the XRD system (Bruker diffractometer) at St Andrews.

"""

# Phil King 08/06/2021
# Brendan Edwards 09/02/2024


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

    raise Exception('StA_Bruker is not currently supported')


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

    raise Exception('StA_Bruker is not currently supported')
