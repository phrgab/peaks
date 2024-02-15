"""Functions to load data from the LOREA beamline at ALBA.

"""

# Phil King 04/01/2023
# Brendan Edwards 09/02/2024


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
    from peaks.core.fileIO.loaders.LOREA import _load_LOREA_data

    fname = 'C:/User/Documents/Research/disp1.nxs'

    # Extract data from file obtained at the LOREA beamline at ALBA
    data = _load_LOREA_data(fname)

    """

    raise Exception('ALBA LOREA is not currently supported')


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
    from peaks.core.fileIO.loaders.LOREA import _load_LOREA_metadata

    fname = 'C:/User/Documents/Research/disp1.nxs'

    # Extract metadata from file obtained at the LOREA beamline at ALBA
    metadata = _load_LOREA_metadata(fname)

    """

    raise Exception('ALBA LOREA is not currently supported')
