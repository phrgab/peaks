"""Functions to load data from the Bloch beamline at MAX IV Laboratory.

"""

# Liam Trzaska 15/04/2020
# Brendan Edwards 15/06/2021
# Phil King 24/07/2022
# Brendan Edwards 08/02/2024


def _load_Bloch_data(fname):
    """This function loads data that was obtained at the A branch of the Bloch beamline at MAX IV Laboratory.

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

        from peaks.core.fileIO.loaders.Bloch import _load_Bloch_data

        fname = 'C:/User/Documents/Research/disp1.ibw'

        # Extract data from file obtained at the A branch of the Bloch beamline at MAX IV Laboratory
        data = _load_Bloch_data(fname)

    """

    raise Exception('MAX IV Bloch is not currently supported')


def _load_Bloch_metadata(fname, scan_type):
    """This function loads metadata from data that was obtained at the A branch of the Bloch beamline at MAX IV
    Laboratory.

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

        from peaks.core.fileIO.loaders.Bloch import _load_Bloch_metadata

        fname = 'C:/User/Documents/Research/disp1.ibw'

        # Extract metadata from file obtained at the A branch of the Bloch beamline at MAX IV Laboratory
        metadata = _load_Bloch_metadata(fname)

    """

    raise Exception('MAX IV Bloch-spin is not currently supported')


def _load_Bloch_spin_data(fname):
    """This function loads data that was obtained at the spin branch of the Bloch beamline at MAX IV Laboratory.

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

        from peaks.core.fileIO.loaders.Bloch import _load_Bloch_spin_data

        fname = 'C:/User/Documents/Research/disp1.ibw'

        # Extract data from file obtained at the spin branch of the Bloch beamline
        data = _load_Bloch_spin_data(fname)

    """

    raise Exception('MAX IV Bloch-spin is not currently supported')


def _load_Bloch_spin_metadata(fname, scan_type):
    """This function loads metadata from data that was obtained at the spin branch of the Bloch beamline at MAX IV
    Laboratory.

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

        from peaks.core.fileIO.loaders.Bloch import _load_Bloch_spin_metadata

        fname = 'C:/User/Documents/Research/disp1.ibw'
    
        # Extract metadata from file obtained at the spin branch of the Bloch beamline
        metadata = _load_Bloch_spin_metadata(fname)

    """

    raise Exception('MAX IV Bloch-spin is not currently supported')
