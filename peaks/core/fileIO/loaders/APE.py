"""Functions to load data from the APE beamline at Elettra.

"""

# Liam Trzaska 15/04/2020
# Brendan Edwards 15/06/2021
# Phil King 24/07/2022
# Brendan Edwards 08/02/2024


def _load_APE_data(fname):
    """This function loads data that was obtained at the APE beamline at Elettra.

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

        from peaks.core.fileIO.loaders.APE import _load_APE_data

        fname = 'C:/User/Documents/Research/disp1.ibw'

        # Extract data from file obtained at the APE beamline at Elettra
        data = _load_APE_data(fname)

    """

    raise Exception('Elettra APE is not currently supported')


def _load_APE_metadata(fname, scan_type):
    """This function loads metadata from data that was obtained at the APE beamline at Elettra.

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

        from peaks.core.fileIO.loaders.APE import _load_APE_metadata

        fname = 'C:/User/Documents/Research/disp1.ibw'

        # Extract metadata from file obtained at the APE beamline at Elettra
        metadata = _load_APE_metadata(fname)

    """

    raise Exception('Elettra APE is not currently supported')
