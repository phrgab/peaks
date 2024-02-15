"""Functions to load data from the Artemis facility at Diamond Light Source.

"""

# Phil King 14/10/2022
# Brendan Edwards 09/02/2024


def _load_Artemis_data(fname):
    """This function loads data that was obtained at the Artemis beamline at Diamond Light Source.

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
    from peaks.core.fileIO.loaders.Artemis import _load_Artemis_data

    fname = 'C:/User/Documents/Research/scan1'

    # Extract data from file obtained at the Artemis beamline at Diamond Light Source.
    data = _load_Artemis_data(fname)

    """

    raise Exception('Diamond Artemis is not currently supported')


def _load_Artemis_metadata(fname, scan_type):
    """This function loads metadata from data that was obtained at the Artemis beamline at Diamond Light Source.

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
    from peaks.core.fileIO.loaders.Artemis import _load_Artemis_metadata

    fname = 'C:/User/Documents/Research/scan1'

    # Extract metadata from file obtained at the Artemis beamline at Diamond Light Source
    metadata = _load_Artemis_metadata(fname)

    """

    raise Exception('Diamond Artemis is not currently supported')
