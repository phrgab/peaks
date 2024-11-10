"""Functions to load data from the Artemis facility at the Central Laser Facility.

"""

from peaks.core.options import _BaseARPESConventions, _register_location

# Phil King 14/10/2022
# Brendan Edwards 09/02/2024


def _load_Artemis_data(fname):
    """This function loads data that was obtained at the Artemis beamline at the Central Laser Facility.

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

        from peaks.core.fileIO.loaders.Artemis import _load_Artemis_data

        fname = 'C:/User/Documents/Research/scan1'

        # Extract data from file obtained at the Artemis beamline at the Central Laser Facility.
        data = _load_Artemis_data(fname)

    """

    raise Exception("CLF Artemis is not currently supported")


def _load_Artemis_metadata(fname, scan_type):
    """This function loads metadata from data that was obtained at the Artemis beamline at the Central Laser Facility.

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

        from peaks.core.fileIO.loaders.Artemis import _load_Artemis_metadata

        fname = 'C:/User/Documents/Research/scan1'

        # Extract metadata from file obtained at the Artemis beamline at the Central Laser Facility
        metadata = _load_Artemis_metadata(fname)

    """

    raise Exception("CLF Artemis is not currently supported")


@_register_location
class _ArtemisConventions(_BaseARPESConventions):
    loc_name = "CLF Artemis"
    loader = _load_Artemis_data
    metadata_loader = _load_Artemis_metadata
    ana_type = "II"
    tilt_name = None
    x1_name = "X"
    x2_name = "Z"
    x3_name = "Y"
