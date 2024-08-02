"""Functions to load data from the APE beamline at Elettra.

"""

# Liam Trzaska 15/04/2020
# Brendan Edwards 15/06/2021
# Phil King 24/07/2022
# Brendan Edwards 28/02/2024

from peaks.core.fileIO.loaders.SES import _load_SES_data, _load_SES_metalines, _SES_find
from ..fileIO_opts import _BaseARPESConventions, _register_location


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

    # Currently all Elettra APE measurements are SES format, so use the _load_SES_data function to load the data
    data = _load_SES_data(fname)

    # If the data is an ibw file, the scan type will not yet be identified
    if data["scan_type"] is None:
        # If scan_no is a dimension in data, the scan was run in add dimension mode. Thus, update scan type to a
        # multiple scan dispersion
        if "scan_no" in list(data):
            data["scan_type"] = "dispersion (multiple scans)"

        # If not set the scan type to a dispersion by default
        else:
            data["scan_type"] = "dispersion"

    return data


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

    # Extract the lines containing metadata
    metalines = _load_SES_metalines(fname)

    # Define dictionary to store metadata
    metadata = {}

    # Extract the scan name from the file full path
    fname_split = fname.split("/")
    metadata["scan_name"] = fname_split[len(fname_split) - 1].split(".")[0]

    # Define initial attributes
    metadata["scan_type"] = scan_type
    metadata["sample_description"] = None
    metadata["eV_type"] = "kinetic"
    metadata["beamline"] = _APEConventions.loc_name
    metadata["analysis_history"] = []
    metadata["EF_correction"] = None

    # Define attributes, using the _SES_find function to obtain metadata where possible
    metadata["PE"] = float(_SES_find(metalines, "Pass Energy"))
    hv = _SES_find(metalines, "Excitation Energy")
    hv = hv.replace(",", ".")
    metadata["hv"] = round(float(hv), 2)
    metadata["pol"] = None
    metadata["sweeps"] = int(_SES_find(metalines, "Number of Sweeps"))
    metadata["dwell"] = float(_SES_find(metalines, "Step Time")) / 1000
    metadata["ana_mode"] = _SES_find(metalines, "Lens Mode")
    metadata["ana_slit"] = None
    metadata["ana_slit_angle"] = 90
    metadata["exit_slit"] = None
    metadata["defl_par"] = None
    try:
        defl_Y_high = _SES_find(metalines, "Thetay_High")
        defl_Y_low = _SES_find(metalines, "Thetay_Low")
        defl_delta = _SES_find(metalines, "Thetay_StepSize")
        metadata["defl_perp"] = defl_Y_low + ":" + defl_Y_high + " (" + defl_delta + ")"
    except UnboundLocalError:
        metadata["defl_perp"] = None

    # Define other attributes that are not saved in the metadata
    metadata["x1"] = None
    metadata["x2"] = None
    metadata["x3"] = None
    metadata["polar"] = None
    metadata["tilt"] = None
    metadata["azi"] = None
    metadata["norm_polar"] = None
    metadata["norm_tilt"] = None
    metadata["norm_azi"] = None
    metadata["temp_sample"] = None
    metadata["temp_cryo"] = None

    return metadata


@_register_location
class _APEConventions(_BaseARPESConventions):
    loc_name = "Elettra APE"
    loader = _load_APE_data
    metadata_loader = _load_APE_metadata
    azi_name = None
    defl_par_name = "theta_x"
    defl_perp_name = "theta_y"
