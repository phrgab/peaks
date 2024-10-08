"""Functions to load data from the Bloch beamline at MAX IV Laboratory.

"""

# Liam Trzaska 15/04/2020
# Brendan Edwards 15/06/2021
# Phil King 24/07/2022
# Brendan Edwards 29/02/2024

import numpy as np
from peaks.core.fileIO.loaders.SES import (
    _load_SES_data,
    _load_SES_metalines,
    _load_SES_manipulator_coords,
    _SES_find,
)
from ..fileIO_opts import _BaseARPESConventions, _register_location
from ..base_arpes_data_classes import BaseSESDataLoader
from ..loc_registry import register_loader


@register_loader
class BlochArpesLoader(BaseSESDataLoader):
    _loc_name = "MAXIV_Bloch_A"
    _loc_description = "MAX IV Bloch beamline A branch"
    _loc_url = "https://www.maxiv.lu.se/beamlines-accelerators/beamlines/bloch/"

    _manipulator_name_conventions = {
        "polar": "P",
        "tilt": "T",
        "azi": "A",
        "x1": "X",
        "x2": "Y",
        "x3": "Z",
    }
    _SES_metadata_units = {
        f"manipulator_{dim}": ("mm" if dim in ["x1", "x2", "x3"] else "deg")
        for dim in _manipulator_name_conventions.keys()
    }


def _load_Bloch_data(fname):
    """This function loads data that was obtained at the A-branch of the Bloch beamline at MAX IV Laboratory.

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

    # Currently all Bloch A-branch are SES format, so use the _load_SES_data function to load the data
    data = _load_SES_data(fname)

    # If the data is an ibw file, the scan type will not yet be identified, and the coordinate names may not be in peaks
    # format. Ensure these issue are resolved
    if data["scan_type"] is None:
        # Set the scan type to a dispersion by default
        data["scan_type"] = "dispersion"

        # If scan_no is a dimension in data, the scan was run in add dimension mode. Thus, update scan type to a
        # multiple scan dispersion
        if "scan_no" in list(data):
            data["scan_type"] = "dispersion (multiple scans)"

        # If hv is a dimension in data, update the scan type to an hv scan
        elif "hv" in list(data):
            data["scan_type"] = "hv scan"

        # If dim0 is a dimension in data, we have at least one unidentified mapping coordinate
        elif "dim0" in list(data):
            # Define Bloch to peaks naming conventions
            Bloch_to_peaks_conventions = {
                "X": "x1",
                "Y": "x2",
                "Z": "x3",
                "P": "polar",
                "T": "tilt",
                "A": "azi",
            }

            # Extract manipulator coordinates information to help identify mapping coordinate(s)
            metadata_lines = _load_SES_metalines(fname)
            manipulator_coords = _load_SES_manipulator_coords(
                metadata_lines, unique=True
            )

            # Loop through the dimensions in manipulator_coords, and extract the mapping dimensions
            mapping_dims = []
            for dim in manipulator_coords:
                # If any of the coordinates along a dimension varies by more than 0.01, append dimension to mapping_dims
                if manipulator_coords[dim].max() - manipulator_coords[dim].min() > 0.01:
                    if dim in list(Bloch_to_peaks_conventions):
                        mapping_dims.append(dim)

            # If dim1 is a dimension in data, we have a 4D scan which will almost certainly be a spatial map
            if "dim1" in list(data):
                # Update the scan type to a spatial map
                data["scan_type"] = "spatial map"

                # Redefine the mapping coordinates
                data["dim0"] = manipulator_coords[mapping_dims[0]]
                data["dim1"] = manipulator_coords[mapping_dims[1]]

                # Rename the mapping dimensions using peaks conventions
                data[Bloch_to_peaks_conventions[mapping_dims[0]]] = data.pop("dim0")
                data[Bloch_to_peaks_conventions[mapping_dims[1]]] = data.pop("dim1")

                # Ensure the spectrum is shaped x2 then x1, so that if we have equal x2 and x1 sizes, the DataArray
                # generated in _make_DataArray correctly assumes x2 is first dimension. The slow axis is the first part
                # of the spectrum shape, so if x1 is the slow axis, flip the x1 and x2 ordering of spectrum so that x2
                # is first

                # Extract all manipulator coords during a scan (not just the unique values)
                all_manipulator_coords = _load_SES_manipulator_coords(
                    metadata_lines, unique=False
                )

                # If there is a negligible change between the first and second coordinate during a scan, for a
                # dimension, that will be the fast axis
                if (
                    abs(
                        all_manipulator_coords[mapping_dims[0]][1]
                        - all_manipulator_coords[mapping_dims[0]][0]
                    )
                    < 0.001
                ):
                    slow_axis = Bloch_to_peaks_conventions[mapping_dims[0]]
                else:
                    slow_axis = Bloch_to_peaks_conventions[mapping_dims[1]]

                # If x1 is the slow axis, flip the x1 and x2 ordering of spectrum so that x2 is first
                if slow_axis == "x1":
                    data["spectrum"] = np.transpose(data["spectrum"], (1, 0, 2, 3))

            # If dim0 is present and not dim1, we have a 3D scan with one mapping dimension
            else:
                # Extract mapping dimension
                mapping_dim = mapping_dims[0]

                # If the mapping dimension is polar or tilt, we have a Fermi map
                if mapping_dim == "P" or mapping_dim == "T":
                    data["scan_type"] = "FS map"

                # If not, we should have a line scan
                elif mapping_dim == "X" or mapping_dim == "Y" or mapping_dim == "Z":
                    data["scan_type"] = "line scan"

                # Redefine the mapping coordinates
                data["dim0"] = manipulator_coords[mapping_dim]

                # Rename the mapping dimensions using peaks conventions
                data[Bloch_to_peaks_conventions[mapping_dim]] = data.pop("dim0")

    return data


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
    metadata["beamline"] = _BlochAConventions.loc_name
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

    # Attempt to extract sample position coordinates using the function _load_SES_manipulator_coords
    try:
        manipulator_coords = _load_SES_manipulator_coords(metalines, unique=True)
        metadata["x1"] = _Bloch_find(manipulator_coords["X"])
        metadata["x2"] = _Bloch_find(manipulator_coords["Y"])
        metadata["x3"] = _Bloch_find(manipulator_coords["Z"])
        metadata["polar"] = _Bloch_find(manipulator_coords["P"])
        metadata["tilt"] = _Bloch_find(manipulator_coords["T"])
        metadata["azi"] = _Bloch_find(manipulator_coords["A"])

    # Attempt to extract sample position metadata using the _SES_find function
    except (ValueError, KeyError):
        try:
            metadata["x1"] = float(_SES_find(metalines, "X="))
        except (UnboundLocalError, ValueError):
            metadata["x1"] = None
        try:
            metadata["x2"] = float(_SES_find(metalines, "Y="))
        except (UnboundLocalError, ValueError):
            metadata["x2"] = None
        try:
            metadata["x3"] = float(_SES_find(metalines, "Z="))
        except (UnboundLocalError, ValueError):
            metadata["x3"] = None
        try:
            metadata["polar"] = float(_SES_find(metalines, "P="))
        except (UnboundLocalError, ValueError):
            metadata["polar"] = None
        try:
            metadata["tilt"] = float(_SES_find(metalines, "T="))
        except (UnboundLocalError, ValueError):
            metadata["tilt"] = None
        try:
            metadata["azi"] = float(_SES_find(metalines, "A="))
        except (UnboundLocalError, ValueError):
            metadata["azi"] = None

    # Define other attributes that are not saved in the metadata
    metadata["norm_polar"] = None
    metadata["norm_tilt"] = None
    metadata["norm_azi"] = None
    metadata["temp_sample"] = None
    metadata["temp_cryo"] = None

    return metadata


def _Bloch_find(coords):
    """Helper function to extract coordinate metadata from a SES format manipulator scan obtained at the A-branch of
    the Bloch beamline at MAX IV Laboratory.

    Parameters
    ------------
    coords : numpy.ndarray
        The coordinates of a given manipulator axis during a manipulator scan.

    Returns
    ------------
    extracted_coord : float, str, NoneType
        The extracted coordinate(s) of a given manipulator axis.

    Examples
    ------------
    Example usage is as follows::

        from peaks.core.fileIO.loaders.SES import _load_SES_metalines, _load_SES_manipulator_coords
        from peaks.core.fileIO.loaders.Bloch import _Bloch_find

        fname = 'C:/User/Documents/Research/SM1.ibw'

        # Extract the lines containing metadata
        metadata_lines = _load_SES_metalines(fname)

        # Extract the manipulator coordinates
        manipulator_coords = _load_SES_manipulator_coords(metadata_lines, unique=True)

        # Extract the X coordinates
        extracted_coord = _Bloch_find(manipulator_coords['X'])

    """

    from peaks.core.fileIO.data_loading import _extract_mapping_metadata

    # If there is a non-negligible step between coordinate values, extract the coordinate metadata using a
    # 'min:max (step)' format (rounding to 3 decimal places)
    if abs(coords[-1] - coords[0]) > 0.001:
        extracted_coord = _extract_mapping_metadata(coords, num_dp=3)
    # If the coordinate values are constant, extract the first value
    else:
        extracted_coord = float(coords[0])

    return extracted_coord


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

    raise Exception("MAX IV Bloch-spin is not currently supported")


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

    raise Exception("MAX IV Bloch-spin is not currently supported")


@_register_location
class _BlochAConventions(_BaseARPESConventions):
    loc_name = "MAX IV Bloch"
    loader = _load_Bloch_data
    metadata_loader = _load_Bloch_metadata
    ana_type = "Ip"
    polar_name = "P"
    tilt_name = "T"
    azi_name = "A"
    defl_par_name = "theta_x"
    defl_perp_name = "theta_y"
    x1_name = "X"
    x2_name = "Y"
    x3_name = "Z"


@_register_location
class _BlochBConventions(_BaseARPESConventions):
    loc_name = "MAX IV Bloch-spin"
    loader = _load_Bloch_spin_data
    metadata_loader = _load_Bloch_spin_metadata
    ana_type = "Ip"
    polar_name = "P"
    tilt_name = "T"
    azi_name = "A"
    defl_par_name = "theta_x"
    defl_perp_name = "theta_y"
    x1_name = "X"
    x2_name = "Y"
    x3_name = "Z"
