"""General data loading helper functions.

"""

# Phil King 15/05/2021
# Brendan Edwards 26/02/2024

import numpy as np
import xarray as xr
import h5py
from termcolor import colored
from peaks.core.fileIO.data_loading import _make_DataArray, _extract_mapping_metadata
from peaks.core.utils.accessors import register_accessor
from peaks.core.utils.misc import analysis_warning


def make_hv_scan(data):
    """Function to combine multiple dispersions (measured at different hv values) into a single hv scan DataArray.

    Parameters
    ------------
    data : list
        Any number of :class:`xarray.DataArray` dispersions (measured at different hv values) to combine into an hv scan.

    Returns
    ------------
    hv_scan : xarray.DataArray
        The resulting hv scan DataArray.

    Examples
    ------------
    Example usage is as follows::

        from peaks import *

        disp_70eV = load('disp70eV.ibw')
        disp_72eV = load('disp72eV.ibw')
        disp_74eV = load('disp74eV.ibw')
        disp_76eV = load('disp76eV.ibw')
        disp_78eV = load('disp78eV.ibw')
        disp_80eV = load('disp80eV.ibw')

        # Combine dispersions (measured at different hv values) into a single hv scan DataArray
        hv_scan = make_hv_scan([disp_70eV, disp_72eV, disp_74eV, disp_76eV, disp_78eV, disp_80eV])


    """

    # Define the new scan type
    scan_type = "hv scan"

    # Ensure the dispersions are arranged in order of increasing hv
    hvs_and_disps = []
    for disp in data:
        hvs_and_disps.append([float(disp.hv), disp])
    hvs_and_disps.sort()

    # Extract the data spectrum and the hv, KE and theta_par coordinates
    spectrum = []
    hv_values = []
    KE_values = []
    for hv, disp in hvs_and_disps:
        # Want theta_par as the first dimension to get the expected spectrum shape in the _make_DataArray function
        if disp.dims[0] == "eV":
            spectrum.append(disp.data.T)
        else:
            spectrum.append(disp.data)
        hv_values.append(hv)
        KE_values.append(disp.coords["eV"].data)
    theta_par_values = data[0].theta_par.data

    # We want to save the kinetic energy coordinates of the first scan, and also the corresponding offsets for
    # successive scans (KE_delta)
    KE0 = np.array(KE_values)[
        :, 0
    ]  # Get  the maximum kinetic energy of the scan as a function of photon energy
    KE_delta = (
        KE0 - KE0[0]
    )  # Get the change in KE value of detector as a function of hv

    # Define dictionary to be sent to the _make_DataArray function to make a xarray.DataArray
    data_dict = {
        "scan_type": scan_type,
        "spectrum": np.array(spectrum),
        "hv": hv_values,
        "theta_par": theta_par_values,
        "eV": KE_values[0],
        "KE_delta": KE_delta,
    }

    # Make an hv scan DataArray
    hv_scan = _make_DataArray(data_dict)

    # Ensure that the dimensions of the hv scan are arranged in the standard order
    hv_scan = hv_scan.transpose(
        "hv", "eV", "defl_par", "theta_par", "k_par", missing_dims="ignore"
    )

    # Add metadata to the new hv scan DataArray
    hv_scan.attrs = data[0].attrs
    hv_scan.attrs["scan_name"] = "Manual hv_scan"
    hv_scan.attrs["scan_type"] = "hv scan"
    hv_scan.attrs["hv"] = _extract_mapping_metadata(hv_values, num_dp=2)

    # Update analysis history
    hv_scan.history.add("Combined multiples dispersions into an hv scan")

    # Display warning explaining how kinetic energy values are saved
    warn_str = (
        "The kinetic energy coordinates saved are that of the first scan. The corresponding offsets "
        "for successive scans are included in the KE_delta coordinate. Run DataArray.disp_from_hv(hv), "
        "where DataArray is the loaded hv scan xarray.DataArray and hv is the relevant photon energy, "
        "to extract a dispersion at using the proper kinetic energy scaling for that photon energy."
    )
    analysis_warning(warn_str, title="Loading info", warn_type="info")

    return hv_scan


@register_accessor(xr.DataArray)
def slant_correct(data, factor=None):
    """Function to remove a slant that was present in data obtained using the Scienta DA30 (9ES210) analyser at the nano
    branch of the I05 beamline at Diamond Light Source in 2021/22.

    Parameters
    ------------
    data : xarray.DataArray
        The data to be corrected.

    factor : float, optional
        The slant factor correction (degrees/eV) to use. Defaults to 8/PE (where PE is the pass energy used).

    Returns
    ------------
    corrected_data : xarray.DataArray
        The corrected data.

    Examples
    ------------
    Example usage is as follows::

        from peaks import *

        disp = load('i05-1-13579.ibw')

        # Remove the slant present in the dispersion, using the default slant factor
        corrected_disp = disp.slant_correct()

    """

    # Set the slant factor to the default correction if it has not been supplied
    if factor is None:
        factor = 8 / data.attrs["PE"]

    # Define the new angle mapping
    theta_par_values = data.theta_par - (factor * (data.eV - data.eV.median()))

    # Perform the interpolation onto the corrected grid
    corrected_data = data.interp({"theta_par": theta_par_values, "eV": data.eV})

    # Update the analysis history
    corrected_data.history.add(
        "Slant correction for Diamond nano-ARPES data applied: {factor} deg/eV".format(
            factor=factor
        )
    )

    return corrected_data


def _print_hdf5_structure(
    name, obj, parent_group, indent_level=0, is_last=False, branch=""
):
    """Recursive function to print the structure of the HDF5 file with colored keys and default (black) data, and indentation lines.

    Parameters
    ----------
    name : str
        The name of the current object.

    obj : h5py.Group, h5py.Dataset, h5py.SoftLink, h5py.ExternalLink
        The object to explore.

    parent_group : h5py.Group
        The parent group of the current object.

    indent_level : int, optional
        The current indentation level of the object.

    is_last : bool, optional
        Whether the current object is the last child of the parent group.

    branch : str, optional
        The current branch of the object.

    """
    # Color definitions
    group_color = "cyan"
    dataset_color = "yellow"
    attribute_color = "green"
    link_color = "magenta"
    grey_line = colored("|-- ", "light_grey")

    # Set the line connector based on whether the node is the last child
    connector = "└── " if is_last else "├── "

    # Build the branch line for current level
    new_branch = branch + ("    " if is_last else "│   ")

    if isinstance(obj, h5py.Group):
        # Print groups in cyan
        group_name = colored(
            f"{name.split('/')[-1]}:NX{obj.attrs.get('NX_class', 'unknown')}",
            group_color,
        )
        print(f"{branch}{connector}{group_name}")

        # Recursively explore the contents of the group
        keys = list(obj.keys())
        for i, key in enumerate(keys):
            sub_obj = obj[key]
            _print_hdf5_structure(
                key,
                sub_obj,
                obj,
                indent_level + 1,
                is_last=(i == len(keys) - 1),
                branch=new_branch,
            )

    elif isinstance(obj, h5py.Dataset):
        # Print datasets in yellow, but display data in default black if large data is compacted
        data_info = (
            f"shape={obj.shape}, dtype={obj.dtype}" if obj.size > 10 else f"{obj[()]}"
        )
        dataset_name = colored(f"{name.split('/')[-1]} =", dataset_color)
        print(f"{branch}{connector}{dataset_name} {data_info}")

        # Print dataset attributes in green, but display attribute values in default black
        for attr_name, attr_value in obj.attrs.items():
            attr_display = colored(f"@{attr_name} =", attribute_color)
            print(f"{new_branch}  {attr_display} {attr_value}")

    elif isinstance(obj, h5py.SoftLink) or isinstance(obj, h5py.ExternalLink):
        # Handle links (even though we said we'd resolve them, this is here if needed)
        link_name = colored(f"{name.split('/')[-1]} -> {obj.path} (Link)", link_color)
        print(f"{branch}{connector}{link_name}")


def hdf5_explorer(file_path):
    """Function to explore the structure of an HDF5 file, printing the keys, groups, datasets and attributes.

    Parameters
    ----------
    file_path : str
        The path to the HDF5 file to explore.

    Examples
    --------
    Example usage is as follows::

        import peaks as pks

        # Explore the structure of an HDF5 file
        pks.hdf5_explorer('data.h5')

    Notes
    -----
    Colored output is used to distinguish between groups, datasets, attributes and links:
        - Cyan: For groups (NX classes like NXentry, NXdata, etc.)
        - Yellow: For datasets (the actual data arrays)
        - Green: For attributes (metadata attached to datasets or groups, typically prefixed with @)
        - Magenta: For soft links (references to other datasets or groups within the file)
        - Black: For data values (the actual contents of datasets and attributes)

    Note, soft links may resolve automatically and so not show as links in the output.
    """
    with h5py.File(file_path, "r") as f:
        # Manually explore the root level items
        keys = list(f.keys())
        for i, key in enumerate(keys):
            obj = f[key]
            _print_hdf5_structure(key, obj, f, is_last=(i == len(keys) - 1))


@register_accessor(xr.DataArray)
def display_metadata(da):
    # Recursive function to display dictionary with colored keys
    colours = ["green", "blue", "red", "yellow"]

    # Recursive function to display dictionary with cycling colors for each indent level
    def display_colored_dict(d, indent_level=0, col_cycle=0):
        indent = "    " * indent_level
        current_color = colours[col_cycle % len(colours)]  # Cycle through colors
        for key, value in d.items():
            if isinstance(value, dict):  # Nested dictionary (recursive case)
                print(f"{indent}{colored(key, current_color)}:")
                display_colored_dict(value, indent_level + 1, col_cycle + 1)
            else:  # Base case (simple value)
                print(f"{indent}{colored(key, current_color)}: {value}")

    # Display the model with colored keys
    metadata = {
        key: value.dict()
        for key, value in da.attrs.items()
        if key not in ["analysis_history"]
    }
    display_colored_dict(metadata)
