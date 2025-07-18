"""General data loading helper functions."""

# Phil King 15/05/2021
# Brendan Edwards 26/02/2024

import numpy as np

from peaks.core.fileIO.data_loading import _extract_mapping_metadata, _make_DataArray
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
    KE_delta = KE0 - KE0[0]  # Get the change in KE value of detector as a function of hv

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
