"""Helper functions for acting on metadata.

"""

# Phil King 20/04/2021
# Brendan Edwards 16/10/2023

import xarray as xr
from peaks.utils.OOP_method import add_methods


@add_methods(xr.DataArray)
def update_hist(data, hist):
    """Updates the analysis history metadata of the supplied DataArray.

    Parameters
    ------------
    data : xarray.DataArray
        The :class:`xarray.DataArray` for which the analysis history is to be updated.

    hist : string
        Analysis history to be appended to DataArray analysis history metadata.

    Returns
    ------------
    data : xarray.DataArray
        The :class:`xarray.DataArray` with updated analysis history.

    Examples
    ------------
    Example usage is as follows::

        from peaks import *

        disp = load('disp.ibw')

        disp = disp/2

        # Update the analysis history metadata of the dispersion
        disp = disp.update_hist('Dispersion data divided by 2')

    """

    # If analysis history is not in the DataArray attributes, define it
    if "analysis_history" not in data.attrs:
        data.attrs["analysis_history"] = []

    # Update the DataArray analysis history (the following is done so that the original DataArray is not overwritten)
    analysis_history = []
    for item in data.attrs["analysis_history"]:
        analysis_history.append(item)
    analysis_history.append(hist)
    data.attrs["analysis_history"] = analysis_history

    return data


@add_methods(xr.DataArray)
def _set_normals(data, **angle_kwargs):
    """Function to set normal emissions into the attributes of DataArrays based on arguments passed in angle_kwargs.

    Parameters
    ------------
    data: xarray.DataArray
        The :class:`xarray.DataArray` to write normal emission angles to.

    **angle_kwargs : float, optional
        Attributes to be overwritten in the :class:`xarray.DataArray` in the format attr=float. Valid for polar, tilt,
        azi, norm_polar, norm_tilt and norm_azi. All other angle_kwargs are ignored.

    Examples
    ------------
    Example usage is as follows::

        from peaks import *

        disp = load('disp.ibw')

        # Sets the attributes of the dispersion
        disp._set_normals(polar=0, tilt=0, azi=0, norm_polar=0, norm_tilt=0, norm_azi=0)

    """

    if "polar" in angle_kwargs:
        data.attrs["polar"] = angle_kwargs["polar"]  # Set polar attribute of metadata
    if "tilt" in angle_kwargs:
        data.attrs["tilt"] = angle_kwargs["tilt"]  # Set tilt attribute of metadata
    if "azi" in angle_kwargs:
        data.attrs["azi"] = angle_kwargs["azi"]  # Set azi attribute of metadata
    if "norm_polar" in angle_kwargs:
        data.attrs["norm_polar"] = angle_kwargs[
            "norm_polar"
        ]  # Set norm_polar attribute of metadata
    if "norm_tilt" in angle_kwargs:
        data.attrs["norm_tilt"] = angle_kwargs[
            "norm_tilt"
        ]  # Set norm_tilt attribute of metadata
    if "norm_azi" in angle_kwargs:
        data.attrs["norm_azi"] = angle_kwargs[
            "norm_azi"
        ]  # Set norm_azi attribute of metadata
