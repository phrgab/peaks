"""Master function for interactive display panels.

"""

import xarray as xr
from peaks.core.utils.OOP_method import add_methods
from peaks.core.GUI.disp_panels.disp_2d import _disp_2d
from peaks.core.GUI.disp_panels.disp_3d import _disp_3d
from peaks.core.GUI.disp_panels.disp_4d import _disp_4d


@add_methods(xr.DataArray)
def disp(data, primary_dim=None, exclude_from_centering="eV"):
    """GUI data viewer for 2D, 3D, or 4D data.

    Parameters
    ------------
    data : list or xarray.DataArray
         Either a single 2D, 3D, or 4D :class:`xarray.DataArray` or a list of 2D :class:`xarray.DataArray` objects.

    primary_dim : str or tuple of str or list of str, optional
        The primary dimension for the viewer, should be a single dim for 2D and 3D plots or a tuple or list of two
        dims for a 4D plot. Primary dim will be shown:
        - on the y-axis for 2D plots
        - as the central panel for 3D plots
        - as the primary data explorer panel for 4D plots
        Default behaviour is based on the conventional peaks data structure for ARPES data.

    exclude_from_centering : str or tuple of str or list of str or None, optional
        The dimension to exclude from centering for 2D and 3D data. Default is 'eV'.

    Examples
    ------------
    Example usage is as follows::

        import peaks as pks

        # To display a single dispersion
        # Load the dispersion
        disp = load('disp.ibw')

        # Display it in the UI
        disp.disp()


        # To display a set of 2D dispersions with theta_par axis on the vertical axis
        # Load the dispersions
        disp1 = load('disp1.ibw')
        disp2 = load('disp2.ibw')
        disp3 = load('disp3.ibw')
        disp4 = load('disp4.ibw')

        # Make a list
        disps_to_plot = [disp1, disp2, disp3, disp4]

        # Open dispersions in viewer
        pks.disp(disps_to_plot, primary_dim='theta_par')


        # To display a Fermi surface map
        FS_map = load('FS_map.nxs')
        FS_map.disp()

    """

    if isinstance(data, list):
        for array in data:
            if isinstance(array, xr.DataArray):
                if len(array.dims) != 2:
                    raise ValueError(
                        "Unexpected dimensionality: only 2D xr.DataArrays are supported "
                        "for passing to `.disp` in a list."
                    )
            else:
                raise ValueError(
                    "Unexpected data type: only 2D xr.DataArrays are supported "
                    "for passing to `.disp` in a list."
                )
        _disp_2d(
            data, primary_dim=primary_dim, exclude_from_centering=exclude_from_centering
        )

    elif isinstance(data, xr.DataArray):
        if 1 in data.shape:  # Check if there is a dummy axis that can be squeezed
            data = data.squeeze()
        ndim = len(data.dims)
        if ndim == 2:
            _disp_2d(
                [data],
                primary_dim=primary_dim,
                exclude_from_centering=exclude_from_centering,
            )
        elif ndim == 3:
            _disp_3d(
                data,
                primary_dim=primary_dim,
                exclude_from_centering=exclude_from_centering,
            )
        elif ndim == 4:
            _disp_4d(data, primary_dim=primary_dim)
        else:
            raise ValueError(
                "Number of dimensions not supported for interactive display. "
                "Only 2D, 3D, or 4D xr.DataArrays are supported."
            )
    else:
        raise ValueError(
            "Data type not currently supported for interactive display. Only single 2D, 3D, or 4D xr.DataArrays or"
            "a list of 2D xr.DataArrays are supported."
        )
