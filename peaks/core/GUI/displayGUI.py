# Master function for passing data to interactive display panels
# Phil King 24/10/21

import xarray as xr
import dask

from peaks.utils.OOP_method import add_methods
from .display_panel.disp2d_GUI import disp2d
from .spatial_map.disp4d_GUI import disp4d
from .align.align_GUI import align

@add_methods(xr.DataArray)
def disp(data):
    ''' Master load function for GUI data display panels. Currently works with single 2D dataarray,
    a list of 2D dataarrays, and 4D dataarray of spatial mapping data.

        Input:
            data - the data to display

        Returns:
            null - opens the relevant data display panel '''


    if type(data) is list:
        disp2d(dask.compute(*data))
    else:
        if 1 in data.shape:  # Check if there is a dummy axis that can be squeezed
            data = data.squeeze()
        ndim = len(data.dims)
        if ndim == 2:
            disp2d([data.compute()])
        elif ndim == 3:
            align(data)
        elif ndim == 4:
            disp4d(data.compute())
        else:
            print('Data type not currently supported for interactive display')