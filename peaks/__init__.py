"""peaks: (P)ython (E)lectron Spectroscopy and Diffraction (A)nalysis by (K)ing Group (S)t Andrews.

peaks is a collection of analysis tools for the loading, processing and display of spectroscopic and diffraction data,
with a core focus on tools for angle-resolved photoemission and electron diffraction techniques.

Usage
----------
It is recommended to import peaks as follows::

    import peaks as pks

To load data, use the load function::

    data = pks.load('data.ibw')

"""

# Import peaks modules to namespace
from peaks.core import *
from peaks.ML import *

# Set some default xarray options
from xarray import set_options

set_options(cmap_sequential="binary", use_numbagg=True)

# Import hvplot to enable the hvplot accessor and set some default options
from holoviews import opts as hv_opts
import hvplot.xarray

# Set default options
hv_opts.defaults(hv_opts.Image(invert_axes=True))

# Register a dask progressbar with a minimum 1 second time
from dask.diagnostics import ProgressBar as dask_prog_bar

dask_prog_bar(minimum=1).register()
