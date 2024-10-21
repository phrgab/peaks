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

# Set some default xarray options
import xarray as xr
from datatree import DataTree

# Temporarily define xr.DataTree method until it is moved properly into the xarray codebase
xr.DataTree = DataTree

# Set default xarray options
xr.set_options(
    cmap_sequential="binary",
    use_numbagg=True,
    display_expand_attrs=False,
    display_expand_data=False,
)

# Register a dask progressbar with a minimum 1 second time
from dask.diagnostics import ProgressBar as dask_prog_bar

dask_prog_bar(minimum=1).register()

# Enable pint accessor and sete default options
import pint_xarray

ureg = pint_xarray.unit_registry
ureg.formatter.default_format = (
    "~P"  # Set formatting option to short form (with symbols)
)

# Import the core functions that should be accessible from the main peaks namespace
from peaks.core.options import opts

# Register the relevant data loaders and load method
from peaks.core.fileIO.loc_registry import LOC_REGISTRY
from peaks.core.fileIO.loc_registry import locs
from peaks.core.fileIO.base_data_classes import BaseIBWDataLoader
from peaks.core.fileIO.base_arpes_data_classes import BaseSESDataLoader
from peaks.core.fileIO.loaders import *
from peaks.core.fileIO.data_loading import load
from peaks.core.fileIO.data_saving import save

# Register the relevant accessor functions
from peaks.core.metadata.history import History
from peaks.core.utils.datatree_utils import list_scans
