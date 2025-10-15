# ruff: noqa

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

__version__ = "0.4.4"

# Set some default xarray options
import xarray as xr

# Set default xarray options
xr.set_options(
    cmap_sequential="binary",
    use_numbagg=True,
    display_expand_attrs=False,
    display_expand_data=False,
)

# # Register a progressbar for use during dask compute calls
from peaks.core.utils.misc import DaskTQDMProgressBar

DaskTQDMProgressBar(desc="Evaluating deferred computation", minimum=1.0).register()

# Enable pint accessor and set default options
import pint_xarray

ureg = pint_xarray.unit_registry
ureg.formatter.default_format = (
    "~P"  # Set formatting option to short form (with symbols)
)

# Register the relevant data loaders and load method (do this first to get the loc_registry populated)
from peaks.core.fileIO.loc_registry import LOC_REGISTRY
from peaks.core.fileIO.loc_registry import locs
from peaks.core.fileIO.base_data_classes.base_ibw_class import BaseIBWDataLoader
from peaks.core.fileIO.base_arpes_data_classes.base_ses_class import SESDataLoader
from peaks.core.fileIO.base_arpes_data_classes.base_fesuma_class import (
    BaseFeSuMaDataLoader,
)
from peaks.core.fileIO.base_arpes_data_classes.base_mbs_class import MBSDataLoader
from peaks.core.fileIO.loaders import *

# Import the core functions that should be accessible from the main peaks namespace
from peaks.core.fileIO.data_loading import load
from peaks.core.fitting.fit import load_fit
from peaks.core.options import opts
from peaks.core.display.plotting import (
    plot_grid,
    plot_DCs,
    plot_ROI,
    plot_kpar_cut,
    plot_kz_cut,
    plot_nanofocus,
)
from peaks.core.process.tools import sum_data, subtract_data, merge_data

# Register the relevant accessor functions
from peaks.core.accessors import *
