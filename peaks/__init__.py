"""peaks: (P)ython (E)lectron Spectroscopy and Diffraction (A)nalysis by (K)ing Group (S)t Andrews.

peaks is a collection of analysis tools for the loading, processing and display of spectroscopic and diffraction data,
with a core focus on tools for angle-resolved photoemission and electron diffraction techniques.

Sub-packages
----------
(see peaks.<sub-package> for more information)
- core: Core functionality for data loading, processing and display.
- ML: Machine learning tools for data analysis.
- diffraction: Tools for electron and x-ray diffraction data loading and analysis.
- SARPES: Dedicated tools for analysis of spin- and angle-resolved photoemission spectroscopy.
- TRARPES: Dedicated tools for analysis of time-resolved angle-resolved photoemission spectroscopy.
- XMCD: Dedicated tools for analysis of x-ray absorption and magnetic circular dichroism spectroscopy.
- structure: Helper functions for crystal structure loading and BZ plotting.
- utils: Miscellaneous helper functions and classes.

Modules
----------
- _lazy_import.py: Helper functions for lazy importing of modules.

Usage
----------
It is recommended to import peaks as follows::

    import peaks as pks

"""

# Import core packages to namespace
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# Import peaks modules to namespace
from peaks.core import *
from peaks.ML import *

# Set some default xarray options
xr.set_options(cmap_sequential="binary", use_numbagg=True)

# Import hvplot to enable the hvplot accessor and set some default options
import holoviews as hv
import hvplot.xarray

# Set default options
hv.opts.defaults(hv.opts.Image(invert_axes=True))

# Register a dask progressbar with a minimum 1 second time
from dask.diagnostics import ProgressBar as dask_prog_bar

dask_prog_bar(minimum=1).register()
