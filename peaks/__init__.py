'''peaks: Python Electron spepctroscopy and diffraction Analysis by King group St Andrews.

peaks is a collection of analysis tools for the loading, processing and display of spectroscopic
and diffraction data, with a core focus on tools for angle-resolved photoemission, electron
energy loss spectroscopy, LEED, RHEED, and some other related techniques. It also includes
various functions for efficient log keeping and experimental system monitoring.

It has been developed by the King group in St Andrews, with the following core
developers: Tommaso Antonelli, Brendan Edwards, Edgar Abarca Morales, Lewis Hart, Phil King, and Andela Zivanovic.

Copyright the above authors. Contact pdk6@st-andrews.ac.uk for further information, to contribute,
and for bug reporting.

For documentation, please see /docs/index.html in root folder.

**Installation:**

Suggest to create a dedicated virtual environment.
If using conda, in a terminal (or windows equivalent) type:

    `conda create –-name peaks python=3.10`
    `conda activate peaks`

Then set current directory to the package root directory and run::

    `pip install -e .`

in the terminal. Dependencies will be installed automatically.


**Documentation:**

To make (or update) an .html version of the documentation, in the terminal
set the directory to ``/docs`` run ``make html``.

This puts a set of .html files in the `docs` folder of the package root directory.

**Basic Usage:**

Typically, run in a jupyter notebook or equivalent. To import a core set of functions run::

    from peaks import *

See the individual module and function descriptions for detailed information about the usage,
and follow the example Notebooks for tutorials.

'''

# Import some core packages to namespace
import numpy as np
import xarray as xr
import dask
import matplotlib.pyplot as plt
import gc

# # Import core functions from peaks modules to namespace
# from peaks.core.fileIO import *
# from peaks.core.process import *
# from peaks.core.fit import *
# from peaks.core.display import *
# #from peaks.ptab import *
# #from .LEED import *  #TODO Need to update LEED to PyQt6
# #from peaks.logbook import *
# from peaks.core.GUI import *
# #from peaks.ML import *
#
# # Import some useful utilities (also to avoid circular imports)
# from peaks.utils.estimate_EF import *
# from peaks.utils.align_util import *
# from peaks.utils.sum_spectra import *
# #from peaks.utils.structure import structure, crystal, get_structure
# from peaks.utils.plotting import colours, publish
# from peaks.utils.nano_slant_correct import slant_correct

# Set default cmap option
xr.set_options(cmap_sequential='Purples')