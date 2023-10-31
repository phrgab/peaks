"""peaks: (P)ython (E)lectron Spectroscopy and Diffraction (A)nalysis by (K)ing Group (S)t Andrews.

peaks is a collection of analysis tools for the loading, processing and display of spectroscopic and diffraction data,
with a core focus on tools for angle-resolved photoemission and electron diffraction techniques. It also includes
various functions for efficient log keeping and experimental system monitoring.

It has been developed by the King group in St Andrews, with contributions from the following group members: Phil King,
Brendan Edwards, Tommaso Antonelli, Edgar Abarca Morales, Liam Trzaska, Lewis Hart and Andela Zivanovic

This version of peaks is the result of a complete restructuring of the package that was completed in late 2023 by
Brendan Edwards and Phil King.

Copyright the above authors. Contact pdk6@st-andrews.ac.uk for further information, to contribute,
and for bug reporting.

For documentation, please see /docs/index.html in root folder.


*** Installation ***

    Suggest to create a dedicated virtual environment.
    If using conda, in a terminal/prompt type:

        conda create –-name peaks python=3.12
        conda activate peaks

    Then set current directory to the package root directory and run:

        pip install -e

    in the terminal. Dependencies will be installed automatically.


*** Documentation ***

    To make (or update) an .html version of the documentation, in the terminal
    set the directory to `/docs` run `make html`.

    This puts a set of .html files in the `docs` folder of the package root directory.


*** Basic Usage ***

    Typically, run in a jupyter notebook or equivalent. To import peaks run:

        from peaks import *

    See the individual module and function descriptions for detailed information about the usage,
    and follow the example notebooks for tutorials.

"""

# Import core packages to namespace
import gc
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# Import core modules from to namespace
from peaks.core import *
from peaks.ML import *

# Set default cmap option
xr.set_options(cmap_sequential='binary')
