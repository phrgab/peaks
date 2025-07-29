# ruff: noqa: I001
# ruff: noqa: F401

# Import xarray Class-based accessors
from peaks.core.metadata.history import History
from peaks.core.metadata.metadata_methods import Metadata
from peaks.core.GUI.iplot.hvplot import HVPlotAccessor
from peaks.core.fitting.quick_fit import QuickFit

# Import direct accessor methods
from peaks.core.accessors.dataarray_accessors import *  # noqa: F403
from peaks.core.accessors.dataset_accessors import *  # noqa: F403
from peaks.core.accessors.datatree_accessors import *  # noqa: F403
