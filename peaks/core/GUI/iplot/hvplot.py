"""Thin wrapper around hvplot to set some defaults and automate handling pint dequantify"""

import xarray as xr


@xr.register_dataarray_accessor("iplot")
class HVPlotAccessor:
    """Thin wrapper around hvplot to handle default options and pint dequantify."""

    def __init__(self, xarray_obj):
        import hvplot.xarray  # noqa: F401

        self._obj = xarray_obj

    def __call__(self, *args, **kwargs):
        # Generate the plot
        plot = self._obj.pint.dequantify().hvplot(*args, **kwargs)
        return plot
