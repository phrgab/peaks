"""Thin wrapper around hvplot to set some defaults and automate handling pint dequantify."""

import os

import xarray as xr


@xr.register_dataarray_accessor("iplot")
class HVPlotAccessor:
    """Thin wrapper around hvplot to handle default options and pint dequantify."""

    def __init__(self, xarray_obj):
        import hvplot.xarray  # noqa: F401

        self._obj = xarray_obj

    def __call__(self, *args, **kwargs):
        # Generate the plot
        if (
            os.getenv("FORCE_NB_EXECUTION") == "1"
        ):  # 1: in a docs build see build_docs.sh
            kwargs.setdefault(
                "dynamic", False
            )  # force static plot to avoid issues with dynamic plots but locally keep dynamic=True (default) for performance
        plot = self._obj.pint.dequantify().hvplot(*args, **kwargs)
        return plot
