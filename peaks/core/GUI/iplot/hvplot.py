"""Thin wrapper around hvplot to set some defaults and automate handling pint dequantify"""

import xarray as xr


@xr.register_dataarray_accessor("iplot")
class HVPlotAccessor:
    """Very thin wrapper around `hvplot`. See documentation at https://hvplot.holoviz.org/"""

    def __init__(self, xarray_obj):
        from holoviews import opts as hv_opts
        import hvplot.xarray  # noqa: F401

        # Set default options
        hv_opts.defaults(hv_opts.Image(invert_axes=True))

        self._obj = xarray_obj

    def __call__(self, *args, **kwargs):
        # Call the original hvplot accessor with the provided arguments
        return self._obj.pint.dequantify().hvplot(*args, **kwargs)
