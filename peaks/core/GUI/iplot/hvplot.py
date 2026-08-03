"""Thin wrapper around hvplot to set some defaults and automate handling pint dequantify."""

import os

import xarray as xr


def _thin_for_docs(da, kwargs, max_frames=5):
    """Keep at most `max_frames` slices along each dim hvplot will animate over."""
    import numpy as np

    groupby = kwargs.get("groupby")
    if groupby:
        dims = [groupby] if isinstance(groupby, str) else list(groupby)
    else:
        # No explicit groupby passed. Infer dims to animate over if x or y is passed
        if kwargs.get("x") is None or kwargs.get("y") is None:
            return da
        mapped = {kwargs.get(k) for k in ("x", "y", "by", "col", "row")}
        dims = [d for d in da.dims if d not in mapped]

    idx = {}
    for d in dims:
        n = da.sizes.get(d)
        if n and n > max_frames:
            idx[d] = np.unique(np.linspace(0, n - 1, max_frames).round().astype(int))
    return da.isel(idx) if idx else da


@xr.register_dataarray_accessor("iplot")
class HVPlotAccessor:
    """Thin wrapper around hvplot to handle default options and pint dequantify."""

    def __init__(self, xarray_obj):
        import hvplot.xarray  # noqa: F401

        self._obj = xarray_obj

    def __call__(self, *args, **kwargs):
        """Dequantify the underlying object and return an interactive hvplot."""
        is_building_docs = os.getenv("FORCE_NB_EXECUTION") == "1"
        obj = self._obj.pint.dequantify()

        if is_building_docs:  # 1: in a docs build, see build_docs.sh
            kwargs.setdefault(
                "dynamic", False
            )  # static frames only - embedded plots have no kernel to compute them
            obj = _thin_for_docs(obj, kwargs)  # only chunk 5 frames in during docs build

        plot = obj.hvplot(*args, **kwargs)

        if is_building_docs:
            import panel as pn

            pane = pn.panel(plot)
            if isinstance(pane, pn.pane.HoloViews):
                pane = pane.layout  # bare pane omits the auto-generated widgets
            return pane.embed(max_opts=20)
        return plot
