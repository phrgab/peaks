import pint_xarray
import xarray as xr

from peaks.core.accessors.accessor_methods import (
    _pass_function_to_xarray_class_accessor,
)

ureg = pint_xarray.unit_registry


@xr.register_datatree_accessor("tr")
@xr.register_dataarray_accessor("tr")
class TRAccessors:
    """Helper functions for time-resolved data, accessed via the `.tr` accessor."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    mean = _pass_function_to_xarray_class_accessor(
        "mean", "peaks.time_resolved.data_select"
    )
    static = _pass_function_to_xarray_class_accessor(
        "static", "peaks.time_resolved.data_select"
    )
    diff = _pass_function_to_xarray_class_accessor(
        "diff", "peaks.time_resolved.data_select"
    )
    set_t0 = _pass_function_to_xarray_class_accessor(
        "set_t0", "peaks.time_resolved.utils"
    )
    set_t0_like = _pass_function_to_xarray_class_accessor(
        "set_t0_like", "peaks.time_resolved.utils"
    )
    assign_t0 = _pass_function_to_xarray_class_accessor(
        "assign_t0", "peaks.time_resolved.utils"
    )
    assign_t0_like = _pass_function_to_xarray_class_accessor(
        "assign_t0_like", "peaks.time_resolved.utils"
    )
