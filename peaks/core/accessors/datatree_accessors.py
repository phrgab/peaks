import xarray as xr

from peaks.core.accessors.accessor_methods import (
    PeaksDataTreeIteratorAccessor,
    register_lazy_accessor,
)

# List of function names (and the module they belong to) to directly register the accessor on DataTree
functions_to_register_for_direct_accessor = {
    "fileIO.data_saving": [
        "save",
    ],
    "GUI.disp_panels.disp": ["disp"],
    "utils.datatree_utils": ["view", "add_scan_group", "add", "get_DataArray"],
    "display.plotting": ["plot_grid", "plot_DCs"],
    "process.tools": ["sum_data", "subtract_data", "merge_data"],
}


# List of functions to register where the function should map over the tree
functions_to_register_for_iterable_accessor = {
    "process.data_select": [
        "DC",
        "MDC",
        "EDC",
        "DOS",
        "tot",
    ],
    "process.differentiate": [
        "deriv",
        "d2E",
        "d2k",
        "dEdk",
        "dkdE",
        "curvature",
        "min_gradient",
    ],
    "process.k_conversion": ["k_convert"],
    "process.tools": [
        "norm",
        "bgs",
        "bin_data",
        "bin_spectra",
        "smooth",
        "rotate",
        "sym",
        "sym_nfold",
        "degrid",
    ],
}


# Register each function as a lazy accessor on DataTree
for module_name, func_names in functions_to_register_for_direct_accessor.items():
    for func_name in func_names:
        accessor_property = register_lazy_accessor(
            func_name, f"peaks.core.{module_name}", xr.DataTree
        )

for module_name, func_names in functions_to_register_for_iterable_accessor.items():
    for func_name in func_names:
        accessor_property = register_lazy_accessor(
            func_name,
            f"peaks.core.{module_name}",
            xr.DataTree,
            accessor_class=PeaksDataTreeIteratorAccessor,
        )
