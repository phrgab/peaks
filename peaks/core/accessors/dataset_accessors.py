import xarray as xr

from peaks.core.accessors.accessor_methods import register_lazy_accessor

# List of function names and the module they belong to
functions_to_register = {
    "fileIO.data_saving": ["save"],
    "display.plotting": ["plot_fit"],
    "fitting.fit": ["save_fit"],
}

# Register each function as a lazy accessor on Dataset
for module_name, func_names in functions_to_register.items():
    for func_name in func_names:
        accessor_property = register_lazy_accessor(
            func_name, f"peaks.core.{module_name}", xr.Dataset
        )
