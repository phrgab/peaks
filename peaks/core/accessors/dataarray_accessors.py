import xarray as xr

from peaks.core.accessors.accessor_methods import register_lazy_accessor

# List of function names and the module they belong to
functions_to_register = {
    "display.plotting": [
        "plot_DCs",
        "plot_3d_stack",
        "plot_fit_test",
        "plot_nanofocus",
    ],
    "fileIO.data_saving": [
        "save",
    ],
    "fitting.fit": ["fit", "fit_gold", "estimate_EF"],
    "GUI.disp_panels.disp": ["disp"],
    "process.data_select": [
        "drop_nan_borders",
        "drop_zero_borders",
        "DC",
        "MDC",
        "EDC",
        "DOS",
        "tot",
        "radial_cuts",
        "extract_cut",
        "mask_data",
        "disp_from_hv",
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
        "estimate_sym_point",
    ],
}


# Register each function as a lazy accessor on DataArray
for module_name, func_names in functions_to_register.items():
    for func_name in func_names:
        accessor_property = register_lazy_accessor(
            func_name, f"peaks.core.{module_name}", xr.DataArray
        )
