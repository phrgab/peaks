import dask as da
import dill
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from tqdm.notebook import tqdm

from peaks.core.fitting.models import LinearDosFermiModel
from peaks.core.utils.misc import analysis_warning


def fit(
    data_array,
    model,
    params,
    independent_var=None,
    sequential=True,
    reverse_sequential_fit_order=False,
):
    """
    Fit an :class:`lmfit.Model` to an :class:`xarray.DataArray`, specifying the co-ordinate correspinding to
    the indepednent variable. Fit broadcasts along other dimensions. Returns the results of the fit as a
    :class:`xarray.DataArray` with uncertainties and the full :class:`lmfit.ModelResult`.

    Parameters:
    -----------
    data_array: xarray.DataArray
          The data to fit the model to.
    model: lmfit.Model
        The model to fit to the data.
    params: lmfit.Parameters
        Initial parameters for the model fitting.
    independent_var: str, optional
         The name of the dimension that represents the independent variable. If the data_array only has a single
         dimension, then this parameter can be omitted, but if the data_array has multiple dimensions, then it
         must be included.
    sequential: bool
        If True, and `data_array` is 2D, use the results of the previous fit to update the starting parameters
        for the next iteration. Assumes that the initial parameters are for the first slice
        (i.e. `data_array.isel(dim=0)` slice) along the non-independent dimension.
        Defaults to True.
        If False, fit the model to the entire data array at once.
    reverse_sequential_fit_order: bool
        Use to reverse the order of a sequential fit along the non-independent dimension. Defaults to False.

    Returns:
    -------
    xarray.DataSet
        A DataSet containing the best-fit parameters, their uncertainties, and the :class:`lmfit.ModelResult` object.
    """

    def fit_func(y, x, model, initial_params):
        result = model.fit(y, params=initial_params, x=x)
        best_values = np.array([result.params[param].value for param in result.params])
        uncertainties = np.array(
            [
                (
                    result.params[param].stderr
                    if result.params[param].stderr is not None
                    else np.nan
                )
                for param in result.params
            ]
        )
        return np.concatenate([best_values, uncertainties, [result]])

    # Dequantify the data array
    data_array = data_array.pint.dequantify()

    # Check independent_var is a valid dimension if supplied and set if None
    if independent_var is not None:
        if independent_var not in data_array.dims:
            raise ValueError(
                f"Independent variable {independent_var} is not a valid dimension of the data array."
                f"Expected one of {data_array.dims}"
            )
    else:
        if data_array.ndim != 1:
            raise ValueError(
                "Data array has multiple dimensions. Please specify the independent variable using the "
                "`independent_var` argument."
            )
        else:
            independent_var = data_array.dims[0]

    if data_array.ndim > 2 and sequential:
        sequential = False
        analysis_warning(
            "Sequential fitting only supported for 2D data. Defaulting to non-sequential.",
            title="Analysis info",
            warn_type="info",
        )
    # Sequential fitting, updating params each iteration
    if sequential and len(data_array.dims) == 2:
        if isinstance(data_array.data, da.array.core.Array):
            raise ValueError(
                "Dask arrays are not supported for sequential fitting. Either set `sequential=False` or load your "
                "data into memory first with ``.compute()``"
            )

        non_indep_dim = list(set(data_array.dims) - set([independent_var]))[0]
        non_indep_dim_len = data_array.sizes[non_indep_dim]

        if reverse_sequential_fit_order:
            # Reverse the order of the non-independent dimension
            data_array = data_array.isel({non_indep_dim: slice(None, None, -1)})

        fit_results = []
        # Iterate through all slices along the non-independent dimension
        for i in tqdm(range(non_indep_dim_len), desc="Fitting"):
            data_array_slice = data_array.isel({non_indep_dim: i})
            results_subset = xr.apply_ufunc(
                fit_func,
                data_array_slice,
                data_array.coords[independent_var],
                kwargs={"model": model, "initial_params": params},
                input_core_dims=[[independent_var], [independent_var]],
                output_core_dims=[["fit_params"]],
                vectorize=True,
                output_dtypes=[object],
                keep_attrs=False,
            )
            fit_results.append(results_subset)

            # Update the initial parameters for the next iteration
            params = results_subset.isel(fit_params=-1).item().params
        # Concatenate the results along the non-independent dimension into a single DataArray
        results = xr.concat(fit_results, dim=non_indep_dim)
    else:
        # Apply the fitting function across all dimensions except the independent variable
        results = xr.apply_ufunc(
            fit_func,
            data_array,
            data_array.coords[independent_var],
            kwargs={"model": model, "initial_params": params},
            input_core_dims=[[independent_var], [independent_var]],
            output_core_dims=[["fit_params"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[object],
            dask_gufunc_kwargs={
                "output_sizes": {"fit_params": len(params) * 2 + 1},
                "allow_rechunk": True,
            },
            keep_attrs=False,
        )

    # Create parameter names, adding "_stderr" for uncertainties, and "model_result" for the serialized data
    param_names = list(params.keys())
    all_param_names = (
        param_names + [f"{name}_stderr" for name in param_names] + ["fit_model"]
    )

    # Add the parameter names to the dataarray
    results = results.assign_coords({"fit_params": ("fit_params", all_param_names)})

    # Parse these as a dataset
    results_ds = xr.Dataset()

    # Iterate over all_param_names and assign each corresponding value from results to the new dataset,
    # ensuring dtype of float if not the fit model
    for i, param_name in enumerate(all_param_names):
        data_var = results.isel(fit_params=i).drop_vars("fit_params")
        if param_name != "fit_model":
            data_var = data_var.astype(np.float64)
        results_ds[param_name] = data_var

    results_ds.attrs["independent_var"] = independent_var
    return results_ds


def fit_gold(data, EF_correction_type="poly4", **kwargs):
    """
    Helper function for fitting a gold reference scan to a standard LinearDosFermiModel with parameters:
    - Fermi level (EF)
    - Temperature (T)
    - DOS slope (dos_slope)
    - DOS intercept (dos_intercept)
    - Background slope (bg_slope)
    - Background intercept (bg_intercept)
    - Gaussian convolution (sigma_conv)

    Parameters:
    -----------
    data_array : xarray.DataArray
        The gold reference data to fit.

    EF_correction_type : str, optional
        The type of Fermi level correction to determine for 2D data. Options are:
        - 'poly4' (default): Fit a 4th order polynomial to the extracted Fermi level from all slices along the
        non-independent dimension.
        - 'poly3': Fit a 3rd order polynomial to the extracted Fermi level values.
        - 'quadratic': Fit a quadratic to the extracted Fermi level values.
        - 'linear': Fit a linear function to the extracted Fermi level values.
        - 'average': Average the extracted Fermi level values from all slices

    **kwargs : optional
        Additional keyword arguments to initialise paramaeter values

    Returns:
    -------
    xarray.DataSet
        A DataSet containing the best-fit parameters, their uncertainties, and the :class:`lmfit.ModelResult` object.

    Examples:
    --------
    Example usage is as follows::

        import peaks as pks

        # Load the gold reference data
        gold_data = pks.load(gold_scan)

        # Fit the gold reference data, initialising the background slope to 0
        gold_fit = pks.fit_gold(gold_data, bg_slope=0)

    """
    data = data.pint.dequantify()
    if data.ndim > 2:
        raise ValueError("Expected 1D or 2D DataArray to be supplied.")
    if "eV" not in data.dims:
        raise ValueError("Supplied data must include a dimension of 'eV'")

    gold_model = LinearDosFermiModel()
    if data.ndim == 1:
        params = gold_model.guess(data, **kwargs)
        fit_result = data.fit(gold_model, params)
        fit_result.plot_fit(show_components=False)
        results_text = (
            f"Fermi energy: {str(np.round(fit_result['EF'].data, 3))} ± "
            f"{str(np.round(fit_result['EF_stderr'].data, 3))} eV,  "
            f"Resolution: {str(np.round(fit_result['sigma_conv'].data, 3))} ± "
            f"{str(np.round(fit_result['sigma_conv_stderr'].data, 3))} eV"
        )
        fit_result.attrs["EF_correction"] = float(fit_result["EF"])

    elif data.ndim == 2:
        other_dim = list(set(data.dims) - set(["eV"]))[0]
        first_slice = data.isel({other_dim: 0})
        params = gold_model.guess(first_slice, **kwargs)
        fit_result = data.fit(gold_model, params, independent_var="eV")

        # Fit the Fermi level correction
        if EF_correction_type == "poly4":
            _order = 4
        elif EF_correction_type == "poly3":
            _order = 3
        elif EF_correction_type == "quadratic":
            _order = 2
        elif EF_correction_type == "linear":
            _order = 1
        elif EF_correction_type == "average":
            _order = 0
        EF_correction_fit = fit_result["EF"].quick_fit.poly(_order)

        if EF_correction_type == "average":
            EF_correction = float(EF_correction_fit["c0"])
            results_text = (
                f"Average Fermi level correction: {np.round(EF_correction, 3)}"
            )
        else:
            EF_correction = {
                f"c{i}": float(EF_correction_fit[f"c{i}"]) for i in range(_order + 1)
            }
            results_text = (
                f"Polynomial Fermi level correction (order {_order}): "
                + ", ".join(
                    [f"{key}: {np.round(val, 6)}" for key, val in EF_correction.items()]
                )
            )

        fit_result.attrs["EF_correction"] = EF_correction
        res_av = np.round(fit_result["sigma_conv"].mean().data, 3)
        results_text += f";   Average resolution: {res_av} eV"

        # Make some plots
        fig = plt.figure(figsize=(8, 5))
        gs = fig.add_gridspec(3, 2, width_ratios=[1, 1], height_ratios=[1, 2, 2])
        ax1 = fig.add_subplot(gs[:, 0])
        data.plot(ax=ax1, add_colorbar=False)
        fit_result["EF"].plot(ax=ax1)
        ax1.set_ylabel("eV")

        ax2 = fig.add_subplot(gs[0, 1])
        EF_correction_fit.fit_model.data[()].plot_residuals(ax=ax2)
        ax2.xaxis.set_visible(False)
        ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)
        EF_correction_fit.fit_model.data[()].plot_fit(ax=ax3)
        ax3.xaxis.set_visible(False)
        ax3.set_title("")
        ax3.set_ylabel("Fermi level (eV)")
        ax4 = fig.add_subplot(gs[2, 1], sharex=ax2)
        fit_result["sigma_conv"].plot(ax=ax4)
        ax4.set_title("Resolution")
        ax4.axhline(res_av, color="r", ls="--")

        plt.tight_layout()
        plt.show()

    analysis_warning(results_text, "success", "Au fitting results")
    return fit_result


def _estimate_EF(y, x):
    """
    Estimate the Fermi level from the first negative peak in the first derivative of the data.

    Parameters:
    -----------
    y : numpy.ndarray
        The data array to estimate the Fermi level from.
    x : numpy.ndarray
        The corresponding x-axis values.

    Returns:
    --------
    float
        The estimated Fermi level.
    """

    def noise_level(y):
        # Calculate the average of nearest neighbour differences to evaluate the noise level of a signal
        NNdist = [abs(y[i] - y[i - 1]) for i in range(1, len(y))]
        noise = np.average(NNdist)
        return noise

    # Smooth the data and compute its derivative
    sigma_px = 0.01 / (abs(x[-1] - x[0]) / len(x))
    deriv = np.gradient(gaussian_filter1d(y, sigma=sigma_px), x)
    y_filtered = -gaussian_filter1d(deriv, 2)
    noise = noise_level(y_filtered)

    # Find all the peaks with prominence >= 2.5 * noise level and with width at least 3 points
    peaks_index, _ = find_peaks(y_filtered, prominence=noise * 2.5, width=3)
    EF = np.round(x[peaks_index].max(), 3)  # Estimated EF
    return EF


def estimate_EF(da):
    """Make an approximate guess for the Fermi level from the corresponding peak in the derivative of the data
    :::{warning}
    This is only a very approximate method for use in making estiamtes to feed into fit functions and GUIs etc.
    It should not be used as a true determination of the Fermi level.
    :::

    Parameters:
    -----------
    data: xarray.DataArray
        The data to estimate the Fermi level from.

    Returns:
    --------
    float
        The estimated Fermi level.
    """

    if "eV" not in da.dims:
        raise ValueError("Data must have an 'eV' dimension to estimate the Fermi level.")

    # Check for an hv scan
    if "hv" in da.dims and "kinetic" in da.metadata.analyser.scan.eV_type.lower():
        # Iterate through the photon energies and estimate at each
        EF_values = []
        for hv in da.hv.data:
            data_hv = da.disp_from_hv(hv=hv)
            EF = _estimate_EF(data_hv.DOS().fillna(0).data, data_hv.eV.data)
            EF_values.append(EF)
        EF_data = xr.DataArray(EF_values, dims=["hv"], coords={"hv": da.hv.data})
        # Fit the result to a 2nd order polynomial
        fit_order = 3
        fit_result = EF_data.quick_fit.poly(fit_order)
        fit_model = fit_result.fit_model.data[()]
        params = {
            f"c{i}": f"{fit_model.params[f'c{i}'].value:.5f}"
            for i in range(fit_order + 1)
        }
        analysis_warning(
            f"Estimated Fermi level correction across the photon energy range via an order {fit_order} polynomial fit as {params}",
            "info",
            "Analysis info",
        )
        fit_result.plot_fit(ylabel="$E_\\mathrm{F}$ (eV)", xlabel="$h\\nu$ (eV)")
        # Back-calculate the Fermi level from the fit at each photon energy
        EF_values_out = fit_model.eval(x=EF_data.hv.data)
        return EF_values_out
    else:
        try:
            return _estimate_EF(da.DOS().fillna(0).pint.dequantify().data, da.eV.data)
        except Exception:
            return None


def save_fit(fit_result, filename):
    """
    Save the results of a fit.

    Parameters:
    -----------
    fit_result : xarray.DataSet
        The results of the fit to save.
    filename : str
        The name of the file to save the fit results to.

    Returns:
    --------
    None
    """

    result_as_dict = fit_result.to_dict()
    with open(filename, "wb") as f:
        dill.dump(result_as_dict, f, protocol=-1)


def load_fit(filename):
    """
    Load the results of a fit from a netCDF file.

    Parameters:
    -----------
    filename : str
        The name of the file to load the fit results from.

    Returns:
    --------
    xarray.DataSet
        The results of the fit.
    """

    with open(filename, "rb") as f:
        result_as_dict = dill.load(f)
    return xr.Dataset.from_dict(result_as_dict)
