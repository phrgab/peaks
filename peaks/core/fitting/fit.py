import numpy as np
import xarray as xr
import dask as da
import matplotlib.pyplot as plt
import panel as pn

from peaks.core.utils.OOP_method import add_methods


@add_methods(xr.DataArray)
def fit(data_array, model, independent_var, params, sequential=True):
    """
    Fit an :class:`lmfit.Model` to an :class:`xarray.DataArray`, specifying the co-ordinate correspinding to
    the indepednent variable. Fit broadcasts along other dimensions. Returns the results of the fit as a
    :class:`xarray.DataArray` with uncertainties and the full :class:`lmfit.ModelResult`.

    Parameters:
    - data_array: xarray.DataArray
          The data to fit the model to.
    - model: lmfit.Model
        The model to fit to the data.
    - independent_var: str
         The name of the dimension that represents the independent variable.
    - initial_params: lmfit.Parameters
        Initial parameters for the model fitting.
    - sequential: bool
        If True, and `data_array` is 2D, use the results of the previous fit to update the starting parameters
        for the next iteration. Assumes that the initial parameters are for the first slice
        (i.e. `data_array.isel(dim=0)` slice) along the non-independent dimension.
        Defaults to True.
        If False, fit the model to the entire data array at once.

    Returns:
    - xarray.DataArray
        A DataArray containing the best-fit parameters, their uncertainties, and the :class:`lmfit.ModelResult` object.
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

    # Check independent_var is a valid dimension
    if independent_var not in data_array.dims:
        raise ValueError(
            f"Independent variable {independent_var} is not a valid dimension of the data array."
            f"Expected one of {data_array.dims}"
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

        fit_results = []
        # Iterate through all slices along the non-independent dimension
        for i in range(non_indep_dim_len):
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

    return results_ds


@xr.register_dataarray_accessor("quick_fit")
class QuickFit:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def fit(self, model, independent_var, params):
        """
        Fit an :class:`lmfit.Model` to an :class:`xarray.DataArray`, specifying the co-ordinate correspinding to
         the indepednent variable. Fit broadcasts along other dimensions. Returns the results of the fit as a
        :class:`xarray.DataArray` with uncertainties and the full :class:`lmfit.ModelResult`.

        Parameters
        ----------
        model : lmfit.Model
            The model to fit to the data.
        independent_var : str
            The name of the coordinate that represents the independent variable.
        params : lmfit.Parameters
            Initial parameters for the model fitting.

        Returns
        -------
        xarray.DataArray
            A DataArray containing the best-fit parameters, their uncertainties, and the full
            :class:`lmfit.ModelResult` object.

        Examples
        --------
        Example usage is as follows::

            import peaks as pks
            import lmfit

            # Load data
            disp1 = load('disp1.ibw')

            # Create a model - in this case a single Lorentzian peak
            model = lmfit.models.LorentzianModel()
            params = model.make_params(center=0, amplitude=1, sigma=1)

            # Fit the model for all EDCs in the dispersion
            fit_results = disp1.fitting.fit(model, 'eV', params)


        """
        return _fit_model_to_dataarray(
            data_array=self._obj,
            model=model,
            independent_var=independent_var,
            params=params,
        )
