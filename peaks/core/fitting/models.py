"""Underlying custom fitting models or extensions for lmfit.
"""

import numpy as np
import xarray as xr
from lmfit import Parameters, Model
import dask.array as da
from scipy.ndimage import gaussian_filter1d


# Function to apply Gaussian convolution
def _apply_gaussian_broadening(y, sigma, x):
    """Apply Gaussian broadening to a 1D array.
    Parameters
    ----------
    y : array_like
        The 1D array to be broadened.
    sigma : float
        The standard deviation of the Gaussian broadening kernel.
    x : array_like
        The independent variable array corresponding to y.

    Returns
    -------
    array_like
        The broadened 1D array.
    """

    sigma_pixels = sigma / (x[1] - x[0])
    return gaussian_filter1d(y, sigma_pixels)


def _gauss_conv(x, sigma):
    """Dummy function used in gaussian convolution to converts sigma_conv in pixel from x axis unit"""
    return sigma / (abs(x[-1] - x[0]) / len(x))


def _convolve_gauss(model, sigma_conv_pxl):
    """Gauss convolution to use in lmfit.ModelComposite as binary operator"""
    return gaussian_filter1d(model, sigma_conv_pxl)


class GaussianConvolvedFitModel(Model):
    """A custom model class that applies Gaussian convolution to the output of another lmfit model."""

    def __init__(
        self, base_model, convolution_sigma_param_name="convolution_sigma", **kwargs
    ):
        """
        Initialize the ConvolvedFitModel.

        Parameters
        ----------
        base_model : lmfit.Model
            The base lmfit model to which Gaussian convolution will be applied.
        convolution_sigma_param_name : str, optional
            The name of the parameter that specifies the standard deviation of the Gaussian convolution.
            Default is "convolution_sigma".
        **kwargs : dict
            Additional keyword arguments passed to the Model superclass.
        """
        super().__init__(self.eval, **kwargs)
        self.base_model = base_model
        self.convolution_sigma_param_name = convolution_sigma_param_name

    def eval(self, params=None, **kwargs):
        """
        Evaluate the model with Gaussian convolution.

        Parameters
        ----------
        params : lmfit.Parameters, optional
            The parameters for the model evaluation, including the Gaussian convolution sigma.
        **kwargs : dict
            Additional keyword arguments, including the independent variable array `x`.

        Returns
        -------
        numpy.ndarray
            The result of the base model evaluation after applying Gaussian convolution.
        """
        result = self.base_model.eval(params=params, **kwargs)  # Base model
        sigma = params[self.convolution_sigma_param_name].value  # Sigma for convolution
        x = kwargs["x"]  # Independent variable array
        return _apply_gaussian_broadening(result, sigma, x)


def fermi_function():
    pass


def _gauss_conv():
    pass


def _Shirley(data, num_avg=1, offset_start=0, offset_end=0, max_iterations=10):
    """Function to calculate the Shirley background of 1D data.

    Parameters
    ------------
    data : numpy.ndarray, list, xarray.DataArray
        The 1D data (y values) to find the Shirley background of.

    num_avg : int, optional
        The number of points to consider when calculating the average value of the data start and end points. Useful for
        noisy data. Defaults to 1.

    offset_start : float, optional
        The offset to subtract from the data start value. Useful when data range does not completely cover the start
        (left) tail of the peak. Defaults to 0.

    offset_end : float, optional
        The offset to subtract from the data end value. Useful when data range does not completely cover the end (right)
        tail of the peak. Defaults to 0.

    max_iterations : int, optional
        The maximum number of iterations to allow for convergence of Shirley background. Defaults to 10.

    Returns
    ------------
    Shirley_bkg : numpy.ndarray
        The Shirley background of the 1D data.

    Examples
    ------------
    Example usage is as follows::

        import peaks as pks

        S2p_XPS = pks.load('XPS1.ibw').DOS()

        # Extract Shirley background of the XPS scan
        S2p_XPS_Shirley_bkg = pks._Shirley(S2p_XPS)

        # Extract Shirley background of the XPS scan, using 3 points to calculate the average value of the data start
        and end points
        S2p_XPS_Shirley_bkg = pks._Shirley(S2p_XPS, num_avg=3)

        # Extract Shirley background of the XPS scan, applying an offset to the data start value
        S2p_XPS_Shirley_bkg = pks._Shirley(S2p_XPS, offset_start=0.3)

    """

    # Ensure data is of type numpy.ndarray
    if isinstance(data, np.ndarray):  # if data is a numpy array
        pass
    elif isinstance(data, xr.core.dataarray.DataArray):  # if data is a DataArray
        data = data.data
    elif isinstance(data, list):  # if data is a list
        data = np.array(data)
    else:
        raise Exception(
            "Inputted data must be a 1D numpy.ndarray, list or xarray.DataArray."
        )

    # Ensure data is 1D
    if len(data.shape) != 1:
        raise Exception(
            "Inputted data must be a 1D numpy.ndarray, list or xarray.DataArray."
        )

    # Ensure num_avg and max_iterations are integers
    try:
        num_avg = int(num_avg)
        max_iterations = int(max_iterations)
    except ValueError:
        raise Exception("The inputs num_avg and max_iterations must both be integers")

    # Ensure offset_start and offset_end are floats
    try:
        offset_start = float(offset_start)
        offset_end = float(offset_end)
    except ValueError:
        raise Exception("The inputs offset_start and offset_end must both be floats")

    # Get number of points in data and define tolerance
    num_points = len(data)
    tolerance = 1e-5

    # Determine start and end limits of Shirley background
    y_start = data[0:num_avg].mean() - offset_start
    y_end = data[(num_points - num_avg) : num_points].mean() - offset_end

    # Initialise the bkg shape B, where total Shirley bkg is given by Shirley_bkg = y_end + B
    B = np.zeros(data.shape)

    # First B value is equal to y_start - y_end, i.e. Shirley_bkg[0] = y_start as expected
    B[0] = y_start - y_end

    # Define function to determine Shirley bkg
    dk = lambda i: sum(
        0.5 * (data[i:-1] + data[i + 1 :] - 2 * y_end - B[i:-1] + B[i + 1 :])
    )

    # Perform iterative procedure to converge to Shirley bkg, stopping if maximum number of iterations is reached
    num_iterations = 0
    while num_iterations < max_iterations:
        # Calculate new k = (y_start - y_end) / (int_(xl)^(xr) J(x') - y_end - B(x') dx')
        k_sum = sum(0.5 * (data[:-1] + data[1:] - 2 * y_end - B[:-1] + B[1:]))
        k = (y_start - y_end) / k_sum
        # Calculate new B
        y_sum = np.array(list(map(dk, range(0, num_points))))
        new_B = k * y_sum
        # If new_B is close to B (within tolerance), stop the loop
        if sum(abs(new_B - B)) < tolerance:
            B = new_B
            break
        else:
            B = new_B
        num_iterations += 1

    # Raise an error if the maximum allowed number of iterations is exceeded
    if num_iterations >= max_iterations:
        raise Exception(
            "Maximum number of iterations exceeded before convergence of Shirley background was achieved."
        )

    # Determine Shirley bkg
    Shirley_bkg = y_end + B

    return Shirley_bkg


# def apply_lmfit_model(
#     dataarray,
#     model,
#     indep_dim,
#     initial_params,
#     strategy="constant",
#     parallelize=True,
# ):
#     # Convert to Dask array if parallelization is requested and dataarray is not already Dask-based
#     if parallelize and not isinstance(dataarray.data, da.Array):
#         dataarray = dataarray.chunk({indep_dim: -1})
#         indep_var = dataarray[indep_dim]
#     else:
#         indep_var = dataarray[indep_dim].values
#
#     # Prepare the initial parameters
#     prepared_initial_params = initial_params if initial_params is not None else {}
#
#     if strategy == "constant":
#         # Use apply_ufunc for constant initial parameters with optional parallelization
#         fit_results = xr.apply_ufunc(
#             fit_lmfit_model,
#             indep_var,
#             dataarray,
#             input_core_dims=[[indep_dim], [indep_dim]],
#             output_core_dims=[[]],
#             vectorize=True,
#             dask="parallelized" if parallelize else None,
#             kwargs={"model": model, "initial_params": prepared_initial_params},
#             output_dtypes=[object],
#         )
#
#     elif strategy == "sequential":
#         # Sequential initialization requires manual looping
#         fit_results = np.empty_like(dataarray.isel({indep_dim: 0}).values, dtype=object)
#
#         for index in np.ndindex(
#             *[dataarray.sizes[dim] for dim in dataarray.dims if dim != indep_dim]
#         ):
#             # Extract the slice
#             slice_data = dataarray.isel(
#                 {dim: i for dim, i in zip(dataarray.dims, index)}
#             ).values
#
#             # Perform the fit
#             result = fit_lmfit_model(
#                 indep_var, slice_data, model, prepared_initial_params
#             )
#
#             # Update the initial parameters for the next iteration
#             prepared_initial_params = {
#                 name: {"value": param.value} for name, param in result.params.items()
#             }
#
#             # Store the results
#             fit_results[index] = result
#
#     # Extract relevant information from the lmfit results
#     params = model.param_names
#     param_values = np.array(
#         [
#             [fit_results[i].params[param].value for param in params]
#             for i in range(len(fit_results))
#         ]
#     )
#     param_errors = np.array(
#         [
#             [fit_results[i].params[param].stderr for param in params]
#             for i in range(len(fit_results))
#         ]
#     )
#     chisqr = np.array([fit_results[i].chisqr for i in range(len(fit_results))])
#     redchi = np.array([fit_results[i].redchi for i in range(len(fit_results))])
#     aic = np.array([fit_results[i].aic for i in range(len(fit_results))])
#     bic = np.array([fit_results[i].bic for i in range(len(fit_results))])
#
#     # Create a Dataset for the extracted results
#     other_dims = [dim for dim in dataarray.dims if dim != indep_dim]
#     fit_dataset = xr.Dataset(
#         {
#             "params": (other_dims + ["parameter"], param_values),
#             "param_errors": (other_dims + ["parameter"], param_errors),
#             "chisqr": (other_dims, chisqr),
#             "redchi": (other_dims, redchi),
#             "aic": (other_dims, aic),
#             "bic": (other_dims, bic),
#         },
#         coords={dim: dataarray.coords[dim] for dim in dataarray.dims},
#     )
#
#     return fit_dataset
#
#
# # Example usage:
#
# # Prepare data (same as previous example)
# time = np.linspace(0, 10, 100)
# lat = np.linspace(-90, 90, 10)
# lon = np.linspace(-180, 180, 10)
# data = np.sin(time[:, None, None]) + 0.1 * np.random.randn(100, 10, 10)
# dataarray = xr.DataArray(
#     data, coords=[time, lat, lon], dims=["time", "latitude", "longitude"]
# )
#
# # Define a composite model (for demonstration purposes)
# model1 = Model(example_func, prefix="m1_")
# model2 = Model(example_func, prefix="m2_")
# composite_model = model1 + model2
#
# # Define initial parameters and constraints
# initial_params = {
#     "m1_a": {"value": 1, "min": 0},
#     "m1_b": {"value": 0.5, "min": 0, "max": 1},
#     "m1_c": {"value": 0, "vary": True},
#     "m2_a": {"value": 1, "min": 0},
#     "m2_b": {"value": 0.5, "min": 0, "max": 1},
#     "m2_c": {"value": 0, "vary": True},
# }
#
# # Apply the model to the dataarray with 'constant' strategy and parallelization
# fit_dataset_constant = apply_model_to_dataarray(
#     dataarray,
#     composite_model,
#     "time",
#     initial_params,
#     strategy="constant",
#     parallelize=True,
# )
# print(fit_dataset_constant)
#
# # Apply the model to the dataarray with 'sequential' strategy without parallelization
# fit_dataset_sequential = apply_model_to_dataarray(
#     dataarray,
#     composite_model,
#     "time",
#     initial_params,
#     strategy="sequential",
#     parallelize=False,
# )
# print(fit_dataset_sequential)

from lmfit import Model, Parameters
from xarray import DataArray, Dataset, broadcast
import numpy as np


def _lm_model_fit(
    dataarray,
    coords,
    model,
    reduce_dims=None,
    p0=None,
    method="leastsq",
    reduce_func="mean",
    skipna=True,
    errors="raise",
    **kwargs,
):
    if isinstance(coords, (str, DataArray)) or not isinstance(coords, Iterable):
        coords = [coords]
    coords = [dataarray[coord] if isinstance(coord, str) else coord for coord in coords]

    if reduce_dims:
        if isinstance(reduce_dims, str):
            reduce_dims = [reduce_dims]
        else:
            reduce_dims = list(reduce_dims)

    coords = broadcast(*coords)
    coords = [coord.broadcast_like(dataarray, exclude=reduce_dims) for coord in coords]

    reduce_dims = reduce_dims or []
    preserved_dims = list(set(dataarray.dims) - set(reduce_dims))

    def apply_fit(data_array, *coords):
        if skipna:
            mask = ~np.isnan(data_array) & np.all(
                [~np.isnan(c) for c in coords], axis=0
            )
            if not np.any(mask):
                return np.full(len(model.param_names), np.nan)
            x_data = np.vstack([c[mask] for c in coords]).T
            y_data = data_array[mask]
        else:
            x_data = np.vstack([c.ravel() for c in coords]).T
            y_data = data_array.ravel()

        if x_data.shape[0] != y_data.shape[0]:
            raise ValueError("Mismatch between x_data and y_data lengths")

        params = p0 or model.make_params()
        try:
            result = model.fit(y_data, params, x=x_data, method=method, **kwargs)
            return np.array([result.params[name].value for name in model.param_names])
        except Exception as e:
            if errors == "raise":
                raise
            else:
                return np.full(len(model.param_names), np.nan)

    if isinstance(dataarray, xr.DataArray):
        dataarray = dataarray.to_dataset(name="data")

    result = Dataset()
    for name, da in dataarray.data_vars.items():
        fit_result = xr.apply_ufunc(
            apply_fit,
            da,
            *coords,
            vectorize=True,
            input_core_dims=[reduce_dims] * (len(coords) + 1),
            output_core_dims=[["param"]],
            exclude_dims=set(reduce_dims),
            output_dtypes=[float],
            dask="parallelized",
            kwargs=kwargs,
        )
        result[name + "_fit_coefficients"] = fit_result

    result = result.assign_coords({"param": list(model.param_names)})
    result.attrs = dataarray.attrs.copy()

    return result


def create_fit_model():
    pass


def save_fit_model():
    pass
