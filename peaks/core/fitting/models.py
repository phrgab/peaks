"""Underlying custom fitting models or extensions for lmfit."""

import re

import lmfit.models as lm_models
import numpy as np
import xarray as xr
from lmfit import CompositeModel, Model
from scipy.ndimage import gaussian_filter1d

from peaks.core.fitting.fit_functions import _fermi_function, _linear_dos_fermi


def create_xarray_compatible_lmfit_model(model):
    """Dynamic class factory function to generate modified versions of lmfit models which accept 1D
    :class:`xarray.DataArray`'s as input for the guess parameters function.

    Parameters
    ----------
    model : lmfit.Model
        The :class:`lmfit` model.

    Returns
    -------
    WrappedModel : lmfit.Model
        The modified lmfit model which accepts 1D xarray.DataArray's as input for the guess parameters function.
    """

    class WrappedModel(model):
        def guess(self, data, **kws):
            if not isinstance(data, xr.DataArray):
                raise TypeError(
                    "This is a modified lmfit model which expects the data to be supplied as a 1D xarray.DataArray. "
                    "To pass numpy arrays, use the original lmfit model."
                )
            if data.ndim != 1:
                raise ValueError(
                    "Supplied xr.DataArray should be one dimensional, with the included dimension representing the "
                    "independent variable for the fit."
                )
            return super().guess(
                data=data.pint.dequantify().data, x=data[data.dims[0]].data, **kws
            )

        # Modify the docstring
        _original_guess_docstring = model.guess.__doc__
        _modified_guess_docstring = (
            f"Modified version of :class:`lmfit.model.Model.guess` method to accept a 1D "
            f"xarray.DataArray instead of seperate numpy arrays for data and independent "
            f"variable.\n{_original_guess_docstring}."
        )
        _new_guess_parameters = """
            Parameters
            ----------
            data : xarray.DataArray
                The data to guess fit parameters for. Should be a 1D DataArray where the dimension coresponds to the
                independent variable.
            **kws : optional
                Additional keyword arguments, passed to model function.
            """

        # Replace text between "Parameters" and "Returns"
        pattern = re.compile(
            r"(Parameters\s*[-]+\s*)(.*?)(\s*Returns\s*[-]+\s*)", re.DOTALL
        )
        _modified_guess_docstring = re.sub(
            pattern, r"\1" + _new_guess_parameters + r"\3", _modified_guess_docstring
        )
        guess.__doc__ = _modified_guess_docstring

    WrappedModel.__doc__ = (
        f"Modified version of :class:`lmfit.models.{model.__name__}` model with some key methods "
        f"modified to accept xarray.DataArray as inputs.\n{model.__doc__}"
    )
    return WrappedModel


# Get wrapped versions of all standard lmfit models
for model in lm_models.lmfit_models.values():
    model_name = model.__name__
    wrapped_model = create_xarray_compatible_lmfit_model(model)
    globals()[model_name] = wrapped_model


class GaussianConvolvedFitModel(create_xarray_compatible_lmfit_model(CompositeModel)):
    """Create a Gaussian convolved model for fitting, useful for e.g. including experimental resolution.

    Parameters
    ----------
    model : lmfit.Model or lmfit.CompositeModel
        The base or composite model to convolve.

    Examples
    --------
    Example usage is as follows::

        import peaks as pks
        import lmfit

        # Create a model - in this case a single Lorentzian peak
        model = lmfit.models.LorentzianModel()

        # Create a convolved model
        convolved_model = pks.GaussianConvolvedFitModel(model)
    """

    def __init__(self, model):
        def _gauss_conv(x, sigma_conv):
            """Convert sigma_conv from eV units to pixels."""
            return sigma_conv / (abs(x[-1] - x[0]) / len(x))

        def _convolve_gauss(model, sigma_conv_pxl):
            """Apply Gaussian convolution."""
            return gaussian_filter1d(model, sigma_conv_pxl)

        gauss_model = Model(_gauss_conv)
        gauss_model.set_param_hint("sigma_conv", min=0, value=0.001, vary=False)
        super().__init__(model, gauss_model, _convolve_gauss)


class FermiFunctionModel(create_xarray_compatible_lmfit_model(Model)):
    """lmfit compatible model for the Fermi function."""

    def __init__(self, *args, **kwargs):
        super().__init__(_fermi_function, *args, **kwargs)
        self.set_param_hint("EF", min=-np.inf, max=np.inf)
        self.set_param_hint("T", value=10, min=0, vary=False)

    def guess(self, data, **kws):
        """Guess the parameters of the model."""
        params = self.make_params()
        params[f"{self._prefix}EF"].set(value=data.estimate_EF())
        params[f"{self._prefix}T"].set(value=data.attrs.get("temp_sample", 10))
        return params


class LinearDosFermiModel(GaussianConvolvedFitModel):
    """Obtain a lmfit model for fitting typical poly-Au Fermi edge data. Includes: linear bg above and below E_F (dos_),
    Fermi cutoff (fermi_), linear background above E_F accounting for e.g. inhomogeneous detector efficiency (bg_) and
    Gaussian broadening for experimental energy resolution (conv_sigma).

    Attributes
    ----------
    model : lmfit.CompositeModel
        The composite model for fitting.
    """

    def __init__(self, prefix="", *args, **kwargs):
        self.base_model_prefix = prefix
        base_model = Model(_linear_dos_fermi, *args, prefix=prefix, **kwargs)
        base_model.set_param_hint("EF", min=-np.inf, max=np.inf)
        base_model.set_param_hint("T", value=10, min=0, vary=False)
        base_model.set_param_hint("bg_slope", value=0)
        base_model.set_param_hint("bg_intercept", value=0)
        base_model.set_param_hint("dos_slope", value=0)
        super().__init__(base_model)

    def guess(self, data, **kws):
        """Guess the parameters of the model."""
        pars = self.make_params()

        dim = data.dims[0]
        x_xr = data[dim].data

        # Estimate the Fermi level
        EF_estimate = data.estimate_EF()
        pars[f"{self.base_model_prefix}EF"].set(
            value=EF_estimate, max=max(x_xr), min=min(x_xr)
        )
        pars[f"{self.base_model_prefix}T"].set(
            value=data.attrs.get("temp_sample", 10), min=0, max=400, vary=False
        )

        # Guess background parameters
        cutoff = np.percentile(x_xr, 90)  # Take top 10% of data range
        bg_guess = data.sel({dim: slice(cutoff, None)}).quick_fit.linear()
        pars[f"{self.base_model_prefix}bg_slope"].set(value=bg_guess["slope"].data[()])
        pars[f"{self.base_model_prefix}bg_intercept"].set(
            value=bg_guess["intercept"].data[()]
        )

        # Guess DOS parameters
        cutoff = np.percentile(x_xr, 15)  # Take bottom 15% of data range
        dos_guess = data.sel({dim: slice(None, cutoff)}).quick_fit.linear()
        pars[f"{self.base_model_prefix}dos_slope"].set(value=dos_guess["slope"].data[()])
        pars[f"{self.base_model_prefix}dos_intercept"].set(
            value=dos_guess["intercept"].data[()]
        )

        # Set initial guess for Gaussian convolution
        pars["sigma_conv"].set(value=0.01, vary=True)

        return lm_models.update_param_vals(pars, self.base_model_prefix, **kws)


def _shirley_bg(data, num_avg=1, offset_start=0, offset_end=0, max_iterations=10):
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
    except ValueError as e:
        raise Exception(
            "The inputs num_avg and max_iterations must both be integers"
        ) from e

    # Ensure offset_start and offset_end are floats
    try:
        offset_start = float(offset_start)
        offset_end = float(offset_end)
    except ValueError as e:
        raise Exception(
            "The inputs offset_start and offset_end must both be floats"
        ) from e

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
    def dk(i):
        return sum(0.5 * (data[i:-1] + data[i + 1 :] - 2 * y_end - B[i:-1] + B[i + 1 :]))

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
