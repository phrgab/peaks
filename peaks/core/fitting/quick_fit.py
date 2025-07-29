import numpy as np
import xarray as xr

from peaks.core.fitting.fit import fit

QUICK_FIT_COMMON_DOC = """

Parameters
-----------
indepndent_var : str, optional
    The dimension corresponding to the indpendent variable. Must be specified if data has >1 dimension.
**kwargs : optional
    Additional keyword arguments to initialise parameter values.

Returns
---------
xarray.DataSet
    A DataSet containing the best-fit parameters, their uncertainties, and the :class:`lmfit.ModelResult` object.
"""


@xr.register_dataarray_accessor("quick_fit")
class QuickFit:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def _get_data_for_guess(self, model, independent_var, **kwargs):
        if independent_var is None and self._obj.ndim > 1:
            raise ValueError(
                f"Supplied data is {self._obj.ndim}-dimensional. Please specify the independent variable "
                f"with the argument `independent_var=_dim_`."
            )
        if self._obj.ndim > 1:
            dependent_vars = set(self._obj.dims) - set([independent_var])

            return self._obj.isel({dim: 0 for dim in dependent_vars}).squeeze().compute()
        else:
            return self._obj.compute()

    def _get_percentile_data(self, data, percentile=10, region="start"):
        """Get the data below or above a certain percentile of the dimension range

        Parameters
        -----------
        data : xarray.DataArray
            The data to extract the percentile from.
        percentile : int, optional
            The percentile to use for the cutoff. Defaults to 10.
        region : str, optional
            The region of the data to use for the fit. Options are 'start' or 'end'. Defaults to 'start'.

        Returns
        --------
        xarray.DataArray
            The data below or above the percentile cutoff.
        """

        dim = data.dims[0]
        cutoff = np.percentile(data[dim].data, percentile)
        if region == "start":
            return data.sel({dim: slice(None, cutoff)})
        elif region == "end":
            return data.sel({dim: slice(cutoff, None)})
        else:
            raise ValueError("Region must be 'start' or 'end'")

    def linear(self, independent_var=None, **kwargs):
        """Quick fit to a linear model"""
        from .models import LinearModel

        data_for_guess = self._get_data_for_guess(
            LinearModel(), independent_var, **kwargs
        )
        params = LinearModel().guess(data_for_guess, **kwargs)
        return fit(self._obj, LinearModel(), params, independent_var)

    linear.__doc__ = linear.__doc__ + QUICK_FIT_COMMON_DOC

    def poly(self, degree=3, independent_var=None, **kwargs):
        """Quick fit to a polynomial model

        Parameters
        -----------
        independent_var : str, optional
            The dimension corresponding to the independent variable. Must be specified if data has >1 dimension.
        degree : int, optional
            The degree of the polynomial model. Defaults to 3.
        **kwargs : optional
            Additional keyword arguments to initialise parameter values.

        Returns
        ---------
        xarray.DataSet
            A DataSet containing the best-fit parameters, their uncertainties, and the :class:`lmfit.ModelResult` object.
        """
        from .models import PolynomialModel

        data_for_guess = self._get_data_for_guess(
            PolynomialModel(degree=degree), independent_var
        )
        params = PolynomialModel(degree=degree).guess(data_for_guess, **kwargs)
        return fit(self._obj, PolynomialModel(degree=degree), params, independent_var)

    def _peak_model(self, peak, independent_var, **kwargs):
        """Quick fit to a Gaussian model"""
        from .models import LinearModel

        if peak == "gaussian":
            from .models import GaussianModel

            peak_model = GaussianModel()
        elif peak == "lorentzian":
            from .models import LorentzianModel

            peak_model = LorentzianModel()

        model = peak_model + LinearModel()
        data_for_guess = self._get_data_for_guess(model, independent_var, **kwargs)
        params = model.make_params(**kwargs)
        params.update(peak_model.guess(data_for_guess, **kwargs))
        params.update(
            LinearModel().guess(self._get_percentile_data(data_for_guess), **kwargs)
        )
        return fit(self._obj, model, params, independent_var)

    def gaussian(self, independent_var=None, **kwargs):
        """Quick fit to a Gaussian model"""
        return self._peak_model("gaussian", independent_var, **kwargs)

    gaussian.__doc__ = gaussian.__doc__ + QUICK_FIT_COMMON_DOC

    def lorentzian(self, independent_var=None, **kwargs):
        """Quick fit to a Lorentzian model"""
        return self._peak_model("lorentzian", independent_var, **kwargs)

    lorentzian.__doc__ = lorentzian.__doc__ + QUICK_FIT_COMMON_DOC
