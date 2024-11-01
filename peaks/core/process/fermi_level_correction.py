"""Functions used to apply Fermi level corrections to data.

"""

import numpy as np
import xarray as xr
import numexpr as ne
import copy
from datatree import register_datatree_accessor

from peaks.core.utils.misc import analysis_warning
from peaks.core.metadata.base_metadata_models import EFCorrectionModel
from peaks.core.utils.datatree_utils import _map_over_dt_containing_single_das


@xr.register_dataarray_accessor("EF_correction")
class EFCorrection:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __repr__(self):
        return self._obj.metadata.EF_correction.__repr__()

    def set(self, EF_correction):
        """Sets the Fermi level correction for a :class:`xarray.DataArray`.

        Parameters
        ----------
        EF_correction : float, int, dict or xarray.Dataset
            Fermi level correction to apply. This should be:

            - a dictionary of the form {'c0': 16.82, 'c1': -0.001, ...} specifying the coefficients of a
            polynomial fit to the Fermi edge.
            - a float or int, this will be taken as a constant shift in energy.
            - an xarray.Dataset containing the fit_result as returned by the `peaks` `.fit_gold` method

          Returns
         -------
         None
             Adds the Fermi level correction to the data attributes.
        """

        # Do some checks on the EF_correction format
        if isinstance(EF_correction, xr.Dataset):
            EF_correction = copy.deepcopy(EF_correction.attrs.get("EF_correction"))
            correction_type = "gold fit result"
        else:
            correction_type = type(EF_correction)
        if not isinstance(EF_correction, (float, int, dict)):
            raise ValueError(
                "EF_correction must be a float, int, dict of fit coefficients or xarray.Dataset containing the fit_result "
                "of the `.fit_gold` function."
            )
        if isinstance(EF_correction, dict):
            expected_keys = [f"c{i}" for i in range(len(EF_correction))]
            if not all(key in EF_correction for key in expected_keys):
                raise ValueError(
                    f"EF_correction dictionary must contain keys {expected_keys} for the polynomial fit."
                )
            for value in EF_correction.values():
                if not isinstance(value, (float, int)):
                    raise ValueError(
                        "EF_correction dictionary must contain only floats or ints as values in the form "
                        "{'c0': 16.82, 'c1': -0.001, ...} specifying the coefficients of a polynomial fit to the Fermi edge."
                    )

        self._obj.attrs["_EF_correction"] = EFCorrectionModel(
            EF_correction=copy.deepcopy(EF_correction)
        )
        self._obj.history.add(
            f"EF_correction set to {EF_correction} from a passed {correction_type}."
        )

    def set_like(self, da):
        """Sets the Fermi level correction for a :class:`xarray.DataArray` to be the same as another DataArray.

        Parameters
        ----------
        da : xarray.DataArray
            DataArray to copy the Fermi level correction from.

        Returns
        -------
        None
            Adds the Fermi level correction to the data attributes.
        """

        self.set(da.EF_correction.get())
        # Patch the analysis history
        current_history = self._obj._analysis_history.records[-1].record
        new_history = (
            current_history
            + f" [set from existing EF_correction attributes of scan `{da.name}`]."
        )
        self._obj._analysis_history.records[-1].record = new_history

    def get(self):
        """Gets the Fermi level correction from a :class:`xarray.DataArray`."""

        EF_correction_model = self._obj.attrs.get("_EF_correction")
        try:
            return EF_correction_model.EF_correction
        except AttributeError:
            return None


@register_datatree_accessor("EF_correction")
class EFCorrectionDt:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __repr__(self):
        if not self._obj.is_empty and hasattr(self._obj, "data"):
            return self._obj.data.metadata.EF_correction.__repr__()
        else:
            return "Tree node is empty - no EF correction."

    def set(self, EF_correction):
        _map_over_dt_containing_single_das(
            self._obj, lambda da: da.EF_correction.set(EF_correction)
        )

    def set_like(self, da_to_set_like):
        _map_over_dt_containing_single_das(
            self._obj, lambda da: da.EF_correction.set_like(da_to_set_like)
        )

    set.__doc__ = EFCorrection.set.__doc__
    set_like.__doc__ = EFCorrection.set_like.__doc__


def _get_EF_at_theta_par0(da):
    """Gets the kinetic energy of the DataArray corresponding to the Fermi level at theta_par=0.
    The `EF_correction` attribute will be taken if set, or estimated if not.

    Parameters
    ----------
    da : xarray.DataArray
        Data to use.

    Returns
    -------
    float or np.ndarray
        Kinetic energy of the Fermi level at the given theta_par value(s).
    """

    # Get EF_correction, estimating if not present
    if "hv" not in da.dims:
        if not da.EF_correction.get():
            _add_estimated_EF(da)
        EF_fn = da.EF_correction.get()
        if isinstance(EF_fn, dict):
            return EF_fn["c0"]
        else:
            return EF_fn
    else:  # For photon energy scan, we need this from the coords
        if "EF" not in da.coords:
            _add_estimated_EF(da)
        return da.coords["EF"].data  # EF vs hv


def _get_E_shift_at_theta_par(da, theta_par, Ek=None):
    """Gets the energy shift that should be applied to a kinetic energy to correct for the curvature of the Fermi
    level as a function of theta_par. The `EF_correction` attribute should be set first.
    Uses numexpr to allow for fast evaluation even over large theta_par grids.

    Parameters
    ----------
    da : xarray.DataArray
        Underlying data
    theta_par : float or np.ndarray
        Theta_par value to extract the shift at.
    Ek : float or np.ndarray, optional
        Kinetic energy value(s) to apply the shift to. If not provided, the raw shift will be returned.

    Returns
    -------
    float or np.ndarray
        Energy shift of the Fermi level at the given theta_par value(s).
    """

    # Get EF_correction
    EF_fn = da.EF_correction.get()
    if isinstance(EF_fn, dict) and len(EF_fn) > 1:  # Theta-par-dep EF correction
        shift_str = ""
        for i in range(1, len(EF_fn)):
            shift_str += f"+ {str(EF_fn[f'c{i}'])} * theta_par ** {i}"
        if Ek is not None:
            shift_str += f"+ Ek"
        return ne.evaluate(shift_str)
    else:
        if Ek is not None:
            return Ek
        else:
            return np.zeros_like(theta_par)


def _get_wf(da):
    """Gets the relevant work function to use for the data processing.

    Parameters
    ----------
    da : xarray.DataArray
        Data to extract the work function from.

    Returns
    -------
    wf : float or np.ndarray
        Work function to use for the data processing. Will be an array if an hv scan passed
    """

    # Get EF values, estimating if not already set
    Ek_at_EF = _get_EF_at_theta_par0(da)

    # Get photon energy
    if "hv" in da.dims:
        hv = da.hv.data
    else:
        hv = da.metadata.photon.hv
        if hv is not None:
            hv = hv.to("eV").magnitude
    if hv is None:
        hv = _add_estimated_hv(da)

    return hv - Ek_at_EF


def _get_BE_scale(da):
    """Gets the relevant binding energy scale from the kinetic energy scale, padding to account for any theta-par
    dependence of the Fermi level on the detector and any shift of EF within the detector for hv-dep data.

    Parameters
    ----------
    da : xarray.DataArray
        Data for extracting BE scale from

    Returns
    -------
    tuple
        (start, end, step) of the binding energy scale. Step is the same as the input data.
        Convention is used for negative values below Fermi level (i.e. :math:`\\omega=E-E_F` returned).
    """
    # Function implementation

    # Get current scales
    theta_par = da.theta_par.data
    # Work out theta_par-dep EF shift
    EF_shift = _get_E_shift_at_theta_par(da, theta_par)

    # Get current energy scale, taking a single hv if a hv scan
    if "hv" in da.dims:
        hv0 = da.hv.data[0]
        disp = da.disp_from_hv(hv0)
        EF_values = da.EF.sel(hv=hv0).data + EF_shift
    else:
        disp = da
        EF_values = _get_EF_at_theta_par0(da) + EF_shift
    eV = disp.eV.data
    eV_step = (eV[-1] - eV[0]) / (len(eV) - 1)

    # Work out new energy range, padding to avoid loosing real data at the edges due to theta_par dependence
    Emin, Emax = np.min(eV), np.max(eV)
    EF_min, EF_max = np.min(EF_values), np.max(EF_values)
    BE_min, BE_max = Emin - EF_max, Emax - EF_min

    # If hv scan, add extra padding to account for shift in EF on detector with hv
    if "hv" in da.dims:
        KE_shift_vs_hv = da.KE_delta.data - da.KE_delta.data[0]
        EF_shift_vs_hv = da.EF.data - da.EF.data[0]
        shift_on_detector = KE_shift_vs_hv - EF_shift_vs_hv
        if min(shift_on_detector) < 0:
            BE_max += np.abs(min(shift_on_detector))
        if max(shift_on_detector) > 0:
            BE_min -= max(shift_on_detector)

    return BE_min, BE_max + eV_step, eV_step


def _add_estimated_EF(da):
    """Adds estimated Fermi level to :class:`xarray.DataArray` attributes if existing EF correction is not present.

    Parameters
    ----------
    da : xarray.DataArray
        Data to add estimated Fermi level to.

    Returns
    -------
    None
        Adds the estimated Fermi level to the data attributes or coordinates as appropraite.

    """

    # Check no existing EF correction exists
    if hasattr(da, "EF_correction") and da.EF_correction.get():
        if "hv" in da.dims and "EF" not in da.coords:
            pass
        else:
            return

    # Estimate EF and add to data attributes
    estimated_EF = da.estimate_EF()
    if "hv" not in da.dims:
        da.EF_correction.set(estimated_EF)
        estimated_EF_str = f"{estimated_EF} eV."
    else:
        da.coords.update({"EF": ("hv", estimated_EF)})
        estimated_EF_str = (
            f"{estimated_EF[0]:.2f}-{estimated_EF[-1]:.2f} eV (hv dependent)"
        )

    # Update history and warning
    update_str = (
        f"EF_correction set from automatic estimation of Fermi level to: {estimated_EF_str}. "
        f"NB may not be accurate."
    )
    analysis_warning(update_str, "warning", "Analysis warning")
    da.history.add(update_str)


def _add_estimated_hv(da):
    """Adds estimated photon energy to :class:`xarray.DataArray` attributes if existing hv value is not set.

    Parameters
    ----------
    da : :class:`xarray.DataArray`
        Data to add estimated photon energy to.

    Returns
    -------
    None
        Adds the estimated photon energy to the data attributes.

    """

    # Check no existing hv value exists
    if "hv" in da.dims or (
        hasattr(hasattr(da.metadata, "photon"), "hv") and da.metadata.photon.hv
    ):
        return

    # Check if EF correction is present and estimate if not
    if not da.EF_correction.get():
        _add_estimated_EF(da)

    EF = da.EF_correction.get()
    if isinstance(EF, dict):
        EF = EF["c0"]

    if EF > 16.5 and EF < 17:  # Likely to be He-I
        hv_est = 21.2182
        hv_est_str = str(hv_est) + " eV (He I)."
    elif EF > 1 and EF < 2:  # Likely to be 6.05 eV laser
        hv_est = 6.05
        hv_est_str = str(hv_est) + " eV (6 eV laser)."
    elif EF > 6 and EF < 7:  # Likely to be 11 eV laser
        hv_est = 10.897
        hv_est_str = str(hv_est) + " eV (11 eV laser)."
    else:  # Estimate from EF_est
        hv_est = EF + 4.4  # Taking 4.4 eV as a reasonable work function
        hv_est_str = str(hv_est) + " eV."

    da.metadata.photon.hv = hv_est

    # Update history and warning
    update_str = (
        f"hv set from automatic estimation of photon energy to: {hv_est_str}. "
        f"Check for accuracy."
    )
    analysis_warning(update_str, "warning", "Analysis warning")
    da.history.add(update_str)
