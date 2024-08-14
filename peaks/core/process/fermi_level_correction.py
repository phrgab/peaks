"""Functions used to apply Fermi level corrections to data.

"""

import numpy as np
import numexpr as ne

from ...utils import analysis_warning


# Brendan Edwards 26/04/2021


def apply_EF():
    pass


def hv_align():
    pass


def _get_EF_at_theta_par(data, theta_par):
    """Gets the kinetic energy of the DataArray corresponding to the Fermi level at a given theta_par value.
    The `EF_correction` attribute should be set first, but will be estimated if not present.

    Parameters
    ----------
    data : xarray.DataArray
        Data to use.
    theta_par : float or np.ndarray
        Theta_par value to extract the Fermi level kinetic energy at.

    Returns
    -------
    float or np.ndarray
        Kinetic energy of the Fermi level at the given theta_par value(s).
    """

    # Get EF_correction, estimating if not present
    EF_fn = data.attrs.get("EF_correction", _add_estimated_EF(data))
    if isinstance(EF_fn, dict):
        Ek_at_EF = np.polyval(list(EF_fn.values())[::-1], theta_par)
    else:
        Ek_at_EF = EF_fn

    return Ek_at_EF


def _get_E_shift_at_theta_par(data, theta_par, Ek=None):
    """Gets the energy shift that should be applied to a kinetic energy to correct for the curvature of the Fermi
    level as a function of theta_par. The `EF_correction` attribute should be set first, but will be estimated
    if not present. Uses numexpr to allow for fast evaluation even over large theta_par grids.

    Parameters
    ----------
    data : xarray.DataArray
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

    # Get EF_correction, estimating if not present
    EF_fn = data.attrs.get("EF_correction", _add_estimated_EF(data))
    if not isinstance(EF_fn, dict):  # No theta_par-dep energy shift
        if Ek is not None:
            return Ek  # Return original energy array if present
        else:
            return np.zeros_like(theta_par)  # Return zeros if no energy array
    else:
        Ek_at_EF_th0 = _get_EF_at_theta_par(data, 0)
        shift_str = f"-{Ek_at_EF_th0}"
        for i in range(len(EF_fn)):
            shift_str += f"+ {str(EF_fn[f'c{i}'])} * theta_par ** {i}"

        if Ek is not None:
            return ne.evaluate(f"{shift_str} + Ek")
        else:
            return ne.evaluate(shift_str)


def _get_wf(data):
    """Gets the relevant work function to use for the data processing.

    Parameters
    ----------
    data : xarray.DataArray
        Data to extract the work function from.

    Returns
    -------
    float
        Work function to use for the data processing.
    """

    # Get current theta_par scale
    theta_par_data = data.theta_par.data

    # Get EF_correction, estimating if not present
    Ek_at_EF = _get_EF_at_theta_par(data, 0)

    # Photon energy
    hv = data.attrs.get("hv", _add_estimated_hv(data))

    return hv - Ek_at_EF


def _get_BE_scale(data):
    """Gets the relevant binding energy scale from the kinetic energy scale, padding to account for any theta-par
    dependence of the Fermi level on the detector.

    Parameters
    ----------
    data : xarray.DataArray
        Data for extracting BE scale from

    Returns
    -------
    tuple
        (start, end, default_step) of the binding energy scale. Default step is the same as the input data.
        Convention is used for negative BE value below Fermi level.
    """

    # Get current scales
    theta_par = data.theta_par.data
    eV = data.eV.data
    eV_step = (eV[-1] - eV[0]) / (len(eV) - 1)

    # Work out theta_par-dep EF from EF_correction
    EF_values = _get_EF_at_theta_par(data, theta_par)

    # Work out new energy range, padding to avoid loosing real data at the edges
    Emin, Emax = np.min(eV), np.max(eV)
    EF_min, EF_max = np.min(EF_values), np.max(EF_values)
    BE_min, BE_max = Emin - EF_max, Emax - EF_min

    return BE_min, BE_max + eV_step, eV_step


def _add_estimated_EF(data):
    """Adds estimated Fermi level to :class:`xarray.DataArray` attributes if existing EF correction is not present.

    Parameters
    ----------
    data : :class:`xarray.DataArray`
        Data to add estimated Fermi level to.

    Returns
    -------
    None
        Adds the estimated Fermi level to the data attributes.

    Examples
    --------
    Example usage is as follows::

        from peaks.core.process.process import _add_estimated_EF

        disp = load('disp.ibw')

        # Add estimated Fermi level to the dispersion data
        disp = _add_estimated_EF(disp)
    """

    # Check no existing EF correction exists
    if data.attrs.get("EF_correction"):
        return

    # Estimate EF and add to data attributes
    estimated_EF = data.estimate_EF()
    data.attrs["EF_correction"] = estimated_EF

    # Update history and warning
    update_str = (
        f"EF_correction set from automatic estimation of Fermi level to: {estimated_EF} eV. "
        f"NB may not be accurate."
    )
    analysis_warning(update_str, "warning", "Analysis warning")
    data.update_hist(update_str)


def _add_estimated_hv(data):
    """Adds estimated photon energy to :class:`xarray.DataArray` attributes if existing hv value is not set.

    Parameters
    ----------
    data : :class:`xarray.DataArray`
        Data to add estimated photon energy to.

    Returns
    -------
    None
        Adds the estimated photon energy to the data attributes.


    Examples
    --------
    Example usage is as follows::

        from peaks.core.process.process import _add_estimated_hv

        disp = load('disp.ibw')

        # Add estimated photon energy to the dispersion data
        disp = _add_estimated_hv(disp)

    """

    # Check no existing hv value exists
    if data.attrs.get("hv"):
        return

    # Check if EF correction is present and estimate if not
    if not data.attrs.get("EF_correction"):
        _add_estimated_EF(data)

    EF = data.attrs["EF_correction"]
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

    data.attrs["hv"] = hv_est

    # Update history and warning
    update_str = (
        f"hv set from automatic estimation of photon energy to: {hv_est_str}. "
        f"Check for accuracy."
    )
    analysis_warning(update_str, "warning", "Analysis warning")
    data.update_hist(update_str)
