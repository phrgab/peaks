import numba
import numpy as np
from scipy.constants import physical_constants
from scipy.special import expit

TINY = 1.0e-15
kb_eV, _, _ = physical_constants["Boltzmann constant in eV/K"]


@numba.njit(cache=True)
def _linear_dos_fermi(
    x,
    EF,
    T,
    dos_slope,
    dos_intercept,
    bg_slope,
    bg_intercept,
):
    """Base function for fitting typical poly-Au Fermi edge data. Includes: Fermi cutoff (EF, T), linear DOS (dos_),
    and linear background above E_F accounting for e.g. inhomogeneous detector efficiency (bg_).

    Parameters:
    -----------
    x : np.ndarray
        Energy values in eV.
    EF : float
        The Fermi level in eV.
    T : float
        The temperature in K
    dos_slope : float
        The slope of the linear DOS below the Fermi level
    dos_intercept : float
        The intercept of the linear DOS below the Fermi level
    bg_slope : float
         The slope of the linear background above the Fermi level
    bg_intercept : float
        The intercept of the linear background above the Fermi level

    Returns
    -------
    numpy.ndarray
        The calculated Fermi function + DOS and backgrounds vs. x
    """

    return (bg_intercept + bg_slope * x) + (
        dos_intercept - bg_intercept + (dos_slope - bg_slope) * x
    ) / (1 + np.exp((1.0 * x - EF) / max(TINY, T * kb_eV)))


def _fermi_function(x, EF, T):
    """
    Fermi function in eV units.

    Parameters
    ----------
    x : np.ndarray or list
        Current energy value in eV.
    EF : float
        The Fermi level in eV.
    T : float
        The temperature in K.

    Returns
    -------
    numpy.ndarray
        The calculated Fermi function vs. x
    """

    return expit((EF - x) / max(kb_eV * T, TINY))
