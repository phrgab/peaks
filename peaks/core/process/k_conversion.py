"""Functions used to apply k-space conversions to data."""

import numba_progress
import numexpr as ne
import numpy as np
import pint
import pint_xarray
import xarray as xr
from scipy.constants import angstrom, electron_volt, hbar, m_e
from tqdm.notebook import tqdm

from peaks.core.fileIO.base_data_classes.base_data_class import BaseDataLoader
from peaks.core.process.fermi_level_correction import (
    _get_BE_scale,
    _get_E_shift_at_theta_par,
    _get_wf,
)
from peaks.core.utils.interpolation import (
    _fast_bilinear_interpolate,
    _fast_bilinear_interpolate_rectilinear,
    _fast_trilinear_interpolate,
    _fast_trilinear_interpolate_rectilinear,
    _is_linearly_spaced,
)
from peaks.core.utils.misc import analysis_warning

ureg = pint_xarray.unit_registry

# Calculate kvac_const
KVAC_CONST = (2 * m_e / (hbar**2)) ** 0.5 * (electron_volt**0.5) * angstrom
PI = np.pi


# --------------------------------------------------------- #
# Mapping functions: angle -> k-space (in plane)            #
# following conventions and nomenclature of Ishida and Shin #
# Rev. Sci. Instrum. 89 (2018) 043903.                      #
# --------------------------------------------------------- #


def ensure_radians(func):
    def wrapper(*args, **kwargs):
        processed_args = []
        for arg in args:
            if isinstance(arg, pint.Quantity):
                if arg.dimensionality != ureg.radian.dimensionality:
                    raise ValueError("All angles must be in radians.")
                else:
                    processed_args.append(arg.magnitude)
            else:
                processed_args.append(arg)

        processed_kwargs = {}
        for key, arg in kwargs.items():
            if isinstance(arg, pint.Quantity):
                if arg.dimensionality != ureg.radian.dimensionality:
                    raise ValueError("All angles must be in radians.")
                else:
                    processed_kwargs[key] = arg.magnitude
            else:
                processed_kwargs[key] = arg

        return func(*processed_args, **processed_kwargs)

    return wrapper


def _f_dispatcher(
    ana_type, Ek, alpha, beta, beta_0, chi, chi_0, delta, delta_0, xi, xi_0
):
    if ana_type == "I":
        return _fI(alpha, beta - beta_0, delta - delta_0, xi - xi_0, Ek)
    elif ana_type == "II":
        return _fII(alpha, beta - beta_0, delta - delta_0, xi - xi_0, Ek)
    elif ana_type == "Ip":
        return _fIp(alpha, beta, delta - delta_0, xi - xi_0, chi - chi_0, Ek)
    elif ana_type == "IIp":
        return _fIIp(alpha, beta, delta - delta_0, xi - xi_0, chi - chi_0, Ek)


def _f_inv_dispatcher(
    ana_type, Ek, kx, ky, beta_0, chi, chi_0, delta, delta_0, xi, xi_0
):
    if ana_type == "I":
        return _fI_inv(kx, ky, delta - delta_0, xi - xi_0, Ek, beta_0)
    elif ana_type == "II":
        return _fII_inv(kx, ky, delta - delta_0, xi - xi_0, Ek, beta_0)
    elif ana_type == "Ip":
        return _fIp_inv(kx, ky, delta - delta_0, xi - xi_0, chi - chi_0, Ek)
    elif ana_type == "IIp":
        return _fIIp_inv(kx, ky, delta - delta_0, xi - xi_0, chi - chi_0, Ek)


def _fI(alpha, beta_, delta_, xi_, Ek):  # Type I, no deflector
    """Convert angles to k-space for a type-I analyser with no deflector,
    following the conventions and nomenclature of Ishida and Shin, Rev. Sci. Instrum. 89 (2018) 043903.

    Parameters
    -----------
    alpha : float or np.ndarray
        theta_par angle (analyser slit angle, rad)
    beta_ : float or np.ndaarray
        polar angle - reference angle (rad)
    delta_ : float
        azi angle - reference angle (rad)
    xi_ : float or np.ndarray
        tilt angle - reference angle (rad)
    Ek : float or np.nadarray
        Kinetic energy (eV)

    Returns
    --------
    kx : float or np.ndarray
        k-vector along analyser slit (1/A)
    ky : float or np.ndarray
        k-vector perp to analyser slit (1/A)
    """

    # k_vacuum from KE
    kvac_str = "KVAC_CONST * sqrt(Ek)"

    # Mapping functions
    kx = ne.evaluate(
        f"{kvac_str} * (((sin(delta_) * sin(beta_) + cos(delta_) * sin(xi_) * cos(beta_)) * cos(alpha)) "
        "- (cos(delta_) * cos(xi_) * sin(alpha)))"
    )
    ky = ne.evaluate(
        f"{kvac_str} * (((-cos(delta_) * sin(beta_) + sin(delta_) * sin(xi_) * cos(beta_)) * cos(alpha)) "
        "- (sin(delta_) * cos(xi_) * sin(alpha)))"
    )

    return kx, ky


def _fII(alpha, beta_, delta_, xi_, Ek):  # Type II, no deflector
    """Convert angles to k-space for a type-II analyser with no deflector,
    following the conventions and nomenclature of Ishida and Shin, Rev. Sci. Instrum. 89 (2018) 043903.

       Parameters
       ----------
       alpha : float or np.ndarray
           theta_par angle (analyser slit angle, rad)
       beta_ : float or np.ndarray
           tilt angle - reference angle (rad)
       delta_ : float
           azi angle - reference angle (rad)
       xi_ : float or np.ndarray
           polar angle - reference angle (rad)
       Ek : float or np.ndarray
           Kinetic energy (eV)

       Returns
       -------
       kx : float or np.ndarray
           k-vector perp to analyser slit (1/A)
       ky : float or np.ndarray
           k-vector along analyser slit (1/A)
    """
    # k_vacuum from KE
    kvac_str = "KVAC_CONST * sqrt(Ek)"

    # Mapping functions
    kx = ne.evaluate(
        f"{kvac_str} * (((sin(delta_) * sin(xi_)) + (cos(delta_) * sin(beta_) * cos(xi_))) * cos(alpha) - "
        "((sin(delta_) * cos(xi_) - (cos(delta_) * sin(beta_) * sin(xi_))) * sin(alpha)))"
    )
    ky = ne.evaluate(
        f"{kvac_str} * (((-cos(delta_) * sin(xi_)) + (sin(delta_) * sin(beta_) * cos(xi_))) * cos(alpha) + "
        "((cos(delta_) * cos(xi_) + (sin(delta_) * sin(beta_) * sin(xi_))) * sin(alpha)))"
    )

    return kx, ky


def _fIp(alpha, beta, delta_, xi_, chi_, Ek):  # Type I with deflector
    """
    Convert angles to k-space for a type-I analyser with deflector, following the conventions and nomenclature of Ishida and Shin, Rev. Sci. Instrum. 89 (2018) 043903.

    Parameters
    ----------
    alpha : float or np.ndarray
        theta_par angle (analyser slit angle) + defl_par (deflector angle along the slit) (rad)
    beta : float or np.ndarray
        defl_perp (deflector angle perpendicular to the slit, rad)
    delta_ : float
        azi angle - reference angle (rad)
    xi_ : float or np.ndarray
        tilt angle - reference angle (rad)
    chi_ : float or np.ndarray
        polar angle - reference angle (rad)
    Ek : float or np.ndarray
        Kinetic energy (eV)

    Returns
    -------
    kx : float or np.ndarray
        k-vector along analyser slit (1/A)
    ky : float or np.ndarray
        k-vector perp to analyser slit (1/A)
    """

    # k_vacuum from KE
    kvac_str = "KVAC_CONST * sqrt(Ek)"

    # Mapping functions
    sinc_a2b2_str = "sin(sqrt(alpha**2 + beta**2))/(sqrt(alpha**2 + beta**2))"
    kx = ne.evaluate(
        f"{kvac_str} * ("
        "(((-alpha * cos(delta_) * cos(xi_)) + (beta * sin(delta_) * cos(chi_)) "
        f"- (beta * cos(delta_) * sin(xi_) * sin(chi_))) * {sinc_a2b2_str}) + "
        "(((sin(delta_) * sin(chi_)) + (cos(delta_) * sin(xi_) * cos(chi_))) * cos(sqrt(alpha**2 + beta**2)))"
        ")"
    )
    ky = ne.evaluate(
        f"{kvac_str} * ("
        "(((-alpha * sin(delta_) * cos(xi_)) - (beta * cos(delta_) * cos(chi_)) "
        f"- (beta * sin(delta_) * sin(xi_) * sin(chi_))) * {sinc_a2b2_str}) - "
        "(((cos(delta_) * sin(chi_)) - (sin(delta_) * sin(xi_) * cos(chi_))) * cos(sqrt(alpha**2 + beta**2)))"
        ")"
    )

    return kx, ky


def _fIIp(alpha, beta, delta_, xi_, chi_, Ek):  # Type II with deflector
    """
    Convert angles to k-space for a type-II analyser with deflector, following the conventions and nomenclature of Ishida and Shin, Rev. Sci. Instrum. 89 (2018) 043903.

    Parameters
    ----------
    alpha : float or np.ndarray
        theta_par angle (analyser slit angle) + defl_par (deflector angle along the slit) (rad)
    beta : float or np.ndarray
        defl_perp (deflector angle perpendicular to the slit, rad)
    delta_ : float
        azi angle - reference angle (rad)
    xi_ : float or np.ndarray
        tilt angle - reference angle (rad)
    chi_ : float or np.ndarray
        polar angle - reference angle (rad)
    Ek : float or np.ndarray
        Kinetic energy (eV)

    Returns
    -------
    kx : float or np.ndarray
        k-vector perp to analyser slit (1/A)
    ky : float or np.ndarray
        k-vector along analyser slit (1/A)
    """

    # k_vacuum from KE
    kvac_str = "KVAC_CONST * sqrt(Ek)"

    # Mapping function
    sinc_a2b2_str = "sin(sqrt(alpha**2 + beta**2))/(sqrt(alpha**2 + beta**2))"
    kx = ne.evaluate(
        f"{kvac_str} * ("
        "(((-beta * cos(delta_) * cos(xi_)) - (alpha * sin(delta_) * cos(chi_)) "
        f"+ (alpha * cos(delta_) * sin(xi_) * sin(chi_))) * {sinc_a2b2_str}) + "
        "(((sin(delta_) * sin(chi_)) + (cos(delta_) * sin(xi_) * cos(chi_))) * cos(sqrt(alpha**2 + beta**2)))"
        ")"
    )
    ky = ne.evaluate(
        f"{kvac_str} * ("
        "(((-beta * sin(delta_) * cos(xi_)) + (alpha * cos(delta_) * cos(chi_)) "
        f"+ (alpha * sin(delta_) * sin(xi_) * sin(chi_))) * {sinc_a2b2_str}) - "
        "(((cos(delta_) * sin(chi_)) - (sin(delta_) * sin(xi_) * cos(chi_))) * cos(sqrt(alpha**2 + beta**2)))"
        ")"
    )

    return kx, ky


# Inverse mapping functions for converting between angle and k-space: k --> angle


def _fI_inv(kx, ky, delta_, xi, Ek, beta_0):  # Type I, no deflector
    """
    Convert k-space to angles for a type-I analyser with no deflector, following the conventions and nomenclature
    of Ishida and Shin, Rev. Sci. Instrum. 89 (2018) 043903.

    Parameters
    ----------
    kx : float or np.ndarray
        k-vector along analyser slit (1/A)
    ky : float or np.ndarray
        k-vector perp to analyser slit (1/A)
    delta_ : float
        azi angle - reference angle (rad)
    xi : float
        tilt angle (rad)
    Ek : float
        Kinetic energy (eV)
    beta_0 : float, optional
        polar angle offset

    Returns
    -------
    alpha : float or np.ndarray
        theta_par angle (analyser slit angle, rad)
    beta : float or np.ndarray
        polar angle (rad)
    """
    # k_vacuum from KE
    kvac_str = "(KVAC_CONST * sqrt(Ek))"

    # Mapping function (include convert to degrees directly for speed)
    alpha = ne.evaluate(
        f"arcsin((sin(xi) * sqrt({kvac_str}**2 - kx**2 - ky**2) - cos(xi) * (kx * cos(delta_) + ky * sin(delta_))) "
        f"/ {kvac_str})"
    )

    beta = ne.evaluate(
        "(beta_0 + (arctan((kx * sin(delta_) - ky * cos(delta_)) /"
        " (kx * sin(xi) * cos(delta_) + ky * sin(xi) * sin(delta_) "
        f"+ cos(xi) * sqrt({kvac_str}**2 - kx**2 - ky**2)))))"
    )

    return alpha, beta


def _fII_inv(kx, ky, delta_, xi, Ek, beta_0):  # Type I, no deflector
    """
    Convert k-space to angles for a type-II analyser with no deflector, following the conventions and nomenclature of Ishida and Shin, Rev. Sci. Instrum. 89 (2018) 043903.

    Parameters
    ----------
    kx : float or np.ndarray
        k-vector perp to analyser slit (1/A)
    ky : float or np.ndarray
        k-vector along analyser slit (1/A)
    delta_ : float
        azi angle - reference angle (rad)
    xi : float
        polar angle (rad)
    Ek : float or np.ndarray
        Kinetic energy (eV)
    beta_0 : float, optional
        tilt angle offset

    Returns
    -------
    alpha : float or np.ndarray
        theta_par angle (analyser slit angle, rad)
    beta : float or np.ndarray
        tilt angle (rad)
    """
    # k_vacuum from KE
    kvac_str = "(KVAC_CONST * sqrt(Ek))"

    # Mapping function (include convert to degrees directly for speed)
    alpha = ne.evaluate(
        f"arcsin((sin(xi) * sqrt({kvac_str}**2 - ((kx * sin(delta_) - ky * cos(delta_))**2)) "
        f"- cos(xi) * (kx * sin(delta_) - ky * cos(delta_))) / {kvac_str})"
    )
    beta = ne.evaluate(
        f"(beta_0 + arctan((kx * cos(delta_) + ky * sin(delta_)) / sqrt({kvac_str}**2 - kx**2 - ky**2)))"
    )

    return alpha, beta


def _tij(ij):
    """
    Defines the inverse of the rotation matrix T_rot to obtain the elements of the inverse functions of type I' and II'
     manipulators, Eqn A9 of Ishida and Shin, Rev. Sci. Instrum. 89 (2018) 043903.

    Parameters
    ----------
    ij : int
        Index to return.

    Returns
    -------
    str
        String representation of relevant element of T_rot^-1 for passing to ne.evaluate().
    """

    expressions = {
        11: "cos(xi_) * cos(delta_)",
        12: "cos(xi_) * sin(delta_)",
        13: "-sin(xi_)",
        21: "(sin(chi_) * sin(xi_) * cos(delta_)) - (cos(chi_) * sin(delta_))",
        22: "(sin(chi_) * sin(xi_) * sin(delta_)) + (cos(chi_) * cos(delta_))",
        23: "sin(chi_) * cos(xi_)",
        31: "(cos(chi_) * sin(xi_) * cos(delta_)) + (sin(chi_) * sin(delta_))",
        32: "(cos(chi_) * sin(xi_) * sin(delta_)) - (sin(chi_) * cos(delta_))",
        33: "cos(chi_) * cos(xi_)",
    }

    return expressions[ij]


def _fIp_inv(kx, ky, delta_, xi_, chi_, Ek):  # Type I, with deflector
    """
    Convert k-space to angles for a type-I analyser with deflector, following the conventions and nomenclature of Ishida and Shin, Rev. Sci. Instrum. 89 (2018) 043903.

    Parameters
    ----------
    kx : float or np.ndarray
        k-vector along analyser slit (1/A)
    ky : float or np.ndarray
        k-vector perp to analyser slit (1/A)
    delta_ : float
        azi angle (rad), relative to 'normal' emission
    xi_ : float
        tilt angle (rad), relative to normal emission
    chi_ : float
        polar angle (rad), relative to normal emission
    Ek : float
        Kinetic energy (eV)

    Returns
    -------
    alpha : float or np.ndarray
        theta_par angle (analyser slit angle, rad)
    beta : float or np.ndarray
        polar angle (rad)
    """

    # k_vacuum from KE
    kvac_str = "(KVAC_CONST * sqrt(Ek))"
    k2p_str = f"sqrt({kvac_str}**2 - kx**2 - ky**2)"

    # Mapping functions
    arg1_str = (
        f"(({_tij(31)}) * kx) + (({_tij(32)}) * ky) + (({_tij(33)}) * ({k2p_str}))"
    )
    arg2_str = (
        f"(({_tij(11)}) * kx) + (({_tij(12)}) * ky) + (({_tij(13)}) * ({k2p_str}))"
    )
    arg3_str = (
        f"(({_tij(21)}) * kx) + (({_tij(22)}) * ky) + (({_tij(23)}) * ({k2p_str}))"
    )

    alpha = ne.evaluate(
        f"-arccos(({arg1_str}) / ({kvac_str})) * ({arg2_str}) / sqrt(({kvac_str})**2 - ({arg1_str})**2)"
    )
    beta = ne.evaluate(
        f"-arccos(({arg1_str}) / ({kvac_str})) * ({arg3_str}) / sqrt(({kvac_str})**2 - ({arg1_str})**2)"
    )

    return alpha, beta


def _fIIp_inv(kx, ky, delta_, xi_, chi_, Ek):  # Type II, with deflector
    """
    Convert k-space to angles for a type-II analyser with deflector, following the conventions and nomenclature of Ishida and Shin, Rev. Sci. Instrum. 89 (2018) 043903.

    Parameters
    ----------
    kx : float or np.ndarray
        k-vector perp to analyser slit (1/A)
    ky : float or np.ndarray
        k-vector along analyser slit (1/A)
    delta_ : float
        azi angle (rad), relative to 'normal' emission
    xi_ : float
        tilt angle (rad), relative to normal emission
    chi_ : float
        polar angle (rad), relative to normal emission
    Ek : float
        Kinetic energy (eV)

    Returns
    -------
    alpha : float or np.ndarray
        theta_par angle (analyser slit angle, rad)
    beta : float or np.ndarray
        mapping angle (rad)
    """

    # k_vacuum from KE
    kvac_str = "(KVAC_CONST * sqrt(Ek))"
    k2p_str = f"sqrt({kvac_str}**2 - kx**2 - ky**2)"

    # Mapping functions
    arg1_str = (
        f"(({_tij(31)}) * kx) + (({_tij(32)}) * ky) + (({_tij(33)}) * ({k2p_str}))"
    )
    arg2_str = (
        f"(({_tij(21)}) * kx) + (({_tij(22)}) * ky) + (({_tij(23)}) * ({k2p_str}))"
    )
    arg3_str = (
        f"(({_tij(11)}) * kx) + (({_tij(12)}) * ky) + (({_tij(13)}) * ({k2p_str}))"
    )

    alpha = ne.evaluate(
        f"arccos(({arg1_str}) / ({kvac_str})) * ({arg2_str}) / sqrt(({kvac_str})**2 - ({arg1_str})**2) "
    )
    beta = ne.evaluate(
        f"-arccos(({arg1_str}) / ({kvac_str})) * ({arg3_str}) / sqrt(({kvac_str})**2 - ({arg1_str})**2) "
    )

    return alpha, beta


def _f_kz(Ek, k_along_slit, k_perp_slit, V0):
    """Forward transform to kz.

    Parameters
    ----------
    Ek : float or np.ndarray
        Kinetic energy (eV).
    k_along_slit : float or np.ndarray
        k-vector along analyser slit (1/A).
    k_perp_slit : float or np.ndarray
        k-vector perp to analyser slit (1/A).
    V0 : float
        Inner potential (eV).

    Returns
    -------
    kz : float or np.ndarray
        k-vector for out-of-plan direction (1/A).
    """

    return ne.evaluate(
        "KVAC_CONST * sqrt(Ek - V0 - ((k_along_slit**2 + k_perp_slit**2)/(KVAC_CONST**2)))"
    )


def _f_inv_kz(kz, k_along_slit, k_perp_slit, V0, BE, wf):
    """Inverse transform from kz.

    Parameters
    ----------
    kz : float or np.ndarray
        k-vector for out-of-plan direction (1/A).
    k_along_slit : float or np.ndarray
        k-vector along analyser slit (1/A).
    k_perp_slit : float or np.ndarray
        k-vector perp to analyser slit (1/A).
    V0 : float
        Inner potential (eV).
    BE : float or np.ndarray
        Binding energy (eV).
    wf : float
        Work function (eV).

    Returns
    -------
    hv : float or np.ndarray
        Photon energy (eV).
    """

    return ne.evaluate(
        "(kz**2 + k_along_slit**2 + k_perp_slit**2) / (KVAC_CONST**2) + V0 + wf - BE"
    )


# --------------------------------------------------------- #
#      Helper functions for k-space conversion              #
# --------------------------------------------------------- #
def _reshape_for_2d(arr1, arr2):
    """
    Reshape arrays to make them broadcastable for 2D interpolation methods.

    Parameters
    ----------
    arr1 : np.ndarray
        First array to reshape.
    arr2 : np.ndarray
        Second array to reshape.

    Returns
    -------
    np.ndarray
        Reshaped array 1.
    np.ndarray
        Reshaped array 2.
    """

    arr1 = np.asarray(arr1).reshape(-1, 1)
    arr2 = np.asarray(arr2).reshape(1, -1)

    return arr1, arr2


def _reshape_for_3d(arr1, arr2, arr3):
    """
    Reshape arrays to make them broadcastable for 3D interpolation methods.

    Parameters
    ----------
    arr1 : np.ndarray
        First array to reshape.
    arr2 : np.ndarray
        Second array to reshape.
    arr3 : np.ndarray
        Third array to reshape.

    Returns
    -------
    np.ndarray
        Reshaped array 1.
    np.ndarray
        Reshaped array 2.
    np.ndarray
        Reshaped array 3.
    """

    arr1 = np.asarray(arr1).reshape(-1, 1, 1)
    arr2 = np.asarray(arr2).reshape(1, -1, 1)
    arr3 = np.asarray(arr3).reshape(1, 1, -1)

    return arr1, arr2, arr3


def _get_k_along_slit(kx, ky, ana_type):
    if ana_type in ["I", "Ip"]:
        return kx
    elif ana_type in ["II", "IIp"]:
        return ky
    else:
        raise ValueError(f"Invalid analyser type: {ana_type}")


def _get_k_perpto_slit(kx, ky, ana_type):
    if ana_type in ["I", "Ip"]:
        return ky
    elif ana_type in ["II", "IIp"]:
        return kx
    else:
        raise ValueError(f"Invalid analyser type: {ana_type}")


def get_kpar_cut(
    hv=21.2,
    Eb=0,
    theta_par_range=(-15, 15),
    polar=0,
    tilt=0,
    defl_perp=0,
    ana_type="I",
):
    """Extract the k_par cut along the analyser slit for a given set of angle parameters.

    Parameters
    ----------
    hv : float, optional
        Photon energy (eV) (default=21.2 eV).
    Eb : float, optional
        Binding energy (eV) (positive for below the Fermi level, default=0).
    theta_par_range : tuple, optional
        Range of theta_par values to calculate over in the form (start, stop);
        default=(-15, 15)).
    polar : float, optional
        Polar angle (deg) (default=0).
    tilt : float, optional
        Tilt angle (deg) (default=0).
    defl_perp : float, optional
        Deflector angle perpendicular to the slit (deg) (default=0).
    ana_type : str, optional
        Analyser type, one of I, II, Ip, IIp (default='I').
    """
    # Force deflector type if non-zero deflector angle
    if defl_perp and not ana_type.endswith("p"):
        ana_type = ana_type + "p"

    alpha = np.radians(np.linspace(*theta_par_range, 101))
    if ana_type == "I":
        beta, xi = np.radians(polar), np.radians(tilt)
        chi = None
    elif ana_type == "II":
        beta, xi = np.radians(tilt), np.radians(polar)
        chi = None
    else:
        beta, xi, chi = np.radians(defl_perp), np.radians(tilt), np.radians(polar)

    # Calculate kpar
    Ek = hv - 4.35 - Eb
    kx, ky = _f_dispatcher(ana_type, Ek, alpha, beta, 0, chi, 0, 0, 0, xi, 0)

    return kx, ky


def get_kz_cut(
    hv=21.2,
    Eb=0,
    polar_or_tilt=0,
    theta_par_range=(-15, 15),
    V0=12,
):
    """Extract the kz-dep of the cut along the analyser slit for given parameters.

    Parameters
    ----------
    hv : float, optional
        Photon energy (eV) (default=21.2 eV).
    Eb : float, optional
        Binding energy (eV) (positive for below the Fermi level, default=0).
    theta_par_range : tuple, optional
        Range of theta_par values to calculate over in the form (start, stop);
        default=(-15, 15)).
    polar_or_tilt : float, optional
        Angle offset in direction along the slit (deg) (default=0).
    V0 : float, optional
        Inner potential (eV) (default=12).
    """

    # Calculate for a type I analyser
    alpha = np.radians(np.linspace(*theta_par_range, 101))

    # Calculate kpar
    Ek = hv - 4.35 - Eb
    kx, ky = _f_dispatcher(
        "I", Ek, alpha, 0, 0, 0, 0, 0, 0, np.radians(polar_or_tilt), 0
    )
    kz = _f_kz(Ek, kx, ky, V0)

    return kx, kz


# --------------------------------------------------------- #
#      Main k-space conversion functions                    #
# --------------------------------------------------------- #


def k_convert(
    da,
    eV=None,
    eV_slice=None,
    kx=None,
    ky=None,
    kz=None,
    return_kz_scan_in_hv=False,
    quiet=False,
):
    """Perform k-conversion of angle dispersion or mapping data.

    Parameters
    ----------
    da : xarray.DataArray
        Data to convert to k-space.
    eV : slice, optional
        Binding energy range to calculate over for the final converted data in the form slice(start, stop, step).
        If not provided, the full energy range of the data will be used.
        Only energies within the limits of the data will be considered.
    eV_slice : float or tuple, optional
        Optional argument to return a single slice in energy, optionally integrated over some range, akin to the MDC
        method. Should be specified as a tuple of (energy, width) where energy is the central energy to integrate over
        and width is the total width of the integration window. Takes precedence over eV if provided.
    kx : slice, optional
        kx range to calculate over for the final converted data in the form slice(start, stop, step).
        Use `None` for any componet you do not wish to restrict, e.g. `(None,None,0.01)` to set the step size to 0.01.
        Only considered if kx is a dispersive direction of the data.
        Only kx values within the limits of the data will be considered.
        If not provided, the full kx range of the data will be used.
    ky : slice, optional
        ky range to calculate over for the final converted data in the form slice(start, stop, step).
        See Also: kx.
    kz : slice, optional
        kz range to calculate over for the final converted data in the form slice(start, stop, step).
        See Also: kx.
    return_kz_scan_in_hv : bool, optional
        If True, returns the converted data as (hv, eV, k_||) not (kz, eV, k_||).
    quiet : bool, optional
        If True, suppresses warnings and hides progress bar after k-space conversion completion.

    Returns
    -------
    xarray.DataArray
        Data converted to k-space.
    """

    # Parse basic data properties
    loader = BaseDataLoader.get_loader(da.metadata.scan.loc)
    angles = loader._get_angles_Ishida_Shin(da, quiet=quiet)  # Get angles for k-conv
    # Ensure all angles are in radians, da energies are in eV and then dequantify
    unit_stripped_angles = {
        axis: (angle.to("rad").magnitude if isinstance(angle, pint.Quantity) else angle)
        for axis, angle in angles.items()
    }
    angles = unit_stripped_angles
    da = da.pint.to({"hv": "eV", "eV": "eV"}).pint.dequantify()

    # Make a progressbar
    pb_steps = 3
    if "hv" in da.dims and not return_kz_scan_in_hv:
        pb_steps += 1
    pbar = tqdm(
        total=pb_steps,
        desc="Converting data to k-space - initialising",
        leave=not quiet,
    )

    # Get relevant energy scale information
    wf = _get_wf(da)  # Work fn, array if hv scan else single value
    if "hv" in da.dims:
        hv = da.hv.data
    else:
        hv = da.metadata.photon.hv.to("eV").magnitude

    BE_scale = _get_BE_scale(da)  # Tuple of start, stop, step
    # Restrict to manual energy range if specified
    if eV is not None:
        BE_scale = (
            np.max([eV.start if eV.start is not None else float("-inf"), BE_scale[0]]),
            np.min([eV.stop if eV.stop is not None else float("inf"), BE_scale[1]]),
            eV.step if eV.step is not None else BE_scale[2],
        )
    if eV_slice is not None:  # Get the full slice
        BE_scale = (
            eV_slice[0] - eV_slice[1] / 2,
            eV_slice[0] + eV_slice[1] / 2,
            BE_scale[2],
        )

    # Get bounds of data for k-conversion - use highest hv for hv scan
    if "hv" in da.dims and hv[-1] > hv[0]:
        EK_range = [hv[-1] + BE_scale[0] - wf[-1], hv[-1] + BE_scale[1] - wf[-1]]
    elif "hv" in da.dims and hv[-1] < hv[0]:
        EK_range = [hv[0] + BE_scale[0] - wf[0], hv[0] + BE_scale[1] - wf[0]]
    else:
        EK_range = [hv + BE_scale[0] - wf, hv + BE_scale[1] - wf]
    alpha_range = np.asarray(
        [
            np.min(angles["alpha"]),
            angles["xi_0"],
            np.max(angles["alpha"]),
        ]
    )

    # Check if a 2D or 3D conversion is required & reshape arrays to make them broadcastable
    if np.min(angles["beta"]) != np.max(angles["beta"]):
        beta_range = np.asarray([np.min(angles["beta"]), np.max(angles["beta"])])
        n_interpolation_dims = 3
        EK_range, alpha_range, beta_range = _reshape_for_3d(
            EK_range, alpha_range, beta_range
        )
    else:
        n_interpolation_dims = 2
        EK_range, alpha_range = _reshape_for_2d(EK_range, alpha_range)
        beta_range = angles["beta"]

    # Get k-space values corresponding to extremes of range
    kx_, ky_ = _f_dispatcher(
        ana_type=angles["type"],
        Ek=EK_range,
        alpha=alpha_range,
        beta=beta_range,
        beta_0=angles["beta_0"],
        chi=angles["chi"],
        chi_0=angles["chi_0"],
        delta=angles["delta"],
        delta_0=angles["delta_0"],
        xi=angles["xi"],
        xi_0=angles["xi_0"],
    )

    # Determine ranges
    k_along_slit = _get_k_along_slit(kx_, ky_, angles["type"])
    default_k_step = np.ptp(k_along_slit) / (len(angles["alpha"]) - 1)
    kx_range = (np.min(kx_), np.max(kx_) + default_k_step, default_k_step)
    ky_range = (np.min(ky_), np.max(ky_) + default_k_step, default_k_step)

    # Restrict to manual k ranges if specified and if a relevant axis
    if kx is not None and (n_interpolation_dims == 3 or angles["type"] in ["I", "Ip"]):
        kx_range = (
            np.max([kx.start if kx.start is not None else float("-inf"), kx_range[0]]),
            np.min([kx.stop if kx.stop is not None else float("inf"), kx_range[1]]),
            kx.step if kx.step is not None else kx_range[2],
        )
    if ky is not None and (n_interpolation_dims == 3 or angles["type"] in ["II", "IIp"]):
        ky_range = (
            np.max([ky.start if ky.start is not None else float("-inf"), ky_range[0]]),
            np.min([ky.stop if ky.stop is not None else float("inf"), ky_range[1]]),
            ky.step if ky.step is not None else ky_range[2],
        )

    # Make the arrays of required angle and energy values
    kx_values = np.arange(*kx_range)
    ky_values = np.arange(*ky_range)

    if "hv" not in da.dims:
        KE_values_no_curv = np.arange(*BE_scale) + hv - wf
    else:
        # For an hv scan, need to extract different KE values for each hv value,
        # but then flatten them for interpolation as we still only need bilinear interpolation
        KE_values_no_curv = (
            np.arange(*BE_scale).reshape(1, -1) + hv.reshape(-1, 1) - wf.reshape(-1, 1)
        )
        KE_values_no_curv_shape = KE_values_no_curv.shape  # Keep for later
        KE_values_no_curv = KE_values_no_curv.flatten()

    # Create meshgrid of angle and energy values for the interpolation
    if n_interpolation_dims == 3:
        KE_values_no_curv, kx_values, ky_values = _reshape_for_3d(
            KE_values_no_curv, kx_values, ky_values
        )
    else:
        if angles["type"] in ["II", "IIp"]:
            # Take the average k value of the perp to slit direction
            kx_values = np.mean(kx_values)
            KE_values_no_curv, ky_values = _reshape_for_2d(KE_values_no_curv, ky_values)
        else:
            KE_values_no_curv, kx_values = _reshape_for_2d(KE_values_no_curv, kx_values)
            ky_values = np.mean(ky_values)

    pbar.update(1)
    pbar.set_description_str(
        "Converting data to k-space - calculating inverse angle transformations"
    )

    alpha, beta = _f_inv_dispatcher(
        ana_type=angles["type"],
        Ek=KE_values_no_curv,
        kx=kx_values,
        ky=ky_values,
        beta_0=angles["beta_0"],
        chi=angles["chi"],
        chi_0=angles["chi_0"],
        delta=angles["delta"],
        delta_0=angles["delta_0"],
        xi=angles["xi"],
        xi_0=angles["xi_0"],
    )

    # Determine the KE values including curvature correction
    if n_interpolation_dims == 2:
        Ek_new = _get_E_shift_at_theta_par(
            da,
            alpha * 180 / np.pi,
            np.broadcast_to(
                KE_values_no_curv.reshape(-1, 1),
                (
                    KE_values_no_curv.size,
                    _get_k_along_slit(kx_values, ky_values, angles["type"]).size,
                ),
            ),
        )
    elif n_interpolation_dims == 3:
        Ek_new = _get_E_shift_at_theta_par(
            da,
            alpha * 180 / np.pi,
            np.broadcast_to(
                KE_values_no_curv.reshape(-1, 1, 1),
                (KE_values_no_curv.size, kx_values.size, ky_values.size),
            ),
        )

    # Interpolate onto the desired range
    pbar.update(1)
    pbar.set_description_str("Converting data to k-space - interpolating")
    # Check if we can use the faster rectilinear methods
    is_rectilinear = True
    for i in ["eV", "theta_par"]:
        if not _is_linearly_spaced(
            da[i].data, tol=(da[i].data[1] - da[i].data[0]) * 1e-3
        ):
            is_rectilinear = False

    if n_interpolation_dims == 2:
        if is_rectilinear:
            interpolation_fn = _fast_bilinear_interpolate_rectilinear
        else:
            interpolation_fn = _fast_bilinear_interpolate

        k_along_slit_label = _get_k_along_slit("kx", "ky", angles["type"])

        # Deal with any sign changes required for theta_par scale conventions and any stacked axes
        loader._get_sign_convention("theta_par")

        if "hv" in da.dims:
            interpolated_data = []
            for i, hv in enumerate(da.hv.data):
                data_hv = da.disp_from_hv(hv)
                start_index = i * KE_values_no_curv_shape[1]
                end_index = start_index + KE_values_no_curv_shape[1]
                interpolated_data_hv_slice = xr.apply_ufunc(
                    interpolation_fn,
                    Ek_new[start_index:end_index, :],
                    alpha[start_index:end_index, :],
                    data_hv.eV.data,
                    angles["alpha"],
                    data_hv,
                    input_core_dims=[
                        ["eV", k_along_slit_label],
                        ["eV", k_along_slit_label],
                        ["eV"],
                        ["theta_par"],
                        ["eV", "theta_par"],
                    ],
                    output_core_dims=[["eV", k_along_slit_label]],
                    exclude_dims={"eV"},
                    vectorize=True,
                    dask="parallelized",
                    keep_attrs=True,
                )
                interpolated_data.append(interpolated_data_hv_slice)
            interpolated_data = xr.concat(interpolated_data, dim="hv")
            # Add co-ordinates
            interpolated_data.coords.update(
                {
                    k_along_slit_label: _get_k_along_slit(
                        kx_values, ky_values, angles["type"]
                    ).squeeze(),
                    "eV": np.arange(*BE_scale),
                    "hv": da.hv.data,
                }
            )
            interpolated_data = interpolated_data.pint.quantify(
                {k_along_slit_label: "1/angstrom", "eV": "eV", "hv": "eV"}
            )

        else:
            interpolated_data = xr.apply_ufunc(
                interpolation_fn,
                Ek_new,
                alpha,
                da.eV.data,
                angles["alpha"],
                da,
                input_core_dims=[
                    ["eV", k_along_slit_label],
                    ["eV", k_along_slit_label],
                    ["eV"],
                    ["theta_par"],
                    ["eV", "theta_par"],
                ],
                output_core_dims=[["eV", k_along_slit_label]],
                exclude_dims={"eV"},
                vectorize=True,
                dask="parallelized",
                keep_attrs=True,
            )
            # Add co-ordinates
            interpolated_data.coords.update(
                {
                    k_along_slit_label: _get_k_along_slit(
                        kx_values, ky_values, angles["type"]
                    ).squeeze(),
                    "eV": np.arange(*BE_scale),
                }
            )
            interpolated_data = interpolated_data.pint.quantify(
                {k_along_slit_label: "1/angstrom", "eV": "eV"}
            )
        # Add the perpendicular momentum to the data attributes
        interpolated_data.attrs[_get_k_perpto_slit("kx", "ky", angles["type"])] = (
            _get_k_perpto_slit(kx_values, ky_values, angles["type"]).squeeze()
        )
    else:  # Should be Fermi map
        # Other angular dimension - assume the only remaining dimension
        other_dim = list(set(da.dims) - set(["eV", "theta_par"]))[0]

        # Check linearity of remaining dimension
        is_rectilinear = is_rectilinear and _is_linearly_spaced(
            da[other_dim].data,
            tol=(da[other_dim].data[1] - da[other_dim].data[0]) * 1e-3,
        )

        if is_rectilinear:
            interpolation_fn = _fast_trilinear_interpolate_rectilinear
        else:
            interpolation_fn = _fast_trilinear_interpolate

        with numba_progress.ProgressBar(
            total=Ek_new.size,
            dynamic_ncols=True,
            delay=0.2,
            desc="Interpolating onto new grid",
            leave=False,
        ) as nb_pbar:
            interpolated_data = xr.apply_ufunc(
                interpolation_fn,
                Ek_new,
                alpha,
                beta,
                da.eV.data,
                angles["alpha"],
                angles["beta"],
                da,
                nb_pbar,
                input_core_dims=[
                    ["eV", "kx", "ky"],
                    ["eV", "kx", "ky"],
                    ["eV", "kx", "ky"],
                    ["eV"],
                    ["theta_par"],
                    [other_dim],
                    ["eV", "theta_par", other_dim],
                    [],
                ],
                output_core_dims=[["eV", "kx", "ky"]],
                exclude_dims={"eV"},
                vectorize=True,
                dask="parallelized",
                keep_attrs=True,
            ).transpose(
                _get_k_perpto_slit("kx", "ky", angles["type"]),
                "eV",
                _get_k_along_slit("kx", "ky", angles["type"]),
            )
        interpolated_data.coords.update(
            {
                "kx": kx_values.squeeze(),
                "eV": np.arange(*BE_scale),
                "ky": ky_values.squeeze(),
            }
        )
        interpolated_data = interpolated_data.pint.quantify(
            {"kx": "1/angstrom", "eV": "eV", "ky": "1/angstrom"}
        )

    if "hv" in da.dims and not return_kz_scan_in_hv:
        # If hv scan and if required, do kz-interpolation step to return the data in the form (kz, eV, k_||)
        pbar.update(1)
        pbar.set_description_str("Converting data to k-space - converting to kz")
        # Convert to kz
        interpolated_data = _convert_to_kz(
            interpolated_data, kz, wf, [np.min(Ek_new), np.max(Ek_new)]
        )

    # Do a hack to remove some noise at the boundary which can give negative values, screwing up the plots
    if da.min() <= 0:
        interpolated_data = interpolated_data.where(interpolated_data > 0, 0)

    # Update the energy type in data attributes
    interpolated_data.metadata.analyser.scan.eV_type = "Binding Energy"

    # Update the history
    hist_str = "Converted to k-space using the following parameters: "
    reference_angles = {k: v for k, v in angles.items() if ("_0" in k and v is not None)}
    hist_str += f"Reference angles: {reference_angles}, "
    if "hv" in da.dims:
        hist_str += f"Inner potential: {da.metadata.calibration.V0 or 12 * ureg('eV')}, "
    hist_str += f"Time taken: {pbar.format_dict['elapsed']:.2f}s."
    interpolated_data.history.add(hist_str)

    if eV_slice is not None:
        interpolated_data = interpolated_data.mean("eV")
        interpolated_data.history.add(
            f"Data integrated in energy about {eV_slice[0]} eV +/- {eV_slice[1] / 2} eV."
        )

    pbar.update(1)
    pbar.set_description_str("Converting data to k-space - complete")

    if pbar.format_dict["elapsed"] < 0.2:  # Hide if only ran for a short time
        pbar.leave = False
    pbar.close()

    return interpolated_data.pint.quantify()


def _convert_to_kz(da, kz, wf, Ek_range):
    # Get relevant parameters
    k_along_slit_str = list(set(da.dims) - {"hv", "eV"})[0]
    k_perp_slit_str = "ky" if k_along_slit_str == "kx" else "kx"
    k_perp_slit = da.attrs.get(k_perp_slit_str, 0)
    k_along_slit = da[k_along_slit_str].data
    V0 = da.metadata.calibration.V0
    if V0 is None:
        V0 = 12 * ureg("eV")
        analysis_warning(
            f"No inner potential provided, defaulting to V0={V0}.",
            "warning",
            "Missing inner potential",
        )
    V0 = V0.to("eV").magnitude

    # Get kz values corresponding to the extremes of the range
    kz_ = _f_kz(
        np.asarray(Ek_range).reshape(-1, 1),
        np.asarray([min(k_along_slit), 0, max(k_along_slit)]).reshape(1, -1),
        k_perp_slit,
        V0,
    )
    # Determine the kz values to interpolate onto, including manual range if specified
    default_k_step = (np.nanmax(kz_) - np.nanmin(kz_)) / (
        2 * len(da.hv)
    )  # Default is based on 0.5 step from the number of hv points
    kz_range = (np.nanmin(kz_), np.nanmax(kz_), default_k_step)
    # Restrict to manual kz range if specified
    if kz is not None:
        kz_range = (
            np.max([kz.start if kz.start is not None else float("-inf"), kz_range[0]]),
            np.min([kz.stop if kz.stop is not None else float("inf"), kz_range[1]]),
            kz.step if kz.step is not None else kz_range[2],
        )
    kz_values = np.arange(kz_range[0], kz_range[1] + kz_range[2], kz_range[2])

    # Do inverse transform to get the required hv values for interpolation [kz, eV, k_||]
    # Take the first work function value for getting this hv scale - need to account for this in the final interpolation
    hv_values = _f_inv_kz(
        kz_values.reshape(-1, 1, 1),
        k_along_slit.reshape(1, 1, -1),
        k_perp_slit,
        V0,
        da.eV.data.reshape(1, -1, 1),
        wf[0],
    )

    # Get true hv values accounting for the change which was previously encoded in an effective wf change
    wf_diff = wf - wf[0]
    true_hv = da.hv.data + wf_diff

    # Do the interpolation, data originally [hv, eV, k_||]
    k_along_slit_vectorised = np.broadcast_to(
        k_along_slit.reshape(1, 1, -1), hv_values.shape
    )
    BE_vectorised = np.broadcast_to(da.eV.data.reshape(1, -1, 1), hv_values.shape)
    with numba_progress.ProgressBar(
        total=hv_values.size,
        dynamic_ncols=True,
        delay=0.2,
        desc="Interpolating onto kz grid",
        leave=False,
    ) as nb_pbar:
        interpolated_data = xr.apply_ufunc(
            _fast_trilinear_interpolate,
            hv_values,
            BE_vectorised,
            k_along_slit_vectorised,
            true_hv,
            da.eV.data,
            k_along_slit,
            da.pint.dequantify(),
            nb_pbar,
            input_core_dims=[
                ["kz", "eV", k_along_slit_str],
                ["kz", "eV", k_along_slit_str],
                ["kz", "eV", k_along_slit_str],
                ["hv"],
                ["eV"],
                [k_along_slit_str],
                ["hv", "eV", k_along_slit_str],
                [],
            ],
            output_core_dims=[["kz", "eV", k_along_slit_str]],
            exclude_dims={},
            vectorize=True,
            dask="parallelized",
            keep_attrs=True,
        )

        # Add the kz axis to the data
        interpolated_data.coords.update(
            {
                "kz": kz_values.squeeze(),
            }
        )

    return interpolated_data.pint.quantify({"kz": "1/angstrom"})
