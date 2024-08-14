"""Functions used to apply k-space conversions to data.

"""

import numpy as np
import numexpr as ne
import xarray as xr
from scipy.constants import m_e, hbar, electron_volt, angstrom
import numba_progress
from tqdm.notebook import tqdm

from .fermi_level_correction import _get_wf, _get_BE_scale, _get_E_shift_at_theta_par
from ...utils.misc import analysis_warning
from ...utils.interpolation import (
    _is_linearly_spaced,
    _fast_bilinear_interpolate,
    _fast_trilinear_interpolate,
    _fast_bilinear_interpolate_rectilinear,
    _fast_trilinear_interpolate_rectilinear,
)
from ..fileIO.fileIO_opts import LocOpts

# Calculate kvac_const
KVAC_CONST = (2 * m_e / (hbar**2)) ** 0.5 * (electron_volt**0.5) * angstrom
PI = np.pi


# --------------------------------------------------------- #
# Mapping functions: angle -> k-space (in plane)            #
# following conventions and nomenclature of Ishida and Shin #
# Rev. Sci. Instrum. 89 (2018) 043903.                      #
# --------------------------------------------------------- #


def _f_dispatcher(ana_type, Ek, alpha, beta, delta_, xi_, beta_0=None, chi_=None):
    if ana_type == "I":
        return _fI(alpha, beta - beta_0, delta_, xi_, Ek)
    elif ana_type == "II":
        return _fII(alpha, beta - beta_0, delta_, xi_, Ek)
    elif ana_type == "Ip":
        return _fIp(alpha, beta, delta_, xi_, chi_, Ek)
    elif ana_type == "IIp":
        return _fIIp(alpha, beta, delta_, xi_, chi_, Ek)


def _f_inv_dispatcher(ana_type, Ek, kx, ky, delta_, xi, xi_0, beta_0=None, chi_=None):
    if ana_type == "I":
        return _fI_inv(kx, ky, delta_, xi - xi_0, Ek, beta_0)
    elif ana_type == "II":
        return _fII_inv(kx, ky, delta_, xi - xi_0, Ek, beta_0)
    elif ana_type == "Ip":
        return _fIp_inv(kx, ky, delta_, xi - xi_0, chi_, Ek)
    elif ana_type == "IIp":
        return _fIIp_inv(kx, ky, delta_, xi - xi_0, chi_, Ek)


def _fI(alpha, beta_, delta_, xi_, Ek):  # Type I, no deflector
    """Convert angles to k-space for a type-I analyser with no deflector,
    following the conventions and nomenclature of Ishida and Shin, Rev. Sci. Instrum. 89 (2018) 043903.

    Parameters
    -----------
    alpha : float or np.ndarray
        theta_par angle (analyser slit angle, deg)
    beta_ : float or np.ndaarray
        polar angle - reference angle (deg)
    delta_ : float
        azi angle - reference angle (deg)
    xi_ : float or np.ndarray
        tilt angle - reference angle (deg)
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

    # Convert angles to radians
    alpha = np.radians(alpha)
    beta_ = np.radians(beta_)
    delta_ = np.radians(delta_)
    xi_ = np.radians(xi_)

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
           theta_par angle (analyser slit angle, deg)
       beta_ : float or np.ndarray
           tilt angle - reference angle (deg)
       delta_ : float
           azi angle - reference angle (deg)
       xi_ : float or np.ndarray
           polar angle - reference angle (deg)
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

    # Convert angles to radians
    alpha = np.radians(alpha)
    beta_ = np.radians(beta_)
    delta_ = np.radians(delta_)
    xi_ = np.radians(xi_)

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
        theta_par angle (analyser slit angle) + defl_par (deflector angle along the slit) (deg)
    beta : float or np.ndarray
        defl_perp (deflector angle perpendicular to the slit, deg)
    delta_ : float
        azi angle - reference angle (deg)
    xi_ : float or np.ndarray
        tilt angle - reference angle (deg)
    chi_ : float or np.ndarray
        polar angle - reference angle (deg)
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

    # Convert angles to radians
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    delta_ = np.radians(delta_)
    xi_ = np.radians(xi_)
    chi_ = np.radians(chi_)

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
        theta_par angle (analyser slit angle) + defl_par (deflector angle along the slit) (deg)
    beta : float or np.ndarray
        defl_perp (deflector angle perpendicular to the slit, deg)
    delta_ : float
        azi angle - reference angle (deg)
    xi_ : float or np.ndarray
        tilt angle - reference angle (deg)
    chi_ : float or np.ndarray
        polar angle - reference angle (deg)
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

    # Convert angles to radians
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    delta_ = np.radians(delta_)
    xi_ = np.radians(xi_)
    chi_ = np.radians(chi_)

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
        azi angle - reference angle (deg)
    xi : float
        tilt angle (deg)
    Ek : float
        Kinetic energy (eV)
    beta_0 : float, optional
        polar angle offset

    Returns
    -------
    alpha : float or np.ndarray
        theta_par angle (analyser slit angle, deg)
    beta : float or np.ndarray
        polar angle (deg)
    """
    # k_vacuum from KE
    kvac_str = "(KVAC_CONST * sqrt(Ek))"

    # Convert angles to radians
    delta_ = np.radians(delta_)
    xi = np.radians(xi)
    beta_0 = np.radians(beta_0)

    # Mapping function (include convert to degrees directly for speed)
    alpha = ne.evaluate(
        f"arcsin((sin(xi) * sqrt({kvac_str}**2 - kx**2 - ky**2) - cos(xi) * (kx * cos(delta_) + ky * sin(delta_))) "
        f"/ {kvac_str}) * 180 / PI"
    )

    beta = ne.evaluate(
        "(beta_0 + (arctan((kx * sin(delta_) - ky * cos(delta_)) /"
        " (kx * sin(xi) * cos(delta_) + ky * sin(xi) * sin(delta_) "
        f"+ cos(xi) * sqrt({kvac_str}**2 - kx**2 - ky**2))))) * 180 / PI"
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
        azi angle - reference angle (deg)
    xi : float
        polar angle (deg)
    Ek : float or np.ndarray
        Kinetic energy (eV)
    beta_0 : float, optional
        tilt angle offset

    Returns
    -------
    alpha : float or np.ndarray
        theta_par angle (analyser slit angle, deg)
    beta : float or np.ndarray
        tilt angle (deg)
    """

    # k_vacuum from KE
    kvac_str = "(KVAC_CONST * sqrt(Ek))"

    # Convert angles to radians
    delta_ = np.radians(delta_)
    xi = np.radians(xi)
    beta_0 = np.radians(beta_0)

    # Mapping function (include convert to degrees directly for speed)
    alpha = ne.evaluate(
        f"arcsin((sin(xi) * sqrt({kvac_str}**2 - ((kx * sin(delta_) - ky * cos(delta_))**2)) "
        f"- cos(xi) * (kx * sin(delta_) - ky * cos(delta_))) / {kvac_str}) * 180 / PI"
    )
    beta = ne.evaluate(
        f"(beta_0 + arctan((kx * cos(delta_) + ky * sin(delta_)) / sqrt({kvac_str}**2 - kx**2 - ky**2))) * 180 / PI"
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
        azi angle (deg), relative to 'normal' emission
    xi_ : float
        tilt angle (deg), relative to normal emission
    chi_ : float
        polar angle (deg), relative to normal emission
    Ek : float
        Kinetic energy (eV)

    Returns
    -------
    alpha : float or np.ndarray
        theta_par angle (analyser slit angle, deg)
    beta : float or np.ndarray
        polar angle (deg)
    """

    # k_vacuum from KE
    kvac_str = "(KVAC_CONST * sqrt(Ek))"
    k2p_str = f"sqrt({kvac_str}**2 - kx**2 - ky**2)"

    # Convert angles to radians
    delta_ = np.radians(delta_)
    xi_ = np.radians(xi_)
    chi_ = np.radians(chi_)

    # Mapping functions
    arg1_str = f"({_tij(31)} * kx) + ({_tij(32)} * ky) + ({_tij(33)} * {k2p_str})"
    arg2_str = f"({_tij(11)} * kx) + ({_tij(12)} * ky) + ({_tij(13)} * {k2p_str})"
    arg3_str = f"({_tij(21)} * kx) + ({_tij(22)} * ky) + ({_tij(23)} * {k2p_str})"

    alpha = ne.evaluate(
        f"-arccos({arg1_str} / {kvac_str}) * {arg2_str} / sqrt({kvac_str}**2 - {arg1_str}**2) * 180 / PI"
    )
    beta = ne.evaluate(
        f"-arccos({arg1_str} / {kvac_str}) * {arg3_str} / sqrt({kvac_str}**2 - {arg1_str}**2) * 180 / PI"
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
        azi angle (deg), relative to 'normal' emission
    xi_ : float
        tilt angle (deg), relative to normal emission
    chi_ : float
        polar angle (deg), relative to normal emission
    Ek : float
        Kinetic energy (eV)

    Returns
    -------
    alpha : float or np.ndarray
        theta_par angle (analyser slit angle, deg)
    beta : float or np.ndarray
        mapping angle (deg)
    """

    # k_vacuum from KE
    kvac_str = "(KVAC_CONST * sqrt(Ek))"
    k2p_str = f"sqrt({kvac_str}**2 - kx**2 - ky**2)"

    # Convert angles to radians
    delta_ = np.radians(delta_)
    xi_ = np.radians(xi_)
    chi_ = np.radians(chi_)

    # Mapping functions
    arg1_str = f"({_tij(31)} * kx) + ({_tij(32)} * ky) + ({_tij(33)} * {k2p_str})"
    arg2_str = f"({_tij(21)} * kx) + ({_tij(22)} * ky) + ({_tij(23)} * {k2p_str})"
    arg3_str = f"({_tij(11)} * kx) + ({_tij(12)} * ky) + ({_tij(13)} * {k2p_str})"

    alpha = ne.evaluate(
        f"-arccos({arg1_str} / {kvac_str}) * {arg2_str} / sqrt({kvac_str}**2 - {arg1_str}**2) * 180 / PI"
    )
    beta = ne.evaluate(
        f"-arccos({arg1_str} / {kvac_str}) * {arg3_str} / sqrt({kvac_str}**2 - {arg1_str}**2) * 180 / PI"
    )

    return alpha, beta


def _get_angles_for_k_conv(data):
    """
    Get the angles for the k-space conversion.

    Parameters
    ----------
    data : xarray.DataArray
        Data to convert to k-space.

    Returns
    -------
    tuple
        Angles for the k-space conversion.
    """

    # Determine if analyser should be treated as type I, II, I' or II'
    # using nomenclature from Ishida and Shin, Rev. Sci. Instrum. 89 (2018) 043903
    loc = data.attrs.get("beamline")
    _conventions = LocOpts.get_conventions(loc)
    ana_type = _conventions.get("ana_type")

    angles_to_extract = [
        "theta_par",
        "polar",
        "tilt",
        "azi",
        "ana_polar",
        "defl_par",
        "defl_perp",
    ]

    # Get the raw angles from the data
    angles_to_warn = ["polar", "tilt", "azi"]
    angles = {}
    warn_str = ""
    for i in angles_to_extract:
        if i in data.coords:
            angles[i] = data.coords[i].data
        else:
            angle = data.attrs.get(i)
            if angle is not None:
                angles[i] = angle
            else:
                angles[i] = 0
                if i in angles_to_warn:
                    warn_str += f"{i}: 0, "

    # Extract normal emissions, trying to make some sensible guesses for parameters not specified
    norm_angles_to_extract = [
        "norm_polar",
        "norm_tilt",
        "norm_azi",
    ]
    for i in norm_angles_to_extract:
        angle = data.attrs.get(i)
        if angle is not None:
            angles[i] = angle
        elif i == "norm_azi":
            norm_azi = angles["azi"]
            angles[i] = norm_azi
            warn_str += f"{i}: {norm_azi}, "
        elif i == "norm_polar" and (ana_type == "I" or ana_type == "Ip"):
            norm_polar = angles["polar"]
            if isinstance(norm_polar, np.ndarray):
                norm_polar = 0
            angles[i] = norm_polar
            warn_str += f"{i}: {norm_polar}, "
        elif i == "norm_tilt" and "II" in ana_type:
            norm_tilt = angles["tilt"]
            if isinstance(norm_tilt, np.ndarray):
                norm_tilt = 0
            angles[i] = norm_tilt
            warn_str += f"{i}: {norm_tilt}, "
        else:
            angles[i] = 0
            if i in angles_to_warn:
                warn_str += f"{i}:"

    # Give a warning if parameters have been assumed
    if warn_str:
        warn_str = (
            f"Some manipulator data and/or normal emission data was missing or could not be passed. "
            f"Assuming default values of: {warn_str.rstrip(", ")}."
        )
        analysis_warning(warn_str, "warning", "Analysis warning")

    # Above analyser determination gives the analyser capabilities, but for deflector types we need to check if
    # mapping was actually being performed without deflectors - if not, fall back to the non-deflector type
    if ana_type == "Ip" or ana_type == "IIp":
        if np.all(angles["defl_par"] == 0) and np.all(angles["defl_perp"] == 0):
            ana_type = "I" if ana_type == "Ip" else "II"

    # Put angles in correct notation cf. Ishida and Shin, Rev. Sci. Instrum. 89 (2018) 043903
    angles_out = {}
    angles_out["ana_type"] = ana_type
    angles_out["delta_"] = (angles["azi"] - angles["norm_azi"]) * _conventions.get(
        "azi"
    )
    if ana_type == "I":  # Type I
        angles_out["alpha"] = angles["theta_par"] * _conventions.get("theta_par")
        angles_out["beta"] = (angles["polar"] * _conventions.get("polar")) + (
            angles["ana_polar"] * _conventions.get("ana_polar")
        )
        angles_out["beta_0"] = angles["norm_polar"] * _conventions.get("polar")
        angles_out["xi"] = angles["tilt"] * _conventions.get("tilt")
        angles_out["xi_0"] = angles["norm_tilt"] * _conventions.get("tilt")
    elif ana_type == "II":  # Type II
        angles_out["alpha"] = angles["theta_par"] * _conventions.get("theta_par")
        angles_out["beta"] = angles["tilt"] * _conventions.get("tilt")
        angles_out["beta_0"] = angles["norm_tilt"] * _conventions.get("tilt")
        angles_out["xi"] = (angles["polar"] * _conventions.get("polar")) + (
            angles["ana_polar"] * _conventions.get("ana_polar")
        )
        angles_out["xi_0"] = angles["norm_polar"] * _conventions.get("polar")
    elif ana_type == "Ip" or ana_type == "IIp":  # Type I' or Type II'
        angles_out["alpha"] = angles["theta_par"] * _conventions.get("theta_par") + (
            angles["defl_par"] * _conventions.get("defl_par")
        )
        angles_out["beta"] = angles["defl_perp"] * _conventions.get("defl_perp")
        angles_out["beta_0"] = None
        angles_out["xi"] = (angles["tilt"]) * _conventions.get("tilt")
        angles_out["xi_0"] = angles["norm_tilt"] * _conventions.get("tilt")
        angles_out["chi"] = (angles["polar"] * _conventions.get("polar")) + (
            angles["ana_polar"] * _conventions.get("ana_polar")
        )
        angles_out["chi_0"] = angles["norm_polar"] * _conventions.get("polar")

    return angles_out


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


# --------------------------------------------------------- #
#      Main k-space conversion functions                    #
# --------------------------------------------------------- #


def k_conv(data, eV=None, kx=None, ky=None, quiet=False):
    """Perform k-conversion of angle dispersion or mapping data.

    Parameters
    ----------
    data : xarray.DataArray
        Data to convert to k-space.
    eV : slice, optional
        Binding energy range to calculate over for the final converted data in the form slice(start, stop, step).
        If not provided, the full energy range of the data will be used.
        Only energies within the limits of the data will be considered.
    kx : slice, optional
        kx range to calculate over for the final converted data in the form slice(start, stop, step).
        Only considered if kx is a dispersive direction of the data.
        Only kx values within the limits of the data will be considered.
        If not provided, the full kx range of the data will be used.
    ky : slice, optional
        ky range to calculate over for the final converted data in the form slice(start, stop, step).
        Only considered if ky is a dispersive direction of the data.
        Only ky values within the limits of the data will be considered.
        If not provided, the full ky range of the data will be used.
    quiet : bool, optional
        If True, suppresses warnings and hides progress bar after k-space conversion completion.

    Returns
    -------
    xarray.DataArray
        Data converted to k-space.
    """
    # Make a progressbar
    pbar = tqdm(
        total=5, desc="Converting data to k-space - initialising", leave=not quiet
    )

    # Get relevant energy scale information
    wf = _get_wf(data)
    hv = data.attrs.get("hv")
    BE_scale = _get_BE_scale(data)  # Tuple of start, stop, step
    # Restrict to manual energy range if specified
    if eV is not None:
        BE_scale = (
            np.max([eV.start if eV.start is not None else float("-inf"), BE_scale[0]]),
            np.min([eV.stop if eV.stop is not None else float("inf"), BE_scale[1]]),
            eV.step if eV.step is not None else BE_scale[2],
        )

    # Get dictionary of relevant angles from the data in correct format for k-conversion functions
    angles = _get_angles_for_k_conv(data)
    ana_type = angles["ana_type"]

    # Get bounds of data for k-conversion
    EK_range = [hv + BE_scale[0] - wf, hv + BE_scale[1] - wf]
    alpha_range = np.asarray(
        [np.min(angles["alpha"]), angles["xi_0"], np.max(angles["alpha"])]
    )

    # Check if a 2D or 3D conversion is required & reshape arrays to make them broadcastable
    if isinstance(angles["beta"], np.ndarray):
        beta_range = (
            np.asarray([np.min(angles["beta"]), np.max(angles["beta"])])
            + angles["beta_0"]
        )
        ndim = 3
        EK_range, alpha_range, beta_range = _reshape_for_3d(
            EK_range, alpha_range, beta_range
        )
    else:
        ndim = 2
        EK_range, alpha_range = _reshape_for_2d(EK_range, alpha_range)
        beta_range = angles["beta"] + angles["beta_0"]

    # Get k-space values corresponding to extremes of range
    kx_, ky_ = _f_dispatcher(
        ana_type,
        EK_range,
        alpha_range,
        beta_range,
        angles["delta_"],
        angles["xi"] - angles["xi_0"],
        angles["beta_0"],
        angles.get("chi", 0) - angles.get("chi_0", 0),
    )

    # Determine ranges
    k_along_slit = _get_k_along_slit(kx_, ky_, ana_type)
    default_k_step = np.ptp(k_along_slit) / (len(angles["alpha"]) - 1)
    kx_range = (np.min(kx_), np.max(kx_) + default_k_step, default_k_step)
    ky_range = (np.min(ky_), np.max(ky_) + default_k_step, default_k_step)
    # Restrict to manual k ranges if specified and if a relevant axis
    if kx is not None and (ndim == 3 or ana_type in ["I", "Ip"]):
        kx_range = (
            np.max([kx.start if kx.start is not None else float("-inf"), kx_range[0]]),
            np.min([kx.stop if kx.stop is not None else float("inf"), kx_range[1]]),
            kx.step if kx.step is not None else kx_range[2],
        )
    if ky is not None and (ndim == 3 or ana_type in ["II", "IIp"]):
        ky_range = (
            np.max([ky.start if ky.start is not None else float("-inf"), ky_range[0]]),
            np.min([ky.stop if ky.stop is not None else float("inf"), ky_range[1]]),
            ky.step if ky.step is not None else ky_range[2],
        )

    # Make the arrays of required angle and energy values
    kx_values = np.arange(*kx_range)
    ky_values = np.arange(*ky_range)
    KE_values_no_curv = np.arange(*BE_scale) + hv - wf

    # Create meshgrid of angle and energy values for the interpolation
    if ndim == 3:
        KE_values_no_curv, kx_values, ky_values = _reshape_for_3d(
            KE_values_no_curv, kx_values, ky_values
        )
    else:
        if ana_type in ["II", "IIp"]:
            # Take the average k value of the perp to slit direction
            kx_values = np.mean(kx_values)
            KE_values_no_curv, ky_values = _reshape_for_2d(KE_values_no_curv, ky_values)
        else:
            KE_values_no_curv, kx_values = _reshape_for_2d(KE_values_no_curv, kx_values)
            ky_values = np.mean(ky_values)

    alpha, beta = _f_inv_dispatcher(
        ana_type,
        KE_values_no_curv,
        kx_values,
        ky_values,
        angles["delta_"],
        angles["xi"],
        angles["xi_0"],
        angles["beta_0"],
        angles.get("chi", 0) - angles.get("chi_0", 0),
    )

    # Determine the KE values including curvature correction
    if ndim == 2:
        Ek_new = _get_E_shift_at_theta_par(
            data,
            alpha,
            np.broadcast_to(
                KE_values_no_curv.reshape(-1, 1),
                (
                    KE_values_no_curv.size,
                    _get_k_along_slit(kx_values, ky_values, ana_type).size,
                ),
            ),
        )
    elif ndim == 3:
        Ek_new = _get_E_shift_at_theta_par(
            data,
            alpha,
            np.broadcast_to(
                KE_values_no_curv.reshape(-1, 1, 1),
                (KE_values_no_curv.size, kx_values.size, ky_values.size),
            ),
        )

    # Interpolate onto the desired range
    # Check if we can use the faster rectilinear methods
    is_rectilinear = False
    for i in data.dims:
        if _is_linearly_spaced(
            data[i].data, tol=(data[i].data[1] - data[i].data[0]) * 1e-3
        ):
            is_rectilinear = True

    if ndim == 2:
        if is_rectilinear:
            interpolated_data = _fast_bilinear_interpolate_rectilinear(
                Ek_new, alpha, data.eV.data, data.theta_par.data, data.data
            )
        else:
            interpolated_data = _fast_bilinear_interpolate(
                Ek_new, alpha, data.eV.data, data.theta_par.data, data.data
            )
    else:
        other_dim = list(set(data.dims) - set(["eV", "theta_par"]))[0]
        with numba_progress.ProgressBar(
            total=Ek_new.size,
            dynamic_ncols=True,
            delay=0.2,
            desc="Interpolating onto new grid",
            leave=False,
        ) as nb_pbar:
            if is_rectilinear:
                interpolated_data = _fast_trilinear_interpolate_rectilinear(
                    Ek_new,
                    alpha,
                    beta,
                    data.eV.data,
                    data.theta_par.data,
                    data[other_dim].data,
                    data.transpose("eV", "theta_par", other_dim).data,
                    nb_pbar,
                )
            else:
                interpolated_data = _fast_trilinear_interpolate(
                    Ek_new,
                    alpha,
                    beta,
                    data.eV.data,
                    data.theta_par.data,
                    data[other_dim].data,
                    data.transpose("eV", "theta_par", other_dim).data,
                    nb_pbar,
                )

    # Create the new data array
    if ndim == 2:
        if ana_type in ["II", "IIp"]:
            interpolated_data = xr.DataArray(
                interpolated_data,
                coords={"eV": np.arange(*BE_scale), "ky": ky_values.squeeze()},
                dims=["eV", "ky"],
            )
        elif ana_type in ["I", "Ip"]:
            interpolated_data = xr.DataArray(
                interpolated_data,
                coords={"eV": np.arange(*BE_scale), "kx": kx_values.squeeze()},
                dims=["eV", "kx"],
            )
    elif ndim == 3:
        interpolated_data = xr.DataArray(
            interpolated_data,
            coords={
                "eV": np.arange(*BE_scale),
                "kx": kx_values.squeeze(),
                "ky": ky_values.squeeze(),
            },
            dims=["eV", "kx", "ky"],
        ).transpose("kx", "eV", "ky")

    # Do a hack to remove some noise at the boundary which can give negative values, screwing up the plots
    if data.min() >= 0:
        interpolated_data = interpolated_data.where(interpolated_data > 0, 0)

    # If 2D data, add the perpendicular momentum to the data attributes
    if ndim == 2:
        if ana_type in ["II", "IIp"]:
            interpolated_data.attrs["kx"] = kx_values.squeeze()
        elif ana_type in ["I", "Ip"]:
            interpolated_data.attrs["ky"] = ky_values.squeeze()

    pbar.update(5)
    pbar.set_description_str("Done")

    if pbar.format_dict["elapsed"] < 0.2:  # Hide if only ran for a short time
        pbar.leave = False
    pbar.close()

    return interpolated_data


def kz_conv():
    pass
