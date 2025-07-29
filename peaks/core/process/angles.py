import numbers

import numpy as np
import xarray as xr

from peaks.core.options import LocOpts
from peaks.core.utils.misc import analysis_warning

# Constants for angle names
ANGLE_NAMES = [
    "theta_par",
    "polar",
    "tilt",
    "azi",
    "ana_polar",
    "defl_par",
    "defl_perp",
]

NORM_ANGLE_NAMES = [
    "norm_polar",
    "norm_tilt",
    "norm_azi",
]

ALLOWED_ANGLES = ANGLE_NAMES + NORM_ANGLE_NAMES


def _get_conventions(data):
    """Get the angle conventions for the data.

    Parameters
    ----------
    data : xarray.DataArray
        Data for which to get the angle conventions.

    Returns
    -------
    class
        Relevant data convention Class instance.
    """

    loc = data.attrs.get("beamline")
    return LocOpts.get_conventions(loc)


def _get_raw_angles(
    data,
    angle_names,
    default_value=0,
    quiet=False,
    warn_prefix="",
):
    """
    Get the raw angles from the data, with sign conventions of the beamline.

    Parameters
    ----------
    data : xarray.DataArray
        Data from which to get the angles.
    angle_names : list
        List of angle names to retrieve from the data.
    default_value : int or float, optional
        Default value to use if an angle is missing. Defaults to 0.
    quiet : bool, optional
        Whether to suppress warnings of missing angles. Defaults to False.
    warn_prefix : str, optional
        Prefix to add to warning messages. Defaults to an empty string.

    Returns
    -------
    dict
        Dictionary of angles retrieved from the data.
    """
    angles = {}
    warn_str = ""
    for angle_name in angle_names:
        if angle_name in data.coords:
            angles[angle_name] = data.coords[angle_name].data
        else:
            angle = data.attrs.get(angle_name)
            if angle is not None:
                angles[angle_name] = angle
            else:
                angles[angle_name] = default_value
                if not quiet:
                    warn_str += f"{angle_name}: {default_value}, "
    if warn_str:
        warn_message = f"{warn_prefix}Assuming default values for missing angles: {warn_str.rstrip(', ')}."
        analysis_warning(warn_message, "warning", "Analysis warning", quiet)
    return angles


def _parse_normal_angle(angles, norm_angles, norm_angle_list, conventions):
    """
    Parse a single normal emission angle from the supplied angle specification with sign conventions of Ishida & Shin.

    Parameters
    ----------
    angles : dict
        Dictionary of angles from the data.
    norm_angles : dict
        Dictionary of normal emission angles to set.
    norm_angle_list : list
        List of angle names to consider for normal emission.
    conventions : dict
        Dictionary of angle conventions.

    Returns
    -------
    float or None
        Parsed normal emission angle (in `peaks` [Ishida&Shin compatible] co-ordinate system),
        or `None` if nothing relevant is specified for this axis.
    """
    specified_angle = [i for i in norm_angle_list if i in norm_angles]
    if len(specified_angle) > 1:
        raise ValueError(f"Only one of {norm_angle_list} can be specified.")
    elif len(specified_angle) == 0:
        return None
    angle_name = specified_angle[0]
    norm_angle = norm_angles[angle_name] * conventions.get(angle_name, 1)

    # Handle the sign conventions to make consistent with Ishida and Shin definitions
    if angle_name in ["theta_par", "azi"]:
        norm_angle *= -1
    if angle_name == "defl_perp" and conventions.get("ana_type") in ["II", "IIp"]:
        norm_angle *= -1

    # Deal with the other relevant angles
    other_angles = [i for i in norm_angle_list if i != angle_name]
    for i in other_angles:
        angle_to_add = angles.get(i, 0)
        angle_to_add *= conventions.get(i, 1)
        if isinstance(angle_to_add, numbers.Number):
            norm_angle += angle_to_add
        else:
            raise ValueError(
                f"Angle {i} is not a number. Cannot parse a unique normal emission value from the supplied data "
                f"and normal emission angle choice."
            )
    return norm_angle


def _parse_norm_angles(data, norm_angles, quiet):
    """Parse the normal emission angles from the data, given a set of angle specifiers.

    Parameters
    ----------
    data : xarray.DataArray
        Data for which to set the normal emission angles.
    norm_angles : dict
        Dictionary of angles to set normal emission from. Supplied in common peaks angle name
        (e.g. `theta_par`, `polar`, etc.).
    quiet : bool
        Whether to suppress warnings.

    Returns
    -------
    dict :
        Dictionary of normal emission angles that could be parsed from the supplied arguments.
    """

    # Get angle conventions and analyser type
    conventions = _get_conventions(data)
    ana_type = conventions.get("ana_type")

    # Check if the angles are in the correct format
    if not norm_angles:
        raise ValueError(
            "No angles supplied. Either supply a dictionary of angles or pass as keyword arguments."
        )
    invalid_angles = [i for i in norm_angles.keys() if i not in ALLOWED_ANGLES]
    if invalid_angles:
        raise ValueError(
            f"Angles {invalid_angles} are not allowed. Allowed angles are {ALLOWED_ANGLES}."
        )

    # Get the raw angles from the data
    angles = _get_raw_angles(data, ANGLE_NAMES, quiet=quiet)

    # Calculate the normal emission angles
    if "norm_polar" in norm_angles:
        norm_polar = norm_angles["norm_polar"]
    else:
        if ana_type in ["I", "Ip"]:
            norm_polar = _parse_normal_angle(
                angles, norm_angles, ["polar", "ana_polar", "defl_perp"], conventions
            )
        elif ana_type in ["II", "IIp"]:
            norm_polar = _parse_normal_angle(
                angles,
                norm_angles,
                ["polar", "ana_polar", "theta_par", "defl_par"],
                conventions,
            )
    if "norm_tilt" in norm_angles:
        norm_tilt = norm_angles["norm_tilt"]
    else:
        if ana_type in ["I", "Ip"]:
            norm_tilt = _parse_normal_angle(
                angles, norm_angles, ["theta_par", "tilt", "defl_par"], conventions
            )
        elif ana_type in ["II", "IIp"]:
            norm_tilt = _parse_normal_angle(
                angles, norm_angles, ["tilt", "defl_perp"], conventions
            )
    if "norm_azi" in norm_angles:
        norm_azi = norm_angles["norm_azi"]
    else:
        norm_azi = _parse_normal_angle(angles, norm_angles, ["azi"], conventions)

    # Make a dictionary of available normal emissions to return
    norm_emissions = {
        key: val
        for key, val in zip(
            ["norm_polar", "norm_tilt", "norm_azi"],
            [norm_polar, norm_tilt, norm_azi],
            strict=True,
        )
        if val is not None
    }

    return norm_emissions


def _set_norm_angles(
    data, norm_angles=None, quiet=False, update_in_place=True, **kwargs
):
    """
    Set the normal emission angles for the data. This can be called with updating the attributes of the existing `data`
    :class:`xarray.DataArray` (accessed via the `.norm_angles.set()` accessor) or returning a copy with the updated
    metadata (accessed via the `norm_angles.assign()`), for chaining method calls together.

    Parameters
    ----------
    data : xarray.DataArray
        Data for which to set the normal emission angles.
    norm_angles : dict, optional
        Dictionary of angles to set normal emission from. Supplied as original angle name
        (e.g. `{theta_par: 12, polar: 5}`), or using `norm_polar`, `norm_tilt`, `norm_azi`.
        Defaults to None, in which case angles must be defined as keyword arguments.
        If supplied, takes precedence over kwargs.
    quiet : bool, optional
        Whether to suppress warnings. Defaults to False.
    update_in_place : bool, optional
        Whether to update the data in place. Defaults to True.
    **kwargs : dict
        Keyword arguments of angles to set (e.g. `theta_par=12`, `polar=5`, `norm_azi=-5`).

    Returns
    -------
    xarray.DataArray : optional
        If `update_in_place` is False, returns a copy of the original `data` :class:`xarray.DataArray` with the normal
        emission angles applied. Otherwise, returns None with the emission angle metadata updated in place on the
        original :class:`xarray.DataArray`.

    Examples
    --------
    Example usage is as follows::
        import peaks as pks

        # Load data
        FS = pks.load("path/to/data")

        # Set the normal emission angles in FS
        FS.norm_angles.set(theta_par=12, polar=5)

        # Alternatively, to return a copy of the data with the normal emission angles applied
        FS_with_norm = FS.norm_angles.assign(theta_par=12, polar=5)
    """
    if norm_angles is None:
        norm_angles = kwargs

    # Parse the angles
    norm_angles_out = _parse_norm_angles(data, norm_angles, quiet)

    # History to update
    hist = (
        f"Normal emission angles set to: {norm_angles_out} from user input {norm_angles}"
    )

    if update_in_place:
        # Update the data attributes and history
        data.attrs.update(norm_angles_out)
        data.history.add(hist)
    else:
        # Assign the data attributes and history
        return data.assign_attrs(norm_angles_out).history.assign(
            hist,
        )


def _parse_manip_angles_from_norm(data, norm_angles):
    """Parse the manipulator angles from the normal emission angles.

    Parameters
    ----------
    data : xarray.DataArray
        Data for which to parse the manipulator angles.
    norm_angles : dict
        Dictionary of normal emission angles.

    Returns
    -------
    dict
        Dictionary of manipulator angles in the format {f"norm_{axis}_manip": {axis_name: norm_value}} for available
        axes.
    """
    # Get angle conventions and analyser type
    conventions = _get_conventions(data)

    # Calculate the manipulator angles from the normal emission angles
    manip_normals = dict()
    norm_polar_name = conventions.get("polar_name")
    if norm_polar_name and norm_angles.get("norm_polar") is not None:
        norm_polar_manip = {
            norm_polar_name: norm_angles["norm_polar"] * conventions.get("polar", 1)
        }
        manip_normals["norm_polar_manip"] = norm_polar_manip
    norm_tilt_name = conventions.get("tilt_name")
    if norm_tilt_name and norm_angles.get("norm_tilt") is not None:
        norm_tilt_manip = {
            norm_tilt_name: norm_angles["norm_tilt"] * conventions.get("tilt", 1)
        }
        manip_normals["norm_tilt_manip"] = norm_tilt_manip
    norm_azi_name = conventions.get("azi_name")
    if norm_azi_name and norm_angles.get("norm_azi") is not None:
        norm_azi_manip = {
            norm_azi_name: norm_angles["norm_azi"] * conventions.get("azi", 1)
        }
        manip_normals["norm_azi_manip"] = norm_azi_manip

    return manip_normals


def _get_norm_angles(data):
    """Get the normal emission angles from the data.

    Parameters
    ----------
    data : xarray.DataArray
        Data for which to get the normal emission angles.

    Returns
    -------
    dict
        Dictionary of normal emission angles in both `peaks` format and where applicable the true manipulator angles.
    """

    # Get the peaks normal emission angles from the attributes
    norm_angles = {name: data.attrs.get(name) for name in NORM_ANGLE_NAMES}

    # Parse the manipulator angles from the normal emission angles
    norm_angles.update(_parse_manip_angles_from_norm(data, norm_angles))

    return norm_angles


@xr.register_dataarray_accessor("norm_angles")
class NormAngles:
    """Custom `peaks` accessor for the normal emission metadata of a `peaks`-compatible :class:`xarray.DataArray`.

    Access via the `.norm_angles` method.

    Methods
    -------
    __call__
        Displays the currently set normal emission values of the :class:`xarray.DataArray`, including displaying the
        relevant axis notation for the physical manipulator angles.
    """

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __call__(self):
        """
        Displays the currently set normal emission values of self, including displaying the relevant axis notation
        for the physical manipulator angles.

        Returns
        -------
        None :
            Displays the normal emission values of the :class:`xarray.DataArray` in the console.

        Examples
        --------
        Example usage is as follows::
            import peaks as pks

            # Load data and set its normal emission
            disp = load('disp.ibw')
            disp.norm_angles.set(theta_par=12, polar=5)

            # Display the current normal emission metadata
            disp.norm_angles()

        """

        norm_angles = _get_norm_angles(self._obj)
        display_str = "<br>"
        axes = ["norm_polar", "norm_tilt", "norm_azi"]
        for axis in axes:
            if norm_angles.get(axis) is not None:
                display_str += f"&nbsp;&nbsp;&nbsp;&nbsp;{axis} = {norm_angles[axis]}"
                if f"{axis}_manip" in norm_angles:
                    manip_name = list(norm_angles[f"{axis}_manip"].keys())[0]
                    manip_value = list(norm_angles[f"{axis}_manip"].values())[0]
                    display_str += f" ({manip_name} = {manip_value})"
                display_str += "<br>"

        analysis_warning(display_str, title="Normal emission angles")

    def set(self, norm_angles=None, quiet=False, **kwargs):
        """
        Set the normal emission angles for the data, updating the data in place.

        See Also
        --------
        peaks.utils.angles._set_norm_angles

        """
        return _set_norm_angles(
            self._obj,
            norm_angles=norm_angles,
            quiet=quiet,
            update_in_place=True,
            **kwargs,
        )

    def assign(self, norm_angles=None, quiet=False, **kwargs):
        """
        Set the normal emission angles for the data, returning a copy with the updated metadata.

        See Also
        --------
        peaks.utils.angles._set_norm_angles

        """
        return _set_norm_angles(
            self._obj,
            norm_angles=norm_angles,
            quiet=quiet,
            update_in_place=False,
            **kwargs,
        )

    def get(self):
        """
        Get the normal emission angles from the data.

        Returns
        -------
        dict
            Dictionary of normal emission angles in both `peaks` format and where applicable the true manipulator angles.

        See Also
        --------
        peaks.utils.angles._get_norm_angles

        """
        return _get_norm_angles(self._obj)


def _convert_angles_to_Ishida_Shin(data, quiet=False):
    """
    Convert the angles from friendly manipulator names into the conventions from Y. Ishida and S. Shin,
    Functions to map photoelectron distributions in a variety of setups in angle-resolved photoemission spectroscopy,
    Rev. Sci. Instrum. 89, 043903 (2018), taking care of the sign conventionsi.

    Parameters
    ----------
    data : xarray.DataArray
        Data for converting to k-space.
    quiet : bool, optional
        Whether to suppress warnings. Defaults to False.

    Returns
    -------
    dict
        Angles of converted angles.
    """

    # Get angle conventions and analyser type
    conventions = _get_conventions(data)
    ana_type = conventions.get("ana_type")

    # Get the raw angles from the data
    angles = _get_raw_angles(data, ANGLE_NAMES, quiet=quiet)

    # Adjust analyser type if necessary
    if ana_type in ["Ip", "IIp"]:
        if np.all(angles.get("defl_par", 0) == 0) and np.all(
            angles.get("defl_perp", 0) == 0
        ):
            ana_type = "I" if ana_type == "Ip" else "II"

    # Put angles in correct notation cf. Ishida and Shin
    angles_out = dict()
    angles_out["ana_type"] = ana_type
    angles_out["delta"] = angles.get("azi", 0) * conventions.get("azi", 1)
    if ana_type == "I":  # Type I
        angles_out["alpha"] = angles.get("theta_par", 0) * conventions.get(
            "theta_par", 1
        )
        angles_out["beta"] = (angles.get("polar", 0) * conventions.get("polar", 1)) + (
            angles.get("ana_polar", 0) * conventions.get("ana_polar", 1)
        )
        angles_out["xi"] = angles.get("tilt", 0) * conventions.get("tilt", 1)
    elif ana_type == "II":  # Type II
        angles_out["alpha"] = angles.get("theta_par", 0) * conventions.get(
            "theta_par", 1
        )
        angles_out["beta"] = angles.get("tilt", 0) * conventions.get("tilt", 1)
        angles_out["xi"] = (angles.get("polar", 0) * conventions.get("polar", 1)) + (
            angles.get("ana_polar", 0) * conventions.get("ana_polar", 1)
        )
    elif ana_type in ["Ip", "IIp"]:  # Type I' or Type II'
        angles_out["alpha"] = angles.get("theta_par", 0) * conventions.get(
            "theta_par", 1
        ) + (angles.get("defl_par", 0) * conventions.get("defl_par", 1))
        angles_out["beta"] = angles.get("defl_perp", 0) * conventions.get("defl_perp", 1)
        angles_out["xi"] = angles.get("tilt", 0) * conventions.get("tilt", 1)
        angles_out["chi"] = (angles.get("polar", 0) * conventions.get("polar", 1)) + (
            angles.get("ana_polar", 0) * conventions.get("ana_polar", 1)
        )

    return angles_out


def _get_angles_for_k_conv(data, return_raw=False, quiet=False, warn_norm=True):
    """
    Get the angles for the k-space conversion.

    Parameters
    ----------
    data : xarray.DataArray
        Data for converting to k-space.
    return_raw : bool, optional
        Whether to return the raw angles (polar, norm_tilt etc.), or the angles in the convention for
        k-conversion (alpha, beta etc.). Defaults to False.
    quiet : bool, optional
        Whether to suppress warnings. Defaults to False.
    warn_norm : bool, optional
        Whether to warn if normal emission angles are assumed. Defaults to True.

    Returns
    -------
    dict
        Angles for the k-space conversion (if return_raw=False) or raw angles (if return_raw=True).
    """

    # Get angle conventions and analyser type
    conventions = _get_conventions(data)
    ana_type = conventions.get("ana_type")

    # Get the raw angles from the data
    angles = _get_raw_angles(data, ANGLE_NAMES, quiet=quiet)

    # Extract normal emission angles
    norm_angles = data.norm_angles.get()

    # Make sensible guesses for missing normal emission angles
    warn_str = ""
    if warn_norm:
        for norm_angle in NORM_ANGLE_NAMES:
            if norm_angles.get(norm_angle) is None:
                if norm_angle == "norm_azi":
                    norm_angles[norm_angle] = angles.get("azi", 0) * conventions.get(
                        "azi", 1
                    )
                    warn_str += f"{norm_angle}: {norm_angles[norm_angle]}, "
                elif norm_angle == "norm_polar" and ana_type in ["I", "Ip"]:
                    norm_angles[norm_angle] = angles.get("polar", 0) * conventions.get(
                        "polar", 1
                    )
                    warn_str += f"{norm_angle}: {norm_angles[norm_angle]}, "
                elif norm_angle == "norm_tilt" and ana_type in ["II", "IIp"]:
                    norm_angles[norm_angle] = angles.get("tilt", 0) * conventions.get(
                        "tilt", 1
                    )
                    warn_str += f"{norm_angle}: {norm_angles[norm_angle]}, "
                else:
                    norm_angles[norm_angle] = 0
                    warn_str += f"{norm_angle}: 0, "
    if warn_str:
        _warn_str = f"Some normal emission data was missing or could not be passed. Assuming default values of: {warn_str.rstrip(', ')}."
        analysis_warning(_warn_str, "warning", "Analysis warning", quiet)

    # Combine angles
    angles.update(norm_angles)

    if return_raw:
        return angles

    # Adjust analyser type if necessary
    if ana_type in ["Ip", "IIp"]:
        if np.all(angles.get("defl_par", 0) == 0) and np.all(
            angles.get("defl_perp", 0) == 0
        ):
            ana_type = "I" if ana_type == "Ip" else "II"

    # Put angles in correct notation cf. Ishida and Shin, Rev. Sci. Instrum. 89 (2018) 043903
    angles_out = dict()
    angles_out["ana_type"] = ana_type
    angles_out["delta_"] = angles.get("azi", 0) * conventions.get("azi", 1) - angles.get(
        "norm_azi", 0
    )
    if ana_type == "I":  # Type I
        angles_out["alpha"] = angles.get("theta_par", 0) * conventions.get(
            "theta_par", 1
        )
        angles_out["beta"] = (angles.get("polar", 0) * conventions.get("polar", 1)) + (
            angles.get("ana_polar", 0) * conventions.get("ana_polar", 1)
        )
        angles_out["beta_0"] = angles.get("norm_polar", 0)
        print(angles_out["beta_0"])
        angles_out["xi"] = angles.get("tilt", 0) * conventions.get("tilt", 1)
        angles_out["xi_0"] = angles.get("norm_tilt", 0) * conventions.get("tilt", 1)
    elif ana_type == "II":  # Type II
        angles_out["alpha"] = angles.get("theta_par", 0) * conventions.get(
            "theta_par", 1
        )
        angles_out["beta"] = angles.get("tilt", 0) * conventions.get("tilt", 1)
        angles_out["beta_0"] = angles.get("norm_tilt", 0)
        angles_out["xi"] = (angles.get("polar", 0) * conventions.get("polar", 1)) + (
            angles.get("ana_polar", 0) * conventions.get("ana_polar", 1)
        )
        angles_out["xi_0"] = angles.get("norm_polar", 0)
    elif ana_type in ["Ip", "IIp"]:  # Type I' or Type II'
        angles_out["alpha"] = angles.get("theta_par", 0) * conventions.get(
            "theta_par", 1
        ) + (angles.get("defl_par", 0) * conventions.get("defl_par", 1))
        angles_out["beta"] = angles.get("defl_perp", 0) * conventions.get("defl_perp", 1)
        angles_out["beta_0"] = None
        angles_out["xi"] = angles.get("tilt", 0) * conventions.get("tilt", 1)
        angles_out["xi_0"] = angles.get("norm_tilt", 0)
        angles_out["chi"] = (angles.get("polar", 0) * conventions.get("polar", 1)) + (
            angles.get("ana_polar", 0) * conventions.get("ana_polar", 1)
        )
        angles_out["chi_0"] = angles.get("norm_polar", 0)

    # Add the normal emission angle sign
    angles_out["theta_par_sign"] = conventions.get("theta_par", 1)

    return angles_out
