"""Functions to load data from the Bloch beamline at MAX IV Laboratory."""

import pint_xarray

from peaks.core.fileIO.base_arpes_data_classes.base_ses_class import SESDataLoader
from peaks.core.fileIO.loc_registry import register_loader

ureg = pint_xarray.unit_registry


@register_loader
class BlochArpesLoader(SESDataLoader):
    _loc_name = "MAXIV_Bloch_A"
    _loc_description = "A branch (ARPES) of Bloch beamline at Max-IV"
    _loc_url = "https://www.maxiv.lu.se/beamlines-accelerators/beamlines/bloch/"
    _analyser_slit_angle = 0 * ureg("deg")

    _manipulator_name_conventions = {
        "polar": "P",
        "tilt": "T",
        "azi": "A",
        "x1": "X",
        "x2": "Y",
        "x3": "Z",
    }
    _manipulator_sign_conventions = {
        "polar": -1,
        "tilt": -1,
    }
    _analyser_sign_conventions = {
        "deflector_perp": -1,
    }
    _SES_metadata_units = {
        f"manipulator_{dim}": ("mm" if dim in ["x1", "x2", "x3"] else "deg")
        for dim in _manipulator_name_conventions.keys()
    }
