import pint_xarray

from peaks.core.fileIO.base_arpes_data_classes.base_mbs_class import MBSDataLoader
from peaks.core.fileIO.loc_registry import register_loader

ureg = pint_xarray.unit_registry


@register_loader
class StAndrewsMBS(MBSDataLoader):
    """Data loader for St Andrews MBS ARPES system, measured using A1 Soft."""

    _loc_name = "StA_MBS"
    _loc_description = "MBS A1 in King Group at St Andrews"
    _loc_url = "https://www.quantummatter.co.uk/king"
    _analyser_slit_angle = 90 * ureg.deg
    _manipulator_name_conventions = {
        "polar": "sapolar",
        "tilt": "satilt",
        "azi": "saazi",
        "x1": "say",
        "x2": "saz",
        "x3": "sax",
    }
    _temperature_exclude_from_metadata_warn = [
        "sample",
        "cryostat",
        "shield",
        "setpoint",
    ]
    _photon_exclude_from_metadata_warn = ["hv", "polarisation", "exit_slit"]
    _manipulator_exclude_from_metadata_warn = [
        axis for axis in _manipulator_name_conventions.keys()
    ]

    _manipulator_sign_conventions = {}

    _analyser_sign_conventions = {
        "theta_par": -1,
        "deflector_parallel": -1,  # consistent with theta_par
    }

    _MBS_metadata_key_mappings = {}
    _MBS_metadata_units = {}
