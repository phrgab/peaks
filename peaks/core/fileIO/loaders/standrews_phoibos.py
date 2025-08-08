import pint_xarray

from peaks.core.fileIO.base_arpes_data_classes.base_specs_class import SpecsDataLoader
from peaks.core.fileIO.loc_registry import register_loader

ureg = pint_xarray.unit_registry


@register_loader
class StAndrewsSpecs(SpecsDataLoader):
    """Data loader for St Andrews Specs ARPES system, measured using Specs Prodigy.

    Define _scan_axis_resolution_order to define preferences for the primary dimension of a
    3D scan where more than one user axis varies."""

    _loc_name = "StA_Phoibos"
    _loc_description = "SPECS Phoibos 225 in King Group at St Andrews"
    _loc_url = "https://www.quantummatter.co.uk/king"
    _analyser_slit_angle = 90 * ureg.deg
    _scan_axis_resolution_order = ["tilt", "polar"]
    _manipulator_name_conventions = {
        "polar": "Theta",
        "tilt": "Phi",
        "azi": "Azi",
        "x1": "Y",
        "x2": "Z",
        "x3": "X",
    }
    _temperature_exclude_from_metadata_warn = [
        "sample",
        "cryostat",
        "shield",
        "setpoint",
    ]
    _manipulator_exclude_from_metadata_warn = [
        axis for axis in _manipulator_name_conventions.keys()
    ]

    _manipulator_sign_conventions = {
        "azi": -1,
        "x1": -1,
        "x2": -1,
        "x3": -1,
    }
    _analyser_sign_conventions = {}

    _SPECS_metadata_key_mappings = {}
    _SPECS_metadata_units = {}

    @classmethod
    def _load_data(cls, fpath, lazy):
        data_dict = super()._load_data(fpath, lazy)

        if "azi" in data_dict["coords"] and "tilt" in data_dict["coords"]:
            # The azi should be a secondary axis to the tilt, and should be removed
            new_coords = {
                dim: coord for dim, coord in data_dict["coords"].items() if dim != "azi"
            }
            data_dict["coords"] = new_coords

        return data_dict
