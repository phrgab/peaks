"""Functions to load data from the APE beamline at Elettra."""

import pint_xarray

from peaks.core.fileIO.base_arpes_data_classes.base_ses_class import SESDataLoader
from peaks.core.fileIO.loc_registry import register_loader

ureg = pint_xarray.unit_registry


@register_loader
class APEArpesLoader(SESDataLoader):
    """Data loader for APE ARPES system at Elettra.

    Notes
    ------------
    Loading spin data is not yet supported.

    """

    _loc_name = "Elettra_APE_LE"
    _loc_description = "LE (SARPES) branch of APE beamline at Elettra"
    _loc_url = "https://www.elettra.eu/elettra-beamlines/ape.html"
    _analyser_slit_angle = 0 * ureg("deg")

    _manipulator_name_conventions = {
        "polar": "polar",
        "tilt": "tilt",
        "x1": "X",
        "x2": "Z",
        "x3": "Y",
    }
    _SES_metadata_units = {
        f"manipulator_{dim}": ("mm" if dim in ["x1", "x2", "x3"] else "deg")
        for dim in _manipulator_name_conventions.keys()
    }
