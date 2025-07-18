import importlib
import json
import re
from typing import Optional

import pint_xarray  # noqa: F401
import xarray as xr
from pydantic import create_model

from peaks.core.fileIO.base_data_classes.base_data_class import BaseDataLoader
from peaks.core.fileIO.loc_registry import register_loader
from peaks.core.metadata.base_metadata_models import AxisMetadataModelWithReference
from peaks.core.options import opts


# Classes for data loaders
@register_loader
class NetCDFLoader(BaseDataLoader):
    """Class for re-opening NetCDF files saved from :class:`peaks.core.fileIO.data_saving.save`.

    Notes
    -----
    If this is a NetCDF file, it should be a single DataArray or DataSet
    """

    _loc_name = "NetCDF"

    @classmethod
    def _load(cls, fpath, lazy, metadata, quiet):
        """Load the data from a NetCDF file.

        Parameters
        ----------
        fpath : str
            Path to the file to be loaded.
        lazy : bool
            Whether to load the data lazily.
        metadata : bool
            Whether to attempt to parse the metadata into full `peaks` format.
        quiet : bool
            Whether to suppress print output.

        Returns
        -------
        data : xarray.DataArray or xarray.DataSet
            The loaded data.
        """

        # Open NetCDF file as xarray.DataArray or xarray.Dataset
        try:
            data = xr.open_dataarray(fpath)
        except ValueError:
            data = xr.open_dataset(fpath)

        # Parse the metadata if requested
        if metadata:
            cls._parse_metadata(data)

        # Actually load the data
        if (lazy is False) or (lazy is None and data.nbytes > opts.FileIO.lazy_size):
            data = data.compute()

        # Quantify the data if it has units
        try:
            data = data.pint.quantify()
        except AttributeError:
            pass

        cls._add_load_history(data, fpath)

        return data

    @classmethod
    def _parse_metadata(cls, data):
        """Parse the metadata from the loaded data, returning to `peaks` format."""

        if data.attrs.get("metadata_models"):
            # Try to parse the loc
            loc = None
            loc_pattern = r'"loc":"(.*?)"'
            if data.attrs.get("_scan"):
                match = re.search(loc_pattern, data.attrs.get("_scan"))
                if match:
                    loc = match.group(1)

            metadata_models = json.loads(data.attrs.pop("metadata_models"))
            for attr_name, attr in data.attrs.items():
                model = metadata_models.get(attr_name)
                if model == "json":
                    data.attrs[attr_name] = json.loads(attr)
                elif model and "ManipulatorMetadataModel" in model:
                    # Need to handle this as a special case as the ManipulatorMetadataModel
                    # is created dynamically in loaders

                    # Get the axes for the original loader
                    manipulator_axes = cls.get_loader(loc)._manipulator_axes

                    # Rebuild manipulator metadata model
                    fields = {
                        axis: (Optional[AxisMetadataModelWithReference], None)
                        for axis in manipulator_axes
                    }
                    ManipulatorMetadataModel = create_model(
                        "ManipulatorMetadataModel", **fields
                    )
                    data.attrs[attr_name] = ManipulatorMetadataModel.parse_raw(attr)
                elif model:
                    model_class = cls._get_metadata_model(model)
                    data.attrs[attr_name] = model_class.parse_raw(attr)

    @classmethod
    def _get_metadata_model(cls, class_path):
        """Dynamically load the relevant metadata class from the fully qualified class name

        Parameters
        ----------
        class_path : str
            Fully qualified class name

        Returns
        -------
        class
            The class object
        """

        # Split the class_path into module and class parts
        module_name, class_name = class_path.rsplit(".", 1)

        # Dynamically import the module
        module = importlib.import_module(module_name)

        # Retrieve the class from the module
        return getattr(module, class_name)
