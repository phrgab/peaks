from xarray import open_datatree

from peaks.core.fileIO.loaders.netcdf import NetCDFLoader
from peaks.core.fileIO.loc_registry import register_loader
from peaks.core.utils.misc import analysis_warning


@register_loader
class ZarrLoader(NetCDFLoader):
    """Class for re-opening Zarr files saved from :class:`peaks.core.fileIO.data_saving.save`.

    Notes
    -----
    If this is a Zarr file, it should be a DataTree
    """

    _loc_name = "Zarr"

    @classmethod
    def _load(cls, fpath, lazy, metadata, quiet):
        """Load the data from a Zarr file.

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

        # Load Zarr file as xarray.DataArray or xarray.Dataset
        data = open_datatree(fpath, engine="zarr", chunks={})

        # Parse the metadata if requested
        if metadata:
            data = data.map_over_datasets(ZarrLoader._parse_ds_metadata)

        # Quantify the data
        data = data.map_over_datasets(ZarrLoader._quantify_da_in_dt)

        # Actually load the data
        if not lazy:
            data = data.map_over_datasets(ZarrLoader._load_all)
        elif not quiet:
            analysis_warning(
                "The data is lazily loaded by default for loading from a Zarr store. "
                "Use the .compute() method on the individual data entries to load each into memory. "
                "To eagerly load the entire DataTree, pass the `lazy=False` argument to the load "
                "function. Ensure that the total data contents is not too large to fit in memory.",
                title="Loading info",
                warn_type="info",
            )

        return data

    @staticmethod
    def _parse_ds_metadata(ds):
        return ZarrLoader._parse_da_metadata(ds).map(ZarrLoader._parse_da_metadata)

    @staticmethod
    def _parse_da_metadata(data):
        ZarrLoader._parse_metadata(data)
        return data

    @staticmethod
    def _quantify_da_in_dt(data):
        return data.pint.quantify()

    @staticmethod
    def _load_all(data):
        return data.compute()
