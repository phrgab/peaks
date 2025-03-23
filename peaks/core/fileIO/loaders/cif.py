from peaks.core.fileIO.base_data_classes.base_data_class import BaseDataLoader
from peaks.core.fileIO.loc_registry import register_loader


@register_loader
class BaseStructureLoader(BaseDataLoader):
    """Base class for data loaders for .cif structure files."""

    _loc_name = "cif"

    @classmethod
    def _load(cls, fpath, lazy, metadata, quiet, **kwargs):
        """Load the structure data."""

        try:
            import ase.io
        except ImportError as e:
            raise ImportError(
                "The 'ase' module is required to use this module. Please install it using \
        'pip install ase'."
            ) from e

        return ase.io.read(fpath)
