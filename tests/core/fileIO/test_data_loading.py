import dask.array
import numpy as np
import pytest
import xarray as xr

from peaks.core.fileIO.data_loading import load
from peaks.core.utils.sample_data import ZenodoDownloader


@pytest.fixture(scope="session")
def test_disp_path():
    # temporary example data before the data simulation module is available
    with ZenodoDownloader(["i05-59819.nxs"]) as downloader:
        downloader.download()
        yield downloader.downloaded_files["i05-59819.nxs"]


class TestLoad:
    def test_returns_xarray(self, test_disp_path):
        result = load(test_disp_path)
        assert isinstance(result, (xr.DataArray, xr.Dataset, xr.DataTree))

    def test_data_not_empty(self, test_disp_path):
        result = load(test_disp_path)
        assert result.size > 0


class TestLoadMetadata:
    def test_metadata_loaded_by_default(self, test_disp_path):
        result = load(test_disp_path)
        eV = result.attrs["_analyser"].scan.eV
        assert len(result.attrs) > 0
        assert eV is not None

    def test_metadata_skipped(self, test_disp_path):
        result = load(test_disp_path, metadata=False)
        eV = result.attrs["_analyser"].scan.eV
        assert eV is None


class TestLoadLazy:
    def test_lazy_true(self, test_disp_path):
        result = load(test_disp_path, lazy=True)
        assert type(result.data.magnitude) is dask.array.Array

    def test_lazy_false(self, test_disp_path):
        result = load(test_disp_path, lazy=False)
        assert type(result.data.magnitude) is np.ndarray


class TestLoadErrorHandling:
    def test_missing_file_raises(self):
        with pytest.raises(Exception, match="No valid file paths could be found"):
            load("/whatever/path.ext")
