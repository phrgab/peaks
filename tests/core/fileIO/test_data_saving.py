import pytest

from peaks.core.fileIO.data_loading import load
from peaks.core.metadata.metadata_methods import display_metadata
from peaks.core.utils.sample_data import ExampleData


@pytest.fixture(scope="session")
def disp():
    return ExampleData.dispersion()


class TestSave:
    def test_save_load_roundtrip_preserves_metadata(self, disp, tmp_path):
        fpath = tmp_path / "peaks_saved_disp.nc"
        original_data = disp.copy(deep=True)
        original_data.save(str(fpath))
        result = load(str(fpath))

        assert set(result.metadata.keys()) == set(original_data.metadata.keys())
        assert display_metadata(result) == display_metadata(original_data)

    def test_save_with_wrong_extension_raises(self, disp, tmp_path):
        with pytest.raises(ValueError, match="File path must have a .nc extension"):
            disp.save(str(tmp_path / "disp.zarr"))
