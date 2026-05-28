import numpy as np
import pytest
import xarray as xr

import peaks  # noqa: F401


@pytest.fixture
def single_scan_datatree():
    da = xr.DataArray(
        np.arange(16, dtype=float).reshape(4, 4),
        dims=("eV", "theta_par"),
        coords={"eV": np.linspace(-0.3, 0.3, 4), "theta_par": np.linspace(-1, 1, 4)},
        name="data",
    ).history.assign("seed history")
    return xr.DataTree.from_dict({"scan": da.to_dataset()})


class TestIterableDataTreeAccessor:
    def test_bin_data_updates_history(self, single_scan_datatree):
        result = single_scan_datatree.bin_data(2)

        history = result["scan"].ds["data"].history.get()

        assert len(history) == 2
        assert history[-1]["record"].startswith("Binned data using the bins:")

    def test_mdc_updates_history(self, single_scan_datatree):
        result = single_scan_datatree.MDC(E=0, dE=0.1)

        history = result["scan"].ds["data"].history.get()

        assert len(history) == 2
        assert history[-1]["record"] == "MDC(s) extracted, integration window: 0.1"
