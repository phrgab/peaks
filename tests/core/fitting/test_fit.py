from unittest.mock import patch

import matplotlib
import numpy as np
import pint_xarray
import pytest
import xarray as xr

from peaks.core.fitting.fit import _estimate_EF  # type: ignore
from peaks.core.fitting.models import GaussianModel  # type: ignore
from peaks.core.utils.sample_data import ExampleData

matplotlib.use("Agg")  # suppress plots
ureg = pint_xarray.unit_registry


def _simulate_fake_scan(
    data,
    dims=("eV", "theta_par"),
    y=None,
    x=None,
    name="fake-scan",
    unit="count/s",
):
    from peaks.core.metadata.base_metadata_models import BaseScanMetadataModel

    coords = {}
    for i, dim in enumerate(dims):
        n_pts = data.shape[i]
        if dim == "eV":
            coords[dim] = y if y is not None else np.linspace(16.3, 16.8, n_pts)
        elif dim == "theta_par":
            coords[dim] = x if x is not None else np.linspace(-17, 17, n_pts)
        elif dim == "kx":
            coords[dim] = x if x is not None else np.linspace(-1.5, 1.5, n_pts)
        elif dim == "ky":
            coords[dim] = y if y is not None else np.linspace(-1.5, 1.5, n_pts)
        else:
            coords[dim] = np.arange(n_pts, dtype=float)
    da = xr.DataArray(data, dims=dims, coords=coords, name=name)
    da = da.pint.quantify(units=unit)
    if "eV" in dims:
        da["eV"].attrs["units"] = ureg.eV
    if "theta_par" in dims:
        da["theta_par"].attrs["units"] = ureg.deg
    if "kx" in dims:
        da["kx"].attrs["units"] = ureg.angstrom**-1
    if "ky" in dims:
        da["ky"].attrs["units"] = ureg.angstrom**-1
    da = da.history.assign("creating a fake scan for tests")
    da.attrs["_scan"] = BaseScanMetadataModel(
        name=name,
        filepath="/home/kinggroup/fake-scan.ext",
        loc="St Andrews",
        timestamp="2000-01-06 00:00:00",
    )
    return da


@pytest.fixture
def fake_disp():
    # A linear dispersion with Gaussian EDCs
    # The peak is centered at 16.55 eV and theta_par = 0, and has a width of 0.02 eV.
    eV = np.linspace(16.3, 16.8, 100)
    theta_par = np.linspace(-17, 17, 200)
    T, E = np.meshgrid(theta_par, eV)
    centres = 0.03 * T + 16.55
    values = 1000 * np.exp(-((E - centres) ** 2) / (2 * 0.02**2))
    rng = np.random.default_rng(seed=42)
    values += np.abs(
        rng.normal(0, 50, values.shape)
    )  # chunk in a wee bit of noise for robustness testing
    da = _simulate_fake_scan(values, y=eV, x=theta_par)
    return da


@pytest.fixture
def poly_gold():
    return ExampleData.gold_reference4().sel(
        eV=slice(16.4, 17), theta_par=slice(-10, 10)
    )


class TestFit:
    def test_fit_recovers_gaussian_centres(self, fake_disp):
        model = GaussianModel()
        params = model.make_params(center=16.55, sigma=0.02, height=1000)
        result = fake_disp.sel(theta_par=slice(-7, 7)).fit(
            model, params, independent_var="eV", sequential=True
        )
        fitted_centres = result["center"]
        expected_centres = 0.03 * fake_disp.theta_par.sel(theta_par=slice(-7, 7)) + 16.55
        np.testing.assert_allclose(
            fitted_centres.values, expected_centres.values, rtol=0.005
        )

    def test_fit_returns_stderr(self, fake_disp):
        model = GaussianModel()
        params = model.make_params(center=16.55, sigma=0.02, height=1000)
        result = fake_disp.sel(theta_par=slice(-7, 7)).fit(
            model, params, independent_var="eV", sequential=True
        )
        assert "sigma_stderr" in result.data_vars

    def test_fit_invalid_independent_var(self, fake_disp):
        model = GaussianModel()
        params = model.make_params(center=16.55, sigma=0.02, height=1000)
        with pytest.raises(
            ValueError, match="is not a valid dimension of the data array"
        ):
            fake_disp.fit(model, params, independent_var="whatever")

    def test_fit_2D_without_independent_var(self, fake_disp):
        model = GaussianModel()
        params = model.make_params(center=16.55, sigma=0.02, height=1000)
        with pytest.raises(
            ValueError,
            match="Please specify the independent variable using the",
        ):
            fake_disp.fit(model, params)


class TestEstimateEF:
    def test_estimate_EF(self):
        from peaks.core.fitting.fit_functions import _fermi_function

        eV = np.linspace(0, 0.2, 500)
        values = 100 * _fermi_function(eV, EF=0.05, T=50)
        rng = np.random.default_rng(seed=42)
        values += rng.normal(0, 1, values.shape)
        result = _estimate_EF(values, eV)
        assert float(result) == pytest.approx(0.05, rel=0.01)

    def test_estimate_EF_on_real_data(self, poly_gold):
        result = poly_gold.isel(theta_par=0).estimate_EF()
        assert float(result) == pytest.approx(16.767, rel=0.01)


class TestFitGold:
    @pytest.fixture(autouse=True)
    def _suppress_plt_show(self):
        with patch("peaks.core.fitting.fit.plt.show"):
            yield

    @pytest.mark.parametrize(
        "EF_correction_type, expected",
        [
            (
                "poly4",
                {
                    "c0": 16.803629590704013,
                    "c1": 0.00022636528328194577,
                    "c2": -0.0002222775835049918,
                    "c3": -6.453745990279035e-07,
                    "c4": -1.380527081025937e-06,
                },
            ),
            (
                "poly3",
                {
                    "c0": 16.804822069260307,
                    "c1": 0.00022636523093286966,
                    "c2": -0.00034106675111900375,
                    "c3": -6.453737296917145e-07,
                },
            ),
            (
                "quadratic",
                {
                    "c0": 16.804822069261,
                    "c1": 0.00018749217574237317,
                    "c2": -0.0003410667511398363,
                },
            ),
            (
                "linear",
                {"c0": 16.79340882215599, "c1": 0.00018749217555738323},
            ),
        ],
    )
    def test_fit_gold(self, poly_gold, EF_correction_type, expected):
        result = poly_gold.fit_gold(EF_correction_type=EF_correction_type)
        for order, value in expected.items():
            assert result.attrs["EF_correction"][order] == pytest.approx(value, rel=1e-5)

    def test_fit_gold_average(self, poly_gold):
        result = poly_gold.fit_gold(EF_correction_type="average")
        assert result.attrs["EF_correction"] == pytest.approx(
            16.79340882215599, rel=1e-5
        )


class TestSaveLoadFitResults:
    @pytest.fixture
    def fit_results(self, fake_disp):
        model = GaussianModel()
        params = model.make_params(center=16.55, sigma=0.02, height=1000)
        result = fake_disp.sel(theta_par=slice(-7, 7)).fit(
            model, params, independent_var="eV", sequential=True
        )
        return result

    def test_roundtrip_preserves_params(self, fit_results, tmp_path):
        import peaks as pks

        tmp_file = tmp_path / "fit_results.pkl"
        fit_results.save_fit(tmp_file)
        loaded = pks.load_fit(tmp_file)
        params_to_check = ["center", "sigma", "height"]
        for param in params_to_check:
            np.testing.assert_array_almost_equal(
                fit_results[param].values, loaded[param].values
            )
