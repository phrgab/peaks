import numpy as np
import pint_xarray
import pytest
import xarray as xr

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


class TestDeriv:
    def test_dE_quadratic_returns_linear(self):
        theta_par = np.linspace(-17, 17, 100)
        eV = np.linspace(16.3, 16.8, 200)
        T, E = np.meshgrid(theta_par, eV)
        values = E**2
        da = _simulate_fake_scan(values, dims=("eV", "theta_par"), x=theta_par, y=eV)
        result = da.deriv("eV")
        expected = 2 * da.eV.values
        np.testing.assert_allclose(
            result.isel(eV=slice(5, -5), theta_par=50).values, expected[5:-5]
        )  # trim the boundary

    def test_dk_quadratic_returns_linear(self):
        theta_par = np.linspace(-17, 17, 100)
        eV = np.linspace(16.3, 16.8, 200)
        T, E = np.meshgrid(theta_par, eV)
        values = T**2
        da = _simulate_fake_scan(values, dims=("eV", "theta_par"), x=theta_par, y=eV)
        result = da.deriv("theta_par")
        expected = 2 * da.theta_par.values
        np.testing.assert_allclose(
            result.isel(eV=100, theta_par=slice(5, -5)).values,
            expected[5:-5],
        )  # trim the boundary

    def test_dE_sin_returns_cos(self):
        theta_par = np.linspace(-17, 17, 100)
        eV = np.linspace(16.3, 16.8, 200)
        T, E = np.meshgrid(theta_par, eV)
        values = np.sin(E)
        da = _simulate_fake_scan(values, dims=("eV", "theta_par"), x=theta_par, y=eV)
        result = da.deriv("eV")
        expected = np.cos(da.eV.values)
        np.testing.assert_allclose(
            result.isel(eV=slice(5, -5), theta_par=50).values, expected[5:-5], atol=1e-5
        )  # trim the boundary

    def test_deriv_perserves_shape(self):
        rng = np.random.default_rng(seed=42)
        values = rng.random((50, 100))
        da = _simulate_fake_scan(values, dims=("eV", "theta_par"))
        result = da.deriv("eV")
        assert result.shape == da.shape

    def test_deriv_invalid_dims(self):
        rng = np.random.default_rng(seed=42)
        values = rng.random((50, 100))
        da = _simulate_fake_scan(values, dims=("eV", "theta_par"))
        with pytest.raises(
            Exception, match="is not a valid dimension of the inputted DataArray"
        ):
            da.deriv("whatever")


class TestD2E:
    def test_d2E_quadratic_returns_const(self):
        theta_par = np.linspace(-17, 17, 100)
        eV = np.linspace(16.3, 16.8, 200)
        T, E = np.meshgrid(theta_par, eV)
        values = E**2
        da = _simulate_fake_scan(values, dims=("eV", "theta_par"), x=theta_par, y=eV)
        result = da.d2E()
        np.testing.assert_allclose(
            result.isel(eV=slice(5, -5), theta_par=50).values,
            2.0,
        )


class TestD2k:
    def test_d2k_quadratic_returns_const(self):
        theta_par = np.linspace(-17, 17, 100)
        eV = np.linspace(16.3, 16.8, 200)
        T, E = np.meshgrid(theta_par, eV)
        values = T**2
        da = _simulate_fake_scan(values, dims=("eV", "theta_par"), x=theta_par, y=eV)
        result = da.d2k()
        np.testing.assert_allclose(
            result.isel(eV=100, theta_par=slice(5, -5)).values,
            2.0,
        )


class TestDEdk:
    def test_dEdk_Exk_product_returns_const(self):
        theta_par = np.linspace(-17, 17, 100)
        eV = np.linspace(16.3, 16.8, 200)
        T, E = np.meshgrid(theta_par, eV)
        values = E * T
        da = _simulate_fake_scan(values, dims=("eV", "theta_par"), x=theta_par, y=eV)
        result = da.dEdk()
        np.testing.assert_allclose(
            result.isel(eV=slice(5, -5), theta_par=slice(5, -5)).values,
            1.0,
        )


class TestDkdE:
    def test_dkdE_Exk_product_returns_const(self):
        theta_par = np.linspace(-17, 17, 100)
        eV = np.linspace(16.3, 16.8, 200)
        T, E = np.meshgrid(theta_par, eV)
        values = E * T
        da = _simulate_fake_scan(values, dims=("eV", "theta_par"), x=theta_par, y=eV)
        result = da.dkdE()
        np.testing.assert_allclose(
            result.isel(eV=slice(5, -5), theta_par=slice(5, -5)).values,
            1.0,
        )

    def test_dkdE_Exk_product_equals_dEdk(self):
        theta_par = np.linspace(-17, 17, 100)
        eV = np.linspace(16.3, 16.8, 200)
        T, E = np.meshgrid(theta_par, eV)
        values = E * T
        da = _simulate_fake_scan(values, dims=("eV", "theta_par"), x=theta_par, y=eV)
        dEdk = da.dEdk()
        dkdE = da.dkdE()
        np.testing.assert_allclose(
            dEdk.isel(eV=slice(5, -5), theta_par=slice(5, -5)).values,
            dkdE.isel(eV=slice(5, -5), theta_par=slice(5, -5)).values,
        )


class TestCurvature:
    def test_curvature_quadratic_form_min_returns_constant(self):
        theta_par = np.linspace(-5, 5, 100)
        eV = np.linspace(-5, 5, 100)
        T, E = np.meshgrid(theta_par, eV)
        values = E**2 + T**2  # minimum at (0, 0) with curvature 4
        da = _simulate_fake_scan(values, dims=("eV", "theta_par"), x=theta_par, y=eV)
        result = da.curvature(theta_par=1.0, eV=1.0)
        np.testing.assert_allclose(
            result.sel(eV=0.0, theta_par=0.0, method="nearest").values, 4.0, atol=0.1
        )

    def test_curvature_quadratic_form_saddle_returns_zero(self):
        theta_par = np.linspace(-5, 5, 100)
        eV = np.linspace(-5, 5, 100)
        T, E = np.meshgrid(theta_par, eV)
        values = E**2 - T**2  # origin (0, 0) has zero curvature
        da = _simulate_fake_scan(values, dims=("eV", "theta_par"), x=theta_par, y=eV)
        result = da.curvature(theta_par=1.0, eV=1.0)
        np.testing.assert_allclose(
            result.sel(eV=0.0, theta_par=0.0, method="nearest").values, 0.0, atol=0.1
        )

    def test_curvature_plane_returns_zero(self):
        theta_par = np.linspace(-5, 5, 100)
        eV = np.linspace(-5, 5, 100)
        T, E = np.meshgrid(theta_par, eV)
        values = E + T  # plane with zero curvature
        da = _simulate_fake_scan(values, dims=("eV", "theta_par"), x=theta_par, y=eV)
        result = da.curvature(theta_par=1.0, eV=1.0)
        np.testing.assert_allclose(result.values, 0.0, atol=1e-10)

    def test_curvature_rejects_1D_data(self):
        da = _simulate_fake_scan(np.ones((10,)), dims=("eV",))
        with pytest.raises(Exception, match="Function only acts on 2D data"):
            da.curvature(theta_par=1.0, eV=1.0)

    def test_curvature_missing_free_params(self):
        da = _simulate_fake_scan(np.ones((10, 10)), dims=("eV", "theta_par"))
        with pytest.raises(
            Exception,
            match="Function requires free parameters to be defined for both axes of the data",
        ):
            da.curvature()


class TestMinGradient:
    def test_min_gradient_rejects_1D_data(self):
        da = _simulate_fake_scan(np.ones((10,)), dims=("eV",))
        with pytest.raises(Exception, match="Function cannot act on 1D data"):
            da.min_gradient(theta_par=0.05, eV=0.5)

    def test_min_gradient_preserves_shape(self):
        da = _simulate_fake_scan(np.ones((10, 10)), dims=("eV", "theta_par"))
        result = da.min_gradient(theta_par=0.05, eV=0.5)
        assert result.shape == da.shape


class TestDifferentiatePreservesUnits:
    def test_dEdk_preserves_units(self):
        theta_par = np.linspace(-17, 17, 100)
        eV = np.linspace(16.3, 16.8, 200)
        T, E = np.meshgrid(theta_par, eV)
        values = E * T
        da = _simulate_fake_scan(values, dims=("eV", "theta_par"), x=theta_par, y=eV)
        result = da.dEdk()
        assert result.data.units == da.data.units

    def test_curvature_preserves_units(self):
        theta_par = np.linspace(-5, 5, 100)
        eV = np.linspace(-5, 5, 100)
        T, E = np.meshgrid(theta_par, eV)
        values = E**2 + T**2
        da = _simulate_fake_scan(values, dims=("eV", "theta_par"), x=theta_par, y=eV)
        result = da.curvature(theta_par=1.0, eV=1.0)
        assert result.data.units == da.data.units

    def test_min_gradient_preserves_units(self):
        da = _simulate_fake_scan(np.ones((10, 10)), dims=("eV", "theta_par"))
        result = da.min_gradient(theta_par=0.05, eV=0.5)
        assert result.data.units == da.data.units
