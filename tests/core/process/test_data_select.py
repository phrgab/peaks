import numpy as np
import pint_xarray
import pytest
import xarray as xr

from peaks.core.utils.sample_data import ExampleData

ureg = pint_xarray.unit_registry

# to-do:
# TestTot - a spatial map needed
# TestDispFromHv - this needs an hv scan


# temporary example data before the data simulation module is available
@pytest.fixture(scope="session")
def disp():
    return ExampleData.dispersion2a()


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
    # physically meaningless but is a valid xarray da with proper file structure
    _n_eV = 50
    _n_theta_par = 100
    rng = np.random.default_rng(seed=42)
    values = rng.random((_n_eV, _n_theta_par))
    return _simulate_fake_scan(values)


class TestDropNaNBorders:
    def test_trims_nan_rows(self):
        values = np.ones((5, 5))
        values[0, :] = np.nan
        values[-1, :] = np.nan
        da = _simulate_fake_scan(values)
        result = da.drop_nan_borders()
        assert result.shape == (3, 5)
        assert not np.isnan(result).any()
        assert result["eV"].size == 3

    def test_trims_nan_columns(self):
        values = np.ones((5, 5))
        values[:, 0] = np.nan
        values[:, -1] = np.nan
        da = _simulate_fake_scan(values)
        result = da.drop_nan_borders()
        assert result.shape == (5, 3)
        assert not np.isnan(result).any()
        assert result["theta_par"].size == 3

    def test_non_nan_borders_untouched(self):
        values = np.ones((5, 5))
        da = _simulate_fake_scan(values)
        result = da.drop_nan_borders()
        assert result.shape == (5, 5)
        assert np.isnan(result).sum() == 0

    def test_interior_nans_untouched(self):
        values = np.ones((5, 5))
        values[2, 2] = np.nan
        da = _simulate_fake_scan(values)
        result = da.drop_nan_borders()
        assert result.shape == (5, 5)
        assert np.isnan(result).sum() == 1
        assert np.isnan(result[2, 2])


class TestDropZeroBorders:
    def test_trims_zero_rows(self):
        values = np.ones((5, 5))
        values[0, :] = 0
        values[-1, :] = 0
        da = _simulate_fake_scan(values)
        result = da.drop_zero_borders()
        assert result.shape == (3, 5)
        assert not (result == 0).any()
        assert result["eV"].size == 3

    def test_trims_zero_columns(self):
        values = np.ones((5, 5))
        values[:, 0] = 0
        values[:, -1] = 0
        da = _simulate_fake_scan(values)
        result = da.drop_zero_borders()
        assert result.shape == (5, 3)
        assert not (result == 0).any()
        assert result["theta_par"].size == 3

    def test_non_zero_borders_untouched(self):
        values = np.ones((5, 5))
        da = _simulate_fake_scan(values)
        result = da.drop_zero_borders()
        assert result.shape == (5, 5)
        assert (result == 0).sum() == 0

    def test_interior_zeros_untouched(self):
        values = np.ones((5, 5))
        values[2, 2] = 0
        da = _simulate_fake_scan(values)
        result = da.drop_zero_borders()
        assert result.shape == (5, 5)
        assert (result == 0).sum() == 1
        assert result[2, 2] == 0


class TestDC:
    def test_DC_single_val_returns_1D_array(self, fake_disp):
        result = fake_disp.DC(val=16.5)  # extract along eV by default
        assert isinstance(result, xr.DataArray)
        assert result.ndim == 1
        assert "eV" not in result.dims
        assert "theta_par" in result.dims

    def test_DC_single_val_selects_nearest(self, fake_disp):
        eV_coord = fake_disp.eV.values
        target_val = (eV_coord[10] + eV_coord[11]) / 2
        result = fake_disp.DC(coord="eV", val=target_val)
        selected_eV = float(result.eV)
        assert selected_eV in [eV_coord[10], eV_coord[11]]

    def test_DC_list_returns_multi_DCs(self, fake_disp):
        result = fake_disp.DC(val=[16.4, 16.5, 16.6])
        assert result["eV"].size == 3
        for da in result:
            assert isinstance(da, xr.DataArray)
            assert da.ndim == 1
            assert "eV" not in da.dims
            assert "theta_par" in da.dims

    def test_DC_tuple_returns_range(self, fake_disp):
        result = fake_disp.DC(val=(16.4, 16.6, 0.05))
        assert result["eV"].size == pytest.approx((16.6 - 16.4) / 0.05 + 1)

    def test_integration_window_averages(self, fake_disp):
        result = fake_disp.DC(val=16.5, dval=0.05)
        expected = fake_disp.sel(eV=slice(16.5 - 0.05 / 2, 16.5 + 0.05 / 2)).mean("eV")
        diff = result.values - expected.values
        np.testing.assert_allclose(diff, 0, atol=1e-14)

    def test_invalid_tuple_args(self, fake_disp):
        with pytest.raises(Exception, match="Tuple argument must be in the format"):
            fake_disp.DC(val=(16.4, 16.6))


class TestMDC:
    def test_MDC_on_real_data(self, disp):
        eV_midpoint = disp["eV"].values[len(disp.eV) // 2]
        result = disp.MDC(E=eV_midpoint, dE=0.01)
        expected = disp.sel(
            eV=slice(eV_midpoint - 0.01 / 2, eV_midpoint + 0.01 / 2)
        ).mean("eV")
        diff = result.values - expected.values
        np.testing.assert_allclose(diff, 0, atol=1e-3)


class TestEDC:
    def test_EDC_on_real_data(self, disp):
        theta_par_midpoint = disp["theta_par"].values[len(disp.theta_par) // 2]
        result = disp.EDC(k=theta_par_midpoint, dk=0.2)
        expected = disp.sel(
            theta_par=slice(theta_par_midpoint - 0.2 / 2, theta_par_midpoint + 0.2 / 2)
        ).mean("theta_par")
        diff = result.values - expected.values
        np.testing.assert_allclose(diff, 0, atol=1e-3)


class TestDOS:
    def test_DOS_returns_1D_along_eV(self, fake_disp):
        result = fake_disp.DOS()
        assert result.ndim == 1
        assert "eV" in result.dims

    def test_DOS_vals_are_mean_along_theta_par(self, fake_disp):
        result = fake_disp.DOS()
        expected = fake_disp.mean("theta_par")
        diff = result.values - expected.values
        np.testing.assert_allclose(diff, 0, atol=1e-14)

    def test_DOS_on_real_data(self, disp):
        result = disp.DOS()
        expected = disp.mean("theta_par")
        diff = result.values - expected.values
        np.testing.assert_allclose(diff, 0, atol=1e-10)
        assert "eV" in result.dims
        assert result.size == disp["eV"].size


class TestRadialCuts:
    def test_radial_cuts_on_circle_returns_straight_line(self):
        kx = np.linspace(-2, 2, 401)
        ky = np.linspace(-2, 2, 401)
        KX, KY = np.meshgrid(kx, ky)
        values = 1000 * np.exp(-((np.sqrt(KX**2 + KY**2) - 1) ** 2) / 0.001)
        # making a circular Fermi surface centred round gamma
        da = _simulate_fake_scan(values, dims=("ky", "kx"), x=kx, y=ky)
        result = da.radial_cuts(radius=1.5, kx=0, ky=0)
        peak_positions = result.k[result.argmax("k")]
        expected = xr.ones_like(peak_positions)  # all peaks should be at radius 1.0
        xr.testing.assert_allclose(peak_positions, expected, atol=1e-2)

    def test_radial_cuts_azi_0_is_deterministic(self):
        """azi=0 should always point along the positive dim[0] irrespective of dimension naming.
        Two dataarrays here are created from the same underlying arc-shaped array but with
        different dimension orderings. If radial_cuts consistently treats azi=0 as the
        positive direction of dim[0], both should give the same result.
        """
        kx = np.linspace(-2, 2, 401)
        ky = np.linspace(-2, 2, 401)
        KX, KY = np.meshgrid(kx, ky)
        circle_values = 1000 * np.exp(-((np.sqrt(KX**2 + KY**2) - 1) ** 2) / 0.001)
        azi = np.degrees(np.arctan2(KY, KX)) % 360
        mask = (azi >= 45) & (azi <= 135)
        arc_values = circle_values * mask
        da_dim0_is_ky = _simulate_fake_scan(arc_values, dims=("ky", "kx"), x=kx, y=ky)
        da_dim0_is_kx = _simulate_fake_scan(arc_values, dims=("kx", "ky"), x=kx, y=ky)
        result_ky = da_dim0_is_ky.radial_cuts(radius=1.5, kx=0, ky=0)
        result_kx = da_dim0_is_kx.radial_cuts(radius=1.5, kx=0, ky=0)
        np.testing.assert_allclose(
            result_ky.sel(azi=0).values, result_kx.sel(azi=0).values, atol=1e-14
        )

    def test_radial_cuts_have_k_azi_dims(self):
        kx = np.linspace(-2, 2, 401)
        ky = np.linspace(-2, 2, 401)
        KX, KY = np.meshgrid(kx, ky)
        values = 1000 * np.exp(-((np.sqrt(KX**2 + KY**2) - 1) ** 2) / 0.001)
        # making a circular Fermi surface centred round gamma
        da = _simulate_fake_scan(values, dims=("ky", "kx"), x=kx, y=ky)
        result = da.radial_cuts(radius=1.5, kx=0, ky=0)
        assert len(result.dims) == 2
        assert "k" in result.dims
        assert "azi" in result.dims

    def test_radial_cuts_custom_centre(self):
        kx = np.linspace(-2, 2, 401)
        ky = np.linspace(-2, 2, 401)
        KX, KY = np.meshgrid(kx, ky)
        cx, cy = 0.2, -0.3
        values = 1000 * np.exp(
            -((np.sqrt((KX - cx) ** 2 + (KY - cy) ** 2) - 1) ** 2) / 0.001
        )
        da = _simulate_fake_scan(values, dims=("ky", "kx"), x=kx, y=ky)
        result = da.radial_cuts(radius=1.5, kx=cx, ky=cy)
        peak_positions = result.k[result.argmax("k")]
        expected = xr.ones_like(peak_positions)
        xr.testing.assert_allclose(peak_positions, expected, atol=1e-2)

    def test_radial_cuts_num_azi_num_points(self):
        kx = np.linspace(-2, 2, 401)
        ky = np.linspace(-2, 2, 401)
        KX, KY = np.meshgrid(kx, ky)
        values = 1000 * np.exp(-((np.sqrt(KX**2 + KY**2) - 1) ** 2) / 0.001)
        da = _simulate_fake_scan(values, dims=("ky", "kx"), x=kx, y=ky)
        result = da.radial_cuts(radius=1.5, kx=0, ky=0, num_azi=91, num_points=100)
        assert result.sizes["azi"] == 91
        assert result.sizes["k"] == 100


class TestExtractCut:
    @pytest.fixture
    def fake_FS(self):
        sapolar = np.linspace(-2, 2, 201)
        theta_par = np.linspace(-2, 2, 201)
        eV = np.linspace(16.65, 16.7, 2)
        X, Y = np.meshgrid(theta_par, sapolar)
        values = 1000 * np.exp(-((np.sqrt(X**2 + Y**2) - 1) ** 2) / 0.001)
        values_3D = np.repeat(values[:, :, np.newaxis], len(eV), axis=2)

        da = _simulate_fake_scan(
            values_3D, dims=("sapolar", "theta_par", "eV"), y=eV, x=theta_par
        )
        da["sapolar"] = sapolar
        da["sapolar"].attrs["units"] = ureg.deg
        return da

    def test_extract_cut_returns_expected_slice(self, fake_FS):
        result_h = fake_FS.extract_cut(
            start_point={"theta_par": 0.3, "sapolar": -1.5},
            end_point={"theta_par": 0.3, "sapolar": 1.5},
            num_points=201,
        )
        result_v = fake_FS.extract_cut(
            start_point={"theta_par": -1.5, "sapolar": 0.3},
            end_point={"theta_par": 1.5, "sapolar": 0.3},
            num_points=201,
        )
        xr.testing.assert_allclose(result_h, result_v, atol=1e-10)

    def test_extract_cut_has_proj_dim(self, fake_FS):
        result = fake_FS.extract_cut(
            start_point={"theta_par": 0.3, "sapolar": -1.5},
            end_point={"theta_par": 0.3, "sapolar": 1.5},
        )
        assert "proj" in result.dims

    def test_extract_cut_num_points(self, fake_FS):
        num_points = 100
        result = fake_FS.extract_cut(
            start_point={"theta_par": 0.3, "sapolar": -1.5},
            end_point={"theta_par": 0.3, "sapolar": 1.5},
            num_points=num_points,
        )
        assert result.sizes["proj"] == num_points

    def test_extract_cut_invalid_dims(self, fake_FS):
        with pytest.raises(
            ValueError,
            match="Ensure the dimensions specified in the start and end points are present in the data",
        ):
            fake_FS.extract_cut(
                start_point={"kx": 0.3, "ky": -1.5},
                end_point={"kx": 0.3, "ky": 1.5},
            )

    def test_extract_cut_mismatched_dims(self, fake_FS):
        with pytest.raises(
            ValueError,
            match="Start and end points must be specified as dictionaries",
        ):
            fake_FS.extract_cut(
                start_point={"theta_par": 0.3, "sapolar": -1.5},
                end_point={"theta": 0.3, "polar": 1.5},
            )


class TestMaskData:
    def test_mask_data_returns_expected(self, fake_disp):
        ROI = {
            "theta_par": [-5, 5, 5, -5],
            "eV": [16.4, 16.4, 16.6, 16.6],
        }
        result = fake_disp.mask_data(ROI=ROI, return_integrated=False)
        expected = fake_disp.sel(theta_par=slice(-5, 5), eV=slice(16.4, 16.6))
        xr.testing.assert_allclose(result, expected, atol=1e-14)
        assert len(result.dims) == 2

    def test_integrated_true_returns_a_single_val(self, fake_disp):
        ROI = {
            "theta_par": [-5, 5, 5, -5],
            "eV": [16.4, 16.4, 16.6, 16.6],
        }
        result = fake_disp.mask_data(ROI=ROI, return_integrated=True)
        assert result.size == 1

    def test_mask_data_invalid_ROI(self, fake_disp):
        ROI = {
            "theta_par": [-5, 5, 5, -5],
            "eV": [16.4, 16.6],
        }
        with pytest.raises(
            Exception,
            match="ROI must be a dictionary containing two entries for the relevant axes",
        ):
            fake_disp.mask_data(ROI=ROI)


class TestDataSelectPreservesUnits:
    def test_DC_preserves_units(self, fake_disp):
        result = fake_disp.DC(val=16.5)
        assert result.data.units == fake_disp.data.units

    def test_MDC_preserves_units(self, fake_disp):
        eV_midpoint = fake_disp["eV"].values[len(fake_disp.eV) // 2]
        result = fake_disp.MDC(E=eV_midpoint, dE=0.01)
        assert result.data.units == fake_disp.data.units

    def test_DOS_preserves_units(self, fake_disp):
        result = fake_disp.DOS()
        assert result.data.units == fake_disp.data.units

    def test_radial_cuts_preserves_units(self):
        kx = np.linspace(-2, 2, 401)
        ky = np.linspace(-2, 2, 401)
        KX, KY = np.meshgrid(kx, ky)
        values = 1000 * np.exp(-((np.sqrt(KX**2 + KY**2) - 1) ** 2) / 0.001)
        da = _simulate_fake_scan(values, dims=("ky", "kx"), x=kx, y=ky)
        result = da.radial_cuts(radius=1.5, kx=0, ky=0)
        assert result.data.units == da.data.units
