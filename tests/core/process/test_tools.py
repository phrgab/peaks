import matplotlib
import numpy as np
import pint_xarray
import pytest
import xarray as xr

from peaks.core.process.tools import (
    _sum_or_subtract_data,
    drift_correction,
    estimate_sym_point,
    merge_data,
)
from peaks.core.utils.sample_data import ExampleData

matplotlib.use("Agg")  # suppress plots
ureg = pint_xarray.unit_registry


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


class TestNorm:
    def test_norm_by_max(self, disp):
        result = disp.norm()
        assert float(result.max().data) == pytest.approx(1.0)

    def test_norm_by_mean(self, disp):
        result = disp.norm(dim="all")
        assert float(result.data.magnitude.mean()) == pytest.approx(1.0)

    def test_norm_by_integrated_MDC(self, disp):
        result = disp.norm(dim="eV")
        mean_da = result.mean("eV")
        expected_da = xr.ones_like(mean_da)
        xr.testing.assert_allclose(mean_da, expected_da)

    def test_norm_by_intergrated_EDC(self, disp):
        result = disp.norm(dim="theta_par")
        mean_da = result.mean("theta_par")
        expected_da = xr.ones_like(mean_da)
        xr.testing.assert_allclose(mean_da, expected_da)

    def test_norm_by_slices(self, disp):
        result = disp.norm(eV=slice(15.8, 16.0), theta_par=slice(3.0, 4.0))
        normalised_region = result.sel(eV=slice(15.8, 16.0), theta_par=slice(3.0, 4.0))
        assert float(normalised_region.mean().data) == pytest.approx(1.0)

    def test_original_data_not_overwritten(self, disp):
        original_max = float(disp.data.magnitude.max())
        _normalised = disp.norm()
        assert original_max == float(disp.data.magnitude.max())

    def test_invalid_dim(self, fake_disp):
        with pytest.raises(
            Exception, match="is not a valid dimension of the inputted DataArray"
        ):
            fake_disp.norm(dim="whatever")


class TestBgs:
    def test_subtract_const_int(self, disp):
        result = disp.bgs(10)
        expected = disp.values - 10
        np.testing.assert_allclose(result.data.magnitude, expected)

    def test_substract_const_float(self, disp):
        result = disp.bgs(10.5)
        expected = disp.values - 10.5
        np.testing.assert_allclose(result.data.magnitude, expected)

    def test_substract_mean(self, disp):
        result = disp.bgs("all")
        expected = disp.values - float(disp.mean().values)
        np.testing.assert_allclose(result.data.magnitude, expected)

    def test_substract_eV(self, disp):
        result = disp.bgs("eV")
        expected = disp - disp.mean("eV")
        xr.testing.assert_allclose(result, expected)

    def test_substract_theta_par(self, disp):
        result = disp.bgs("theta_par")
        expected = disp - disp.mean("theta_par")
        xr.testing.assert_allclose(result, expected)


class TestBinData:
    def test_bin_all_dims(self, fake_disp):
        result = fake_disp.bin_data(2)
        assert result.shape == (25, 50)

    def test_bin_specific_dim(self, fake_disp):
        result = fake_disp.bin_data(theta_par=2, eV=5)
        assert result.sizes["eV"] == 10
        assert result.sizes["theta_par"] == 50

    def test_bin_values_are_means(self):
        data = np.arange(16, dtype=float).reshape(4, 4)
        da = _simulate_fake_scan(
            data, y=np.arange(4, dtype=float), x=np.arange(4, dtype=float)
        )
        result = da.bin_data(2)
        assert result.values[0, 0] == da.values[:2, :2].mean()
        assert result.values[0, 1] == da.values[:2, 2:].mean()
        assert result.values[1, 0] == da.values[2:, :2].mean()
        assert result.values[1, 1] == da.values[2:, 2:].mean()

    def test_bin_a_real_disp(self, disp):
        original_shape = disp.shape
        result = disp.bin_data(2)
        for original, binned in zip(original_shape, result.shape, strict=True):
            assert binned == original // 2

    def test_bin_non_int(self, fake_disp):
        with pytest.raises(Exception, match="Binning value must be an integer"):
            fake_disp.bin_data(1.5)

    def test_bin_invalid_dim(self, fake_disp):
        with pytest.raises(
            Exception, match="is not a valid dimension of the inputted DataArray"
        ):
            fake_disp.bin_data(oops=2)

    def test_bin_no_args(self, fake_disp):
        with pytest.raises(Exception, match="No binning parameters set"):
            fake_disp.bin_data()


class TestBinSpectra:
    def test_bin_spectra(self, fake_disp):
        result = fake_disp.bin_spectra()
        assert result.shape == (25, 50)

    def test_bin_no_spectral_dims(self):
        da = _simulate_fake_scan(np.ones((4, 4)), dims=("x1", "x2"))
        with pytest.raises(
            Exception, match="No spectral dimensions found in the inputted DataArray"
        ):
            da.bin_spectra()


class TestSmooth:
    def test_smooth_single_axis(self, disp):
        result = disp.smooth(eV=0.2)
        assert float(result.std("eV").mean()) < float(disp.std("eV").mean())

    def test_smooth_preserves_shape(self, disp):
        result = disp.smooth(eV=0.2, theta_par=0.5)
        assert disp.shape == result.shape

    def test_smooth_broadens_spikes(self):
        theta_par = np.linspace(-17, 17, 100)
        eV = np.array([0.0])
        values = np.zeros((1, 100))
        values[0, 50] = 1000.0
        # making an MDC with a delta function shape
        da = _simulate_fake_scan(values, y=eV, x=theta_par)
        result = da.smooth(theta_par=1.5)
        assert float(result.max()) < 1000.0
        assert np.allclose(
            float(da.sum()), float(result.sum())
        )  # total intensity should be preserved ish

    def test_smooth_no_kwargs(self, fake_disp):
        with pytest.raises(
            Exception, match="Function requires axes to be smoothed over to be defined"
        ):
            fake_disp.smooth()


class TestRotate:
    def test_rotate_360_returns_same(self, fake_disp):
        result = fake_disp.rotate(360)
        xr.testing.assert_identical(result, fake_disp)

    def test_rotate_4fold_data_by90_returns_same_ish(self):
        kx = np.linspace(-5, 5, 51)
        ky = np.linspace(-5, 5, 51)
        KX, KY = np.meshgrid(kx, ky)
        values = np.cos(KX) * np.cos(KY)
        da = _simulate_fake_scan(values, dims=("ky", "kx"), x=kx, y=ky)
        result_trimmed = da.rotate(90, kx=0, ky=0).sel(
            kx=slice(-3.8, 3.8), ky=slice(-3.8, 3.8)
        )
        xr.testing.assert_allclose(
            da.sel(kx=slice(-3.8, 3.8), ky=slice(-3.8, 3.8)),
            result_trimmed,
        )

    def test_rotate_FS_preserves_eV_ax(self):
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
        result = da.rotate(30, sapolar=0, theta_par=0)
        np.testing.assert_array_equal(result.eV.values, da.eV.values)
        assert set(result.dims) == set(da.dims)

    def test_rotate_round_invalid_centre(self):
        kx = np.linspace(-5, 5, 51)
        ky = np.linspace(-5, 5, 51)
        KX, KY = np.meshgrid(kx, ky)
        values = np.cos(KX) * np.cos(KY)
        da = _simulate_fake_scan(values, dims=("ky", "kx"), x=kx, y=ky)
        with pytest.raises(
            Exception, match="is not a valid dimension of the inputted DataArray"
        ):
            da.rotate(30, x1=0, x2=0)

    def test_rotate_rejects_1D_data(self):
        da = _simulate_fake_scan(np.ones((10,)), dims=("eV",))
        with pytest.raises(Exception, match="Function only acts on 2D or 3D data"):
            da.rotate(10)


class TestSym:
    def test_sym_doubles_data(self):
        theta_par = np.linspace(-5, 5, 51)
        values = np.exp(-(theta_par**2))  # gaussian symmetric about 0
        eV = np.array([0.0])
        # making an MDC symmetic about 0
        da = _simulate_fake_scan(values[np.newaxis, :], y=eV, x=theta_par)
        result = da.sym(theta_par=0)
        np.testing.assert_allclose(result.values, da.values * 2)

    def test_sym_flip_returns_mirror(self):
        theta_par = np.linspace(-5, 5, 51)
        values = theta_par  # a random antisymmetric linear ramp
        eV = np.array([0.0])
        da = _simulate_fake_scan(values[np.newaxis, :], y=eV, x=theta_par)
        result = da.sym(theta_par=0, flipped=True)
        np.testing.assert_allclose(da.values, -result.values)

    def test_sym_not_crash(self, disp):
        result = disp.sym(eV=16.8)
        assert result is not None

    def test_sym_multiple_axes(self, fake_disp):
        with pytest.raises(
            Exception, match="Function can only be called with single axis"
        ):
            fake_disp.sym(theta_par=0, eV=0)

    def test_sym_invalid_axis(self, fake_disp):
        with pytest.raises(
            Exception,
            match="Provided symmetrisation axis is not a valid dimension of inputted DataArray",
        ):
            fake_disp.sym(x1=0)

    def test_sym_out_of_range(self, fake_disp):
        with pytest.raises(Exception, match="not within"):
            fake_disp.sym(eV=999)


class TestSymNfold:
    def test_sym_nfold_on_4fold_symmetric_data(self):
        # idea is to make a 4-fold symmetric scan, symmetrise it, and compare the symmetrised result with the original data
        kx = np.linspace(-5, 5, 51)
        ky = np.linspace(-5, 5, 51)
        KX, KY = np.meshgrid(kx, ky)
        values = np.cos(KX) * np.cos(KY)
        da = _simulate_fake_scan(values, dims=("ky", "kx"), x=kx, y=ky)
        trimmed_4fold = da.sym_nfold(nfold=4, kx=0, ky=0, expand=False).sel(
            kx=slice(-4, 4), ky=slice(-4, 4)
        )
        trimmed_2fold = da.sym_nfold(nfold=2, kx=0, ky=0, expand=False).sel(
            kx=slice(-4, 4), ky=slice(-4, 4)
        )
        max_trimmed_4fold = trimmed_4fold.max().item()
        max_trimmed_2fold = trimmed_2fold.max().item()
        xr.testing.assert_allclose(
            da.sel(kx=slice(-4, 4), ky=slice(-4, 4)).pint.dequantify(),
            trimmed_4fold / max_trimmed_4fold,
            atol=1e-2,
        )
        xr.testing.assert_allclose(
            da.sel(kx=slice(-4, 4), ky=slice(-4, 4)).pint.dequantify(),
            trimmed_2fold / max_trimmed_2fold,
            atol=1e-2,
        )

    def test_sym_nfold_on_3fold_symmetric_data(self):
        kx = np.linspace(-5, 5, 51)
        ky = np.linspace(-5, 5, 51)
        KX, KY = np.meshgrid(kx, ky)
        q1r = KX
        q2r = -0.5 * KX + (np.sqrt(3) / 2) * KY
        q3r = -0.5 * KX - (np.sqrt(3) / 2) * KY
        # summing 3 plane waves to make a 3-fold symmetric scan (actually 6-fold symmetric? but hey ho)
        values = np.cos(q1r) + np.cos(q2r) + np.cos(q3r)
        da = _simulate_fake_scan(values, dims=("ky", "kx"), x=kx, y=ky)
        da_trimmed = da.sel(kx=slice(-3, 3), ky=slice(-3, 3))
        da_trimmed_normalised = da_trimmed / da_trimmed.max().item()
        trimmed_3fold = da.sym_nfold(nfold=3, kx=0, ky=0, expand=False).sel(
            kx=slice(-3, 3), ky=slice(-3, 3)
        )
        max_trimmed_3fold = trimmed_3fold.max().item()
        xr.testing.assert_allclose(
            da_trimmed_normalised.pint.dequantify(),
            trimmed_3fold / max_trimmed_3fold,
            atol=1e-2,
        )

    def test_sym_nfold_expand_false_preserves_shape(self):
        da = _simulate_fake_scan(np.ones((10, 10)), dims=("ky", "kx"))
        result = da.sym_nfold(nfold=4, expand=False)
        assert da.shape == result.shape

    def test_sym_nfold_rejects_1D_data(self):
        da = _simulate_fake_scan(np.ones((10,)), dims=("eV",))
        with pytest.raises(Exception, match="Function only acts on 2D or 3D data"):
            da.sym_nfold(nfold=3)

    def test_sym_nfold_invalid_dim(self):
        da = _simulate_fake_scan(np.ones((10, 10)), dims=("ky", "kx"))
        with pytest.raises(
            Exception, match="is not a valid dimension of the inputted DataArray"
        ):
            da.sym_nfold(nfold=2, x1=0, x2=0)


class TestDegrid:
    def test_degrid(self):
        theta_par = np.linspace(-17, 17, 500)
        eV = np.linspace(16.3, 16.8, 500)
        theta_range = theta_par[-1] - theta_par[0]
        eV_range = eV[-1] - eV[0]
        T, E = np.meshgrid(theta_par, eV)
        band_peaks = 0.005 * T**2 + 16.4
        # make a fake dispersion - one is so broad that I'll never measure anything like this...hopefully!
        disp = 1000 * np.exp(-((E - band_peaks) ** 2) / 0.003)
        mesh = 0.9 + 0.1 * np.cos(2 * np.pi * T * 100 / theta_range) * np.cos(
            2 * np.pi * E * 100 / eV_range
        )
        gridded_disp = disp * mesh
        gridded_da = _simulate_fake_scan(gridded_disp)
        result = gridded_da.degrid()
        error_original = float(np.mean(((gridded_da.values - disp) ** 2)))
        error_degridded = float(np.mean(((result.values - disp) ** 2)))
        assert float(error_degridded) < float(error_original)

    def test_degrid_rejects_non_2D_data(self):
        da = _simulate_fake_scan(np.ones((10,)), dims=("eV",))
        with pytest.raises(Exception, match="Inputted data must be a 2D scan"):
            da.degrid()

    def test_degrid_preserves_shape(self, fake_disp):
        result = fake_disp.degrid()
        assert fake_disp.shape == result.shape


class TestSumSubtractData:
    def test_sum_two_arrays(self, fake_disp):
        result = _sum_or_subtract_data([fake_disp, fake_disp], _sum=True, quiet=False)
        expected = fake_disp.values * 2
        np.testing.assert_allclose(result.values, expected)

    def test_subtract_two_arrays(self, fake_disp):
        result = _sum_or_subtract_data([fake_disp, fake_disp], _sum=False, quiet=False)
        expected = fake_disp.values - fake_disp.values
        np.testing.assert_allclose(result.values, expected)

    def test_subtract_requires_two_arrays(self):
        with pytest.raises(
            Exception,
            match="Data subtraction only accepts two DataArrays or a DataTree with two leaves",
        ):
            _sum_or_subtract_data(
                [fake_disp, fake_disp, fake_disp], _sum=False, quiet=False
            )

    def test_sum_mismatched_coords_interpolated(self, fake_disp):
        theta_par = np.linspace(-17.01, 17.01, 101)
        eV = np.linspace(16.31, 16.81, 51)
        rng = np.random.default_rng(seed=42)
        fake_disp_mismatched = _simulate_fake_scan(
            rng.random((51, 101)), y=eV, x=theta_par
        )
        result = _sum_or_subtract_data(
            [fake_disp, fake_disp_mismatched], _sum=True, quiet=True
        )
        # the second one should be interpolated onto the first's grid
        assert result.shape == fake_disp.shape

    def test_sum_fires_warning_on_mismatched_metadata(self, fake_disp, monkeypatch):
        from peaks.core.metadata.base_metadata_models import TemperatureMetadataModel

        fake_disp_1 = fake_disp.copy(deep=True)
        fake_disp_2 = fake_disp.copy(deep=True)
        fake_disp_1.attrs["_temperature"] = TemperatureMetadataModel(
            sample=18.1 * ureg.K
        )
        fake_disp_2.attrs["_temperature"] = TemperatureMetadataModel(
            sample=18.2 * ureg.K
        )
        warnings_received = []

        def fake_warning(msg, *args, **kwargs):
            warnings_received.append(msg)

        monkeypatch.setattr("peaks.core.process.tools.analysis_warning", fake_warning)
        _sum_or_subtract_data([fake_disp_1, fake_disp_2], _sum=True, quiet=False)
        assert len(warnings_received) == 1
        assert "do not match those of scan" in warnings_received[0]

    def test_sum_mismatched_dims(self, fake_disp):
        fake_disp_2 = _simulate_fake_scan(np.ones((50, 100)), dims=("ky", "kx"))
        with pytest.raises(
            Exception, match="Inputted DataArrays must have the same dimensions"
        ):
            _sum_or_subtract_data([fake_disp, fake_disp_2])


class TestMergeData:
    # to-do: test merging two hv scans into one
    @pytest.fixture
    def da1(self):
        # making a huge flat band ?!
        from peaks.core.metadata.base_metadata_models import ARPESCalibrationModel

        theta_par = np.linspace(-17, 17, 500)
        eV = np.linspace(16.3, 16.8, 500)
        T, E = np.meshgrid(theta_par, eV)
        band_peaks = 16.5
        disp = 1000 * np.exp(-((E - band_peaks) ** 2) / 0.00005)
        da = _simulate_fake_scan(disp, y=eV, x=theta_par)
        da.attrs["_calibration"] = ARPESCalibrationModel(EF_correction=16.7)
        return da

    def test_merge_extends_range(self, da1):
        da2 = da1.copy(deep=True)
        result = merge_data([da1, da2], offsets=[0, 12])
        original_angular_range = da1.theta_par[-1] - da1.theta_par[0]
        merged_angular_range = result.theta_par[-1] - result.theta_par[0]
        assert float(original_angular_range) < float(merged_angular_range)

    def test_merge_preserves_features(self, da1):
        # the merged one should also be a flat band but longer
        da2 = da1.copy(deep=True)
        result = merge_data([da1, da2], offsets=[0, 12])
        merged_peaks = result.eV[result.argmax("eV")].values
        assert len(np.unique(merged_peaks)) == 1
        assert merged_peaks[0] == pytest.approx(16.5, abs=1e-2)

    def test_merge_preserves_intensity(self, da1):
        da2 = da1.copy(deep=True)
        result = merge_data([da1, da2], offsets=[0, 12])
        np.testing.assert_almost_equal(
            result.max("eV").mean().values, da1.max("eV").mean().values
        )

    def test_merge_invalid_dim(self, da1):
        with pytest.raises(
            Exception, match="is not a valid dimension of the inputted data"
        ):
            merge_data([da1, da1], dim="whatever")


class TestEstimateSymPoint:
    def test_est_sym_centred_circ(self):
        kx = np.linspace(-2, 2, 401)
        ky = np.linspace(-2, 2, 401)
        KX, KY = np.meshgrid(kx, ky)
        values = 1000 * np.exp(-((np.sqrt(KX**2 + KY**2) - 1) ** 2) / 0.001)
        # making a circular Fermi surface centred round gamma
        da = _simulate_fake_scan(values, dims=("ky", "kx"), x=kx, y=ky)
        result = estimate_sym_point(da)
        assert float(result["kx"]) == pytest.approx(0.0)
        assert float(result["ky"]) == pytest.approx(0.0)

    def test_est_sym_offset_square(self):
        kx = np.linspace(-2, 2, 401)
        ky = np.linspace(-2, 2, 401)
        KX, KY = np.meshgrid(kx, ky)
        cx, cy = 0.7, -0.5
        sq = np.maximum(np.abs(KX - cx), np.abs(KY - cy))
        values = 1000 * np.exp(-((sq - 0.9) ** 2) / 0.001)
        da = _simulate_fake_scan(values, dims=("ky", "kx"), x=kx, y=ky)
        result = estimate_sym_point(da)
        assert float(result["kx"]) == pytest.approx(0.7)
        assert float(result["ky"]) == pytest.approx(-0.5)

    def test_est_sym_offset_hexagon(self):
        kx = np.linspace(-2, 2, 401)
        ky = np.linspace(-2, 2, 401)
        KX, KY = np.meshgrid(kx, ky)
        cx, cy = 0.7, -0.5
        angles = np.arange(0, 6) * np.pi / 3
        hex = np.zeros_like(KX)
        for a in angles:
            hex = np.maximum(
                hex,
                np.abs((KX - cx) * np.cos(a) + (KY - cy) * np.sin(a)),
            )
        values = 1000 * np.exp(-((hex - 1.0) ** 2) / 0.001)
        da = _simulate_fake_scan(values, dims=("ky", "kx"), x=kx, y=ky)
        result = estimate_sym_point(da)
        assert float(result["kx"]) == pytest.approx(0.7)
        assert float(result["ky"]) == pytest.approx(-0.5)


class TestDriftCorrection:
    def test_drift_correction_for_known_shift(self):
        x = np.arange(100, dtype=float)
        y = np.arange(100, dtype=float)
        X, Y = np.meshgrid(x, y)
        ref_data = np.exp(-((X - 30) ** 2 + (Y - 30) ** 2) / 10)
        mov_data = np.exp(-((X - 25) ** 2 + (Y - 37) ** 2) / 10)
        ref_da = _simulate_fake_scan(ref_data, dims=("y", "x"), x=x, y=y)
        mov_da = _simulate_fake_scan(mov_data, dims=("y", "x"), x=x, y=y)
        result = drift_correction(
            reference_data=ref_da, moving_data=mov_da, orig_pos={"x": 30, "y": 30}
        )
        assert float(result[1]["x"]) == pytest.approx(25.0)
        assert float(result[1]["y"]) == pytest.approx(37.0)

    def test_drift_correction_for_zero_shift(self):
        x = np.arange(100, dtype=float)
        y = np.arange(100, dtype=float)
        X, Y = np.meshgrid(x, y)
        ref_data = np.exp(-((X - 30) ** 2 + (Y - 30) ** 2) / 10)
        ref_da = _simulate_fake_scan(ref_data, dims=("y", "x"), x=x, y=y)
        mov_da = ref_da.copy(deep=True)
        result = drift_correction(
            reference_data=ref_da, moving_data=mov_da, orig_pos={"x": 30, "y": 30}
        )
        assert float(result[0]["x"]) == pytest.approx(0.0)
        assert float(result[0]["y"]) == pytest.approx(0.0)

    def test_drift_correction_rejects_3D_data(self):
        ref_da = _simulate_fake_scan(np.ones((5, 5, 5)), dims=("y", "x", "z"))
        mov_da = ref_da.copy(deep=True)
        with pytest.raises(
            Exception,
            match="Supplied data has more than 2 dimensions. Select a 2D slice",
        ):
            drift_correction(ref_da, mov_da)


class TestToolsPreserveUnits:
    def test_bgs_preserves_units(self, fake_disp):
        result = fake_disp.bgs(0.5)
        assert result.data.units == fake_disp.data.units

    def test_bin_data_preserves_units(self, fake_disp):
        result = fake_disp.bin_data(2)
        assert result.data.units == fake_disp.data.units

    def test_bin_spectra_preserves_units(self, fake_disp):
        result = fake_disp.bin_spectra()
        assert result.data.units == fake_disp.data.units

    def test_smooth_preserves_units(self, fake_disp):
        result = fake_disp.smooth(eV=0.05, theta_par=0.2)
        assert result.data.units == fake_disp.data.units

    def test_rotate_preserves_units(self, fake_disp):
        result = fake_disp.rotate(360)
        assert result.data.units == fake_disp.data.units

    def test_sym_preserves_units(self, fake_disp):
        result = fake_disp.sym(eV=16.7)
        assert result.data.units == fake_disp.data.units

    def test_sym_nfold_preserves_units(self):
        da = _simulate_fake_scan(np.ones((10, 10)), dims=("ky", "kx"))
        result = da.sym_nfold(nfold=4, expand=True)
        assert result.data.units == da.data.units

    def test_degrid_preserves_units(self, fake_disp):
        result = fake_disp.degrid()
        assert result.data.units == fake_disp.data.units

    def test_sum_subtract_preserves_units(self, fake_disp):
        result = _sum_or_subtract_data([fake_disp, fake_disp], _sum=True, quiet=False)
        assert result.data.units == fake_disp.data.units

    def test_merge_preserves_units(self, fake_disp):
        from peaks.core.metadata.base_metadata_models import ARPESCalibrationModel

        fake_disp.attrs["_calibration"] = ARPESCalibrationModel(EF_correction=16.7)
        result = merge_data([fake_disp, fake_disp], offsets=[0, 12])
        assert result.data.units == fake_disp.data.units
