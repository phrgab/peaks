import matplotlib
import numpy as np
import pint_xarray
import pytest
import xarray as xr

from peaks.core.process.tools import (
    _sum_or_subtract_data,
    correct_isolated_bad_pixels,
    correct_swept_scan_bad_pixels,
    drift_correction,
    estimate_sym_point,
    flat_field_correction,
    merge_data,
    norm_by_I0,
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


class TestCorrectSweptScanBadPixels:
    def _simple_dispersion(self):
        eV = np.linspace(16.3, 16.8, 6)
        theta_par = np.linspace(-2.5, 2.5, 7)
        values = 3 * eV[:, None] + 2 * theta_par[None, :]
        return _simulate_fake_scan(values, y=eV, x=theta_par)

    def test_corrects_single_swept_line_from_coordinate(self):
        da = self._simple_dispersion()
        damaged = da.copy(deep=True)
        damaged.data.magnitude[:, 3] = -999.0

        corrected = correct_swept_scan_bad_pixels(
            damaged, theta_par=float(da.theta_par[3])
        )
        xr.testing.assert_allclose(corrected, da)
        assert (
            corrected.attrs["_analysis_history"].records[-1].record
            == "Swept bad pixels corrected by interpolation along theta_par, defined from "
            f"single theta_par value {float(da.theta_par[3])}"
        )

    def test_corrects_swept_stripe_from_two_anchors(self):
        da = self._simple_dispersion()
        damaged = da.copy(deep=True)
        damaged.data.magnitude[0, 1:3] = -999.0
        damaged.data.magnitude[1, 2:4] = -999.0
        damaged.data.magnitude[2, 2:4] = -999.0
        damaged.data.magnitude[3, 3:5] = -999.0
        damaged.data.magnitude[4, 3:5] = -999.0
        damaged.data.magnitude[5, 4:6] = -999.0

        corrected = correct_swept_scan_bad_pixels(
            damaged,
            theta_par=[
                (float(da.eV[0]), float(da.theta_par[1]), float(da.theta_par[2])),
                (float(da.eV[-1]), float(da.theta_par[4]), float(da.theta_par[5])),
            ],
        )
        xr.testing.assert_allclose(corrected, da)

    def test_requires_eV_dimension(self):
        da = _simulate_fake_scan(np.ones((5, 5)), dims=("ky", "kx"))
        with pytest.raises(
            ValueError, match="requires a 2D DataArray with an 'eV' dimension"
        ):
            correct_swept_scan_bad_pixels(da, kx=0.0)

    def test_rejects_unbounded_swept_region(self):
        da = self._simple_dispersion()
        damaged = da.copy(deep=True)
        damaged.data.magnitude[:, :2] = -999.0

        with pytest.raises(
            ValueError,
            match="Swept bad pixels must be bounded by good pixels within each eV slice",
        ):
            correct_swept_scan_bad_pixels(damaged, theta_par=float(da.theta_par[0]))


class TestCorrectIsolatedBadPixels:
    def _simple_grid(self):
        values = np.arange(25, dtype=float).reshape(5, 5)
        return _simulate_fake_scan(
            values, y=np.arange(5, dtype=float), x=np.arange(5, dtype=float)
        )

    def test_corrects_single_pixel_from_raw_indices(self):
        da = self._simple_grid()
        damaged = da.copy(deep=True)
        damaged.data.magnitude[2, 2] = -999.0

        corrected = correct_isolated_bad_pixels(damaged, [(2, 2)])

        expected = da.copy(deep=True)
        expected.data.magnitude[2, 2] = np.mean([7.0, 17.0, 11.0, 13.0])
        xr.testing.assert_allclose(corrected, expected)

    def test_corrects_single_pixel_from_coordinates(self):
        da = self._simple_grid()
        damaged = da.copy(deep=True)
        damaged.data.magnitude[1, 3] = -999.0

        corrected = correct_isolated_bad_pixels(damaged, [{"eV": 1.0, "theta_par": 3.0}])

        expected = da.copy(deep=True)
        expected.data.magnitude[1, 3] = np.mean([3.0, 13.0, 7.0, 9.0])
        xr.testing.assert_allclose(corrected, expected)

    def test_accepts_boolean_mask(self):
        da = self._simple_grid()
        damaged = da.copy(deep=True)
        damaged.data.magnitude[3, 1] = -999.0
        mask = xr.zeros_like(da, dtype=bool)
        mask.data[3, 1] = True

        corrected = correct_isolated_bad_pixels(damaged, mask)

        expected = da.copy(deep=True)
        expected.data.magnitude[3, 1] = np.mean([11.0, 21.0, 15.0, 17.0])
        xr.testing.assert_allclose(corrected, expected)

    def test_adds_history_entry(self):
        da = self._simple_grid()
        result = correct_isolated_bad_pixels(da, [(1, 2)])
        assert (
            result.attrs["_analysis_history"].records[-1].record
            == "Isolated bad pixels corrected from 1 raw pixel indices"
        )


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

    def test_correct_swept_scan_bad_pixels_preserves_units(self, fake_disp):
        damaged = fake_disp.copy(deep=True)
        damaged.data.magnitude[:, 10] = -999.0
        result = correct_swept_scan_bad_pixels(
            damaged, theta_par=float(fake_disp.theta_par[10])
        )
        assert result.data.units == fake_disp.data.units

    def test_sum_subtract_preserves_units(self, fake_disp):
        result = _sum_or_subtract_data([fake_disp, fake_disp], _sum=True, quiet=False)
        assert result.data.units == fake_disp.data.units

    def test_merge_preserves_units(self, fake_disp):
        from peaks.core.metadata.base_metadata_models import ARPESCalibrationModel

        fake_disp.attrs["_calibration"] = ARPESCalibrationModel(EF_correction=16.7)
        result = merge_data([fake_disp, fake_disp], offsets=[0, 12])
        assert result.data.units == fake_disp.data.units


def _flat_field_reference():
    # Vignetted gaussian sensitivity on a 1212x1212 detector, zero outside a disc
    y, x = np.mgrid[:1212, :1212] - 605.5
    sensitivity = np.exp(-(x**2 + y**2) / 600**2)
    sensitivity[x**2 + y**2 > 500**2] = 0
    sensitivity[600, 600] = 0  # dead pixel inside the illuminated region
    return _simulate_fake_scan(
        sensitivity * 1000, dims=("detector_y", "detector_x"), unit="count"
    )


def test_flat_field_correction_flattens_uniform_signal():
    ref = _flat_field_reference()
    data = _simulate_fake_scan(
        3 * ref.pint.dequantify(), dims=("detector_y", "detector_x")
    )
    corrected = flat_field_correction(data, ref)

    illuminated = ref.pint.dequantify() > 0.05 * float(ref.max().pint.dequantify())
    values = corrected.pint.dequantify().values
    assert np.ptp(values[illuminated]) < 1e-9
    assert corrected.pint.units == data.pint.units
    assert corrected.name == data.name
    assert set(corrected.attrs) == set(data.attrs)
    assert len(corrected.history.get()) == len(data.history.get()) + 1
    assert corrected.history.get(-1)["record"]["threshold"] == 0.05


@pytest.mark.parametrize("fill", [1, np.nan])
def test_flat_field_correction_fill(fill):
    ref = _flat_field_reference()
    data = _simulate_fake_scan(np.ones((1212, 1212)), dims=("detector_y", "detector_x"))
    values = flat_field_correction(data, ref, fill=fill).pint.dequantify().values
    # Outside the disc and at the dead pixel
    np.testing.assert_equal(values[0, 0], fill)
    np.testing.assert_equal(values[600, 600], fill)


def test_flat_field_correction_binned_data():
    ref = _flat_field_reference()
    raw = 3 * ref.pint.dequantify().values
    full = flat_field_correction(
        _simulate_fake_scan(raw, dims=("detector_y", "detector_x")), ref
    )
    binned = flat_field_correction(
        _simulate_fake_scan(
            raw.reshape(606, 2, 1212).mean(1), dims=("detector_y", "detector_x")
        ),
        ref,
    )
    # Only compare bins fully inside the illuminated region
    inside = (raw > 0.05 * raw.max()).reshape(606, 2, 1212).all(1)
    full_binned = full.pint.dequantify().coarsen(detector_y=2).mean().values
    np.testing.assert_allclose(
        binned.pint.dequantify().values[inside], full_binned[inside], rtol=1e-6
    )


def test_flat_field_correction_broadcasts_and_stays_lazy():
    ref = _flat_field_reference()
    raw = np.stack([ref.pint.dequantify().values * i for i in (1, 2, 3)])
    data = _simulate_fake_scan(raw, dims=("eV", "detector_y", "detector_x")).chunk(
        {"eV": 1}
    )
    corrected = flat_field_correction(data, ref)
    assert corrected.dims == data.dims
    assert corrected.chunks is not None
    values = corrected.pint.dequantify().values
    np.testing.assert_allclose(values[2, 605, 605] / values[0, 605, 605], 3)


def test_flat_field_correction_custom_mask():
    ref = _flat_field_reference()
    data = _simulate_fake_scan(np.ones((1212, 1212)), dims=("detector_y", "detector_x"))
    mask = xr.zeros_like(ref.pint.dequantify(), dtype=bool)
    mask[500:700, 500:700] = True
    corrected = flat_field_correction(data, ref, mask=mask)
    assert corrected.pint.dequantify().values[100, 600] == 1
    assert corrected.history.get(-1)["record"]["custom mask"]


def test_flat_field_correction_bad_shape_raises():
    ref = _flat_field_reference()
    data = _simulate_fake_scan(np.ones((500, 1212)), dims=("detector_y", "detector_x"))
    with pytest.raises(ValueError, match="integer fraction"):
        flat_field_correction(data, ref)


def test_norm_by_I0_broadcasts_over_higher_dims():
    rng = np.random.default_rng(0)
    values = rng.random((3, 4, 5, 6))
    I0 = rng.random((3, 4)) + 1
    data = xr.DataArray(
        values, dims=("x1", "x2", "eV", "theta_par"), name="spectrum"
    ).assign_coords(I0=(("x1", "x2"), I0))
    data = data.pint.quantify("count")
    normed = norm_by_I0(data)
    np.testing.assert_allclose(
        normed.pint.dequantify().values, values / I0[:, :, None, None]
    )
    assert normed.pint.units == data.pint.units
    assert normed.history.get(-1)["record"] == "Data normalised by I0"
    # I0 coordinate stays aligned when slicing before normalising
    np.testing.assert_allclose(
        norm_by_I0(data.isel(x1=1)).pint.dequantify().values,
        values[1] / I0[1, :, None, None],
    )


def _I0_scan(I0, I0_units=None):
    data = xr.DataArray(
        np.full(len(I0), 10.0), dims="eV", coords={"eV": np.arange(len(I0))}
    ).assign_coords(I0=("eV", np.asarray(I0, dtype=float)))
    return data.pint.quantify({None: "count", "I0": I0_units})


@pytest.mark.parametrize("I0_units", [None, "V"])
def test_norm_by_I0_offset(I0_units):
    data = _I0_scan([0.07, 0.08], I0_units)
    offsets = [-0.001]
    if I0_units:
        # a pint offset in other units, and the mean I0 of a dark scan
        offsets += [
            -1 * pint_xarray.unit_registry("mV"),
            _I0_scan([-0.001] * 3, "V").I0.mean(),
        ]
    for offset in offsets:
        normed = norm_by_I0(data, I0_offset=offset)
        np.testing.assert_allclose(
            normed.pint.dequantify().values, 10 / np.array([0.071, 0.081])
        )
        assert "I0 offset" in normed.history.get(-1)["record"]
    if I0_units:
        assert normed.pint.units == pint_xarray.unit_registry.Unit("count / V")


@pytest.mark.parametrize(
    "I0, offset", [([0.07, -0.001], 0), ([0.07, 0.0], 0), ([0.07, 0.001], 0.002)]
)
def test_norm_by_I0_non_positive_raises(I0, offset):
    with pytest.raises(ValueError, match="<= 0"):
        norm_by_I0(_I0_scan(I0, "V"), I0_offset=offset)


def test_norm_by_I0_without_I0_raises():
    data = xr.DataArray(np.ones((2, 2)), dims=("eV", "theta_par"))
    with pytest.raises(ValueError, match="No I0 coordinate"):
        norm_by_I0(data)
