import numpy as np
import pytest
import xarray as xr
from lmfit.models import GaussianModel

from peaks.core.fitting.models import _shirley_bg, create_xarray_compatible_lmfit_model


class TestShirleyBG:
    @pytest.fixture
    def fake_XPS(self):
        def _make(centre=5.0):
            x = np.linspace(0, 10, 200)
            centre, sigma, amp = centre, 0.8, 15.0
            gaussian = amp * np.exp(-0.5 * ((x - centre) / sigma) ** 2)
            step = 2.0 + 7.0 / (1 + np.exp(3 * (x - centre)))
            return gaussian + step

        return _make

    @pytest.fixture
    def pure_gaussian(self):
        x = np.linspace(0, 10, 200)
        centre, sigma, amp = 5.0, 0.8, 15.0
        gaussian = amp * np.exp(-0.5 * ((x - centre) / sigma) ** 2)
        return gaussian

    def test_pure_gaussian_returns_a_zero_baseline(self, pure_gaussian):
        result = _shirley_bg(pure_gaussian)
        np.testing.assert_allclose(result, np.zeros_like(pure_gaussian), atol=1e-6)

    def test_monotonic_bg(self, fake_XPS):
        result = _shirley_bg(fake_XPS())
        diffs = np.diff(result)
        assert result[0] > result[-1]
        assert np.all(diffs <= 0)

    def test_shirley_bg_subtracts_offset_start(self, fake_XPS):
        result_offset = _shirley_bg(
            fake_XPS(centre=2), offset_start=fake_XPS(centre=2)[0] - 9
        )
        result_no_offset = _shirley_bg(fake_XPS(centre=2))
        assert result_offset[0] < result_no_offset[0]

    def test_shirley_bg_subtracts_offset_end(self, fake_XPS):
        result_offset = _shirley_bg(
            fake_XPS(centre=8), offset_end=fake_XPS(centre=8)[-1] - 2
        )
        result_no_offset = _shirley_bg(fake_XPS(centre=8))
        assert result_offset[-1] < result_no_offset[-1]

    def test_shirley_bg_rejects_2D_input(self, fake_XPS):
        with pytest.raises(
            Exception,
            match="Inputted data must be a 1D numpy.ndarray, list or xarray.DataArray",
        ):
            _shirley_bg(np.array([fake_XPS(), fake_XPS()]))


class TestCreateXarrayCompatibleLmfitModel:
    def test_returns_a_lmfit_model(self):
        wrapped_gaussian = create_xarray_compatible_lmfit_model(GaussianModel)
        assert isinstance(wrapped_gaussian, type)

    def test_returned_model_is_a_subclass_of_orig_lmfit_model(self):
        wrapped_gaussian = create_xarray_compatible_lmfit_model(GaussianModel)
        assert issubclass(wrapped_gaussian, GaussianModel)

    def test_rejects_np_array(self):
        wrapped_gaussian = create_xarray_compatible_lmfit_model(GaussianModel)
        with pytest.raises(
            Exception,
            match="To pass numpy arrays, use the original lmfit model",
        ):
            wrapped_gaussian().guess(np.array([1, 2, 3]))

    def test_rejects_2D_array(self):
        wrapped_gaussian = create_xarray_compatible_lmfit_model(GaussianModel)
        da = xr.DataArray(np.array([[1, 2, 3], [4, 5, 6]]))
        with pytest.raises(
            Exception,
            match="Supplied xr.DataArray should be one dimensional",
        ):
            wrapped_gaussian().guess(da)

    def test_docstring_modified(self):
        wrapped_gaussian = create_xarray_compatible_lmfit_model(GaussianModel)
        assert wrapped_gaussian.__doc__ is not None
        assert "Modified version of" in wrapped_gaussian.__doc__

    def test_wrapped_all_standard_lmfit_models(self):
        from peaks.core.fitting.models import (
            GaussianModel,  # type: ignore
            LinearModel,  # type: ignore
            LorentzianModel,  # type: ignore
            PseudoVoigtModel,  # type: ignore
            VoigtModel,  # type: ignore
        )

        model_gaussian = GaussianModel()
        model_linear = LinearModel()
        model_lorentzian = LorentzianModel()
        model_pseudo_voigt = PseudoVoigtModel()
        model_voigt = VoigtModel()
        assert model_gaussian is not None
        assert model_linear is not None
        assert model_lorentzian is not None
        assert model_pseudo_voigt is not None
        assert model_voigt is not None

    def test_wrapped_model_works(self):
        from peaks.core.fitting.models import GaussianModel  # type: ignore

        x = np.linspace(0, 10, 200)
        centre, sigma, amp = 5.0, 0.8, 15.0
        real_gaussian = amp * np.exp(-0.5 * ((x - centre) / sigma) ** 2)
        da = xr.DataArray(real_gaussian, dims=["x"], coords={"x": x})
        model_gaussian = GaussianModel()
        params = model_gaussian.guess(da)
        x = da.coords["x"].values
        y = da.values
        result = model_gaussian.fit(y, params, x=x)
        assert result.best_fit is not None
        assert result.params["center"].value == pytest.approx(centre, rel=0.05)
        assert result.params["sigma"].value == pytest.approx(sigma, rel=0.05)
        assert result.params["height"].value == pytest.approx(amp, rel=0.05)


class TestGaussianConvolvedFitModel:
    def test_sigma_conv_added(self):
        from peaks.core.fitting.models import GaussianConvolvedFitModel

        base_model = GaussianModel()
        model = GaussianConvolvedFitModel(base_model)
        assert "sigma_conv" in model.param_names
        assert "sigma" in model.param_names

    def test_sigma_conv_vary_False_by_default(self):
        from peaks.core.fitting.models import GaussianConvolvedFitModel

        base_model = GaussianModel()
        model = GaussianConvolvedFitModel(base_model)
        assert not model.make_params().get("sigma_conv").vary


class TestFermiFunctionModel:
    def test_has_expected_param_names(self):
        from peaks.core.fitting.models import FermiFunctionModel

        model = FermiFunctionModel()
        assert "EF" in model.param_names
        assert "T" in model.param_names

    def test_T_vary_False_by_default(self):
        from peaks.core.fitting.models import FermiFunctionModel

        model = FermiFunctionModel()
        assert not model.make_params().get("T").vary

    def test_supports_prefix(self):
        from peaks.core.fitting.models import FermiFunctionModel

        model = FermiFunctionModel(prefix="fd_")
        assert "fd_EF" in model.param_names
        assert "fd_T" in model.param_names


class TestLinearDosFermiModel:
    def test_has_expected_param_names(self):
        from peaks.core.fitting.models import LinearDosFermiModel

        model = LinearDosFermiModel()
        assert "EF" in model.param_names
        assert "T" in model.param_names
        assert "bg_slope" in model.param_names
        assert "bg_intercept" in model.param_names
        assert "dos_slope" in model.param_names
        assert "sigma_conv" in model.param_names  # from convolution wrapper
