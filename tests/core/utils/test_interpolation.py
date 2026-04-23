import numpy as np
import pytest
from scipy import interpolate

from peaks.core.utils.interpolation import (
    _fast_bilinear_interpolate,
    _fast_bilinear_interpolate_rectilinear,
    _fast_trilinear_interpolate,
    _fast_trilinear_interpolate_rectilinear,
    _is_linearly_spaced,
)


@pytest.fixture
def sin_1D():
    x = np.linspace(0, 3 * np.pi, 50)
    x_prime = np.linspace(0, 2 * np.pi, 100)
    return x, np.sin(x), x_prime, np.sin(x_prime)


@pytest.fixture
def sincos_2D():
    x = np.linspace(0, 3 * np.pi, 200)
    y = np.linspace(-np.pi, 3 * np.pi, 250)
    X, Y = np.meshgrid(x, y, indexing="ij")
    values = np.sin(X) * np.cos(Y)

    x_prime = np.linspace(0, 2 * np.pi, 250)
    y_prime = np.linspace(-np.pi, 2 * np.pi, 250)
    XP, YP = np.meshgrid(x_prime, y_prime, indexing="ij")
    expected = np.sin(XP) * np.cos(YP)

    return x, y, values, x_prime, y_prime, XP, YP, expected


@pytest.fixture
def sincos_exp_3D():
    x = np.linspace(0, 3 * np.pi, 300)
    y = np.linspace(-np.pi, 3 * np.pi, 250)
    z = np.linspace(0, 2.0, 20)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    values = np.sin(X) * np.cos(Y) * np.exp(-Z)

    x_prime = np.linspace(0, 2 * np.pi, 250)
    y_prime = np.linspace(-np.pi, 2 * np.pi, 250)
    z_prime = np.linspace(0, 2.5, 250)
    XP, YP, ZP = np.meshgrid(x_prime, y_prime, z_prime, indexing="ij")
    expected = np.sin(XP) * np.cos(YP) * np.exp(-ZP)

    return x, y, z, values, x_prime, y_prime, z_prime, XP, YP, ZP, expected


class TestIsLinearlySpaced:
    def test_linspace_is_linear(self):
        assert _is_linearly_spaced(np.linspace(0, 10, 50))  # should be true

    def test_logspace_is_not_linear(self):
        assert not _is_linearly_spaced(np.logspace(0, 2, 20))  # should be false

    def test_single_point(self):
        assert _is_linearly_spaced(np.array([1.0]))  # should be true

    def test_two_points(self):
        assert _is_linearly_spaced(np.array([1.0, 3.0]))  # should be true

    def test_neg_linspace(self):
        assert _is_linearly_spaced(np.linspace(-5, 5, 50))  # should be true

    def test_uneven_spacing(self):
        assert not _is_linearly_spaced(np.array([0.0, 1.0, 3.0, 7.0]))  # should be false


class TestFastTrilinearInterpolateRectilinear:
    def test_1D_data_close_to_expected(self, sin_1D):
        x, sinx, x_prime, expected = sin_1D

        dummy = np.array([0.0, 1.0])
        orig_values = np.repeat(sinx[:, None, None], 2, axis=1)
        orig_values = np.repeat(orig_values, 2, axis=2)

        result = _fast_trilinear_interpolate_rectilinear(
            x_prime,
            np.full_like(x_prime, 0.0),
            np.full_like(x_prime, 0.0),
            x,
            dummy,
            dummy,
            orig_values,
        )
        np.testing.assert_array_less(np.abs(expected - result), 1e-2)

    def test_1D_data_matches_scipy(self, sin_1D):
        x, sinx, x_prime, _ = sin_1D

        dummy = np.array([0.0, 1.0])
        orig_values = np.repeat(sinx[:, None, None], 2, axis=1)
        orig_values = np.repeat(orig_values, 2, axis=2)

        result = _fast_trilinear_interpolate_rectilinear(
            x_prime,
            np.full_like(x_prime, 0.0),
            np.full_like(x_prime, 0.0),
            x,
            dummy,
            dummy,
            orig_values,
        )
        scipy_interp_func = interpolate.interp1d(x, sinx, kind="linear")
        scipy_interp_result = scipy_interp_func(x_prime)
        np.testing.assert_array_less(np.abs(result - scipy_interp_result), 1e-14)

    def test_2D_data_close_to_expected(self, sincos_2D):
        x, y, values_2D, x_prime, y_prime, XP, YP, expected_2D = sincos_2D

        dummy = np.array([0.0, 1.0])
        orig_values = np.repeat(values_2D[:, :, None], 2, axis=2)

        result = _fast_trilinear_interpolate_rectilinear(
            XP.flatten(),
            YP.flatten(),
            np.zeros(XP.size),
            x,
            y,
            dummy,
            orig_values,
        )
        result_2D = result.reshape(expected_2D.shape)
        np.testing.assert_array_less(np.abs(expected_2D - result_2D), 1e-3)

    def test_3D_close_to_expected(self, sincos_exp_3D):
        (x, y, z, values_3D, x_prime, y_prime, z_prime, XP, YP, ZP, expected_3D) = (
            sincos_exp_3D
        )

        result = _fast_trilinear_interpolate_rectilinear(
            XP.flatten(),
            YP.flatten(),
            ZP.flatten(),
            x,
            y,
            z,
            values_3D,
        )
        result_3D = result.reshape(expected_3D.shape)
        diff = np.abs(expected_3D - result_3D)
        valid = ~np.isnan(diff)
        np.testing.assert_array_less(diff[valid], 1e-2)

    def test_interp_on_original_grid_returns_original_values(self):
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        z = np.linspace(0, 1, 10)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        values = X + Y + Z

        result = _fast_trilinear_interpolate_rectilinear(
            X.flatten(),
            Y.flatten(),
            Z.flatten(),
            x,
            y,
            z,
            values,
        )
        result = result.reshape(values.shape)
        diff = np.abs(values - result)
        valid = ~np.isnan(diff)
        np.testing.assert_array_less(diff[valid], 1e-14)

    def test_out_of_bounds_returns_nan(self):
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        z = np.linspace(0, 1, 10)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        values = X + Y + Z

        x_out = np.array([-0.5, 1.5])
        y_out = np.array([-0.5, 1.5])
        z_out = np.array([-0.5, 1.5])
        X_out, Y_out, Z_out = np.meshgrid(x_out, y_out, z_out, indexing="ij")

        result = _fast_trilinear_interpolate_rectilinear(
            X_out.flatten(),
            Y_out.flatten(),
            Z_out.flatten(),
            x,
            y,
            z,
            values,
        )
        assert np.all(np.isnan(result))


class TestFastTrilinearInterpolate:
    def test_3D_close_to_expected(self, sincos_exp_3D):
        (x, y, z, values_3D, x_prime, y_prime, z_prime, XP, YP, ZP, expected_3D) = (
            sincos_exp_3D
        )

        result = _fast_trilinear_interpolate(
            XP.flatten(),
            YP.flatten(),
            ZP.flatten(),
            x,
            y,
            z,
            values_3D,
        )
        result_3D = result.reshape(expected_3D.shape)
        diff = np.abs(expected_3D - result_3D)
        valid = ~np.isnan(diff)
        np.testing.assert_array_less(diff[valid], 1e-2)

    def test_matches_rectilinear_on_rectilinear_grid(self, sincos_exp_3D):
        (x, y, z, values_3D, x_prime, y_prime, z_prime, XP, YP, ZP, expected_3D) = (
            sincos_exp_3D
        )

        result_rect = _fast_trilinear_interpolate_rectilinear(
            XP.flatten(),
            YP.flatten(),
            ZP.flatten(),
            x,
            y,
            z,
            values_3D,
        )
        result = _fast_trilinear_interpolate(
            XP.flatten(),
            YP.flatten(),
            ZP.flatten(),
            x,
            y,
            z,
            values_3D,
        )
        # the rectilinear and general implementations handle boundary points
        # slightly differently - a wee handful of the points near the grid edge disagree
        # on nan status. Only compare where both are valid. But most points should be valid.
        valid = ~np.isnan(result_rect) & ~np.isnan(result)
        assert (
            valid.sum() > 0.75 * result.size
        )  # Note that this makes sense because z_prime goes from 0 to 2.5 but the original z only covers 0 to 2.0 so roughly 20% of the z range is out of bounds
        np.testing.assert_array_almost_equal(result_rect[valid], result[valid])


class TestFastBilinearInterpolateRectilinear:
    def test_1D_data_close_to_expected(self, sin_1D):
        x, sinx, x_prime, expected = sin_1D

        dummy = np.array([0.0, 1.0])
        orig_values = np.repeat(sinx[:, None], 2, axis=1)

        result = _fast_bilinear_interpolate_rectilinear(
            x_prime,
            np.full_like(x_prime, 0.0),
            x,
            dummy,
            orig_values,
        )
        np.testing.assert_array_less(np.abs(expected - result), 1e-2)

    def test_1D_data_matches_scipy(self, sin_1D):
        x, sinx, x_prime, _ = sin_1D

        dummy = np.array([0.0, 1.0])
        orig_values = np.repeat(sinx[:, None], 2, axis=1)

        result = _fast_bilinear_interpolate_rectilinear(
            x_prime,
            np.full_like(x_prime, 0.0),
            x,
            dummy,
            orig_values,
        )
        scipy_interp_func = interpolate.interp1d(x, sinx, kind="linear")
        scipy_interp_result = scipy_interp_func(x_prime)
        np.testing.assert_array_less(np.abs(result - scipy_interp_result), 1e-14)

    def test_2D_data_close_to_expected(self, sincos_2D):
        x, y, values_2D, x_prime, y_prime, XP, YP, expected_2D = sincos_2D

        result = _fast_bilinear_interpolate_rectilinear(
            XP.flatten(),
            YP.flatten(),
            x,
            y,
            values_2D,
        )
        result = result.reshape(expected_2D.shape)
        np.testing.assert_array_less(np.abs(expected_2D - result), 1e-3)

    def test_interp_on_original_grid_returns_original_values(self):
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        X, Y = np.meshgrid(x, y, indexing="ij")
        values = X + Y

        result = _fast_bilinear_interpolate_rectilinear(
            X.flatten(),
            Y.flatten(),
            x,
            y,
            values,
        )
        result = result.reshape(values.shape)
        diff = np.abs(values - result)
        valid = ~np.isnan(diff)
        np.testing.assert_array_less(diff[valid], 1e-14)

    def test_out_of_bounds_returns_nan(self):
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        X, Y = np.meshgrid(x, y, indexing="ij")
        values = X + Y

        x_out = np.array([-0.5, 1.5])
        y_out = np.array([-0.5, 1.5])
        X_out, Y_out = np.meshgrid(x_out, y_out, indexing="ij")

        result = _fast_bilinear_interpolate_rectilinear(
            X_out.flatten(),
            Y_out.flatten(),
            x,
            y,
            values,
        )
        assert np.all(np.isnan(result))


class TestFastBilinearInterpolate:
    def test_2D_close_to_expected(self, sincos_2D):
        x, y, values_2D, x_prime, y_prime, XP, YP, expected_2D = sincos_2D

        result = _fast_bilinear_interpolate(
            XP.flatten(),
            YP.flatten(),
            x,
            y,
            values_2D,
        )
        result = result.reshape(expected_2D.shape)
        diff = np.abs(expected_2D - result)
        valid = ~np.isnan(diff)
        np.testing.assert_array_less(diff[valid], 1e-3)

    def test_matches_rectilinear_on_rectilinear_grid(self, sincos_2D):
        x, y, values_2D, x_prime, y_prime, XP, YP, expected_2D = sincos_2D

        result_rect = _fast_bilinear_interpolate_rectilinear(
            XP.flatten(),
            YP.flatten(),
            x,
            y,
            values_2D,
        )
        result = _fast_bilinear_interpolate(
            XP.flatten(),
            YP.flatten(),
            x,
            y,
            values_2D,
        )
        # the rectilinear and general implementations handle boundary points
        # slightly differently - a wee handful of the points near the grid edge disagree
        # on nan status. Only compare where both are valid. But most points should be valid.
        valid = ~np.isnan(result_rect) & ~np.isnan(result)
        assert valid.sum() > 0.95 * result.size
        np.testing.assert_array_almost_equal(result_rect[valid], result[valid])
