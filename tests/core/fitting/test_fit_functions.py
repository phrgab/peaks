import numpy as np

from peaks.core.fitting.fit_functions import _fermi_function, _linear_dos_fermi


class TestFermiFunction:
    def test_one_half_at_EF(self):
        EF = 16.7
        x = np.array([EF])
        result = _fermi_function(x, EF=EF, T=20.0)
        np.testing.assert_array_almost_equal(result[0], 0.5)

    def test_low_T_gives_step(self):
        x = np.linspace(-0.15, 0.15, 500)
        result = _fermi_function(x, EF=0.0, T=0.001)
        below_EF = result[x < -0.001]
        above_EF = result[x > 0.001]
        np.testing.assert_array_almost_equal(below_EF, 1.0)
        np.testing.assert_array_almost_equal(above_EF, 0.0)


class TestLinearDOSFermi:
    def test_well_below_EF_closes_to_DOS(self):
        x = np.linspace(-0.15, 0.15, 500)
        result = _linear_dos_fermi(
            x,
            EF=0.0,
            T=10.0,
            dos_slope=2.0,
            dos_intercept=10.0,
            bg_slope=1.0,
            bg_intercept=2.0,
        )
        expected = 10.0 + 2.0 * x
        np.testing.assert_array_almost_equal(result[0], expected[0])

    def test_well_above_EF_closes_to_bg(self):
        x = np.linspace(-0.15, 0.15, 500)
        result = _linear_dos_fermi(
            x,
            EF=0.0,
            T=10.0,
            dos_slope=2.0,
            dos_intercept=10.0,
            bg_slope=1.0,
            bg_intercept=2.0,
        )
        expected = 2.0 + 1.0 * x
        np.testing.assert_array_almost_equal(result[-1], expected[-1])

    def test_flat_DOS_wo_bg_reduces_to_FD(self):
        x = np.linspace(-0.15, 0.15, 500)
        result = _linear_dos_fermi(
            x,
            EF=1.0,
            T=10.0,
            dos_slope=0.0,
            dos_intercept=10.0,
            bg_slope=0.0,
            bg_intercept=0.0,
        )
        expected = 10.0 * _fermi_function(x, EF=1.0, T=10.0)
        np.testing.assert_array_almost_equal(result, expected)
