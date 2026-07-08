import sys
import types
import warnings
from unittest.mock import MagicMock, patch

import pint
import pytest
import xarray as xr

from peaks.core.utils.misc import (
    _in_marimo,
    analysis_warning,
    dequantify_quantify_wrapper,
    format_colored_dict,
    silence_unit_warnings,
)


class TestFormatColouredDict:
    def test_flat_dict_has_all_keys_vals(self):
        d = {"loc": "St Andrews", "temperature": 25}
        result = format_colored_dict(d)
        assert "loc" in result
        assert "St Andrews" in result
        assert "temperature" in result
        assert "25" in result

    def test_colours_are_polulated(self):
        from termcolor import colored

        d = {"loc": "St Andrews"}
        result = format_colored_dict(d)
        assert colored("loc", "green") in result

    def test_nested_dict_w_indentations(self):
        d = {"analyser": {"slit": 0.2}}
        result = format_colored_dict(d)
        for line in result.splitlines():
            if "slit" in line:
                assert line.startswith("    ")

    def test_deeply_nested_dict(self):
        from termcolor import colored

        d = {"manipulator": {"polar": {"local_name": "sapolar"}}}
        result = format_colored_dict(d)
        assert "sapolar" in result
        assert colored("sapolar", "green") in result


class TestInMarimo:
    def test_false_when_marimo_not_imported(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "marimo", raising=False)
        assert _in_marimo() is False

    def test_true_when_running_in_mo_nb(self, monkeypatch):
        fake_mo = types.SimpleNamespace(running_in_notebook=lambda: True)
        monkeypatch.setitem(sys.modules, "marimo", fake_mo)
        assert _in_marimo() is True


class TestAnalysisWarning:
    @patch("peaks.core.utils.misc.display")
    def test_displays(self, mock_display):
        analysis_warning("test warning")
        mock_display.assert_called_once()

    @patch("peaks.core.utils.misc.display")
    def test_quite_true_supresses_display(self, mock_display):
        analysis_warning("test warning", quiet=True)
        mock_display.assert_not_called()

    @patch("peaks.core.utils.misc.display")
    def test_all_warn_types_are_accepted(self, mock_display):
        for warn_type in ("info", "warning", "success", "danger"):
            analysis_warning("a message", warn_type=warn_type)
        assert mock_display.call_count == 4

    @patch("peaks.core.utils.misc._in_marimo", return_value=True)
    def test_appends_mo_callout_not_jupyter_display(self, mock_in_marimo, monkeypatch):
        fake_mo = MagicMock()
        monkeypatch.setitem(sys.modules, "marimo", fake_mo)

        with patch("peaks.core.utils.misc.display") as mock_jupyter_display:
            analysis_warning("a message", warn_type="warning", title="Test")

        fake_mo.output.append.assert_called_once()
        assert fake_mo.callout.call_args.kwargs["kind"] == "warn"
        mock_jupyter_display.assert_not_called()


class TestPintDequantifyQuantifyWrapper:
    @pytest.fixture()
    def da_quantified(self):
        "make a fake quantified DataArray"
        da = xr.DataArray(data=[0.1, 0.2, 0.3], dims="x1")
        da = da.pint.quantify(units="mm")
        return da

    def test_dequantify_requantify_logic(self, da_quantified):
        @dequantify_quantify_wrapper
        def dummpy_process_data(da):
            "Inside this function da should be dequantified"
            assert "units" in da.attrs
            assert da.attrs["units"] == "mm"
            assert not isinstance(da.data, pint.Quantity)

            da.values = da.values * 2
            return da

        result = dummpy_process_data(da_quantified)
        assert hasattr(result.pint, "units")  # should be re-quantified already
        assert result.pint.units == result.pint.registry.mm
        assert list(result.data.magnitude) == [0.2, 0.4, 0.6]

    def test_args_kwargs_passed_through(self, da_quantified):
        @dequantify_quantify_wrapper
        def dummy_attach_args(da, a, b, c=None):
            da.attrs["a"] = a
            da.attrs["b"] = b
            da.attrs["c"] = c
            return da

        result = dummy_attach_args(da_quantified, 1, 2, c="wahey")
        assert result.attrs["a"] == 1
        assert result.attrs["b"] == 2
        assert result.attrs["c"] == "wahey"


class TestSilenceUnitWarnings:
    def test_silence_unit_stripping(self):
        from pint import UnitStrippedWarning

        with pytest.warns(UnitStrippedWarning, match="Unit got stripped"):
            warnings.warn(
                "Unit got stripped", UnitStrippedWarning, stacklevel=2
            )  # mock a real one

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")  # force every single warning to fire

            with silence_unit_warnings():
                warnings.warn(
                    "This should be suppressed", UnitStrippedWarning, stacklevel=2
                )

                warnings.warn("This should raise", UserWarning, stacklevel=2)

        unitstripped_warnings = [
            w for w in captured if issubclass(w.category, UnitStrippedWarning)
        ]
        assert len(unitstripped_warnings) == 0

        user_warnings = [w for w in captured if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 1
