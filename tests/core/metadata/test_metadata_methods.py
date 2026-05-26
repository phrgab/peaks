import pytest

from peaks.core.metadata.metadata_methods import (
    DARK_BG_PALETTE,
    LIGHT_BG_PALETTE,
    display_metadata,
)
from peaks.core.utils.sample_data import ExampleData


@pytest.fixture(scope="session")
def disp():
    return ExampleData.dispersion2a()


@pytest.fixture
def test_dict():
    return {"a": {"b": {"c": {"d": {"e": "value"}}}}}


# for testing ANSI output
# copied from the internal _hex_to_rgb function in metadata_methods
def _hex_to_rgb(hex_col):
    hex_col = hex_col.lstrip("#")
    return tuple(int(hex_col[i : i + 2], 16) for i in (0, 2, 4))


class TestDisplayMetadata:
    def test_colours_applied_to_nesting_levels_html(self, test_dict):
        result = display_metadata(test_dict, mode="HTML")
        expected_pairs = [
            (0, "a"),
            (1, "b"),
            (2, "c"),
            (3, "d"),
        ]
        for level, key in expected_pairs:
            expected = f"<span style='color:{LIGHT_BG_PALETTE[level]}'>{key}:</span>"
            assert expected in result

    def test_colours_applied_to_nesting_levels_html_dark(self, test_dict):
        result = display_metadata(test_dict, palette=DARK_BG_PALETTE, mode="HTML")
        expected_pairs = [
            (0, "a"),
            (1, "b"),
            (2, "c"),
            (3, "d"),
        ]
        for level, key in expected_pairs:
            expected = f"<span style='color:{DARK_BG_PALETTE[level]}'>{key}:</span>"
            assert expected in result

    def test_colour_cycling_wraps(self, test_dict):
        result = display_metadata(test_dict, mode="HTML")
        assert result.count(f"color:{LIGHT_BG_PALETTE[0]}") == 2
        for i in (1, 2, 3):
            assert result.count(f"color:{LIGHT_BG_PALETTE[i]}") == 1
        assert f"<span style='color:{LIGHT_BG_PALETTE[0]}'>a:</span>" in result
        assert f"<span style='color:{LIGHT_BG_PALETTE[0]}'>e:</span>" in result

    def test_invalid_mode_raises(self, disp):
        with pytest.raises(ValueError, match="mode must be"):
            display_metadata(disp, mode="och")

    def test_colours_applied_to_nesting_levels_ansi(self, test_dict):
        result = display_metadata(test_dict, mode="ANSI")
        expected_pairs = [
            (0, "a"),
            (1, "b"),
            (2, "c"),
            (3, "d"),
        ]
        for level, key in expected_pairs:
            r, g, b = _hex_to_rgb(LIGHT_BG_PALETTE[level])
            expected = f"\x1b[38;2;{r};{g};{b}m{key}\x1b[0m"
            assert expected in result

    def test_colours_applied_to_nesting_levels_ansi_dark(self, test_dict):
        result = display_metadata(test_dict, palette=DARK_BG_PALETTE, mode="ANSI")
        expected_pairs = [
            (0, "a"),
            (1, "b"),
            (2, "c"),
            (3, "d"),
        ]
        for level, key in expected_pairs:
            r, g, b = _hex_to_rgb(DARK_BG_PALETTE[level])
            expected = f"\x1b[38;2;{r};{g};{b}m{key}\x1b[0m"
            assert expected in result

    def test_colour_cycling_wraps_ansi(self, test_dict):
        result = display_metadata(test_dict, palette=DARK_BG_PALETTE, mode="ANSI")
        r, g, b = _hex_to_rgb(DARK_BG_PALETTE[0])
        expected_0 = f"\x1b[38;2;{r};{g};{b}m"
        assert result.count(expected_0) == 2  # used at L0 and L4
        for i in (1, 2, 3):
            r, g, b = _hex_to_rgb(DARK_BG_PALETTE[i])
            expected_i = f"\x1b[38;2;{r};{g};{b}m"
            assert result.count(expected_i) == 1  # used once each


class TestMetadata:
    def test_repr_returns_ansi(self, disp):
        result = repr(disp.metadata)
        assert "\x1b[38;2;" in result  # ANSI escape code

    def test_repr_html_returns_html(self, disp):
        result = disp.metadata._repr_html_()
        assert "<span style='color:" in result  # HTML span

    def test_repr_displays_all_metadata_keys(self, disp):
        result = repr(disp.metadata)
        for key in ("calibration", "EF_correction", "lens_mode", "width"):
            assert key in result


class TestMetadataItem:
    def test_repr_returns_ansi(self, disp):
        result = repr(disp.metadata.calibration)
        assert "\x1b[38;2;" in result  # ANSI escape code

    def test_repr_html_returns_html(self, disp):
        result = disp.metadata.calibration._repr_html_()
        assert "<span style='color:" in result  # HTML span

    def test_repr_only_displays_one_metadata_group(self, disp):
        result = repr(disp.metadata.calibration)
        assert "EF_correction" in result
        assert "width" not in result
        assert "lens_mode" not in result

    def test_repr_html_only_displays_one_metadata_group(self, disp):
        result = disp.metadata.calibration._repr_html_()
        assert "EF_correction" in result
        assert "width" not in result
        assert "lens_mode" not in result
