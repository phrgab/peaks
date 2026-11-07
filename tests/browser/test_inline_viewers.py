"""Inline display panels in real JupyterLab and marimo sessions (headless Chromium).

Opt-in: ``pytest -m browser``. See ``conftest.py`` for how notebooks are driven.
"""

import shutil
import subprocess
import time

import pytest

pytestmark = pytest.mark.browser

FRONTENDS = ["jupyter", "marimo"]


def show(name, frontend):
    """A cell displaying a viewer of ``name``; marimo needs a unique variable."""
    return f"{name}.disp()" if frontend == "jupyter" else f"_v = {name}.disp()\n_v"


# ----- lifecycle: replacement, state, suspension, toolbar, deletion

LIFECYCLE_SETUP = """\
data = make_data(shape=(300, 400))

def state():
    import gc
    from peaks.core.GUI.disp_panels.disp_2d import _Disp2D
    from peaks.core.GUI.GUI_utils.qt_runtime import _existing_state

    gc.collect()
    saved = (_existing_state() or {}).get("cell_states", {})
    return dict(
        viewers=[
            dict(
                cell=v.cell_key and v.cell_key[0],
                status=v.status,
                state=v.widget.get_state() if v.widget is not None else None,
            )
            for v in pks.gui.viewers()
        ],
        live=sum(type(o) is _Disp2D for o in gc.get_objects()),
        saved=sorted({key[0] for key in saved}),
    )
"""
MUTATE = """\
_w = viewer().widget
_w.xhs[0].set_pos((-0.5, 5.0))
_w.show_DCs_checkboxes[2].setChecked(True)
_w.colorbar.setLevels(values=(0.2, 0.6))
del _w
"""
LIVE = 'img[decoding="sync"]'
STATIC = 'img[src^="data:image/png"]'


@pytest.mark.parametrize("frontend", FRONTENDS)
def test_lifecycle(notebook, frontend):
    """Re-running replaces a viewer and restores it; the limit suspends; buttons work."""
    nb = notebook(
        frontend,
        LIFECYCLE_SETUP,
        [show("data", frontend), show("data", frontend), MUTATE],
    )
    A, B, MUT = 1, 2, 3
    nb.run(A)  # marimo has already run A and B with the setup cell
    nb.run(MUT)
    s = nb.probe()
    assert [v for v in s["viewers"] if v["status"] == "active"][0]["state"][
        "visible"
    ] == [True, True, True]

    nb.run(A)  # replaced, settings restored
    s = nb.probe()
    [active] = [v for v in s["viewers"] if v["status"] == "active"]
    assert [v["cell"] for v in s["viewers"]].count(active["cell"]) == 1
    assert s["live"] == 1
    restored = active["state"]
    assert restored["visible"] == [True, True, True]
    assert restored["crosshairs"][0][0] == pytest.approx(-0.5, abs=0.01)
    assert restored["levels"][0] == pytest.approx(0.2)
    assert nb.visible_count(A, LIVE) == 1

    nb.run(B)  # identical code in another cell: a new viewer; A is suspended
    s = nb.probe()
    assert sorted(v["status"] for v in s["viewers"]) == ["active", "suspended"]
    assert s["live"] == 1
    assert nb.visible_count(A, STATIC) == 1 and nb.visible_count(A, LIVE) == 0

    nb.run(A)  # a suspended cell re-run is interactive again, with its settings
    s = nb.probe()
    active = [v for v in s["viewers"] if v["status"] == "active"]
    assert len(s["viewers"]) == 2 and len(active) == 1
    assert active[0]["state"]["visible"] == [True, True, True]

    nb.button(B, "Close")
    s = nb.probe()
    assert len(s["viewers"]) == 1
    assert nb.visible_count(B, "img") == 0 and nb.visible_count(B, "button") == 0

    nb.button(A, "Suspend")
    s = nb.probe()
    assert [v["status"] for v in s["viewers"]] == ["suspended"] and s["live"] == 0
    assert nb.visible_count(A, STATIC) == 1

    if frontend == "jupyter":  # deleting the cell releases its viewer and state
        nb.delete(A)
        s = nb.probe()
        assert s["viewers"] == [] and s["saved"] == []


# ----- keyboard and menus (2D)

PANEL_2D_SETUP = """\
data = make_data()

def state():
    v = viewer()
    w = v.widget
    menus = v._live_overlays()
    out = dict(
        plot=item_at(v, w.image_plot.getViewBox()),
        colorbar=item_at(v, w.colorbar),
        target=item_at(v, w.xhs[0].xh),
        xh=[list(w.xhs[0].get_pos()), list(w.xhs[1].get_pos())],
        step=w.step_sizes[1],
        cmap=w.colorbar.colorMap().name,
        menus=len(menus),
    )
    if menus:
        top = menus[-1]
        out["entries"] = {a.text(): action_at(v, top, a.text()) for a in top.actions() if a.text()}
        second = top.actionGeometry(top.actions()[1]).translated(v._local(top).topLeft())
        out["second"] = css(v, second.center())  # the first may be the current map
    return out
"""


@pytest.mark.parametrize("frontend", FRONTENDS)
def test_held_keys_repeat_and_cmd_is_control(notebook, frontend):
    nb = notebook(frontend, PANEL_2D_SETUP, [show("data", frontend)])
    nb.run(1)
    s = nb.probe()
    nb.click(1, s["plot"])  # the browser sends keys to a focused viewer only
    page = nb.page
    page.keyboard.down("ArrowRight")
    for _ in range(9):  # Playwright marks repeated down() calls as repeats
        page.keyboard.down("ArrowRight")
        time.sleep(0.05)
    page.keyboard.up("ArrowRight")
    time.sleep(0.5)
    page.keyboard.down("Meta")
    page.keyboard.press("ArrowRight")
    page.keyboard.up("Meta")
    time.sleep(0.5)
    t = nb.probe()
    moved = [(t["xh"][i][1] - s["xh"][i][1]) / s["step"] for i in (0, 1)]
    # A repeat is skipped while the kernel is still busy with the previous one
    assert 8 <= round(moved[0]) <= 10
    assert round(moved[1]) == 1


@pytest.mark.parametrize("frontend", FRONTENDS)
def test_nested_colour_map_menu(notebook, frontend):
    nb = notebook(frontend, PANEL_2D_SETUP, [show("data", frontend)])
    nb.run(1)
    s = nb.probe()
    nb.click(1, s["colorbar"], button="right")
    menu = nb.probe()
    assert menu["menus"] == 1
    nb.hover(1, menu["entries"]["matplotlib"])
    sub = nb.probe()
    assert sub["menus"] == 2
    nb.hover(1, sub["entries"]["Sequential"])
    third = nb.probe()
    assert third["menus"] == 3
    nb.click(1, third["second"])
    t = nb.probe()
    assert t["menus"] == 0 and t["cmap"] != s["cmap"]


# ----- wheel, double-click and dragging (3D)

PANEL_3D_SETUP = """\
data = make_data(("polar", "theta_par", "eV"), (30, 60, 80))

def state():
    v = viewer()
    w = v.widget
    box = w.cursor_positions_selection[0]
    area = w.cursor_stats.parentWidget().parentWidget()
    return dict(
        area=widget_at(v, area),
        scroll=area.verticalScrollBar().value(),
        arrow=widget_at(v, box, box.width() // 2 - 5, box.height() // 2 - 4),
        target=item_at(v, w.xhs[0].xh),
        pos=[b.value() for b in w.cursor_positions_selection],
    )
"""
LONG_TEXT = """\
_w = viewer().widget
_w.cursor_stats.setText("<br>".join(f"line {i}" for i in range(100)))
del _w
"""


@pytest.mark.parametrize("frontend", FRONTENDS)
def test_wheel_double_click_and_drag(notebook, frontend):
    nb = notebook(frontend, PANEL_3D_SETUP, [show("data", frontend), LONG_TEXT])
    nb.run(1)
    nb.run(2)
    s = nb.probe()

    # The wheel scrolls the metadata box: a label inside a scroll area
    nb.click(1, s["area"])
    for _ in range(3):
        nb.page.mouse.wheel(0, 200)
        time.sleep(0.2)
    time.sleep(0.5)
    t = nb.probe()
    assert t["scroll"] > 0 and t["pos"] == s["pos"]

    # A double-click on a spin box arrow is released: it does not keep stepping
    nb.click(1, t["arrow"], count=2)
    stepped = nb.probe()["pos"][0]
    for k in range(10):  # frames run the Qt timers a held arrow would repeat on
        nb.hover(1, [t["arrow"][0] - 30 - 2 * k, t["arrow"][1] + 40], wait=0.15)
    assert nb.probe()["pos"][0] == stepped

    # Dragging the crosshair moves the cursors
    nb.drag(1, t["target"], (40, 30))
    time.sleep(1)
    assert nb.probe()["pos"][1:] != t["pos"][1:]


SLOW_2D_SETUP = (
    PANEL_2D_SETUP
    + """
import time as _time
from peaks.core.GUI.GUI_utils import notebook_rfb as _rfb

_dispatch_input = _rfb.QWidgetRFB._dispatch_input

def _slow(self, typ, event):  # a viewer slower than the browser's move rate
    if typ == "pointer_move":
        _time.sleep(0.06)
    return _dispatch_input(self, typ, event)

_rfb.QWidgetRFB._dispatch_input = _slow
"""
)


def test_drag_does_not_lag_behind(notebook):
    """Moves are sent one at a time, so a slow viewer does not build a backlog."""
    nb = notebook("jupyter", SLOW_2D_SETUP, [show("data", "jupyter")])
    nb.run(1)
    s = nb.probe()
    # 100 moves in about 2 s; handling them all would take 6 s
    nb.drag(1, s["target"], (120, 80), steps=100, pause=0.02)
    released = time.time()
    t = nb.probe()
    assert time.time() - released < 3.0  # about 1 s for the probe itself
    assert t["xh"][0] != s["xh"][0]  # the last move before the release arrived


# ----- zoom and fit

ZOOM_SETUP = """\
data = make_data(("polar", "theta_par", "eV"), (30, 60, 80))

def state():
    v = viewer()
    return dict(zoom=v.zoom, chosen=v.zoom_chosen, size=[v.widget.width(), v.widget.height()])
"""


@pytest.mark.parametrize("frontend", FRONTENDS)
def test_zoom_fits_output_and_buttons(notebook, frontend):
    nb = notebook(frontend, ZOOM_SETUP, [show("data", frontend)], viewport=(1100, 1300))
    nb.run(1)
    time.sleep(1)
    s = nb.probe()
    out = nb.output(1).bounding_box()
    view = nb.output(1).locator(".renderview-wrapper").first.bounding_box()
    assert s["zoom"] < 1 and not s["chosen"]
    assert view["x"] + view["width"] <= out["x"] + out["width"] + 1

    nb.button(1, "+")
    t = nb.probe()
    assert t["zoom"] > s["zoom"] and t["chosen"] and t["size"] == s["size"]
    label = nb.output(1).locator("span").filter(has_text="%").first.inner_text()
    assert label == f"{round(t['zoom'] * 100)}%"

    nb.button(1, "Fit")
    u = nb.probe()
    assert u["zoom"] == s["zoom"] and not u["chosen"]


# ----- dialogs (4D)

PANEL_4D_SETUP = """\
data = make_data(("x1", "x2", "eV", "theta_par"), (8, 9, 40, 50))

def state():
    from PyQt6 import QtWidgets

    v = viewer()
    w = v.widget
    out = dict(
        clear=widget_at(v, w.clear_ROI_store_button),
        rois=w.roi_list_box.count(),
        dialogs=len(v._live_overlays()),
    )
    for d in v._live_overlays():
        yes = d.button(QtWidgets.QMessageBox.StandardButton.Yes)
        out["yes"] = css(v, v._local(d).topLeft() + yes.mapTo(d, yes.rect().center()))
    return out
"""
ADD_ROIS = """\
_w = viewer().widget
_w.centralWidget().setCurrentIndex(1)
for _n in "ab":
    _w.ROI_store[_n] = {"x1": [-5.0, 5.0, 0.0], "x2": [-5.0, -5.0, 5.0]}
    _w.roi_list_box.addItem(_n)
del _w, _n
"""


def test_confirmation_dialog(notebook):
    nb = notebook("jupyter", PANEL_4D_SETUP, [show("data", "jupyter"), ADD_ROIS])
    nb.run(1)
    nb.run(2)
    time.sleep(1)
    s = nb.probe()
    assert s["rois"] == 2

    nb.click(1, s["clear"])
    t = nb.probe()
    assert t["dialogs"] == 1
    nb.click(1, s["clear"])  # focus the viewer again; the open dialog blocks the click
    nb.page.keyboard.press("Enter")  # the default answer is No
    time.sleep(1)
    t = nb.probe()
    assert t["dialogs"] == 0 and t["rois"] == 2

    nb.click(1, s["clear"])
    t = nb.probe()
    nb.click(1, t["yes"])
    t = nb.probe()
    assert t["dialogs"] == 0 and t["rois"] == 0


# ----- native windows and inline viewers in one kernel (needs an X display)

COEXIST_SETUP = """\
data = make_data(shape=(300, 400))
data2 = make_data(shape=(300, 400), seed=1)
pks.opts.gui.max_viewers = 2

def state():
    inline = [v for v in pks.gui.viewers() if v.backend == "notebook"]
    return dict(
        viewers=[v.status for v in pks.gui.viewers()],
        ranges=[v.widget.get_state()["view_range"] for v in inline],
        plot=item_at(inline[0], inline[0].widget.image_plot.getViewBox()) if inline else None,
    )
"""


def mapped_panels():
    """Viewable X windows of peaks display panels."""
    out = subprocess.run(
        ["xwininfo", "-root", "-tree"], capture_output=True, text=True
    ).stdout
    ids = [ln.split()[0] for ln in out.splitlines() if "Display Panel" in ln]
    return sum(
        "IsViewable"
        in subprocess.run(["xwininfo", "-id", i], capture_output=True, text=True).stdout
        for i in ids
    )


@pytest.mark.skipif(
    shutil.which("xwininfo") is None or not __import__("os").environ.get("DISPLAY"),
    reason="needs an X display (e.g. xvfb-run) and xwininfo",
)
def test_native_window_and_inline_viewer_together(notebook):
    nb = notebook(
        "jupyter",
        COEXIST_SETUP,
        ["data.disp()", "data2.disp(backend='qt')", "pks.gui.close_all()"],
        qt_platform="xcb",
    )
    nb.run(1)
    s = nb.probe()
    nb.run(2)
    t = nb.probe()
    assert t["viewers"] == ["active", "active"]
    assert mapped_panels() == 1  # the inline viewer never maps a window
    nb.drag(1, s["plot"], (60, 40))
    time.sleep(1)
    assert nb.probe()["ranges"] != t["ranges"]  # still responds to the browser
    nb.run(3)
    assert nb.probe()["viewers"] == [] and mapped_panels() == 0
