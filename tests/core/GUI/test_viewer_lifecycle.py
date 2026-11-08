"""Headless tests for the shared viewer registry and the inline notebook backend.

Browser events are injected through ``handle_event`` as jupyter_rfb delivers them.
Native windows are created on Qt's offscreen platform (never mapped).
"""

import gc
import os
import sys
import time
import weakref

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("jupyter_rfb")
if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtCore, QtWidgets, sip  # noqa: E402

import peaks as pks  # noqa: E402
from peaks.core.GUI.GUI_utils import notebook_cells, qt_runtime  # noqa: E402
from peaks.core.GUI.GUI_utils.qt_runtime import TooManyDisplayPanelsError  # noqa: E402


def make_data(n_eV=120, n_theta=150, seed=0, eV0=-1.0):
    rng = np.random.default_rng(seed)
    da = xr.DataArray(
        rng.random((n_eV, n_theta)).astype(np.float32),
        dims=("eV", "theta_par"),
        coords={
            "eV": np.linspace(eV0, eV0 + 1.1, n_eV),
            "theta_par": np.linspace(-15, 15, n_theta),
        },
        name=f"test_{seed}",
    )
    da.eV.attrs["units"] = "eV"
    da.theta_par.attrs["units"] = "deg"
    return da


def ev(event_type, x=0, y=0, button=0, buttons=(), modifiers=(), **extra):
    return dict(
        event_type=event_type,
        x=x,
        y=y,
        button=button,
        buttons=buttons,
        modifiers=modifiers,
        **extra,
    )


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    # Allow invisible "native" windows on the offscreen platform
    monkeypatch.setattr(
        QtWidgets.QApplication, "platformName", staticmethod(lambda: "test")
    )
    monkeypatch.setattr(notebook_cells, "current_cell", lambda state: None)
    pks.opts.gui.reset()
    yield
    pks.gui.close_all()
    pks.opts.gui.reset()
    assert pks.gui.viewers() == []


def inline(data=None, **kwargs):
    return (make_data() if data is None else data).disp(backend="notebook", **kwargs)


def native(data=None):
    state = qt_runtime._get_runtime()[1]
    state["defer_exec"] += 1  # do not block in app.exec()
    try:
        (make_data() if data is None else data).disp(backend="qt")
    finally:
        state["defer_exec"] -= 1
    return [v for v in pks.gui.viewers() if v.backend == "qt"][-1]


def live():
    return [v for v in pks.gui.viewers() if v.status == "active"]


def live_disp2d_count():
    gc.collect()
    from peaks.core.GUI.disp_panels.disp_2d import _Disp2D

    return sum(1 for o in gc.get_objects() if type(o) is _Disp2D)


# ----- options


def test_backend_option():
    assert pks.opts.gui.backend == "qt"
    pks.opts.gui.backend = "notebook"
    assert pks.opts.gui.dict() == {"max_viewers": 1, "backend": "notebook"}
    with pytest.raises(ValueError):
        pks.opts.gui.backend = "rfb"
    pks.opts.gui.set(backend="qt")
    with pks.opts as opts:
        opts.gui.backend = "notebook"
    assert pks.opts.gui.backend == "qt"


def test_global_backend_used_and_overridden():
    pks.opts.gui.backend = "notebook"
    view = make_data().disp()
    assert view.backend == "notebook"
    pks.opts.gui.max_viewers = 2
    assert native().backend == "qt"


# ----- shared limit, max_viewers = 1


def test_native_then_native_raises():
    native()
    with pytest.raises(TooManyDisplayPanelsError):
        native()


def test_inline_when_native_active_raises():
    native()
    with pytest.raises(TooManyDisplayPanelsError):
        inline()


def test_native_suspends_inline():
    view = inline()
    nat = native()
    assert view.status == "suspended" and nat.status == "active"
    assert view._peaks_mode == "suspended"
    assert view._peaks_static.startswith("data:image/png;base64,")


def test_inline_suspends_previous_inline():
    first = inline()
    second = inline()
    assert first.status == "suspended" and second.status == "active"
    assert live() == [second]
    assert live_disp2d_count() == 1


# ----- shared limit, max_viewers = 2


def test_limit_two_combinations():
    pks.opts.gui.max_viewers = 2
    a, b = native(), native()
    assert len(live()) == 2
    with pytest.raises(TooManyDisplayPanelsError):
        native()
    a.close()
    view = inline()
    assert {v.backend for v in live()} == {"qt", "notebook"}
    nat = native()  # suspends the inline viewer
    assert view.status == "suspended" and nat.status == "active"
    with pytest.raises(TooManyDisplayPanelsError):
        native()
    assert b.status == "active"


def test_oldest_inline_is_suspended():
    pks.opts.gui.max_viewers = 2
    first, second = inline(), inline()
    third = inline()
    assert [first.status, second.status, third.status] == [
        "suspended",
        "active",
        "active",
    ]


def test_unlimited():
    pks.opts.gui.max_viewers = None
    views = [inline() for _ in range(4)]
    assert all(v.status == "active" for v in views)


# ----- manual lifecycle


def test_suspend_releases_qt_and_buffers():
    view = inline()
    view.handle_event(ev("resize", width=650, height=770, pixel_ratio=1.0))
    assert view.get_frame() is not None
    disp = weakref.ref(view.widget)
    qt_obj = view.widget.image_plot
    view.suspend()
    view.suspend()  # idempotent
    gc.collect()
    assert disp() is None and sip.isdeleted(qt_obj)
    assert view.widget is None and view._rfb_last_frame is None
    assert view.get_frame() is None
    view.handle_event(ev("pointer_down", 100, 100, 1, (1,)))  # late event: ignored
    assert len(view._peaks_static) < 400_000  # compressed, bounded image
    assert view in pks.gui.viewers() and live() == []
    view.close()
    view.close()
    assert view.status == "closed" and view._peaks_mode == "closed"
    assert pks.gui.viewers() == []


def test_toolbar_messages():
    view = inline()
    view._rfb_handle_msg(view, {"type": "peaks_suspend"}, [])
    assert view.status == "suspended"
    view._rfb_handle_msg(view, {"type": "peaks_close"}, [])
    assert view.status == "closed"


def test_close_during_drag():
    view = inline()
    view.handle_event(ev("resize", width=650, height=770, pixel_ratio=1.0))
    view.handle_event(ev("pointer_down", 200, 200, 1, (1,)))
    time.sleep(0.02)
    view.handle_event(ev("pointer_move", 220, 230, 0, (1,)))
    view.suspend()
    view.handle_event(ev("pointer_move", 240, 260, 0, (1,)))
    view.handle_event(ev("pointer_up", 240, 260, 1, ()))
    view.close()


def test_messages_from_another_thread_run_on_qt_thread():
    """JupyterLab sends widget messages over a kernel subshell (another thread)."""
    import asyncio
    import threading

    async def main():
        view = inline()
        handled = []
        view.handle_event = lambda e: handled.append(threading.current_thread())
        data = {"method": "custom", "content": {"type": "show"}}
        msg = {"content": {"data": data}, "buffers": []}
        thread = threading.Thread(target=view._handle_msg, args=(msg,))
        thread.start()
        thread.join()
        assert handled == []  # not on the subshell thread
        await asyncio.sleep(0.01)
        assert handled == [threading.main_thread()]
        view.close()

    asyncio.run(main())


def test_quick_pointer_moves_not_dropped():
    """The last of several quick moves reaches pyqtgraph (hover state, drag ends)."""
    view = inline()
    view.handle_event(ev("resize", width=650, height=770, pixel_ratio=1.0))
    seen = []
    view.widget.image_plot.scene().sigMouseMoved.connect(seen.append)
    for k in range(6):  # well within pyqtgraph's 10 ms mouseRateLimit
        view.handle_event(ev("pointer_move", 200 + 5 * k, 200))
    gv = view.widget.graphics_layout
    final = gv.mapFromScene(seen[-1])
    assert gv.mapTo(view.widget, final).x() == pytest.approx(225, abs=1)
    assert len(seen) == 6
    view.close()


@pytest.mark.parametrize("final_release", ["click", "double_click"])
def test_hover_after_click_reaches_widget_under_cursor(final_release):
    """After a click, hover moves go to the widget under the cursor, not the clicked one."""
    view = inline()
    view.handle_event(ev("resize", width=650, height=770, pixel_ratio=1.0))
    w = view.widget
    box = w.show_DCs_checkboxes[2]
    bx, by = (lambda p: (p.x(), p.y()))(box.mapTo(w, box.rect().center()))
    view.handle_event(ev("pointer_down", bx, by, 1, (1,)))
    view.handle_event(ev("pointer_up", bx, by, 1, ()))
    if final_release == "double_click":
        view.handle_event(ev("pointer_down", bx, by, 1, (1,)))
        view.handle_event(ev("pointer_up", bx, by, 1, ()))
        view.handle_event(ev("double_click", bx, by, 1, ()))
    view.handle_event(ev("resize", width=600, height=700, pixel_ratio=2.0))  # relayout
    seen = []
    w.image_plot.scene().sigMouseMoved.connect(seen.append)
    gv = w.graphics_layout
    centre = gv.mapTo(w, gv.mapFromScene(w.image_plot.sceneBoundingRect().center()))
    sx, sy = view._scale
    view.handle_event(ev("pointer_move", centre.x() / sx, centre.y() / sy))
    assert seen, "hover move went to the clicked checkbox, not the plot"
    assert w.testAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen)
    view.close()


def test_native_suspend_rejected_and_close():
    nat = native()
    with pytest.raises(TypeError):
        nat.suspend()
    widget = nat.widget
    nat.close()
    nat.close()
    assert sip.isdeleted(widget) and pks.gui.viewers() == []


def test_user_closes_native_window():
    nat = native()
    nat.widget.close()  # as if the user closed the window
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    assert pks.gui.viewers() == [] and nat.status == "closed"


def test_suspend_all_and_close_all():
    pks.opts.gui.max_viewers = 3
    nat = native()
    a, b = inline(), inline()
    pks.gui.suspend_all()
    pks.gui.suspend_all()
    assert (a.status, b.status, nat.status) == ("suspended", "suspended", "active")
    c = inline()
    pks.gui.close_all()
    pks.gui.close_all()
    assert all(v.status == "closed" for v in (nat, a, b, c))
    assert qt_runtime._existing_state()["cell_states"] == {}


def test_modal_dialog_does_not_block():
    view = inline()
    view.widget.shortcuts_action.trigger()  # Help: QMessageBox.information(...)
    assert view.status == "active"


# ----- same-cell replacement and state


class Cells:
    """Simulate notebook cell executions for ``notebook_cells.current_cell``."""

    def __init__(self, monkeypatch):
        self.current = None
        self.runs = 0
        monkeypatch.setattr(notebook_cells, "current_cell", lambda state: self.current)

    def run(self, cell_id):
        self.runs += 1
        self.current = (cell_id, self.runs)


def drag_state(view):
    w = view.widget
    w.xhs[0].set_pos((-0.5, 5.0))
    w.show_DCs_checkboxes[2].setChecked(True)
    w.DC_width_selectors[0].setValue(0.05)
    w.colorbar.setLevels(values=(0.2, 0.6))
    w.image_plot.getViewBox().setRange(xRange=(-5, 5), yRange=(-0.8, -0.2), padding=0)


@pytest.mark.parametrize("max_viewers", [1, 3])
def test_same_cell_replacement_restores_state(monkeypatch, max_viewers):
    pks.opts.gui.max_viewers = max_viewers
    cells = Cells(monkeypatch)
    cells.run("A")
    old = inline()
    drag_state(old)
    expected = old.widget.get_state()

    cells.run("A")
    new_data = make_data(seed=1)
    new = inline(new_data)
    assert old.status == "closed" and live() == [new]
    assert new.widget.current_data.name == "test_1"
    state = new.widget.get_state()
    assert np.allclose(state["crosshairs"][0], expected["crosshairs"][0], atol=0.02)
    assert state["visible"] == [True, True, True]
    assert state["widths"][0] == pytest.approx(0.05, abs=1e-3)
    assert state["levels"] == pytest.approx([0.2, 0.6], abs=1e-3)
    assert np.allclose(state["view_range"], expected["view_range"], atol=0.05)


def test_identical_code_in_other_cells_not_replaced(monkeypatch):
    pks.opts.gui.max_viewers = 3
    cells = Cells(monkeypatch)
    cells.run("A")
    a = inline()
    cells.run("B")
    b = inline()
    assert a.status == b.status == "active"
    cells.run("A")
    a2, a3 = inline(), inline()  # two viewers in one cell
    assert a.status == "closed" and b.status == "active"
    assert a2.cell_key == ("A", 0) and a3.cell_key == ("A", 1)


def test_suspended_cell_rerun_restores(monkeypatch):
    cells = Cells(monkeypatch)
    cells.run("A")
    old = inline()
    drag_state(old)
    cells.run("B")
    inline()  # suspends A (limit 1)
    assert old.status == "suspended"
    cells.run("A")
    new = inline()
    assert old.status == "closed" and new.status == "active"
    assert new.widget.get_state()["visible"] == [True, True, True]


def test_explicit_close_discards_state_and_deleted_cell(monkeypatch):
    cells = Cells(monkeypatch)
    cells.run("A")
    view = inline()
    drag_state(view)
    view.suspend()
    assert ("A", 0) in qt_runtime._existing_state()["cell_states"]
    view.close()
    assert qt_runtime._existing_state()["cell_states"] == {}
    cells.run("A")
    inline().suspend()
    qt_runtime.begin_cell("A", deleted=True)
    assert pks.gui.viewers() == []
    assert qt_runtime._existing_state()["cell_states"] == {}


def test_state_with_incompatible_data(monkeypatch):
    cells = Cells(monkeypatch)
    cells.run("A")
    drag_state(inline())
    cells.run("A")
    # Different energy range: crosshair at eV=-0.5 is out of range and is skipped
    new = inline(make_data(eV0=2.0))
    assert new.status == "active"
    pos = new.widget.xhs[0].get_pos()
    assert 2.0 <= pos[0] <= 3.1
    cells.run("A")
    other = inline(make_data().rename(theta_par="k_par"))  # other dims
    assert other.widget.get_state()["visible"] == [True, True, True]


def test_repeated_cycles_do_not_accumulate(monkeypatch):
    import psutil

    data = make_data(400, 500)
    cells = Cells(monkeypatch)
    rss = []
    for i in range(50):
        cells.run("A")
        inline(data)
        if i == 9:
            rss.append(psutil.Process().memory_info().rss)
    rss.append(psutil.Process().memory_info().rss)
    assert live_disp2d_count() == 1 and len(pks.gui.viewers()) == 1
    for _ in range(50):
        inline(data).suspend()
    assert live_disp2d_count() == 0
    pks.gui.close_all()
    for _ in range(50):
        inline(data).close()
    assert live_disp2d_count() == 0 and pks.gui.viewers() == []
    gc.collect()
    rss.append(psutil.Process().memory_info().rss)
    print(
        "RSS MB after 10/50 replacements, 50 suspends+50 closes:", [r >> 20 for r in rss]
    )
    assert rss[-1] - rss[0] < 150 * 2**20


# ----- 3D and 4D viewers


def make_nd(dims, shape, seed=0):
    rng = np.random.default_rng(seed)
    coords = {
        "eV": np.linspace(-1.0, 0.1, shape[dims.index("eV")]) if "eV" in dims else None,
    }
    coords = {d: np.linspace(-10, 10, n) for d, n in zip(dims, shape, strict=True)}
    if "eV" in dims:
        coords["eV"] = np.linspace(-1.0, 0.1, shape[dims.index("eV")])
    da = xr.DataArray(rng.random(shape).astype(np.float32), dims=dims, coords=coords)
    da.name = "nd"
    return da


def drag(view, start, delta, steps=5):
    x0, y0 = start
    view.handle_event(ev("pointer_move", x0, y0))
    view.handle_event(ev("pointer_down", x0, y0, 1, (1,)))
    for i in range(1, steps + 1):
        time.sleep(0.02)
        view.handle_event(
            ev(
                "pointer_move",
                x0 + delta[0] * i / steps,
                y0 + delta[1] * i / steps,
                0,
                (1,),
            )
        )
    view.handle_event(ev("pointer_up", x0 + delta[0], y0 + delta[1], 1, ()))


@pytest.mark.parametrize(
    "dims, shape",
    [
        (("polar", "theta_par", "eV"), (30, 60, 80)),
        (("x1", "x2", "eV", "theta_par"), (8, 9, 40, 50)),
    ],
)
def test_3d_4d_inline_lifecycle_and_state(monkeypatch, dims, shape):
    from peaks.core.GUI.disp_panels.disp_2d import _Disp2D

    cells = Cells(monkeypatch)
    cells.run("A")
    data = make_nd(dims, shape)
    view = data.disp(backend="notebook")
    cls = type(view.widget)
    view.handle_event(ev("resize", width=900, height=700, pixel_ratio=1.0))
    frame = view.get_frame()
    assert frame is not None and frame[..., :3].std() > 5
    drag(view, (300, 300), (40, 30))  # whatever is under the pointer: no crash
    view.handle_event(ev("key_down", key="ArrowRight"))
    view.handle_event(ev("key_up", key="ArrowRight"))
    w = view.widget
    if cls.__name__ == "_Disp3D":
        w.cursor_positions_selection[2].setValue(-0.3)
        w.cursor_widths_selection[0].setValue(1.0)
        w.mirror_checkbox.setChecked(True)
    else:
        w.centralWidget().setCurrentIndex(1)
        w.primary_dims_xh.set_pos((2.0, -3.0))
        w.roi_dim01.setPoints([[-5, -5], [5, -5], [0, 5]])
        w.ROI_store["test"] = {"x1": [-5.0, 5.0, 0.0], "x2": [-5.0, -5.0, 5.0]}
    w.colorbar.setLevels(values=(0.3, 0.7)) if hasattr(w, "colorbar") else None
    expected = w.get_state()
    del w

    cells.run("A")
    new = data.disp(backend="notebook")
    state = new.widget.get_state()
    if cls.__name__ == "_Disp3D":
        assert state["positions"][2] == pytest.approx(-0.3, abs=0.02)
        assert state["widths"][0] == pytest.approx(1.0, abs=1e-3)
        assert state["mirror"] and state["levels"] == pytest.approx([0.3, 0.7])
    else:
        assert state["tab"] == 1
        assert np.allclose(state["primary_xh"], expected["primary_xh"], atol=0.5)
        assert np.allclose(state["roi_dim01"], expected["roi_dim01"])
        assert "test" in state["roi_store"]

    new.suspend()
    gc.collect()
    assert not [o for o in gc.get_objects() if type(o) in (cls, _Disp2D)]
    assert live() == []


def test_4d_slider_drag_moves_crosshair_to_its_value():
    view = make_nd(("x1", "x2", "eV", "theta_par"), (8, 9, 40, 50)).disp(
        backend="notebook"
    )
    w = view.widget
    slider = w.dim01_sliders[1]
    slider.setSliderDown(True)  # as while dragging its handle
    for index in (2, 5, 7):  # a quick drag: only some positions are seen
        slider.setSliderPosition(index)
    slider.setSliderDown(False)
    assert slider.value() == 7
    assert w.primary_dims_xh.get_pos()[1] == pytest.approx(w.coarsened_coords[1][7])
    # The 2D panel shows the cut there (no binning)
    shown = w.data_explorer_disp2d.data_arrays[0].data
    np.testing.assert_allclose(shown, w.data.data[w.dim01_sliders[0].value(), 7])
    slider.triggerAction(QtWidgets.QAbstractSlider.SliderAction.SliderPageStepSub)
    assert w.primary_dims_xh.get_pos()[1] == pytest.approx(
        w.coarsened_coords[1][slider.value()]
    )  # clicking the groove moves the crosshair too


def test_copy_goes_to_browser():
    view = inline()
    sent = []
    view.send = lambda msg, *a: sent.append(msg)
    view.widget.copy_button.click()
    assert sent and sent[0]["type"] == "peaks_copy"
    assert sent[0]["text"].startswith(".metadata.set_normal_emission(")


def test_marimo_internals_changed_falls_back_to_no_replacement(monkeypatch):
    """If marimo's private lifecycle API breaks, viewers still open, unreplaced."""
    import types

    from peaks.core.GUI.GUI_utils import notebook_cells

    class Registry:
        def add(self, item):
            raise AttributeError("marimo internals changed")

    context = types.SimpleNamespace(cell_id="c1", cell_lifecycle_registry=Registry())
    ctx_mod = types.ModuleType("marimo._runtime.context")
    ctx_mod.get_context = lambda: context
    for name in ("marimo", "marimo._runtime"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "marimo._runtime.context", ctx_mod)

    state = {"hooks": set()}
    assert notebook_cells._marimo_cell(state) is None
    assert "c1" not in state["marimo_cells"]


def test_held_keys_repeat_and_cmd_acts_as_control():
    """Held-key repeats move the crosshair and are acknowledged one at a time.

    Cmd (browser "Meta") selects crosshair 1, as Control does and as Cmd does natively
    on a Mac.
    """
    view = inline()
    sent = []
    view.send = lambda msg, *a: sent.append(msg)
    view.handle_event(ev("resize", width=650, height=770, pixel_ratio=1.0))
    view.handle_event(ev("pointer_down", 250, 250, 1, (1,)))  # focus the plot
    view.handle_event(ev("pointer_up", 250, 250, 1, ()))
    w = view.widget
    csr0, csr1 = w.xhs[0].get_pos(), w.xhs[1].get_pos()
    step = w.step_sizes[1]

    cmd = ("Meta",)
    view.handle_event(ev("key_down", key="Meta", modifiers=cmd))
    view.handle_event(ev("key_down", key="ArrowRight", modifiers=cmd))
    for _ in range(3):
        view.handle_event(ev("key_down", key="ArrowRight", modifiers=cmd, repeat=True))
    view.handle_event(ev("key_up", key="ArrowRight", modifiers=cmd))
    view.handle_event(ev("key_up", key="Meta"))

    assert w.xhs[0].get_pos() == csr0
    assert w.xhs[1].get_pos()[1] == pytest.approx(csr1[1] + 4 * step)
    assert [m["type"] for m in sent] == ["peaks_key_ack"] * 3

    view.handle_event(ev("key_down", key="ArrowRight"))  # Cmd released: crosshair 0
    assert w.xhs[0].get_pos()[1] == pytest.approx(csr0[1] + step)


def test_repeat_after_suspend_is_still_acknowledged():
    view = inline()
    sent = []
    view.send = lambda msg, *a: sent.append(msg)
    view.suspend()
    view.handle_event(ev("key_down", key="ArrowRight", repeat=True))
    assert sent[-1] == {"type": "peaks_key_ack"}


def test_help_text_sent_to_browser():
    assert "Key Binding" in inline()._peaks_help


# ----- popups and dialogs drawn inline


def click(view, x, y, button=1):
    view.handle_event(ev("pointer_move", x, y))
    view.handle_event(ev("pointer_down", x, y, button, (button,)))
    view.handle_event(ev("pointer_up", x, y, button, ()))


def widget_centre(view, widget):
    """Centre of a child QWidget, in the viewer's coordinates."""
    return widget.mapTo(view.widget, widget.rect().center())


def scene_centre(view, item):
    gv = item.scene().views()[0]
    return gv.mapTo(view.widget, gv.mapFromScene(item.sceneBoundingRect().center()))


def action_centre(view, menu, action):
    rect = menu.actionGeometry(action).translated(view._local(menu).topLeft())
    return rect.center()


def wait_qt_timers(view, seconds=0.4):
    time.sleep(seconds)
    view.get_frame()  # runs Qt timers, as each browser frame does


def test_colour_map_menu_inline():
    view = inline()
    view.handle_event(ev("resize", width=650, height=770, pixel_ratio=1.0))
    before = view.get_frame().copy()
    p = scene_centre(view, view.widget.colorbar)
    click(view, p.x(), p.y(), button=2)

    [menu] = view._live_overlays()
    assert "local" in [a.text() for a in menu.actions()]
    rect = view._local(menu)
    assert QtCore.QRect(0, 0, 650, 770).contains(rect)  # kept inside the viewer
    frame = view.get_frame()
    area = np.s_[rect.top() : rect.bottom(), rect.left() : rect.right()]
    assert np.abs(frame[area].astype(int) - before[area]).mean() > 5  # drawn over

    # Hover a submenu, let its timer open it, pick the first colour map
    local = next(a for a in menu.actions() if a.text() == "local")
    q = action_centre(view, menu, local)
    view.handle_event(ev("pointer_move", q.x(), q.y()))
    wait_qt_timers(view)
    sub = view._live_overlays()[-1]
    assert sub is not menu
    first = next(a for a in sub.actions() if not a.isSeparator())
    q = action_centre(view, sub, first)
    click(view, q.x(), q.y())
    assert view._live_overlays() == []
    assert view.widget.colorbar.colorMap().name == "cividis"


def test_plot_menu_inline_without_export():
    view = inline()
    view.handle_event(ev("resize", width=650, height=770, pixel_ratio=1.0))
    p = scene_centre(view, view.widget.image_plot.getViewBox())
    click(view, p.x(), p.y(), button=2)
    [menu] = view._live_overlays()
    texts = [a.text() for a in menu.actions()]
    assert "View All" in texts and "Export..." not in texts
    click(view, 5, 5)  # outside: closes the menu
    assert view._live_overlays() == []


def roi_view():
    view = make_nd(("x1", "x2", "eV", "theta_par"), (8, 9, 40, 50)).disp(
        backend="notebook"
    )
    view.handle_event(ev("resize", width=900, height=700, pixel_ratio=1.0))
    w = view.widget
    w.centralWidget().setCurrentIndex(1)
    for name in ("a", "b"):
        w.ROI_store[name] = {"x1": [-5.0, 5.0, 0.0], "x2": [-5.0, -5.0, 5.0]}
        w.roi_list_box.addItem(name)
    view.get_frame()
    return view


@pytest.mark.parametrize("answer, remaining", [("Yes", 0), ("No", 2)])
def test_confirmation_dialog_inline(answer, remaining):
    view = roi_view()
    w = view.widget
    p = widget_centre(view, w.clear_ROI_store_button)
    click(view, p.x(), p.y())

    [dialog] = view._live_overlays()
    assert isinstance(dialog, QtWidgets.QMessageBox)
    assert dialog.testAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen)
    assert view.get_frame() is not None
    click(view, 5, 5)  # outside the dialog: blocked, as for a modal dialog
    assert view._live_overlays() == [dialog]

    button = dialog.button(getattr(QtWidgets.QMessageBox.StandardButton, answer))
    q = view._local(dialog).topLeft() + button.mapTo(dialog, button.rect().center())
    click(view, q.x(), q.y())
    assert view._live_overlays() == []
    assert w.roi_list_box.count() == remaining == len(w.ROI_store)


def test_confirmation_dialog_native_does_not_block():
    record = native(make_nd(("x1", "x2", "eV", "theta_par"), (8, 9, 40, 50)))
    w = record.widget
    w.ROI_store["a"] = {"x1": [-5.0, 5.0, 0.0], "x2": [-5.0, -5.0, 5.0]}
    w.roi_list_box.addItem("a")
    w.clear_ROI_store_button.click()  # returns at once: the dialog is not modal-exec
    [dialog] = [d for d in w.findChildren(QtWidgets.QMessageBox) if d.isVisible()]
    dialog.button(QtWidgets.QMessageBox.StandardButton.Yes).click()
    assert w.roi_list_box.count() == 0 and w.ROI_store == {}


def test_dropdown_inline():
    view = make_nd(("polar", "theta_par", "eV"), (30, 60, 80)).disp(backend="notebook")
    view.handle_event(ev("resize", width=900, height=700, pixel_ratio=1.0))
    view.get_frame()
    combo = view.widget.alignment_aid
    assert combo.count() > 1
    p = widget_centre(view, combo)
    click(view, p.x(), p.y())
    [popup] = view._live_overlays()
    items = combo.view()
    index = items.model().index(combo.count() - 1, 0)
    q = view._local(popup).topLeft() + items.viewport().mapTo(
        popup, items.visualRect(index).center()
    )
    click(view, q.x(), q.y())
    assert view._live_overlays() == []
    assert combo.currentIndex() == combo.count() - 1


# ----- browser input quirks


def view_3d():
    view = make_nd(("polar", "theta_par", "eV"), (30, 60, 80)).disp(backend="notebook")
    view.handle_event(ev("resize", width=1200, height=700, pixel_ratio=1.0))
    view.get_frame()
    return view


def test_wheel_scrolls_metadata_box():
    """An ignored wheel event passes on to the parent, e.g. from a label to its scroll area."""
    view = view_3d()
    w = view.widget
    w.cursor_stats.setText("<br>".join(f"line {i}" for i in range(100)))
    view.get_frame()
    area = w.cursor_stats.parentWidget().parentWidget()
    p = widget_centre(view, area)
    view.handle_event(ev("wheel", p.x(), p.y(), dx=0, dy=240))
    assert area.verticalScrollBar().value() > 0


def test_double_click_on_spin_box_arrow_is_released():
    """The browser's double-click follows the second release: it must be released too."""
    view = view_3d()
    box = view.widget.cursor_positions_selection[0]
    p = box.mapTo(view.widget, QtCore.QPoint(box.width() - 5, box.height() - 4))
    for _ in range(2):
        click(view, p.x(), p.y())
    view.handle_event(ev("double_click", p.x(), p.y(), 1, ()))
    value = box.value()
    for _ in range(3):
        wait_qt_timers(view, 0.3)  # a held spin box arrow repeats on a Qt timer
    assert box.value() == value


def test_pointer_moves_acknowledged_and_unchanged_frames_skipped():
    view = inline()
    sent = []
    view.send = lambda msg, *a: sent.append(msg)
    view.handle_event(ev("pointer_move", 5, 5))
    assert sent == [{"type": "peaks_move_ack"}]
    view._rfb_last_frame = view.get_frame()
    assert view.get_frame() is None
    view.widget.colorbar.setLevels(values=(0.2, 0.3))
    assert view.get_frame() is not None


# ----- zoom


def test_zoom_fits_output_and_keeps_layout():
    view = inline()
    w = view.widget
    size = (w.width(), w.height())
    assert view._peaks_zoom == 0  # the browser asks for a fit once laid out
    assert view.css_width == f"{size[0] + 2}px"  # plus the view's border

    view.handle_event(ev("peaks_fit", available=500))
    zoom = view.zoom
    assert zoom == view._peaks_zoom < 1 and not view.zoom_chosen
    css = int(view.css_width[:-2])
    assert css <= 500
    # The browser may first report the old size, which must not resize the widget;
    # at a fractional pixel ratio it comes as physical px / ratio
    view.handle_event(
        ev("resize", width=size[0] + 2 / 3, height=size[1] + 2 / 3, pixel_ratio=1.5)
    )
    assert (w.width(), w.height()) == size
    # The browser then reports the new size (inside the border), at a Retina ratio
    view.handle_event(
        ev("resize", width=css - 2, height=round(size[1] * zoom), pixel_ratio=2.0)
    )
    assert abs(w.width() - size[0]) <= 1 and abs(w.height() - size[1]) <= 1
    frame = view.get_frame()
    assert frame.shape[1] == pytest.approx(2 * (css - 2), abs=2)  # screen resolution
    pos, _ = view._pos(ev("pointer_move", 100, 100))
    assert pos.x() == pytest.approx(100 / zoom, rel=0.01)

    view.handle_event(ev("peaks_fit", available=5000))
    assert view.zoom == 1.0  # never zoomed in to fit


def test_zoom_buttons_and_rerun(monkeypatch):
    cells = Cells(monkeypatch)
    cells.run("A")
    view = inline()
    view.handle_event(ev("peaks_fit", available=500))
    fitted = view.zoom
    cells.run("A")
    view = inline()
    assert view.zoom == 1.0 and view._peaks_zoom == 0  # fitted zoom is not kept

    view.handle_event(ev("peaks_zoom", step=-1))
    view.handle_event(ev("peaks_zoom", step=-1))
    assert view.zoom == 0.8 and view.zoom_chosen
    cells.run("A")
    view = inline()
    assert view.zoom == view._peaks_zoom == 0.8  # chosen zoom is kept
    assert view.css_width == f"{round(view.widget.width() * 0.8) + 2}px"
    for _ in range(10):
        view.handle_event(ev("peaks_zoom", step=1))
    assert view.zoom == 1.5
    assert fitted < 1
