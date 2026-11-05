"""Headless smoke tests for the jupyter_rfb Qt prototypes.

Run with:  QT_QPA_PLATFORM=offscreen pytest experiments/qt_rfb
Events are injected through ``handle_event`` exactly as jupyter_rfb delivers them
from the browser; only the browser/websocket leg is not exercised here.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from common import ensure_qapp, make_dataarray  # noqa: E402

ensure_qapp()

from PyQt6 import QtCore  # noqa: E402

from qwidget_rfb import disp_rfb  # noqa: E402
from standalone_rfb import CutViewerRFB  # noqa: E402


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


def drag(view, start, delta, steps=10):
    """Left-drag in browser coordinates, paced like the browser's 20 ms throttle."""
    x0, y0 = start
    view.handle_event(ev("pointer_move", x0, y0))
    view.handle_event(ev("pointer_down", x0, y0, 1, (1,)))
    for i in range(1, steps + 1):
        time.sleep(0.02)  # pyqtgraph drops scene move events above mouseRateLimit
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


@pytest.fixture
def view():
    v = disp_rfb(make_dataarray(200, 300))
    yield v
    v.close()


def crosshair_browser_pos(view, i=0):
    """Browser pixel position of crosshair ``i``'s target in a ``_Disp2D``."""
    w = view.widget
    vb = w.image_plot.getViewBox()
    scene = vb.mapViewToScene(QtCore.QPointF(*w.xhs[i].xh.pos()))
    p = w.graphics_layout.viewport().mapTo(w, w.graphics_layout.mapFromScene(scene))
    return p.x(), p.y()


@pytest.mark.parametrize("dpr", [1.0, 2.0])
def test_disp2d_renders_offscreen(view, dpr):
    view.handle_event(ev("resize", width=700, height=800, pixel_ratio=dpr))
    frame = view.get_frame()
    assert frame.shape == (round(800 * dpr), round(700 * dpr), 4)
    assert frame[..., :3].std() > 10  # not blank
    w = view.widget
    assert w.isVisible() and w.testAttribute(
        QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen
    )
    assert not w.windowHandle().isExposed()  # never mapped on the desktop
    assert (
        not w.menuBar().isVisible()
    )  # menu -> modal Help dialog would block the kernel


def test_disp2d_drag_moves_existing_crosshair_and_cuts(view):
    view.get_frame()
    pos0 = view.widget.xhs[0].get_pos()
    edc0 = view.widget.DC_plot_items[0][0].getData()[0].copy()
    mdc0 = view.widget.DC_plot_items[1][0].getData()[1].copy()
    before = view.get_frame().copy()
    drag(view, crosshair_browser_pos(view), (-40, -30))
    pos1 = view.widget.xhs[0].get_pos()
    assert pos1[0] > pos0[0] and pos1[1] < pos0[1]  # up = higher eV, left = lower theta
    assert not np.array_equal(edc0, view.widget.DC_plot_items[0][0].getData()[0])
    assert not np.array_equal(mdc0, view.widget.DC_plot_items[1][0].getData()[1])
    assert (view.get_frame() != before).any()


def test_disp2d_drag_outside_target_does_not_move_crosshair(view):
    view.get_frame()
    pos0 = view.widget.xhs[0].get_pos()
    x, y = crosshair_browser_pos(view)
    drag(view, (x + 25, y + 25), (-20, -20))  # pans the view instead
    assert view.widget.xhs[0].get_pos() == pos0


def test_disp2d_checkbox_click_reaches_qt_control(view):
    w = view.widget
    box = w.show_DCs_checkboxes[2]
    p = box.mapTo(w, QtCore.QPoint(8, box.height() // 2))
    assert not box.isChecked()
    view.handle_event(ev("pointer_down", p.x(), p.y(), 1, (1,)))
    view.handle_event(ev("pointer_up", p.x(), p.y(), 1, ()))
    assert box.isChecked() and w.xhs[2].xh.isVisible()


def test_disp2d_arrow_key_moves_crosshair_after_click(view):
    view.get_frame()
    x, y = crosshair_browser_pos(view)
    view.handle_event(ev("pointer_down", x + 25, y + 25, 1, (1,)))  # focus the plot
    view.handle_event(ev("pointer_up", x + 25, y + 25, 1, ()))
    pos0 = view.widget.xhs[0].get_pos()
    view.handle_event(dict(event_type="key_down", key="ArrowRight", modifiers=()))
    view.handle_event(dict(event_type="key_up", key="ArrowRight", modifiers=()))
    pos1 = view.widget.xhs[0].get_pos()
    assert pos1[1] == pytest.approx(pos0[1] + view.widget.step_sizes[1])


@pytest.mark.parametrize("window_pos", [(0, 0), (100, 100), (400, 300)])
def test_disp2d_wheel_zooms_image_plot(view, window_pos):
    # QGraphicsScene hit-tests wheel events via the *global* position, so the
    # (invisible) window's desktop position must not matter.
    view.widget.move(*window_pos)
    view.get_frame()
    vb = view.widget.image_plot.getViewBox()
    gv = view.widget.graphics_layout
    corner = gv.mapFromScene(
        vb.mapViewToScene(QtCore.QPointF(vb.viewRange()[0][0], vb.viewRange()[1][1]))
    )
    p = gv.viewport().mapTo(
        view.widget, corner + QtCore.QPoint(15, 15)
    )  # near top-left of plot
    r0 = vb.viewRange()
    view.handle_event(ev("wheel", p.x(), p.y(), dx=0, dy=-120))
    r1 = vb.viewRange()
    assert r1[0][1] - r1[0][0] < r0[0][1] - r0[0][0]


def test_close_releases_widget():
    v = disp_rfb(make_dataarray(50, 60))
    w = v.widget
    destroyed = []
    w.destroyed.connect(lambda: destroyed.append(True))
    v.close()
    QtCore.QCoreApplication.sendPostedEvents(
        None, QtCore.QEvent.Type.DeferredDelete.value
    )
    assert v.widget is None and destroyed
    v.handle_event(ev("pointer_move"))  # late events are ignored, not a crash
    assert v.get_frame() is None


def test_standalone_drag_updates_cuts():
    v = CutViewerRFB(size="medium")
    v.handle_event(ev("resize", width=650, height=600, pixel_ratio=1.0))
    v.get_frame()
    vb = v.v.image_plot.getViewBox()
    p = v.gfxView.mapFromScene(vb.mapViewToScene(QtCore.QPointF(*v.v.target.pos())))
    mdc0 = v.v.mdc.getData()[1].copy()
    drag(v, (p.x(), p.y()), (0, 40))
    assert not np.array_equal(mdc0, v.v.mdc.getData()[1])
    assert v.get_frame()[..., :3].std() > 10
    v.close()
