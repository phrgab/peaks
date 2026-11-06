"""Server-side timing of a continuous crosshair drag, RFB path vs native Qt window.

    QT_QPA_PLATFORM=offscreen python bench_headless.py rfb [A,B,P]  # RFB pipeline
    (A: pyqtgraph.jupyter, B: prototype QWidgetRFB, P: production notebook backend)
    xvfb-run -s "-screen 0 1920x1080x24" python bench_headless.py native

Per drag step the RFB path does: handle_event (Qt dispatch + cut update) -> get_frame
(render to QImage) -> JPEG encode (jupyter_rfb's encoder, quality 80). The native path
does: QMouseEvent to the window -> processEvents (cut update + paint + X flush).
Steps are paced at 20 ms like the browser throttle and pyqtgraph's mouseRateLimit.
"""

import json
import sys
import time

from common import SIZES, Timer, ensure_qapp, make_dataarray, make_image

app = ensure_qapp()

import pyqtgraph as pg  # noqa: E402
from peaks.core.GUI.disp_panels.disp_2d import _Disp2D  # noqa: E402
from PyQt6 import QtCore, QtGui, QtWidgets  # noqa: E402

STEPS = 150


def path(x0, y0, i):
    """Back-and-forth diagonal sweep of +-60 px around the start point."""
    t = (i % 60) / 30
    d = 60 * (t if t <= 1 else 2 - t)
    return x0 - d, y0 - 0.6 * d


def target_pos_disp2d(w):
    vb = w.image_plot.getViewBox()
    scene = vb.mapViewToScene(QtCore.QPointF(*w.xhs[0].xh.pos()))
    p = w.graphics_layout.viewport().mapTo(w, w.graphics_layout.mapFromScene(scene))
    return p.x(), p.y()


def target_pos_standalone(gview, v):
    scene = v.image_plot.getViewBox().mapViewToScene(QtCore.QPointF(*v.target.pos()))
    p = gview.viewport().mapTo(gview.window(), gview.mapFromScene(scene))
    return p.x(), p.y()


def bench_rfb(kind, size, dpr):
    from jupyter_rfb._utils import array2compressed

    from qwidget_rfb import QWidgetRFB
    from standalone_rfb import CutViewerRFB

    if kind == "A":
        view = CutViewerRFB(img=make_image(*SIZES[size]))
        width, height = 650, 600
    elif kind == "P":  # production backend: data.disp(backend="notebook")
        view = make_dataarray(*SIZES[size]).disp(backend="notebook")
        width, height = 650, 770
        view.event_times, view.frame_times = Timer(), Timer()
        handle, render = view.handle_event, view.get_frame

        def timed_handle(event):
            with view.event_times():
                handle(event)

        def timed_frame():
            with view.frame_times():
                return render()

        view.handle_event, view.get_frame = timed_handle, timed_frame
    else:
        view = QWidgetRFB(_Disp2D([make_dataarray(*SIZES[size])], None, "eV"))
        width, height = 650, 770
    view.handle_event(
        dict(event_type="resize", width=width, height=height, pixel_ratio=dpr)
    )
    view.get_frame()
    x0, y0 = (
        target_pos_standalone(view.gfxView, view.v)
        if kind == "A"
        else target_pos_disp2d(view.widget)
    )
    ev = dict(modifiers=())
    view.handle_event(
        {
            **ev,
            "event_type": "pointer_move",
            "x": x0,
            "y": y0,
            "button": 0,
            "buttons": (),
        }
    )
    view.handle_event(
        {
            **ev,
            "event_type": "pointer_down",
            "x": x0,
            "y": y0,
            "button": 1,
            "buttons": (1,),
        }
    )
    view.event_times.clear()
    view.frame_times.clear()
    encode, total, nbytes = Timer(), Timer(), []
    for i in range(1, STEPS + 1):
        time.sleep(0.02)
        x, y = path(x0, y0, i)
        with total():
            view.handle_event(
                {
                    **ev,
                    "event_type": "pointer_move",
                    "x": x,
                    "y": y,
                    "button": 0,
                    "buttons": (1,),
                }
            )
            frame = view.get_frame()
            with encode():
                _, data = array2compressed(frame, 80)
        nbytes.append(len(data))
    moved = view.v.target.pos() if kind == "A" else view.widget.xhs[0].get_pos()
    out = {
        "path": f"rfb-{kind}",
        "size": size,
        "frame_px": list(frame.shape[1::-1]),
        "dpr": dpr,
        "handle_event": view.event_times.summary(),
        "get_frame": view.frame_times.summary(),
        "jpeg_q80": encode.summary(),
        "server_total": total.summary(),
        "jpeg_kB_median": round(sorted(nbytes)[len(nbytes) // 2] / 1024, 1),
        "final_pos": [round(float(c), 3) for c in moved],
    }
    if kind == "A":
        out["cut_update"] = view.v.cut_times.summary()
    else:
        cut = Timer()
        for _ in range(100):
            with cut():
                view.widget._update_DC(0)  # existing peaks cut code, both DCs
        out["cut_update"] = cut.summary()
    view.close()
    return out


def bench_native(kind, size):
    from standalone_rfb import build_cut_layout

    if kind == "A":
        widget = pg.GraphicsLayoutWidget(size=(650, 600))
        v = build_cut_layout(widget, make_image(*SIZES[size]))
        gview = widget
    else:
        widget = _Disp2D([make_dataarray(*SIZES[size])], None, "eV")
        gview = widget.graphics_layout
    widget.show()
    for _ in range(20):
        app.processEvents()
        time.sleep(0.02)
    paints = []

    class PaintCounter(QtCore.QObject):
        def eventFilter(self, obj, e):
            if e.type() == QtCore.QEvent.Type.Paint:
                paints.append(1)
            return False

    counter = PaintCounter()
    gview.viewport().installEventFilter(counter)
    x0, y0 = (
        target_pos_standalone(gview, v) if kind == "A" else target_pos_disp2d(widget)
    )
    win = widget.windowHandle()
    L, N = QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.NoButton

    def send(typ, x, y, btn, btns):
        p = QtCore.QPointF(x, y)
        QtWidgets.QApplication.sendEvent(
            win,
            QtGui.QMouseEvent(
                typ,
                p,
                p,
                win.mapToGlobal(p),
                btn,
                btns,
                QtCore.Qt.KeyboardModifier.NoModifier,
            ),
        )
        app.processEvents()

    send(QtCore.QEvent.Type.MouseMove, x0, y0, N, N)
    send(QtCore.QEvent.Type.MouseButtonPress, x0, y0, L, L)
    total = Timer()
    paints.clear()
    for i in range(1, STEPS + 1):
        time.sleep(0.02)
        x, y = path(x0, y0, i)
        with total():
            send(QtCore.QEvent.Type.MouseMove, x, y, N, L)
    moved = v.target.pos() if kind == "A" else widget.xhs[0].get_pos()
    widget.close()
    return {
        "path": f"native-{kind}",
        "size": size,
        "platform": app.platformName(),
        "event_plus_paint": total.summary(),
        "viewport_paints": len(paints),
        "final_pos": [round(float(c), 3) for c in moved],
    }


if __name__ == "__main__":
    mode = sys.argv[1]
    kinds = sys.argv[2].split(",") if len(sys.argv) > 2 else ("A", "B")
    for kind in kinds:
        for size in SIZES:
            if mode == "rfb":
                for dpr in (1.0, 2.0):
                    print(json.dumps(bench_rfb(kind, size, dpr)), flush=True)
            else:
                print(json.dumps(bench_native(kind, size)), flush=True)
