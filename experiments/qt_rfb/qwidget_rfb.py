"""Experiment B: render an existing top-level QWidget (e.g. peaks ``_Disp2D``) in a notebook.

The widget is shown with ``WA_DontShowOnScreen`` (laid out, "visible" to Qt, but never
mapped to the desktop), rendered into a QImage on each draw, and fed browser input by
sending QMouseEvent/QWheelEvent/QKeyEvent to its QWindow. Qt's own QWidgetWindow
dispatch then picks the child under the pointer (normally the GraphicsView viewport),
translates coordinates and keeps the implicit mouse grab during drags.

No Qt event loop runs: pending Qt events are serviced with ``processEvents()`` after
each browser event and before each frame. Everything runs on the kernel's main
thread, which must be the QApplication's thread.

Usage::

    from qwidget_rfb import disp_rfb
    view = disp_rfb(data)  # 2D xr.DataArray or list of them
    view
"""

import numpy as np
import pyqtgraph as pg
from jupyter_rfb import RemoteFrameBuffer
from PyQt6 import QtCore, QtGui, QtWidgets
from pyqtgraph.jupyter.GraphicsView import MBLUT, TYPLUT, get_buttons, get_modifiers

from common import Timer, ensure_qapp, rfb_esm

_TYPES = {**TYPLUT, "double_click": QtCore.QEvent.Type.MouseButtonDblClick}
_KEYS = {
    "ArrowUp": "Up",
    "ArrowDown": "Down",
    "ArrowLeft": "Left",
    "ArrowRight": "Right",
    " ": "Space",
    "Enter": "Return",
}


def _qt_key(name):
    """Map a browser ``KeyboardEvent.key`` to (Qt.Key, text)."""
    qt_name = _KEYS.get(name, name.upper() if len(name) == 1 else name)
    key = getattr(QtCore.Qt.Key, f"Key_{qt_name}", None)
    return key, (name if len(name) == 1 else "")


class QWidgetRFB(RemoteFrameBuffer):
    """jupyter_rfb view of an existing top-level QWidget, without a desktop window."""

    _esm = rfb_esm()

    def __init__(self, widget, **kwargs):
        self._app = ensure_qapp()
        self._check_thread()
        widget.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        if isinstance(widget, QtWidgets.QMainWindow):
            # Menus are separate (unrendered) popup windows, and _Disp2D's Help action
            # runs a modal QMessageBox.exec() that would block the kernel invisibly.
            widget.menuBar().hide()
        widget.show()
        self._app.processEvents()
        kwargs.setdefault("css_width", f"{widget.width()}px")
        kwargs.setdefault("css_height", f"{widget.height()}px")
        super().__init__(**kwargs)
        self.widget = widget
        self._window = widget.windowHandle()
        self._pixel_ratio = 1.0
        self._scale = (1.0, 1.0)  # widget px per browser css px
        self.event_times = Timer()
        self.frame_times = Timer()

    def _check_thread(self):
        if QtCore.QThread.currentThread() != self._app.thread():
            raise RuntimeError("Qt widgets must be driven from the QApplication thread.")

    def get_frame(self):
        """Render the whole widget (controls included) into an RGBX array."""
        if self.widget is None:
            return None
        with self.frame_times():
            self._app.processEvents()  # pending layout/update requests
            dpr = self._pixel_ratio
            w, h = self.widget.width(), self.widget.height()
            buf = np.empty((round(h * dpr), round(w * dpr), 4), dtype=np.uint8)
            img = pg.functions.ndarray_to_qimage(
                buf, QtGui.QImage.Format.Format_RGBX8888
            )
            img.setDevicePixelRatio(dpr)
            painter = QtGui.QPainter(img)
            self.widget.render(painter)
            painter.end()
            return buf

    def handle_event(self, event):
        """Translate a jupyter_rfb event into Qt events on the widget's QWindow."""
        if self.widget is None:
            return
        self._check_thread()
        typ = event["event_type"]
        with self.event_times():
            if typ == "resize":
                self._pixel_ratio = event["pixel_ratio"]
                self.widget.resize(int(event["width"]), int(event["height"]))
                self._app.processEvents()
                # The widget may refuse to shrink below its minimum size; the frame is
                # then scaled by the browser, so scale pointer coordinates to match.
                self._scale = (
                    self.widget.width() / max(event["width"], 1),
                    self.widget.height() / max(event["height"], 1),
                )
                return  # base class already requested a draw
            if typ == "close":
                self.widget.close()
                self.widget.deleteLater()
                self.widget = self._window = None
                self._app.processEvents()
                return
            if typ in _TYPES:
                pos, gpos = self._pos(event)
                btn = MBLUT.get(event["button"], QtCore.Qt.MouseButton.NoButton)
                qev = QtGui.QMouseEvent(
                    _TYPES[typ],
                    pos,
                    pos,
                    gpos,
                    btn,
                    get_buttons(event["buttons"]),
                    get_modifiers(event["modifiers"]),
                )
            elif typ == "wheel":
                pos, gpos = self._pos(event)
                qev = QtGui.QWheelEvent(
                    pos,
                    gpos,
                    QtCore.QPoint(),
                    QtCore.QPoint(-int(event["dx"]), -int(event["dy"])),
                    get_buttons(event.get("buttons", ())),
                    get_modifiers(event["modifiers"]),
                    QtCore.Qt.ScrollPhase.NoScrollPhase,
                    False,
                )
            elif typ in ("key_down", "key_up"):
                key, text = _qt_key(event["key"])
                if key is None:
                    return
                qev = QtGui.QKeyEvent(
                    QtCore.QEvent.Type.KeyPress
                    if typ == "key_down"
                    else QtCore.QEvent.Type.KeyRelease,
                    key,
                    get_modifiers(event["modifiers"]),
                    text,
                )
            else:
                return
            QtWidgets.QApplication.sendEvent(self._window, qev)
            self._app.processEvents()
        self.request_draw()

    def _pos(self, event):
        """Window-local and global position; QGraphicsScene hit-tests wheel via global.

        Uses the QWidget's mapping: with WA_DontShowOnScreen, widget moves do not
        reach the QWindow, but QGraphicsView maps back with QWidget.mapFromGlobal.
        """
        pos = QtCore.QPointF(event["x"] * self._scale[0], event["y"] * self._scale[1])
        return pos, self.widget.mapToGlobal(pos)

    def timing_summary(self):
        """Server-side timings plus jupyter_rfb delivery stats."""
        w = self.widget
        return {
            "framebuffer": [w.width(), w.height(), self._pixel_ratio] if w else None,
            "handle_event": self.event_times.summary(),
            "get_frame": self.frame_times.summary(),
            "rfb_stats": self.get_stats(),
        }


def disp_rfb(data, primary_dim=None, exclude_from_centering="eV", **kwargs):
    """Experimental notebook-embedded equivalent of ``disp()`` for 2D data."""
    ensure_qapp()
    from peaks.core.GUI.disp_panels.disp_2d import _Disp2D

    data = data if isinstance(data, list) else [data]
    return QWidgetRFB(_Disp2D(data, primary_dim, exclude_from_centering), **kwargs)
