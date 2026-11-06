"""Inline notebook presentation of an existing top-level QWidget via ``jupyter_rfb``.

The widget is shown with ``WA_DontShowOnScreen`` (laid out and painted by Qt, never
mapped to the desktop), rendered into a QImage for each frame, and fed browser input
by sending QMouseEvent/QWheelEvent/QKeyEvent to its QWindow, so Qt itself picks the
child under the pointer and keeps drag grabs. No Qt event loop runs; pending Qt events
are serviced with ``processEvents()``. Everything must run on the QApplication thread.

The same anywidget works in JupyterLab and marimo. It adds a small toolbar (Suspend,
Close) and a static-image mode used once the Qt widget has been released.

Viewer bookkeeping (limits, suspension, same-cell replacement) lives in
:mod:`peaks.core.GUI.GUI_utils.qt_runtime`; this module stays generic over QWidget.
"""

import base64
import gc
import weakref

import numpy as np
import pyqtgraph as pg
from jupyter_rfb import RemoteFrameBuffer
from jupyter_rfb.widget import JS
from PyQt6 import QtCore, QtGui, QtWidgets
from pyqtgraph.jupyter.GraphicsView import MBLUT, TYPLUT, get_buttons, get_modifiers
from traitlets import Unicode

_TYPES = {**TYPLUT, "double_click": QtCore.QEvent.Type.MouseButtonDblClick}
_KEYS = {
    "ArrowUp": "Up",
    "ArrowDown": "Down",
    "ArrowLeft": "Left",
    "ArrowRight": "Right",
    " ": "Space",
    "Enter": "Return",
}
# Longest side of the image kept for a suspended viewer (device pixels)
_STATIC_MAX_SIDE = 1600


def _wrapped_esm():
    """jupyter_rfb's frontend, wrapped with a toolbar and a static-image mode."""
    if JS.count("export default") != 1:
        raise RuntimeError(
            "Unsupported jupyter_rfb version: cannot wrap its frontend module."
        )
    rfb = JS.replace("export default", "const rfbWidget =")
    return (
        rfb
        + r"""
const BAR_CSS = 'display:flex;gap:6px;align-items:center;font:12px sans-serif;' +
  'padding:2px 0 4px 0;color:var(--jp-ui-font-color2,#666)'
export default () => {
  const inner = rfbWidget()
  return {
    initialize (ctx) { return inner.initialize(ctx) },
    render (ctx) {
      const { model, el } = ctx
      const bar = document.createElement('div')
      bar.style.cssText = BAR_CSS
      const button = (text, type) => {
        const b = document.createElement('button')
        b.textContent = text
        b.style.cssText = 'font:12px sans-serif;padding:1px 8px;cursor:pointer;' +
          'border:1px solid #999;border-radius:3px;background:#f3f3f3;color:#222'
        b.onclick = () => model.send({ type })
        bar.appendChild(b)
        return b
      }
      const suspendButton = button('Suspend', 'peaks_suspend')
      button('Close', 'peaks_close')
      const label = document.createElement('span')
      bar.appendChild(label)
      const live = document.createElement('div')
      const img = document.createElement('img')
      img.style.cssText = 'display:block;max-width:100%;opacity:0.85'
      el.append(bar, live, img)
      const cleanupInner = inner.render({ ...ctx, el: live })

      let liveWidth = 0
      const update = () => {
        const mode = model.get('_peaks_mode')
        const wrapper = live.querySelector('.renderview-wrapper')
        if (wrapper && wrapper.offsetWidth) liveWidth = wrapper.offsetWidth
        bar.style.display = mode === 'closed' ? 'none' : 'flex'
        live.style.display = mode === 'live' ? '' : 'none'
        suspendButton.style.display = mode === 'live' ? '' : 'none'
        const src = mode === 'suspended' ? model.get('_peaks_static') : ''
        img.style.display = src ? 'block' : 'none'
        if (src) {
          img.src = src
          img.style.width = liveWidth ? `${liveWidth}px` : model.get('css_width')
        } else {
          img.removeAttribute('src')
        }
        label.textContent = mode === 'suspended'
          ? 'Suspended (static image). Re-run the cell to restore interactivity.'
          : ''
      }
      const onMessage = (msg) => {
        if (msg.type === 'peaks_copy' && navigator.clipboard) {
          navigator.clipboard.writeText(msg.text).catch(() => {})
        }
      }
      model.on('change:_peaks_mode', update)
      model.on('change:_peaks_static', update)
      model.on('msg:custom', onMessage)
      update()
      return () => {
        model.off('change:_peaks_mode', update)
        model.off('change:_peaks_static', update)
        model.off('msg:custom', onMessage)
        if (cleanupInner) cleanupInner()
      }
    }
  }
}
"""
    )


def _qt_key(name):
    """Map a browser ``KeyboardEvent.key`` to (Qt.Key, text)."""
    qt_name = _KEYS.get(name, name.upper() if len(name) == 1 else name)
    key = getattr(QtCore.Qt.Key, f"Key_{qt_name}", None)
    return key, (name if len(name) == 1 else "")


def is_inline(widget):
    """Return whether ``widget`` belongs to an inline (never mapped) viewer window."""
    window = widget.window() if widget is not None else None
    return window is not None and window.testAttribute(
        QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen
    )


def _close_popups():
    """Close popups opened by an inline viewer; they cannot be rendered inline."""
    for _ in range(10):  # guard against a popup that refuses to close
        popup = QtWidgets.QApplication.activePopupWidget()
        if popup is None:
            return
        popup.close()


class _InlineDialogGuard(QtCore.QObject):
    """Reject modal dialogs opened by inline viewers, which would block the kernel.

    ``QDialog.exec()`` runs a nested event loop that no browser event can reach. The
    dialog is rejected from a zero-delay timer, which fires inside that nested loop,
    and the message text is reported in the notebook instead.
    """

    def eventFilter(self, obj, event):  # noqa: N802 (Qt API)
        """Intercept the Show event of dialogs that belong to inline viewers."""
        if (
            event.type() == QtCore.QEvent.Type.Show
            and isinstance(obj, QtWidgets.QDialog)
            and is_inline(obj.parentWidget())
        ):
            text = obj.text() if isinstance(obj, QtWidgets.QMessageBox) else ""
            QtCore.QTimer.singleShot(0, obj.reject)
            from peaks.core.utils.misc import analysis_warning

            analysis_warning(
                f"{obj.windowTitle()}: {text}" if text else obj.windowTitle(),
                warn_type="info",
                title="Dialog not available in notebook mode",
            )
        return False


class QWidgetRFB(RemoteFrameBuffer):
    """Inline notebook view of a top-level QWidget, without a desktop window.

    Instances are returned by ``data.disp()`` when the notebook backend is used.
    Use :meth:`suspend` to keep a static image and release the Qt viewer, or
    :meth:`close` to release everything and clear the output.
    """

    _esm = _wrapped_esm()
    _peaks_mode = Unicode("live").tag(sync=True)
    _peaks_static = Unicode("").tag(sync=True)

    def __init__(self, widget, **kwargs):
        self._app = QtWidgets.QApplication.instance()
        self._check_thread()
        widget.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        if isinstance(widget, QtWidgets.QMainWindow):
            # Menus are separate (unrendered) popup windows
            widget.menuBar().hide()
        for view_box in widget.findChildren(pg.ViewBox):
            view_box.setMenuEnabled(False)  # context menus are unrendered popups
        widget.show()
        self._app.processEvents()
        kwargs.setdefault("css_width", f"{widget.width()}px")
        kwargs.setdefault("css_height", f"{widget.height()}px")
        super().__init__(**kwargs)
        self.widget = widget
        self._window = widget.windowHandle()
        self._pixel_ratio = 1.0
        self._scale = (1.0, 1.0)  # widget px per browser css px
        # Lifecycle, managed by qt_runtime
        self.backend = "notebook"
        self.status = "active"
        self.cell_key = None

    def __repr__(self):
        """Return a short description of the viewer."""
        name = type(self.widget).__name__ if self.widget is not None else "viewer"
        return f"<peaks notebook {name} ({self.status})>"

    def _check_thread(self):
        if QtCore.QThread.currentThread() != self._app.thread():
            raise RuntimeError(
                "Qt widgets must be driven from the QApplication thread. Several "
                "simultaneous `marimo run` sessions are not supported."
            )

    # ----- lifecycle (bookkeeping in qt_runtime)

    def suspend(self):
        """Keep a static image of the viewer and release its Qt resources."""
        from peaks.core.GUI.GUI_utils import qt_runtime

        qt_runtime.suspend_viewer(self)

    def close(self, *args, **kwargs):
        """Close the viewer, release all resources and clear its output."""
        from peaks.core.GUI.GUI_utils import qt_runtime

        qt_runtime.close_viewer(self)

    def _render(self):
        """Render the whole widget (controls included) into a QImage and its buffer."""
        dpr = self._pixel_ratio
        w, h = self.widget.width(), self.widget.height()
        buf = np.empty((round(h * dpr), round(w * dpr), 4), dtype=np.uint8)
        img = pg.functions.ndarray_to_qimage(buf, QtGui.QImage.Format.Format_RGBX8888)
        img.setDevicePixelRatio(dpr)
        painter = QtGui.QPainter(img)
        self.widget.render(painter)
        painter.end()
        return img, buf

    def render_image(self, max_side=_STATIC_MAX_SIDE):
        """Return the current widget appearance as PNG bytes."""
        image, _ = self._render()
        if max(image.width(), image.height()) > max_side:
            image = image.scaled(
                max_side,
                max_side,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
        buffer = QtCore.QBuffer()
        buffer.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
        image.save(buffer, "PNG")
        return bytes(buffer.data())

    def show_static(self, png):
        """Switch the output to a static image (``png`` bytes)."""
        self._peaks_static = "data:image/png;base64," + base64.b64encode(png).decode()
        self._peaks_mode = "suspended"

    def release_widget(self):
        """Destroy the Qt widget tree and drop all frame buffers."""
        if self.widget is None:
            return
        ref = destroy_widget(self.widget)
        self.widget = self._window = None
        self._rfb_cancel_lossless_draw()
        self._rfb_draw_requested = False
        self._rfb_last_frame = None
        flush_deletes(ref)

    def close_widget(self):
        """Clear the output and close the widget comms."""
        self._peaks_static = ""
        self._peaks_mode = "closed"
        self._output_context.close()
        RemoteFrameBuffer.close(self)

    # ----- jupyter_rfb interface

    def get_frame(self):
        """Render the whole widget (controls included) into an RGBX array."""
        if self.widget is None or self.status != "active":
            return None
        run_qt_events()  # pending layout/update requests, timers, deferred deletes
        if self.widget is None:  # released while processing events
            return None
        return self._render()[1]

    def handle_event(self, event):
        """Translate a jupyter_rfb event into Qt events on the widget's QWindow."""
        typ = event["event_type"]
        if typ == "peaks_suspend":
            return self.suspend()
        if typ == "peaks_close":
            return self.close()
        if typ == "close":  # frontend model closed (e.g. marimo cell re-run)
            from peaks.core.GUI.GUI_utils import qt_runtime

            return qt_runtime.close_viewer(self, keep_state=True, frontend=False)
        if self.widget is None or self.status != "active":
            return  # late events after suspension or closing
        self._check_thread()
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
        _close_popups()
        if self.widget is not None:
            self.request_draw()

    def _pos(self, event):
        """Window-local and global position; QGraphicsScene hit-tests wheel via global.

        Uses the QWidget's mapping: with WA_DontShowOnScreen, widget moves do not
        reach the QWindow, but QGraphicsView maps back with QWidget.mapFromGlobal.
        """
        pos = QtCore.QPointF(event["x"] * self._scale[0], event["y"] * self._scale[1])
        return pos, self.widget.mapToGlobal(pos)


def destroy_widget(widget):
    """Close a top-level widget and delete its C++ object tree immediately.

    Returns a weak reference to the Python wrapper; drop all other references, then
    call :func:`flush_deletes` with it.
    """
    from PyQt6 import sip

    ref = weakref.ref(widget)
    if not sip.isdeleted(widget):
        widget.close()
    if not sip.isdeleted(widget):  # WA_DeleteOnClose only posts a deferred delete
        sip.delete(widget)
    return ref


def flush_deletes(ref, attempts=5):
    """Process deferred deletions until the object behind ``ref`` is freed.

    PyQt releases the slot proxies that hold a viewer's bound methods and closures
    (and through them its data) with ``deleteLater()``. Without a running Qt event
    loop these need explicit event processing, which cascades over several passes.
    Returns whether the object was freed.
    """
    for _ in range(attempts):
        if ref() is None:
            return True
        run_qt_events()
        if ref() is not None:
            gc.collect()
    return ref() is None


def run_qt_events():
    """Run Qt's event loop once, without blocking.

    Unlike ``processEvents()``, an event loop also performs the deferred deletions
    (``deleteLater()``) posted while no loop was running, e.g. the PyQt slot proxies
    released by ``disconnect()``. Without this they accumulate in notebook mode.
    """
    loop = QtCore.QEventLoop()
    QtCore.QTimer.singleShot(0, loop.quit)
    loop.exec()
