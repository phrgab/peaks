"""Inline notebook presentation of an existing top-level QWidget via ``jupyter_rfb``.

The widget is shown with ``WA_DontShowOnScreen`` (laid out and painted by Qt, never
mapped to the desktop), rendered into a QImage for each frame, and fed browser input
by sending QMouseEvent/QWheelEvent/QKeyEvent to its QWindow, so Qt itself picks the
child under the pointer and keeps drag grabs. No Qt event loop runs; pending Qt events
are serviced with ``processEvents()``. Everything must run on the QApplication thread.

Popups (menus, dropdowns) and non-blocking dialogs opened by the viewer are separate
Qt windows. They are kept off the desktop too, drawn over the viewer's frame, and
receive its input while open. Blocking dialogs (``exec()``) are rejected instead.

The same anywidget works in JupyterLab and marimo. It adds a small toolbar (Suspend,
Close, Help) and a static-image mode used once the Qt widget has been released.

Viewer bookkeeping (limits, suspension, same-cell replacement) lives in
:mod:`peaks.core.GUI.GUI_utils.qt_runtime`; this module stays generic over QWidget.
"""

import asyncio
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
    "Meta": "Control",  # Cmd on a Mac, which Qt treats as Control natively
}
# Longest side of the image kept for a suspended viewer (device pixels)
_STATIC_MAX_SIDE = 1600
# Height of the title strip drawn above an inline dialog (css px)
_DIALOG_TITLE = 22
# Redraw this long after input while a popup is open: submenus open on a Qt timer (s)
_OVERLAY_REDRAW_DELAY = 0.35
# The inline viewer whose input or drawing Qt is processing; it owns new popups
_dispatching = None


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
const HELP_CSS = 'display:none;position:absolute;top:4px;left:4px;z-index:10;' +
  'max-width:min(560px,90%);max-height:90%;overflow:auto;background:#fff;color:#222;' +
  'border:1px solid #999;border-radius:4px;padding:8px 12px;font:13px sans-serif;' +
  'box-shadow:0 2px 8px rgba(0,0,0,.3)'
// Resend held keys at most this long after an unacknowledged repeat (ms)
const REPEAT_TIMEOUT = 1000
const FOCUS_CONTAINER = '#rendercanvas-focus-element-container'
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
      const live = document.createElement('div')
      live.style.position = 'relative'
      const help = document.createElement('div')
      help.style.cssText = HELP_CSS
      const helpButton = button('Help', null)
      helpButton.onclick = () => {
        help.style.display = help.style.display === 'none' ? 'block' : 'none'
      }
      const helpText = model.get('_peaks_help')
      if (helpText) {
        help.innerHTML = helpText
        const hide = document.createElement('button')
        hide.textContent = 'Close help'
        hide.style.cssText = 'margin-top:8px;font:12px sans-serif'
        hide.onclick = () => { help.style.display = 'none' }
        help.appendChild(hide)
        live.appendChild(help)
      }
      const label = document.createElement('span')
      bar.appendChild(label)
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
        helpButton.style.display = mode === 'live' && helpText ? '' : 'none'
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
      // jupyter_rfb ignores held-key repeats; send them, one at a time. Its key
      // input lives outside the output (on document.body), so remember the input
      // this viewer focuses when clicked.
      let ownFocus = null
      const onPointerDown = () => setTimeout(() => {
        const active = document.activeElement
        if (active && active.closest && active.closest(FOCUS_CONTAINER)) {
          ownFocus = active
        }
      }, 0)
      live.addEventListener('pointerdown', onPointerDown, true)
      let repeatPending = false
      let repeatSent = 0
      const onKeyDown = (ev) => {
        if (!ev.repeat || ev.target !== ownFocus) return
        if (model.get('_peaks_mode') !== 'live') return
        const now = performance.now()
        if (repeatPending && now - repeatSent < REPEAT_TIMEOUT) return
        repeatPending = true
        repeatSent = now
        model.send({
          type: 'key_down',
          key: KEY_MAP[ev.key] || ev.key,
          modifiers: getModifiers(ev),
          repeat: true
        })
      }
      document.addEventListener('keydown', onKeyDown, true)
      const onMessage = (msg) => {
        if (msg.type === 'peaks_copy' && navigator.clipboard) {
          navigator.clipboard.writeText(msg.text).catch(() => {})
        } else if (msg.type === 'peaks_key_ack') {
          repeatPending = false
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
        document.removeEventListener('keydown', onKeyDown, true)
        live.removeEventListener('pointerdown', onPointerDown, true)
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


def _modifiers(event):
    """Qt modifiers of a browser event, with Meta (Cmd on a Mac) as Control."""
    names = event.get("modifiers", ())
    return get_modifiers(tuple("Control" if m == "Meta" else m for m in names))


def _is_popup(window):
    return window.windowType() == QtCore.Qt.WindowType.Popup


def _close_foreign_popups():
    """Close popups in Qt's popup mode, which would take the viewer's input unseen.

    Popups adopted by a viewer are never shown on screen, so never enter popup mode.
    """
    for _ in range(10):  # guard against a popup that refuses to close
        popup = QtWidgets.QApplication.activePopupWidget()
        if popup is None:
            return
        popup.close()


class _InlineWindowGuard(QtCore.QObject):
    """Keep windows opened by inline viewers off the desktop, and adopt them.

    Popups and non-blocking dialogs (``QDialog.open()``) are drawn inline by the
    viewer that opened them. ``QDialog.exec()`` instead runs a nested event loop that
    no browser event can reach: such dialogs are rejected from a zero-delay timer,
    which fires inside that loop, and their message is reported in the notebook.

    ``WA_DontShowOnScreen`` is set at Polish, before Qt first shows the window.
    Setting it later makes Qt register a modal dialog without unregistering it.
    """

    def eventFilter(self, obj, event):  # noqa: N802 (Qt API)
        """Intercept new windows of inline viewers at Polish and Show."""
        typ = event.type()
        if typ not in (QtCore.QEvent.Type.Polish, QtCore.QEvent.Type.Show):
            return False
        if not isinstance(obj, QtWidgets.QWidget) or not obj.isWindow():
            return False
        owner = _dispatching or QWidgetRFB._owner_of(obj.parentWidget())
        if owner is None or obj is owner.widget:
            return False
        dialog = isinstance(obj, QtWidgets.QDialog)
        blocking = (
            dialog and obj.windowModality() == QtCore.Qt.WindowModality.ApplicationModal
        )
        hidden = obj.testAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen)
        if typ == QtCore.QEvent.Type.Polish:
            if not blocking:
                obj.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        elif blocking or (dialog and not hidden):
            _reject_blocking_dialog(obj)
        else:
            if not hidden:  # e.g. a menu first polished outside a viewer event
                obj.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen, True)
            if (_is_popup(obj) or dialog) and obj not in owner._overlays:
                owner._overlays.append(obj)
        return False


def _reject_blocking_dialog(dialog):
    text = dialog.text() if isinstance(dialog, QtWidgets.QMessageBox) else ""
    QtCore.QTimer.singleShot(0, dialog.reject)
    from peaks.core.utils.misc import analysis_warning

    analysis_warning(
        f"{dialog.windowTitle()}: {text}" if text else dialog.windowTitle(),
        warn_type="info",
        title="Dialog not available in notebook mode",
    )


class QWidgetRFB(RemoteFrameBuffer):
    """Inline notebook view of a top-level QWidget, without a desktop window.

    Instances are returned by ``data.disp()`` when the notebook backend is used.
    Use :meth:`suspend` to keep a static image and release the Qt viewer, or
    :meth:`close` to release everything and clear the output.
    """

    _esm = _wrapped_esm()
    _views = weakref.WeakSet()  # live inline viewers, to find a window's owner
    _peaks_mode = Unicode("live").tag(sync=True)
    _peaks_static = Unicode("").tag(sync=True)
    _peaks_help = Unicode("").tag(sync=True)  # HTML for the toolbar Help button

    def __init__(self, widget, **kwargs):
        self._app = QtWidgets.QApplication.instance()
        self._check_thread()
        widget.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        if isinstance(widget, QtWidgets.QMainWindow):
            widget.menuBar().hide()  # the toolbar's Help button replaces it
        for view in widget.findChildren(pg.GraphicsView):
            view.sceneObj.contextMenu = []  # Export... opens a separate window
        widget.show()
        self._app.processEvents()
        kwargs.setdefault("css_width", f"{widget.width()}px")
        kwargs.setdefault("css_height", f"{widget.height()}px")
        kwargs.setdefault("_peaks_help", getattr(widget, "help_text", ""))
        super().__init__(**kwargs)
        self.widget = widget
        self._window = widget.windowHandle()
        self._pixel_ratio = 1.0
        self._scale = (1.0, 1.0)  # widget px per browser css px
        self._overlays = []  # popups and dialogs drawn over the frame, bottom first
        QWidgetRFB._views.add(self)
        # Lifecycle, managed by qt_runtime
        self.backend = "notebook"
        self.status = "active"
        self.cell_key = None

    def __repr__(self):
        """Return a short description of the viewer."""
        name = type(self.widget).__name__ if self.widget is not None else "viewer"
        return f"<peaks notebook {name} ({self.status})>"

    @classmethod
    def _owner_of(cls, widget):
        """The live inline viewer containing ``widget``, or None."""
        window = widget.window() if widget is not None else None
        for view in list(cls._views):
            if view.widget is not None and view.widget is window:
                return view
        return None

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

    def _render(self, overlays=True):
        """Render the whole widget (controls included) into a QImage and its buffer."""
        dpr = self._pixel_ratio
        w, h = self.widget.width(), self.widget.height()
        buf = np.empty((round(h * dpr), round(w * dpr), 4), dtype=np.uint8)
        img = pg.functions.ndarray_to_qimage(buf, QtGui.QImage.Format.Format_RGBX8888)
        img.setDevicePixelRatio(dpr)
        painter = QtGui.QPainter(img)
        self.widget.render(painter)
        if overlays:
            for window in self._live_overlays():
                self._paint_overlay(painter, window)
        painter.end()
        return img, buf

    # ----- popups and dialogs drawn over the frame

    def _live_overlays(self):
        """Open popups and dialogs of this viewer: dialogs first, in opening order."""
        from PyQt6 import sip

        self._overlays = [
            o for o in self._overlays if not sip.isdeleted(o) and o.isVisible()
        ]
        return sorted(self._overlays, key=_is_popup)

    def _local(self, window):
        """Geometry of ``window`` in this viewer's widget coordinates."""
        origin = self.widget.mapToGlobal(QtCore.QPoint(0, 0))
        return window.geometry().translated(-origin)

    def _place_overlays(self):
        """Move open popups and dialogs inside the viewer, so they are drawn whole."""
        width, height = self.widget.width(), self.widget.height()
        for window in self._live_overlays():
            rect = self._local(window)
            top = 0 if _is_popup(window) else _DIALOG_TITLE
            dx = max(-rect.left(), min(0, width - rect.right() - 1))
            dy = max(top - rect.top(), min(0, height - rect.bottom() - 1))
            if dx or dy:
                window.move(window.pos() + QtCore.QPoint(dx, dy))

    def _paint_overlay(self, painter, window):
        rect = self._local(window)
        if not _is_popup(window):  # a dialog has no window frame of its own
            title = QtCore.QRect(
                rect.left() - 1,
                rect.top() - _DIALOG_TITLE,
                rect.width() + 2,
                _DIALOG_TITLE,
            )
            painter.fillRect(title, QtGui.QColor("#d8d8d8"))
            painter.setPen(QtGui.QColor("#202020"))
            painter.drawText(
                title.adjusted(8, 0, -8, 0),
                QtCore.Qt.AlignmentFlag.AlignVCenter,
                window.windowTitle(),
            )
            painter.setPen(QtGui.QColor("#808080"))
            painter.drawRect(title.united(rect.adjusted(-1, 0, 0, 0)))
        window.render(painter, rect.topLeft())

    def _input_target(self, pos, gpos):
        """The QWindow and position that should receive pointer input.

        Like a real popup's mouse grab (or a modal dialog), the topmost open popup or
        dialog takes all input, in its own coordinates: a popup closes itself on a
        click outside it, and a menu passes moves over its parent menu on.
        """
        overlays = self._live_overlays()
        if not overlays:
            return self._window, pos
        top = overlays[-1]
        return top.windowHandle(), QtCore.QPointF(top.mapFromGlobal(gpos))

    def _redraw_later(self):
        """Draw again once Qt timers (e.g. opening a submenu) have run."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.call_later(_OVERLAY_REDRAW_DELAY, self._redraw_if_live)

    def _redraw_if_live(self):
        if self.widget is not None and self.status == "active":
            self.request_draw()

    def render_image(self, max_side=_STATIC_MAX_SIDE):
        """Return the current widget appearance as PNG bytes (without popups)."""
        image, _ = self._render(overlays=False)
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
        for window in self._live_overlays():
            window.close()  # e.g. pyqtgraph menus have no Qt parent
        self._overlays = []
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
        with _dispatch(self):
            run_qt_events()  # pending layout/update requests, timers, deferred deletes
            if self.widget is None:  # released while processing events
                return None
            self._place_overlays()
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
        if event.get("repeat"):
            # The browser sends the next held-key repeat after this acknowledgement,
            # so repeats never queue up behind a slow viewer.
            try:
                return self._handle_input(typ, event)
            finally:
                self.send({"type": "peaks_key_ack"})
        return self._handle_input(typ, event)

    def _handle_input(self, typ, event):
        if self.widget is None or self.status != "active":
            return  # late events after suspension or closing
        self._check_thread()
        with _dispatch(self):
            self._dispatch_input(typ, event)

    def _dispatch_input(self, typ, event):
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
        target = self._window
        if typ in _TYPES:
            pos, gpos = self._pos(event)
            target, pos = self._input_target(pos, gpos)
            btn = MBLUT.get(event["button"], QtCore.Qt.MouseButton.NoButton)
            qev = QtGui.QMouseEvent(
                _TYPES[typ],
                pos,
                pos,
                gpos,
                btn,
                get_buttons(event["buttons"]),
                _modifiers(event),
            )
        elif typ == "wheel":
            pos, gpos = self._pos(event)
            target, pos = self._input_target(pos, gpos)
            qev = QtGui.QWheelEvent(
                pos,
                gpos,
                QtCore.QPoint(),
                QtCore.QPoint(-int(event["dx"]), -int(event["dy"])),
                get_buttons(event.get("buttons", ())),
                _modifiers(event),
                QtCore.Qt.ScrollPhase.NoScrollPhase,
                False,
            )
        elif typ in ("key_down", "key_up"):
            key, text = _qt_key(event["key"])
            if key is None:
                return
            overlays = self._live_overlays()
            if overlays:  # the topmost popup or dialog has the keyboard
                target = overlays[-1].windowHandle()
            qev = QtGui.QKeyEvent(
                QtCore.QEvent.Type.KeyPress
                if typ == "key_down"
                else QtCore.QEvent.Type.KeyRelease,
                key,
                _modifiers(event),
                text,
                bool(event.get("repeat")),
            )
        else:
            return
        QtWidgets.QApplication.sendEvent(target, qev)
        self._app.processEvents()
        if self.widget is None:
            return
        _close_foreign_popups()
        overlays = self._live_overlays()
        self._place_overlays()
        self.request_draw()
        if overlays:
            self._redraw_later()

    def _pos(self, event):
        """Window-local and global position; QGraphicsScene hit-tests wheel via global.

        Uses the QWidget's mapping: with WA_DontShowOnScreen, widget moves do not
        reach the QWindow, but QGraphicsView maps back with QWidget.mapFromGlobal.
        """
        pos = QtCore.QPointF(event["x"] * self._scale[0], event["y"] * self._scale[1])
        return pos, self.widget.mapToGlobal(pos)


class _dispatch:  # noqa: N801 (used as a context manager)
    """Mark ``view`` as the viewer whose input or drawing Qt is processing."""

    def __init__(self, view):
        self.view = view

    def __enter__(self):
        global _dispatching
        self.previous, _dispatching = _dispatching, self.view

    def __exit__(self, *exc):
        global _dispatching
        _dispatching = self.previous


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
