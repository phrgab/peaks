"""Inline notebook presentation of an existing top-level QWidget via ``jupyter_rfb``.

The widget is shown with ``WA_DontShowOnScreen`` (laid out and painted by Qt, never
mapped to the desktop), rendered into a QImage for each frame, and fed browser input
by sending QMouseEvent/QWheelEvent/QKeyEvent to its QWindow, so Qt itself picks the
child under the pointer and keeps drag grabs. No Qt event loop runs; pending Qt events
are serviced with ``processEvents()``, unless IPython's Qt integration is on (``%gui
qt``, enabled by any native viewer): the kernel then waits in Qt's event loop, which is
left whenever asyncio work is due (see ``_wake_kernel``). Everything must run on the
QApplication thread.

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
import time
import weakref

import numpy as np
import pyqtgraph as pg
from jupyter_rfb import RemoteFrameBuffer
from jupyter_rfb.widget import JS
from PyQt6 import QtCore, QtGui, QtWidgets
from pyqtgraph.jupyter.GraphicsView import MBLUT, TYPLUT, get_buttons, get_modifiers
from traitlets import Float, Unicode

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
# Zoom levels of the toolbar's zoom buttons; Fit may choose any value from _MIN_FIT
_ZOOM_STEPS = (0.5, 0.67, 0.75, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5)
_MIN_FIT = 0.25
# Border of jupyter_rfb's view around the frame (css px, both sides together)
_VIEW_BORDER = 2
# Longest side of the image kept for a suspended viewer (device pixels)
_STATIC_MAX_SIDE = 1600
# Height of the title strip drawn above an inline dialog (css px)
_DIALOG_TITLE = 22
# Redraw this often while a popup is open: submenus open and switch on Qt timers (s)
_OVERLAY_REDRAW_DELAY = 0.15
# How often, and how long after the last frontend message, to check whether the
# kernel's asyncio loop has work while IPython runs Qt's event loop (ms, s)
_WAKER_INTERVAL = 15
_WAKER_IDLE = 2.0
# The inline viewer whose input or drawing Qt is processing; it owns new popups
_dispatching = None


def _asyncio_work_due(loop):
    """Whether ``loop`` has callbacks ready or timers due."""
    # shortcut: asyncio private attributes; if they change, assume work is due
    ready = getattr(loop, "_ready", None)
    scheduled = getattr(loop, "_scheduled", None)
    if ready is None or scheduled is None:
        return True
    return bool(ready) or bool(scheduled and scheduled[0].when() <= loop.time())


def _wake_kernel():
    """Let the kernel's asyncio loop run while IPython runs Qt's event loop.

    With IPython's Qt integration (``%gui qt``, enabled by any native viewer), the
    kernel waits in Qt's event loop between messages and only leaves it for activity
    on its shell socket. Work scheduled on the asyncio loop then waits: jupyter_rfb's
    frame draws, the redraws while a menu is open, and browser messages arriving on
    a kernel subshell. Quit that Qt loop when such work is due; ipykernel runs it and
    then re-enters the Qt loop.

    Checked only for a while after each frontend message (``_kick_kernel_waker``),
    while a popup is open or a lossless redraw is pending, so an idle viewer costs
    nothing.
    """
    views = [v for v in QWidgetRFB._views if v.status == "active"]
    qt_loop = getattr(QtWidgets.QApplication.instance(), "qt_event_loop", None)
    if qt_loop is not None and qt_loop.isRunning():
        if any(v._loop and _asyncio_work_due(v._loop) for v in views):
            qt_loop.quit()
    idle = time.monotonic() - _waker_last_kick > _WAKER_IDLE
    # A pending lossless redraw (jupyter_rfb's sharp frame after a JPEG) would else
    # wait for the next frontend message, leaving a blurry frame
    busy = any(v._live_overlays() or v._rfb_lossless_draw_info for v in views)
    if idle and not busy:
        _kernel_waker.stop()


_kernel_waker = None  # QTimer running _wake_kernel
_waker_last_kick = 0.0  # time.monotonic() of the last frontend message


def _kick_kernel_waker():
    """Run ``_wake_kernel`` for the next ``_WAKER_IDLE`` s. Callable from any thread."""
    global _kernel_waker, _waker_last_kick
    _waker_last_kick = time.monotonic()
    app = QtWidgets.QApplication.instance()
    if QtCore.QThread.currentThread() != app.thread():
        if _kernel_waker is not None:  # created with the first viewer
            QtCore.QMetaObject.invokeMethod(
                _kernel_waker, "start", QtCore.Qt.ConnectionType.QueuedConnection
            )
        return
    if _kernel_waker is None:
        _kernel_waker = QtCore.QTimer()
        _kernel_waker.setInterval(_WAKER_INTERVAL)
        _kernel_waker.timeout.connect(_wake_kernel)
    if not _kernel_waker.isActive():
        _kernel_waker.start()


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
// Send held keys and pointer moves at most this long after an unacknowledged one (ms)
const ACK_TIMEOUT = 1000
const FOCUS_CONTAINER = '#rendercanvas-focus-element-container'
export default () => {
  const inner = rfbWidget()
  return {
    initialize (ctx) {
      // Send pointer moves one at a time, each after the kernel has handled the
      // previous one, and only the latest: a slow viewer never falls behind.
      const model = ctx.model
      let moveSent = 0
      let pendingMove = null
      const sendMove = (msg) => {
        moveSent = performance.now()
        model.send(msg)
      }
      const send = (msg, ...rest) => {
        if (msg && msg.type === 'pointer_move') {
          if (moveSent && performance.now() - moveSent < ACK_TIMEOUT) {
            pendingMove = msg
          } else {
            sendMove(msg)
          }
          return
        }
        if (pendingMove) { // keep the order, e.g. the last move before a release
          sendMove(pendingMove)
          pendingMove = null
        }
        return model.send(msg, ...rest)
      }
      const onAck = (msg) => {
        if (msg.type !== 'peaks_move_ack') return
        moveSent = 0
        if (pendingMove) {
          sendMove(pendingMove)
          pendingMove = null
        }
      }
      model.on('msg:custom', onAck)
      const proxy = new Proxy(model, {
        get (target, name) {
          if (name === 'send') return send
          const value = target[name]
          return typeof value === 'function' ? value.bind(target) : value
        }
      })
      const cleanupInner = inner.initialize({ ...ctx, model: proxy })
      return () => {
        model.off('msg:custom', onAck)
        if (cleanupInner) cleanupInner()
      }
    },
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
      // Zoom: scale the viewer (its layout is kept); Fit to the output's width
      const zoomOut = button('\u2212', null)
      const zoomLabel = document.createElement('span')
      zoomLabel.style.cssText = 'min-width:3.2em;text-align:center'
      bar.appendChild(zoomLabel)
      const zoomIn = button('+', null)
      const fitButton = button('Fit', null)
      zoomOut.title = 'Zoom out'
      zoomIn.title = 'Zoom in'
      fitButton.title = 'Fit to the width of the output'
      zoomOut.onclick = () => model.send({ type: 'peaks_zoom', step: -1 })
      zoomIn.onclick = () => model.send({ type: 'peaks_zoom', step: 1 })
      const fit = () => {
        if (el.clientWidth > 0) model.send({ type: 'peaks_fit', available: el.clientWidth })
      }
      fitButton.onclick = fit
      const zoomControls = [zoomOut, zoomLabel, zoomIn, fitButton]
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
        for (const c of zoomControls) c.style.display = mode === 'live' ? '' : 'none'
        const zoom = model.get('_peaks_zoom')
        zoomLabel.textContent = zoom ? `${Math.round(zoom * 100)}%` : ''
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
        if (repeatPending && now - repeatSent < ACK_TIMEOUT) return
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
      model.on('change:_peaks_zoom', update)
      model.on('msg:custom', onMessage)
      update()
      // Not zoomed yet: fit to the output, once it has been laid out
      if (!model.get('_peaks_zoom') && model.get('_peaks_mode') === 'live') {
        requestAnimationFrame(fit)
      }
      return () => {
        model.off('change:_peaks_mode', update)
        model.off('change:_peaks_static', update)
        model.off('change:_peaks_zoom', update)
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
    _peaks_zoom = Float(0.0).tag(sync=True)  # 0 until fitted to the output

    def __init__(self, widget, zoom=None, **kwargs):
        """Show ``widget`` inline, at ``zoom`` or else fitted to the output's width."""
        self._app = QtWidgets.QApplication.instance()
        self._check_thread()
        try:  # the Qt thread's event loop, for messages arriving on other threads
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None
        _kick_kernel_waker()
        widget.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        if isinstance(widget, QtWidgets.QMainWindow):
            widget.menuBar().hide()  # the toolbar's Help button replaces it
        for view in widget.findChildren(pg.GraphicsView):
            view.sceneObj.contextMenu = []  # Export... opens a separate window
        widget.show()
        self._app.processEvents()
        scale = zoom or 1.0
        kwargs.setdefault(
            "css_width", f"{round(widget.width() * scale) + _VIEW_BORDER}px"
        )
        kwargs.setdefault(
            "css_height", f"{round(widget.height() * scale) + _VIEW_BORDER}px"
        )
        kwargs.setdefault("_peaks_zoom", zoom or 0.0)
        kwargs.setdefault("_peaks_help", getattr(widget, "help_text", ""))
        super().__init__(**kwargs)
        self.widget = widget
        self._window = widget.windowHandle()
        self._pixel_ratio = 1.0
        self._scale = (1.0, 1.0)  # widget px per browser css px
        self.zoom = scale  # browser css px per widget px
        self.zoom_chosen = zoom is not None  # by the user, rather than fitted
        self._stale_size = None  # a browser size to ignore after a zoom change
        self._overlays = []  # popups and dialogs drawn over the frame, bottom first
        self._overlay_redraw = None  # pending timer handle while overlays are open
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
        """The live inline viewer containing ``widget``, or None.

        Includes its popups, as a submenu opened from Qt's own event loop (``%gui qt``)
        by a delay timer, outside any of the viewer's input, has its menu as parent.
        """
        window = widget.window() if widget is not None else None
        for view in list(cls._views):
            if view.widget is not None and (
                view.widget is window or window in view._overlays
            ):
                return view
        return None

    def _handle_msg(self, msg):
        """Handle a frontend message (input event or synced trait) on the Qt thread.

        JupyterLab 4.4+ sends widget messages over a kernel subshell when the kernel
        has them (ipykernel 7+), so they arrive on another thread. Qt widgets must
        only be used from the Qt thread: hand them to its event loop.
        """
        _kick_kernel_waker()
        if (
            self._loop is not None
            and QtCore.QThread.currentThread() != self._app.thread()
        ):
            self._loop.call_soon_threadsafe(super()._handle_msg, msg)
            return
        super()._handle_msg(msg)

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
        """Move open popups and dialogs inside the viewer, so they are drawn whole.

        Also move their QWindow there: with WA_DontShowOnScreen, widget moves do not
        reach it, and a new popup's QWindow starts at a stale position. On macOS,
        QWidget.mapFromGlobal maps through the QWindow, so pointer input (ours, and
        a menu's to its parent menu) would miss a popup the first time it opens.
        """
        width, height = self.widget.width(), self.widget.height()
        for window in self._live_overlays():
            rect = self._local(window)
            top = 0 if _is_popup(window) else _DIALOG_TITLE
            dx = max(-rect.left(), min(0, width - rect.right() - 1))
            dy = max(top - rect.top(), min(0, height - rect.bottom() - 1))
            if dx or dy:
                window.move(window.pos() + QtCore.QPoint(dx, dy))
            handle = window.windowHandle()
            if handle is not None and handle.position() != window.pos():
                handle.setPosition(window.pos())

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

    def _send_final_release(self, target, release):
        """Send the release of the last held button to ``target`` (a QWindow).

        A press makes Qt send all mouse input, hover moves included, to the pressed
        widget until the last button is released. Qt does not end this at the
        release in windows with WA_DontShowOnScreen, as inline viewers have, so
        hover moves would keep going to the last clicked widget. Clear the
        attribute for the release, so Qt ends it (and sends enter/leave events).

        Only for the viewer's own window: toggling it on a shown popup menu stops
        the menu acting on the release.
        """
        if target is not self._window:
            QtWidgets.QApplication.sendEvent(target, release)
            return
        attr = QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen
        self.widget.setAttribute(attr, False)
        try:
            QtWidgets.QApplication.sendEvent(target, release)
        finally:
            self.widget.setAttribute(attr, True)

    def _redraw_later(self):
        """Draw periodically while popups or dialogs are open.

        Qt timers (e.g. opening or switching a submenu) only run when a frame is
        drawn; unchanged frames are not sent.
        """
        if self._overlay_redraw is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._overlay_redraw = loop.call_later(
            _OVERLAY_REDRAW_DELAY, self._redraw_overlays
        )

    def _redraw_overlays(self):
        self._overlay_redraw = None
        if self.widget is None or self.status != "active":
            return
        self.request_draw()
        if self._live_overlays():
            self._redraw_later()

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
            frame = self._render()[1]
        last = self._rfb_last_frame
        if (
            last is not None
            and last.shape == frame.shape
            and np.array_equal(last, frame)
        ):
            return None  # e.g. hovering over nothing, or a menu waiting for input
        return frame

    def handle_event(self, event):
        """Translate a jupyter_rfb event into Qt events on the widget's QWindow."""
        typ = event["event_type"]
        if typ == "peaks_suspend":
            return self.suspend()
        if typ == "peaks_close":
            return self.close()
        if typ == "peaks_zoom":
            return self._step_zoom(event.get("step", 0))
        if typ == "peaks_fit":
            return self._fit(event.get("available", 0))
        if typ == "close":  # frontend model closed (e.g. marimo cell re-run)
            from peaks.core.GUI.GUI_utils import qt_runtime

            return qt_runtime.close_viewer(self, keep_state=True, frontend=False)
        if event.get("repeat") or typ == "pointer_move":
            # The browser sends the next held-key repeat or pointer move after this
            # acknowledgement, so they never queue up behind a slow viewer.
            try:
                return self._handle_input(typ, event)
            finally:
                ack = "peaks_key_ack" if event.get("repeat") else "peaks_move_ack"
                self.send({"type": ack})
        return self._handle_input(typ, event)

    def _handle_input(self, typ, event):
        if self.widget is None or self.status != "active":
            return  # late events after suspension or closing
        self._check_thread()
        with _dispatch(self):
            self._dispatch_input(typ, event)

    def _dispatch_input(self, typ, event):
        if typ == "resize":
            stale = self._stale_size
            # At a fractional pixel ratio, sizes arrive as e.g. 650.67 for 650 css px
            if stale and all(
                abs(a - b) < 1
                for a, b in zip((event["width"], event["height"]), stale, strict=True)
            ):
                return  # the size from before a zoom change, sent before the new one
            self._stale_size = None
            # The widget keeps its layout size at any zoom; frames match the screen
            self._pixel_ratio = event["pixel_ratio"] * self.zoom
            self.widget.resize(
                round(event["width"] / self.zoom), round(event["height"] / self.zoom)
            )
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
            _, gpos = self._pos(event)
            overlays = self._live_overlays()
            _send_wheel(
                overlays[-1] if overlays else self.widget,
                gpos,
                QtCore.QPoint(-int(event["dx"]), -int(event["dy"])),
                get_buttons(event.get("buttons", ())),
                _modifiers(event),
            )
            qev = None
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
        if qev is not None and typ == "pointer_move":
            # Moves are already paced by acknowledgements. pyqtgraph would drop one
            # within 10 ms of the last, often the final one, leaving hover state and
            # drags at an earlier position.
            rate_limit = pg.getConfigOption("mouseRateLimit")
            pg.setConfigOption("mouseRateLimit", 0)
            try:
                QtWidgets.QApplication.sendEvent(target, qev)
            finally:
                pg.setConfigOption("mouseRateLimit", rate_limit)
        elif qev is not None and typ == "pointer_up" and not qev.buttons():
            self._send_final_release(target, qev)
        elif qev is not None:
            QtWidgets.QApplication.sendEvent(target, qev)
        if typ == "double_click":
            # The browser sends it after the second release, so Qt would see a press
            # (a double-click) that is never released: e.g. a spin box arrow would
            # keep stepping. Release it, as after a native double-click. Send it to
            # the widget directly: Qt routes a release only to a pressed widget, and
            # a double-click event does not mark one.
            owner = self.widget if target is self._window else self._live_overlays()[-1]
            child = owner.childAt(pos.toPoint()) or owner
            local = QtCore.QPointF(child.mapFrom(owner, pos.toPoint()))
            release = QtGui.QMouseEvent(
                QtCore.QEvent.Type.MouseButtonRelease,
                local,
                pos,
                gpos,
                btn,
                QtCore.Qt.MouseButton.NoButton,
                _modifiers(event),
            )
            QtWidgets.QApplication.sendEvent(child, release)
        self._app.processEvents()
        if self.widget is None:
            return
        _close_foreign_popups()
        overlays = self._live_overlays()
        self._place_overlays()
        self.request_draw()
        if overlays:
            self._redraw_later()

    # ----- zoom

    def _set_zoom(self, zoom, chosen):
        """Show the widget ``zoom`` times its size, keeping its layout."""
        if self.widget is None:
            return
        old = (
            round(self.widget.width() * self.zoom),
            round(self.widget.height() * self.zoom),
        )
        if old != (
            round(self.widget.width() * zoom),
            round(self.widget.height() * zoom),
        ):
            # The browser may still report the old size once: at the new zoom, that
            # would resize the widget (e.g. to 3x its size when zooming to 1/3)
            self._stale_size = old
        self.zoom, self.zoom_chosen = zoom, chosen
        self._peaks_zoom = zoom
        # The browser then sends a resize event, which sets the frame resolution
        self.css_width = f"{round(self.widget.width() * zoom) + _VIEW_BORDER}px"
        self.css_height = f"{round(self.widget.height() * zoom) + _VIEW_BORDER}px"

    def _step_zoom(self, step):
        if step > 0:
            zoom = next((z for z in _ZOOM_STEPS if z > self.zoom + 1e-3), None)
        else:
            zoom = next((z for z in _ZOOM_STEPS[::-1] if z < self.zoom - 1e-3), None)
        if zoom is not None:
            self._set_zoom(zoom, chosen=True)

    def _fit(self, available):
        """Zoom out (never in) to fit ``available`` css px of width."""
        if self.widget is None or available <= 0:
            return
        fit = (available - _VIEW_BORDER) / self.widget.width()
        zoom = max(_MIN_FIT, min(1.0, int(fit * 100) / 100))
        self._set_zoom(zoom, chosen=False)

    def _pos(self, event):
        """Window-local and global position; QGraphicsScene hit-tests wheel via global.

        Uses the QWidget's mapping: with WA_DontShowOnScreen, widget moves do not
        reach the QWindow, but QGraphicsView maps back with QWidget.mapFromGlobal.
        """
        pos = QtCore.QPointF(event["x"] * self._scale[0], event["y"] * self._scale[1])
        return pos, self.widget.mapToGlobal(pos)


def _send_wheel(root, gpos, angle, buttons, modifiers):
    """Send a wheel event to the widget under ``gpos`` in ``root``, then its parents.

    Qt passes an ignored wheel event on to the parent widget only for events from the
    window system, which Python cannot create; e.g. a label in a scroll area would
    otherwise swallow it.
    """
    widget = root.childAt(root.mapFromGlobal(gpos).toPoint()) or root
    while widget is not None:
        ev = QtGui.QWheelEvent(
            widget.mapFromGlobal(gpos),
            gpos,
            QtCore.QPoint(),
            angle,
            buttons,
            modifiers,
            QtCore.Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        QtWidgets.QApplication.sendEvent(widget, ev)
        if (
            ev.isAccepted()
            or widget.isWindow()
            or widget.testAttribute(QtCore.Qt.WidgetAttribute.WA_NoMousePropagation)
        ):
            return
        widget = widget.parentWidget()


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
