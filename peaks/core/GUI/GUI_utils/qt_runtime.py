"""Shared Qt application and viewer lifecycle management.

All live Qt viewers, whether native windows (``"qt"`` backend) or inline notebook
viewers (``"notebook"`` backend), share one registry and the ``opts.gui.max_viewers``
limit. When the limit is reached, the oldest live inline viewer is suspended
(static image kept, Qt viewer destroyed). Native windows are never closed
automatically. Re-running a notebook cell replaces the inline viewers it created and
restores their lightweight settings.
"""

import os
import sys
from contextlib import contextmanager
from functools import partial

import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from peaks.core.GUI.GUI_utils import notebook_cells
from peaks.core.options import opts
from peaks.core.utils.misc import analysis_warning

_STATE_ATTRIBUTE = "_peaks_qt_runtime_state"
_WARNING_THRESHOLD = 3


class TooManyDisplayPanelsError(RuntimeError):
    """Raised when the maximum number of active display panels is reached."""


def _get_ipython_shell():
    """Return the active IPython shell, if present."""
    try:
        from IPython import get_ipython
    except ImportError:
        return None

    return get_ipython()


def _enable_ipython_qt():
    """Enable IPython's Qt6 event-loop integration.

    Returns
    -------
    bool
        True when IPython manages the Qt event loop.
    """
    ipython = _get_ipython_shell()

    if ipython is None:
        return False

    active_eventloop = getattr(ipython, "active_eventloop", None)

    if active_eventloop in {"qt", "qt6"}:
        return True

    if active_eventloop is not None:
        raise RuntimeError(
            "Cannot start the PEAKS Qt6 viewer because IPython already has "
            f"the {active_eventloop!r} GUI event loop active. Restart the "
            "kernel or disable the other GUI event loop first."
        )

    # Equivalent to running `%gui qt6` in the notebook
    ipython.run_line_magic("gui", "qt6")
    active_eventloop = getattr(ipython, "active_eventloop", None)
    if active_eventloop is None or not active_eventloop.startswith("qt"):
        return False
    return True


def _get_runtime(headless_fallback=False):
    """Return the QApplication and persistent `peaks` runtime state.

    With ``headless_fallback``, a Linux host with no display uses Qt's offscreen
    platform. The platform is fixed once the QApplication exists.
    """
    if (
        headless_fallback
        and QtWidgets.QApplication.instance() is None
        and sys.platform.startswith("linux")
        and "QT_QPA_PLATFORM" not in os.environ
        and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    ):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

    app = pg.mkQApp("PEAKS")

    state = getattr(app, _STATE_ATTRIBUTE, None)

    if state is None:
        state = {
            "viewers": {},  # id -> viewer record, in creation order
            "defer_exec": 0,
            "cell_states": {},  # cell key -> lightweight viewer state
            "cell_counters": {},  # cell id -> viewers created in the current run
            "cell_runs": {},  # cell id -> current execution identifier
            "hooks": set(),
        }
        setattr(app, _STATE_ATTRIBUTE, state)

    return app, state


def _existing_state():
    """Return the runtime state if a QApplication already exists, else None."""
    app = QtWidgets.QApplication.instance()
    return getattr(app, _STATE_ATTRIBUTE, None) if app is not None else None


class _NativeViewer:
    """Registry record for a native (pop-out window) viewer."""

    backend = "qt"
    cell_key = None

    def __init__(self, widget):
        self.widget = widget
        self.status = "active"

    def __repr__(self):
        return f"<peaks native {type(self.widget).__name__} ({self.status})>"

    def suspend(self):
        """Native windows cannot be suspended."""
        raise TypeError(
            "Native Qt viewers cannot be suspended; only inline notebook viewers "
            "can. Use `.close()` to close a native viewer."
        )

    def close(self):
        """Close the native window."""
        close_viewer(self)


def _qt_event_loop_is_running():
    """Return whether the current thread is inside a Qt event loop."""
    return QtCore.QThread.currentThread().loopLevel() > 0


def _remove_viewer(app, viewer_id, *_):
    """Remove a destroyed viewer from the application registry."""
    state = getattr(app, _STATE_ATTRIBUTE, None)

    if state is not None:
        record = state["viewers"].pop(viewer_id, None)
        if record is not None:
            record.status = "closed"
            record.widget = None


def _live_viewers(state):
    """Return viewer records that currently own a live Qt viewer."""
    return [v for v in state["viewers"].values() if v.status == "active"]


def _make_room(state):
    """Suspend the oldest live inline viewers until a new viewer fits the limit."""
    max_viewers = opts.gui.max_viewers
    while max_viewers is not None:
        live = _live_viewers(state)
        if len(live) < max_viewers:
            return
        inline = [v for v in live if v.backend == "notebook"]
        if not inline:
            _check_viewer_limit(len(live))
        suspend_viewer(inline[0])
        if inline[0].status == "active":
            raise RuntimeError("Could not suspend an inline viewer to make room.")


def _check_viewer_limit(viewer_count):
    max_viewers = opts.gui.max_viewers

    if max_viewers is None:
        return

    if viewer_count >= max_viewers:
        raise TooManyDisplayPanelsError(
            f"Cannot open another display panel: {viewer_count} viewers "
            f"are already open, and the limit is {max_viewers}.\n\n "
            "To open more display panels simultaneously, increase the "
            "`opts.gui.max_viewers` limit, e.g. "
            f"`pks.opts.gui.max_viewers = {max_viewers + 1}`, or close a "
            "viewer (`pks.gui.close_all()` closes all of them)."
        )


def _warn_if_many_viewers(viewer_count):
    """Warn when many top-level viewer windows are open."""
    if viewer_count < _WARNING_THRESHOLD:
        return

    analysis_warning(
        f"There are currently {viewer_count} active display panels. "
        "This may cause performance issues.",
        warn_type="warning",
        title="Multiple display panels open",
    )


def _run_event_loop_if_needed(app, state, ipython_manages_qt):
    """Run Qt's blocking event loop when no host loop is available."""
    if ipython_manages_qt:
        return

    if state["defer_exec"] > 0:
        return

    if _qt_event_loop_is_running():
        return

    app.exec()


def show_viewer(viewer_factory, *args, backend=None, **kwargs):
    """Create, register and show a top-level viewer.

    Parameters
    ----------
    viewer_factory : callable
        Returns the top-level QWidget, called with ``*args`` and ``**kwargs``.
    backend : {"qt", "notebook"}, optional
        Overrides ``opts.gui.backend`` for this viewer.

    Returns
    -------
    QWidgetRFB or None
        The inline viewer for the notebook backend; None for native windows.
    """
    backend = opts.gui.backend if backend is None else backend
    if backend == "notebook":
        return _show_notebook_viewer(viewer_factory, *args, **kwargs)
    if backend != "qt":
        raise ValueError(f"Unknown GUI backend {backend!r}; use 'qt' or 'notebook'.")

    ipython_manages_qt = _enable_ipython_qt()
    app, state = _get_runtime()

    if QtCore.QThread.currentThread() != app.thread():
        raise RuntimeError(
            "`peaks` viewers must be created from the Qt application thread."
        )
    if app.platformName() == "offscreen":
        raise RuntimeError(
            "Native viewer windows are unavailable because Qt is using the "
            "'offscreen' platform (no display). Use the notebook backend instead: "
            "`pks.opts.gui.backend = 'notebook'`."
        )

    _make_room(state)  # Suspend inline viewers or raise if the limit is reached

    viewer = viewer_factory(*args, **kwargs)
    viewer.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)

    viewer_id = id(viewer)

    # Retain the viewer
    state["viewers"][viewer_id] = _NativeViewer(viewer)

    # Capture the application and integer ID
    viewer.destroyed.connect(partial(_remove_viewer, app, viewer_id))

    viewer.show()
    viewer.raise_()
    viewer.activateWindow()

    _warn_if_many_viewers(len(_live_viewers(state)))  # Warn if too many viewers

    _run_event_loop_if_needed(
        app,
        state,
        ipython_manages_qt,
    )


def _show_notebook_viewer(viewer_factory, *args, **kwargs):
    """Create an inline notebook viewer, replacing this cell's previous one."""
    try:
        from peaks.core.GUI.GUI_utils import notebook_rfb
    except ImportError as e:
        raise ImportError(
            "The notebook GUI backend requires `jupyter_rfb`. Install it with "
            "`pip install 'peaks-arpes[notebook]'`."
        ) from e

    app, state = _get_runtime(headless_fallback=True)
    if QtCore.QThread.currentThread() != app.thread():
        raise RuntimeError(
            "`peaks` viewers must be created from the Qt application thread. "
            "Several simultaneous `marimo run` sessions are not supported."
        )
    _install_dialog_guard(app, state, notebook_rfb)

    # Same-cell replacement: release this cell's previous viewers (state is kept)
    key = None
    cell = notebook_cells.current_cell(state)
    if cell is not None:
        cell_id, run_id = cell
        if state["cell_runs"].get(cell_id) != run_id:
            begin_cell(cell_id)
            state["cell_runs"][cell_id] = run_id
        index = state["cell_counters"].get(cell_id, 0)
        state["cell_counters"][cell_id] = index + 1
        key = (cell_id, index)
        for record in list(state["viewers"].values()):
            if record.cell_key == key:
                close_viewer(record, keep_state=True)

    _make_room(state)  # Suspend older inline viewers or raise

    widget = viewer_factory(*args, **kwargs)
    saved = state["cell_states"].get(key) if key is not None else None
    if saved is not None and hasattr(widget, "set_state"):
        try:
            widget.set_state(saved)
        except Exception as e:  # never fail the viewer over a saved setting
            analysis_warning(
                f"Could not restore the previous viewer settings: {e}",
                warn_type="warning",
                title="Viewer state",
            )

    try:
        view = notebook_rfb.QWidgetRFB(widget, zoom=(saved or {}).get("zoom"))
    except Exception:
        notebook_rfb.destroy_widget(widget)
        raise
    view.cell_key = key
    state["viewers"][id(view)] = view
    _warn_if_many_viewers(len(_live_viewers(state)))
    return view


def _install_dialog_guard(app, state, notebook_rfb):
    """Install (once) the filter that keeps inline viewers' windows inline."""
    if "dialog_guard" not in state:
        state["dialog_guard"] = notebook_rfb._InlineWindowGuard()
        app.installEventFilter(state["dialog_guard"])
        if app.platformName() == "offscreen":
            QtCore.qInstallMessageHandler(_quiet_offscreen_messages)


def _quiet_offscreen_messages(mode, context, message):
    """Drop the offscreen plugin's harmless notices (e.g. raise(), grabbing)."""
    if "This plugin does not support" not in message:
        sys.__stderr__.write(message + "\n")


def _capture_state(record):
    """Store the lightweight state of a live inline viewer under its cell key."""
    state = _existing_state()
    widget = record.widget
    if record.cell_key is None or widget is None or not hasattr(widget, "get_state"):
        return
    try:
        saved = widget.get_state()
        if getattr(record, "zoom_chosen", False):  # else fit the new viewer again
            saved["zoom"] = record.zoom
        state["cell_states"][record.cell_key] = saved
    except Exception as e:  # state is a convenience; never block suspension
        analysis_warning(
            f"Could not save the viewer settings: {e}",
            warn_type="warning",
            title="Viewer state",
        )


def suspend_viewer(record):
    """Suspend an inline viewer: keep a static image, release Qt resources."""
    if record.backend != "notebook":
        record.suspend()  # raises for native viewers
    if record.status != "active":
        return
    record.status = "suspending"  # rejects late browser events and draws
    try:
        _capture_state(record)
        png = record.render_image()
        record.release_widget()
        record.show_static(png)
    except Exception:
        if record.widget is None:  # Qt released: do not count it as live
            record.status = "suspended"
        else:
            record.status = "active"
        raise
    record.status = "suspended"


def close_viewer(record, keep_state=False, frontend=True):
    """Close a viewer and release its resources.

    Parameters
    ----------
    record : QWidgetRFB or _NativeViewer
        The viewer.
    keep_state : bool
        Keep the saved cell state (same-cell replacement) instead of discarding it.
    frontend : bool
        Also clear the notebook output and close the widget comms.
    """
    if record.status == "closed":
        return
    state = _existing_state()
    if record.backend == "qt":
        if record.widget is not None:
            from peaks.core.GUI.GUI_utils.notebook_rfb import (
                destroy_widget,
                flush_deletes,
            )

            # `destroyed` removes the record from the registry
            ref = destroy_widget(record.widget)
            record.widget = None
            flush_deletes(ref)
        record.status = "closed"
        return

    if record.status == "active" and keep_state:
        _capture_state(record)
    record.status = "closed"
    record.release_widget()
    if state is not None:
        state["viewers"].pop(id(record), None)
        if not keep_state and record.cell_key is not None:
            state["cell_states"].pop(record.cell_key, None)
    if frontend:
        record.close_widget()


def begin_cell(cell_id, deleted=False):
    """Release the inline viewers of a cell that is re-run (or deleted).

    Their lightweight states are kept for the new run unless the cell is deleted.
    """
    state = _existing_state()
    if state is None:
        return
    for record in list(state["viewers"].values()):
        key = record.cell_key
        if key is not None and key[0] == cell_id:
            close_viewer(record, keep_state=not deleted)
    state["cell_counters"].pop(cell_id, None)
    state["cell_runs"].pop(cell_id, None)
    if deleted:
        for key in [k for k in state["cell_states"] if k[0] == cell_id]:
            del state["cell_states"][key]


def viewers():
    """Return the `peaks` viewers that are open or suspended, oldest first."""
    state = _existing_state()
    return list(state["viewers"].values()) if state is not None else []


def suspend_all():
    """Suspend every live inline viewer. Native windows are not affected."""
    for record in viewers():
        if record.backend == "notebook":
            suspend_viewer(record)


def close_all():
    """Close every `peaks` viewer: native windows, live and suspended inline ones."""
    for record in viewers():
        close_viewer(record)
    state = _existing_state()
    if state is not None:
        state["cell_states"].clear()


_TOAST_MS = 1000


def copy_text(widget, text, message="Copied to clipboard"):
    """Copy text to the clipboard: the browser's for inline viewers, else the host's.

    ``message`` is shown briefly over the viewer.
    """
    window = widget.window()
    for record in viewers():
        if record.backend == "notebook" and record.widget is window:
            record.send({"type": "peaks_copy", "text": text})
            _show_toast(window, message)
            # Inline viewers only run Qt timers when a frame is drawn
            import asyncio

            try:
                asyncio.get_running_loop().call_later(
                    _TOAST_MS / 1000 + 0.05, record.request_draw
                )
            except RuntimeError:  # no event loop, e.g. in tests
                pass
            return
    QtWidgets.QApplication.clipboard().setText(text)
    _show_toast(window, message)


def _show_toast(window, message):
    """Show ``message`` over the top centre of ``window`` for ``_TOAST_MS``."""
    old = window.findChild(QtWidgets.QLabel, "peaks_toast")
    if old is not None:
        old.deleteLater()
    label = QtWidgets.QLabel(message, window)
    label.setObjectName("peaks_toast")
    label.setStyleSheet(
        "background: rgba(40, 40, 40, 220); color: white;"
        "padding: 6px 12px; border-radius: 6px;"
    )
    label.adjustSize()
    label.move((window.width() - label.width()) // 2, 12)
    label.show()
    label.raise_()
    QtCore.QTimer.singleShot(_TOAST_MS, label.deleteLater)


# ----- helpers for viewer get_state()/set_state()


def apply_state_steps(steps):
    """Run named restore steps; a failing step is skipped and reported once."""
    failed = []
    for name, step in steps:
        try:
            step()
        except Exception:
            failed.append(name)
    if failed:
        analysis_warning(
            f"Some previous viewer settings could not be restored: {failed}.",
            warn_type="info",
            title="Viewer state",
        )


def view_range(plot):
    """Return a plot's view range ``[[x0, x1], [y0, y1]]`` as floats."""
    return [[float(v) for v in axis] for axis in plot.getViewBox().viewRange()]


def set_view_range(plot, view_range):
    """Set a plot's view range; the view box limits clip invalid ranges."""
    plot.getViewBox().setRange(xRange=view_range[0], yRange=view_range[1], padding=0.0)


def colour_state(colorbar):
    """Return a colour bar's colour map stops and levels as plain values."""
    pos, colors = colorbar.colorMap().getStops(mode=pg.ColorMap.BYTE)
    return {
        "cmap": [pos.tolist(), colors.tolist()],
        "levels": [float(v) for v in colorbar.levels()],
    }


def set_colour_state(colorbar, state):
    """Restore the colour map and (where within the data limits) the levels."""
    colorbar.setColorMap(pg.ColorMap(*state["cmap"]))
    colorbar.setLevels(values=state["levels"])


@contextmanager
def viewer_session():
    """Defer the Qt event loop while several viewers are constructed.

    Examples
    --------
    Example usage is as follows::

        import peaks as pks

        from peaks.core.GUI.GUI_utils.qt_runtime import viewer_session

        # Set options to allow mutliple viewers to open (defaults to limit of 1)
        pks.opts.gui.max_viewers = 2

        with viewer_session():
            data1.disp()
            data2.disp()
    """
    ipython_manages_qt = _enable_ipython_qt()
    app, state = _get_runtime()

    state["defer_exec"] += 1
    completed = False

    try:
        yield
        completed = True
    finally:
        state["defer_exec"] -= 1

        native_open = any(
            v.backend == "qt" and v.status == "active" for v in state["viewers"].values()
        )
        if completed and state["defer_exec"] == 0 and native_open:
            _run_event_loop_if_needed(
                app,
                state,
                ipython_manages_qt,
            )
