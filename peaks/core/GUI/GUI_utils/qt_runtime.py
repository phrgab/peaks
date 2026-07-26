"""Shared Qt application and viewer lifecycle management."""

from contextlib import contextmanager
from functools import partial

import pyqtgraph as pg
from PyQt6 import QtCore

from peaks.core.utils.misc import analysis_warning

_STATE_ATTRIBUTE = "_peaks_qt_runtime_state"
_WARNING_THRESHOLD = 3


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
    return True


def _get_runtime():
    """Return the QApplication and persistent `peaks` runtime state."""
    app = pg.mkQApp("PEAKS")

    state = getattr(app, _STATE_ATTRIBUTE, None)

    if state is None:
        state = {
            "viewers": {},
            "defer_exec": 0,
        }
        setattr(app, _STATE_ATTRIBUTE, state)

    return app, state


def _qt_event_loop_is_running():
    """Return whether the current thread is inside a Qt event loop."""
    return QtCore.QThread.currentThread().loopLevel() > 0


def _remove_viewer(app, viewer_id, *_):
    """Remove a destroyed viewer from the application registry."""
    state = getattr(app, _STATE_ATTRIBUTE, None)

    if state is not None:
        state["viewers"].pop(viewer_id, None)


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


def show_viewer(viewer_factory, *args, **kwargs):
    """Create, retain, and show a top-level viewer."""
    ipython_manages_qt = _enable_ipython_qt()
    app, state = _get_runtime()

    if QtCore.QThread.currentThread() != app.thread():
        raise RuntimeError(
            "`peaks` viewers must be created from the Qt application thread."
        )

    viewer = viewer_factory(*args, **kwargs)
    viewer.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)

    viewer_id = id(viewer)

    # Retain the viewer
    state["viewers"][viewer_id] = viewer

    # Capture the application and integer ID
    viewer.destroyed.connect(partial(_remove_viewer, app, viewer_id))

    viewer.show()
    viewer.raise_()
    viewer.activateWindow()

    _warn_if_many_viewers(len(state["viewers"]))

    _run_event_loop_if_needed(
        app,
        state,
        ipython_manages_qt,
    )


@contextmanager
def viewer_session():
    """Defer app.exec() while several viewer windows are constructed.

    Examples
    --------
    with viewer_session():
        data_2d.disp()
        data_3d.disp()
    """
    ipython_manages_qt = _enable_ipython_qt()
    app, state = _get_runtime()

    state["defer_exec"] += 1

    try:
        yield
    finally:
        state["defer_exec"] -= 1

        if state["defer_exec"] == 0 and state["viewers"]:
            _run_event_loop_if_needed(
                app,
                state,
                ipython_manages_qt,
            )
