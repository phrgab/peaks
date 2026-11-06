"""Identify the notebook cell that creates an inline viewer.

Used for same-cell replacement: re-running a cell releases the inline viewers it
created before and passes their settings to the new ones.

* Jupyter (ipykernel >= 6): JupyterLab sends the cell's ``cellId`` (and recently
  ``deletedCells``) in the ``execute_request`` metadata. ``kernel.get_parent()`` and
  IPython's ``pre_run_cell`` event are public interfaces.
* marimo: the running cell's id and its lifecycle hooks come from
  ``marimo._runtime`` (**private** API, isolated here). ``dispose`` runs before a
  cell re-runs and when it is deleted.

If no cell id is available (plain IPython, scripts, frontends that send no
``cellId``), None is returned and viewers are never replaced automatically. Cells
are never identified by their source code.
"""

import sys


def current_cell(state):
    """Return ``(cell_id, run_id)`` for the executing notebook cell, or None.

    ``run_id`` changes each time the cell is executed. Registers the lifecycle
    hooks on first use.
    """
    return _jupyter_cell(state) or _marimo_cell(state)


# ----- Jupyter / IPython


def _kernel_parent():
    try:
        from IPython import get_ipython
    except ImportError:
        return None, None
    ipython = get_ipython()
    kernel = getattr(ipython, "kernel", None)
    if kernel is None or not hasattr(kernel, "get_parent"):
        return ipython, None
    return ipython, kernel.get_parent()


def _jupyter_cell(state):
    ipython, parent = _kernel_parent()
    if not parent:
        return None
    cell_id = parent.get("metadata", {}).get("cellId")
    if not cell_id:
        return None
    if "ipython" not in state["hooks"]:
        ipython.events.register("pre_run_cell", _pre_run_cell)
        state["hooks"].add("ipython")
    return cell_id, parent["header"]["msg_id"]


def _pre_run_cell(info):
    """Release viewers of deleted cells and of the cell about to re-run."""
    from peaks.core.GUI.GUI_utils.qt_runtime import begin_cell

    _, parent = _kernel_parent()
    for cell_id in (parent or {}).get("metadata", {}).get("deletedCells", []):
        begin_cell(cell_id, deleted=True)
    cell_id = getattr(info, "cell_id", None)
    if cell_id:
        begin_cell(cell_id)


# ----- marimo (private API)


def _marimo_cell(state):
    if "marimo" not in sys.modules:
        return None
    try:
        from marimo._runtime.context import get_context

        context = get_context()
        cell_id = context.cell_id
    except Exception:  # not running inside a marimo cell
        return None
    if cell_id is None:
        return None
    registered = state.setdefault("marimo_cells", set())
    if cell_id not in registered:
        try:
            item = _marimo_lifecycle_item(cell_id, registered)
            context.cell_lifecycle_registry.add(item)
        except Exception:  # marimo internals changed: fall back to no replacement
            return None
        registered.add(cell_id)
    # One id per run: the lifecycle item resets the run when it is disposed
    return cell_id, "marimo"


def _marimo_lifecycle_item(cell_id, registered):
    from marimo._runtime.cell_lifecycle_item import CellLifecycleItem
    from PyQt6 import QtCore, QtWidgets

    class _ViewerLifecycle(CellLifecycleItem):
        def create(self, context):
            pass

        def dispose(self, context, deletion):
            from peaks.core.GUI.GUI_utils.qt_runtime import begin_cell

            registered.discard(cell_id)
            app = QtWidgets.QApplication.instance()
            if app is not None and QtCore.QThread.currentThread() == app.thread():
                begin_cell(cell_id, deleted=deletion)
            return True

    return _ViewerLifecycle()
