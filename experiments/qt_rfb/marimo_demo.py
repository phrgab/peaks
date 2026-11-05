"""Experiment C: the existing peaks ``_Disp2D`` inside marimo via ``mo.ui.anywidget``.

marimo edit experiments/qt_rfb/marimo_demo.py
"""

import marimo

app = marimo.App()


@app.cell
def _():
    import sys

    import marimo as mo

    sys.path.insert(0, str(mo.notebook_dir()))
    from common import SIZES, make_dataarray
    from qwidget_rfb import disp_rfb

    return SIZES, disp_rfb, make_dataarray, mo


@app.cell
def _(SIZES, disp_rfb, make_dataarray, mo):
    view = disp_rfb(make_dataarray(*SIZES["medium"]))  # no %gui qt, no Qt event loop
    mo.ui.anywidget(view)
    return (view,)


@app.cell
def _(mo, view):
    from PyQt6 import QtWidgets

    tops = [
        w
        for w in QtWidgets.QApplication.topLevelWidgets()
        if type(w).__name__ == "_Disp2D"
    ]
    mo.md(
        f"`_Disp2D` top-level widgets alive: **{len(tops)}** (current view id {id(view)})"
    )
    return


if __name__ == "__main__":
    app.run()
