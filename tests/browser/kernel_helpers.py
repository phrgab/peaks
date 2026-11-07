"""Helpers imported by the notebook kernels that the browser tests drive.

Positions are returned in the browser's css px, relative to the viewer's canvas.
"""

import json

import numpy as np
import xarray as xr

import peaks as pks

pks.opts.gui.backend = "notebook"


def make_data(dims=("eV", "theta_par"), shape=(120, 150), seed=0):
    """Small random data with peaks-style coordinates and units."""
    rng = np.random.default_rng(seed)
    coords = {d: np.linspace(-10, 10, n) for d, n in zip(dims, shape, strict=True)}
    if "eV" in dims:
        coords["eV"] = np.linspace(-1.0, 0.1, shape[dims.index("eV")])
    da = xr.DataArray(rng.random(shape).astype(np.float32), dims=dims, coords=coords)
    for d in dims:
        da[d].attrs["units"] = "eV" if d == "eV" else "deg"
    da.name = f"test_{seed}"
    return da


def viewer():
    """The most recent live inline viewer."""
    return [v for v in pks.gui.viewers() if v.status == "active"][-1]


def css(view, point):
    """A point in the viewer widget's coordinates, as css px on its canvas."""
    return [point.x() / view._scale[0], point.y() / view._scale[1]]


def widget_at(view, widget, dx=0, dy=0):
    """Centre of a child widget (plus an offset in widget px), as css px."""
    point = widget.mapTo(view.widget, widget.rect().center())
    point.setX(point.x() + dx)
    point.setY(point.y() + dy)
    return css(view, point)


def item_at(view, item):
    """Centre of a pyqtgraph scene item, as css px."""
    gv = item.scene().views()[0]
    point = gv.mapTo(view.widget, gv.mapFromScene(item.sceneBoundingRect().center()))
    return css(view, point)


def action_at(view, menu, text):
    """Centre of a menu entry, as css px."""
    action = next(a for a in menu.actions() if a.text() == text)
    rect = menu.actionGeometry(action).translated(view._local(menu).topLeft())
    return css(view, rect.center())


def report(**values):
    """Write values for the test to read (status.json in the working directory)."""
    with open("status.json", "w") as f:
        json.dump(values, f)
