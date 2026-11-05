"""Experiment A: standalone PyQtGraph image + two linked cuts, rendered via jupyter_rfb.

Reproduces the core `_Disp2D` interaction (image, EDC panel right, MDC panel below,
draggable crosshair target, cuts updated from ``sigPositionChanged``) without peaks.

Notebook::

    from standalone_rfb import CutViewerRFB
    w = CutViewerRFB(size="medium"); w

Native baseline (same layout and update code in a normal Qt window)::

    python standalone_rfb.py --native [--size large]
"""

import argparse
from types import SimpleNamespace

import numpy as np
import pyqtgraph as pg

from common import SIZES, Timer, ensure_qapp, make_image, rfb_esm

ensure_qapp()  # before pyqtgraph.jupyter creates any Qt objects
from pyqtgraph.jupyter import GraphicsLayoutWidget  # noqa: E402


def build_cut_layout(glw, img):
    """Populate a (native or jupyter) GraphicsLayoutWidget with image + linked cuts."""
    n0, n1 = img.shape
    image_plot = glw.addPlot(0, 0)
    edc_plot = glw.addPlot(0, 1)  # right: intensity vs eV
    edc_plot.setYLink(image_plot)
    mdc_plot = glw.addPlot(1, 0)  # below: intensity vs theta
    mdc_plot.setXLink(image_plot)
    layout = glw.ci.layout if hasattr(glw, "ci") else glw.gfxLayout.layout
    layout.setRowStretchFactor(0, 3)
    layout.setColumnStretchFactor(0, 3)

    image = pg.ImageItem(img, axisOrder="row-major")
    image.setColorMap(pg.colormap.get("Greys", source="matplotlib"))
    image_plot.addItem(image)
    vline = pg.InfiniteLine(angle=90, pen=(238, 119, 51))
    hline = pg.InfiniteLine(angle=0, pen=(238, 119, 51))
    target = pg.TargetItem(pos=(n1 / 2, n0 / 2), symbol="crosshair", pen=(238, 119, 51))
    for item in (vline, hline, target):
        image_plot.addItem(item)
    edc = edc_plot.plot(pen=(238, 119, 51))
    mdc = mdc_plot.plot(pen=(238, 119, 51))
    rows, cols = np.arange(n0), np.arange(n1)
    cut_times = Timer()

    def update():
        with cut_times():
            x, y = target.pos()
            ix, iy = int(np.clip(x, 0, n1 - 1)), int(np.clip(y, 0, n0 - 1))
            vline.setPos(x)
            hline.setPos(y)
            edc.setData(img[:, ix], rows)
            mdc.setData(cols, img[iy, :])

    target.sigPositionChanged.connect(update)
    update()
    return SimpleNamespace(
        image_plot=image_plot, target=target, edc=edc, mdc=mdc, cut_times=cut_times
    )


class CutViewerRFB(GraphicsLayoutWidget):
    """pyqtgraph.jupyter widget with image, crosshair and two linked cuts."""

    _esm = rfb_esm()

    def __init__(self, size="medium", img=None, **kwargs):
        kwargs.setdefault("css_width", "650px")
        kwargs.setdefault("css_height", "600px")
        super().__init__(**kwargs)
        self.img = make_image(*SIZES[size]) if img is None else img
        self.v = build_cut_layout(self, self.img)
        self.v.target.sigPositionChanged.connect(self.request_draw)
        self.event_times = Timer()
        self.frame_times = Timer()

    def handle_event(self, event):
        """Time upstream event handling (includes the cut update for drags)."""
        with self.event_times():
            super().handle_event(event)

    def get_frame(self):
        """Time upstream scene rendering into the frame buffer."""
        with self.frame_times():
            return super().get_frame()

    def timing_summary(self):
        """Server-side timings plus jupyter_rfb delivery stats."""
        return {
            "framebuffer": [*self.logical_size, self.pixel_ratio],
            "image_shape": list(self.img.shape),
            "cut_update": self.v.cut_times.summary(),
            "handle_event": self.event_times.summary(),
            "get_frame": self.frame_times.summary(),
            "rfb_stats": self.get_stats(),
        }


def run_native(size):
    """Show the same layout in a native window; print timings every 2 s."""
    app = ensure_qapp()
    glw = pg.GraphicsLayoutWidget(size=(650, 600), title="native baseline")
    v = build_cut_layout(glw, make_image(*SIZES[size]))
    paint_times = Timer()
    orig_paint = glw.paintEvent

    def timed_paint(ev):
        with paint_times():
            orig_paint(ev)

    glw.paintEvent = timed_paint
    glw.show()

    def report():
        n = len(paint_times)
        print(
            "cut",
            v.cut_times.summary(),
            "paint",
            paint_times.summary(),
            f"{n / 2:.0f} paints/s",
        )
        v.cut_times.clear()
        paint_times.clear()

    timer = pg.QtCore.QTimer()
    timer.timeout.connect(report)
    timer.start(2000)
    app.exec()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--native", action="store_true", required=True)
    p.add_argument("--size", default="medium", choices=list(SIZES))
    run_native(p.parse_args().size)
