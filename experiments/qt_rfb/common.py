"""Shared helpers for the Qt-in-notebook (jupyter_rfb) feasibility experiments."""

import os
import sys
from time import perf_counter

import numpy as np

# Representative sizes: (n_eV, n_theta). "medium" ~ a binned DA30 dispersion,
# "large" ~ a full-resolution detector image.
SIZES = {"medium": (500, 700), "large": (2000, 2000)}


def ensure_qapp():
    """Return the QApplication, creating it without any IPython ``%gui`` integration.

    On a Linux host with no display server, fall back to Qt's offscreen platform
    (must be decided before the QApplication exists).
    """
    if (
        sys.platform.startswith("linux")
        and "QT_QPA_PLATFORM" not in os.environ
        and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    ):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    import pyqtgraph as pg

    return pg.mkQApp("PEAKS")


def rfb_esm():
    """jupyter_rfb's frontend JS, with the pointer-move throttle from $RFB_MOVE_THROTTLE_MS.

    Upstream hard-codes 20 ms. Browsers deliver pointermove at display rate (~16.7 ms),
    so a 20 ms throttle forwards only every second move (~30 Hz); < 16 ms forwards all.
    """
    from jupyter_rfb.widget import JS

    ms = int(os.environ.get("RFB_MOVE_THROTTLE_MS", "20"))
    js = JS.replace("this.setThrottle(20)", f"this.setThrottle({ms})")
    if ms != 20 and js == JS:
        raise RuntimeError("jupyter_rfb JS changed; move-throttle patch not applied")
    return js


def make_image(n_eV=500, n_theta=700, seed=0):
    """Return a synthetic ARPES-like dispersion, shape (n_eV, n_theta), float32."""
    rng = np.random.default_rng(seed)
    eV = np.linspace(-1.0, 0.1, n_eV)[:, None]
    k = np.linspace(-15, 15, n_theta)[None, :]
    spec = np.zeros((n_eV, n_theta))
    for k0, a in [(0, 0.004), (4, 0.006), (-4, 0.006)]:
        band = -0.8 + a * (k - k0) ** 2
        spec += 1 / (1 + ((eV - band) / 0.03) ** 2)
    spec /= 1 + np.exp(eV / 0.01)  # Fermi cutoff
    return rng.poisson(200 * spec + 5).astype(np.float32)


def make_dataarray(n_eV=500, n_theta=700, seed=0):
    """Return :func:`make_image` as a peaks-style ``xr.DataArray`` (dims eV, theta_par)."""
    import xarray as xr

    da = xr.DataArray(
        make_image(n_eV, n_theta, seed),
        dims=("eV", "theta_par"),
        coords={
            "eV": np.linspace(-1.0, 0.1, n_eV),
            "theta_par": np.linspace(-15, 15, n_theta),
        },
        name=f"synthetic_{n_eV}x{n_theta}",
    )
    da.eV.attrs["units"] = "eV"
    da.theta_par.attrs["units"] = "deg"
    return da


class Timer(list):
    """List of durations (s) with a context manager to record and a summary."""

    def __call__(self):
        """Return a context manager that appends the elapsed time on exit."""
        timer = self

        class _Ctx:
            def __enter__(self):
                self.t = perf_counter()

            def __exit__(self, *exc):
                timer.append(perf_counter() - self.t)

        return _Ctx()

    def summary(self):
        """Return n / median / p95 / max in milliseconds."""
        if not self:
            return {"n": 0}
        a = np.asarray(self) * 1e3
        return {
            "n": len(a),
            "median_ms": round(float(np.median(a)), 2),
            "p95_ms": round(float(np.percentile(a, 95)), 2),
            "max_ms": round(float(a.max()), 2),
        }
