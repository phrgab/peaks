"""End-to-end browser measurement: JupyterLab + headless Chromium (Playwright).

    python bench_browser.py [--configs A-medium,B-medium,...] [--frontend jupyter|marimo]

For each config it writes a one-cell notebook (or marimo app, served with
``marimo run``), starts the server, runs the cell in Chromium and drives the real jupyter_rfb <img> with real pointer events. The kernel
side writes a status JSON (rfb stats, crosshair pixel position, view range) every
250 ms; nothing is typed into notebook cells.

Measured in the page (performance.now()):
  * step latency: one pointer move during a drag, idle before and after
    -> time until the next frame <img> has loaded and an animation frame has run.
  * continuous drag (~100 Hz pointer moves, 2 s and 8 s): delivered FPS and
    "settle" = last pointer move -> last JPEG frame. Growing settle time with drag
    length would indicate accumulating lag. The final crosshair position (kernel)
    is compared with the final pointer position.
  * wheel zoom and wrapper resize: a new frame arrives and the kernel-side
    view range / widget size changes.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent

CELL = """\
import asyncio, json, os, sys
sys.path.insert(0, {here!r})
from common import make_dataarray, SIZES
from bench_headless import target_pos_disp2d, target_pos_standalone
if {kind!r} == "A":
    from standalone_rfb import CutViewerRFB
    view = CutViewerRFB(size={size!r})
    def _status_extra():
        x, y = target_pos_standalone(view.gfxView, view.v)
        return dict(xh_px=[x, y], view_range=view.v.image_plot.getViewBox().viewRange(),
                    widget_size=list(view.logical_size))
else:
    from qwidget_rfb import disp_rfb
    view = disp_rfb(make_dataarray(*SIZES[{size!r}]))
    def _status_extra():
        w = view.widget
        x, y = target_pos_disp2d(w)
        return dict(xh_px=[x / view._scale[0], y / view._scale[1]],
                    view_range=w.image_plot.getViewBox().viewRange(),
                    widget_size=[w.width(), w.height()])

import collections
_counts = collections.Counter()
_orig_handle = view.handle_event
def _counting_handle(event):
    _counts[event["event_type"]] += 1
    _orig_handle(event)
view.handle_event = _counting_handle

async def _status_loop():
    while True:
        if os.path.exists("reset"):
            os.remove("reset")
            view.reset_stats(); view.event_times.clear(); view.frame_times.clear()
        with open("status.tmp", "w") as f:
            json.dump(dict(view.timing_summary(), event_counts=_counts, **_status_extra()), f)
        os.replace("status.tmp", "status.json")
        await asyncio.sleep(0.25)

_task = asyncio.ensure_future(_status_loop())
view
"""

FIND_JS = """() => {
  window.__find = (sel) => {
    const walk = (root) => {
      const f = root.querySelector(sel);
      if (f) return f;
      for (const el of root.querySelectorAll('*')) {
        if (el.shadowRoot) { const r = walk(el.shadowRoot); if (r) return r; }
      }
      return null;
    };
    return walk(document);
  };
  const i = window.__find('img[decoding="sync"]');
  return !!i && i.naturalWidth > 10;
}"""

PAGE_JS = """() => {
  const img = __find('img[decoding="sync"]');
  window.__loads = []; window.__moves = [];
  img.addEventListener('load', () => {
    const t = performance.now();
    requestAnimationFrame(() => __loads.push([t, performance.now()]));
  });
  img.addEventListener('pointermove', () => __moves.push(performance.now()), true);
  window.__wheels = [];
  img.addEventListener('wheel', () => { const a = document.activeElement; __wheels.push(a.tagName + '.' + a.className); }, true);
  return true;
}"""


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def status(workdir):
    return json.loads((workdir / "status.json").read_text())


def wait_status(workdir, pred, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            s = status(workdir)
            if pred(s):
                return s
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(0.1)
    raise TimeoutError("status condition not met")


def marimo_app(cell):
    body = cell.rstrip().removesuffix("view") + "_w = mo.ui.anywidget(view)\n_w\n"
    body = "import marimo as mo\n" + body
    return (
        "import marimo\n\napp = marimo.App()\n\n\n@app.cell\ndef _():\n"
        + "".join("    " + ln + "\n" if ln else "\n" for ln in body.splitlines())
        + "    return\n"
    )


def run_config(kind, size, dpr, frontend="jupyter"):
    workdir = Path(tempfile.mkdtemp(prefix=f"rfb_{kind}_{size}_"))
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": CELL.format(here=str(HERE), kind=kind, size=size),
            }
        ],
        "metadata": {
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3",
                "language": "python",
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (workdir / "bench.ipynb").write_text(json.dumps(nb))
    (workdir / "bench_app.py").write_text(
        marimo_app(CELL.format(here=str(HERE), kind=kind, size=size))
    )
    port, token = free_port(), "bench"
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    if frontend == "jupyter":
        cmd = [
            sys.executable,
            "-m",
            "jupyter",
            "lab",
            "--no-browser",
            f"--port={port}",
            f"--IdentityProvider.token={token}",
            f"--ServerApp.root_dir={workdir}",
        ]
        url = f"http://127.0.0.1:{port}/lab/tree/bench.ipynb?token={token}"
    else:  # app mode runs all cells on load; one session = one kernel thread
        cmd = [
            sys.executable,
            "-m",
            "marimo",
            "run",
            "--headless",
            "--no-token",
            "-p",
            str(port),
            "bench_app.py",
        ]
        url = f"http://127.0.0.1:{port}/"
    server = subprocess.Popen(
        cmd, cwd=workdir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    out = {"config": f"{kind}-{size}", "dpr": dpr, "frontend": frontend}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(
                viewport={"width": 1400, "height": 1100}, device_scale_factor=dpr
            )
            for _ in range(60):
                try:
                    page.goto(url)
                    break
                except Exception:
                    time.sleep(0.5)
            if frontend == "jupyter":
                page.wait_for_selector(".jp-Notebook .jp-Cell", timeout=60000)
                page.wait_for_selector(
                    ".jp-Notebook-ExecutionIndicator[data-status='idle']",
                    timeout=60000,
                    state="attached",
                )
                page.locator(".jp-Cell").first.click()
                page.keyboard.press("Shift+Enter")
            img = page.locator('img[decoding="sync"]')
            img.wait_for(timeout=60000)
            page.wait_for_function(FIND_JS, timeout=60000)
            img.scroll_into_view_if_needed()
            time.sleep(1.5)
            page.evaluate(PAGE_JS)
            box = img.bounding_box()
            s = wait_status(workdir, lambda s: s["rfb_stats"]["sent_frames"] > 0)
            out["framebuffer_css"] = [box["width"], box["height"]]
            out["widget_size"] = s["widget_size"]

            def to_page(px):
                return box["x"] + px[0], box["y"] + px[1]

            # ---- step latency (drag held, one move, then idle) ----
            x, y = to_page(s["xh_px"])
            page.mouse.move(x, y)
            page.mouse.down()
            time.sleep(0.6)
            latencies = []
            for i in range(20):
                page.evaluate("() => { __loads.length = 0; __moves.length = 0; }")
                page.mouse.move(x - 3 * (i + 1), y - 2 * (i + 1))
                time.sleep(0.6)
                moves, loads = page.evaluate("() => [__moves, __loads]")
                after = [ld for ld in loads if moves and ld[0] >= moves[0]]
                if moves and after:
                    latencies.append([after[0][0] - moves[0], after[0][1] - moves[0]])
            lat = sorted(latencies, key=lambda a: a[1])
            out["step_latency_ms"] = {
                "n": len(lat),
                "median_to_img_load": round(sorted(a[0] for a in lat)[len(lat) // 2], 1),
                "median_to_next_rAF": round(lat[len(lat) // 2][1], 1),
                "max_to_next_rAF": round(lat[-1][1], 1),
            }
            page.mouse.up()
            time.sleep(0.6)

            # ---- continuous drag ----
            for duration in (2.0, 8.0):
                s = status(workdir)
                x, y = to_page(s["xh_px"])
                page.mouse.move(x, y)
                page.mouse.down()
                time.sleep(0.4)
                (workdir / "reset").touch()
                time.sleep(0.4)
                page.evaluate("() => { __loads.length = 0; __moves.length = 0; }")
                t0 = time.perf_counter()
                i = 0
                while time.perf_counter() - t0 < duration:
                    i += 1
                    ph = (i % 200) / 100  # triangle wave, +-40 px around the start
                    d = 40 * (2 * ph - 1 if ph <= 1 else 3 - 2 * ph)
                    page.mouse.move(x - d, y - 0.5 * d)
                    time.sleep(0.008)
                end_px = (x - d - box["x"], y - 0.5 * d - box["y"])
                time.sleep(1.0)
                moves, loads = page.evaluate("() => [__moves, __loads]")
                page.mouse.up()
                s = wait_status(workdir, lambda s: True)
                # jupyter_rfb's lossless PNG redraw comes >= 300 ms after the last
                # JPEG, so frames within 290 ms of the last move are motion frames.
                jpeg = [ld for ld in loads if ld[1] <= moves[-1] + 290]
                t_first, t_last = moves[0], moves[-1]
                during = [ld for ld in jpeg if t_first <= ld[1] <= t_last]
                gaps = [b[1] - a[1] for a, b in zip(during, during[1:], strict=False)]
                out[f"drag_{duration:.0f}s"] = {
                    "pointer_moves_seen": len(moves),
                    "pointer_rate_hz": round(len(moves) / ((t_last - t_first) / 1e3), 1),
                    "frames_during_drag": len(during),
                    "fps_browser": round(len(during) / ((t_last - t_first) / 1e3), 1),
                    "frame_gap_p95_ms": round(sorted(gaps)[int(0.95 * len(gaps))], 1)
                    if gaps
                    else None,
                    "settle_ms": round(max(ld[1] for ld in jpeg) - t_last, 1)
                    if jpeg
                    else None,
                    "final_xh_error_px": round(
                        max(
                            abs(s["xh_px"][0] - end_px[0]),
                            abs(s["xh_px"][1] - end_px[1]),
                        ),
                        1,
                    ),
                    "rfb_stats": {k: round(v, 4) for k, v in s["rfb_stats"].items()},
                    "server": {k: s[k] for k in ("handle_event", "get_frame")},
                }
                time.sleep(0.5)

            # ---- wheel zoom over the image plot ----
            s0 = status(workdir)
            x, y = to_page(s0["xh_px"])
            page.evaluate("() => { __loads.length = 0; }")
            page.mouse.click(
                x + 30, y + 30
            )  # jupyter_rfb only forwards wheel when focused
            for _ in range(5):
                page.mouse.wheel(0, -120)
                time.sleep(0.05)
            time.sleep(0.8)
            s1 = status(workdir)
            out["wheel_zoom"] = {
                "view_range_before": s0["view_range"],
                "view_range_after": s1["view_range"],
                "frames": len(page.evaluate("() => __loads")),
                "focus_at_wheel": page.evaluate("() => __wheels"),
                "kernel_events": s1["event_counts"],
            }

            # ---- resize the wrapper ----
            page.evaluate(
                "() => { __loads.length = 0; const w = __find('.renderview-wrapper'); w.style.width = '900px'; w.style.height = '820px'; }"
            )
            time.sleep(1.0)
            s2 = status(workdir)
            out["resize"] = {
                "widget_size_after": s2["widget_size"],
                "img_natural": page.evaluate(
                    "() => { const i = __find('img[decoding=\"sync\"]'); return [i.naturalWidth, i.naturalHeight]; }"
                ),
                "frames": len(page.evaluate("() => __loads")),
            }
            # kernel still responsive: status loop (an asyncio task in the kernel) kept running
            out["kernel_status_age_s"] = round(
                time.time() - (workdir / "status.json").stat().st_mtime, 2
            )
            page.screenshot(
                path=str(Path.cwd() / f"screenshot_{frontend}_{kind}_{size}.png")
            )
            browser.close()
    finally:
        out["workdir"] = str(workdir)
        server.terminate()
        server.wait(10)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="A-medium,A-large,B-medium,B-large")
    ap.add_argument("--dpr", type=float, default=1.0)
    ap.add_argument("--frontend", default="jupyter", choices=["jupyter", "marimo"])
    a = ap.parse_args()
    for cfg in a.configs.split(","):
        kind, size = cfg.split("-")
        print(json.dumps(run_config(kind, size, a.dpr, a.frontend)), flush=True)
