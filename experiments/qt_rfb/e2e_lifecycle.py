"""End-to-end check of inline viewer lifecycle in a real frontend (headless Chromium).

    python e2e_lifecycle.py [--frontend jupyter|marimo] [--shots DIR]

Uses the production backend (``pks.opts.gui.backend = "notebook"``). Two cells contain
identical ``data.disp()`` code. A probe cell writes the kernel-side registry to
``status.json``. Checks: same-cell replacement with restored state, identical code in
another cell not replaced, automatic suspension at ``max_viewers=1`` with a static image
left in the output, the Close and Suspend buttons, and (Jupyter) cell deletion.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

from bench_browser import free_port

SETUP = """\
import gc, json
import numpy as np
import peaks as pks
from common import make_dataarray
pks.opts.gui.backend = "notebook"
data = make_dataarray(300, 400)
data2 = make_dataarray(300, 400, seed=1)

def probe():
    from peaks.core.GUI.disp_panels.disp_2d import _Disp2D
    from peaks.core.GUI.GUI_utils.qt_runtime import _existing_state
    gc.collect()
    vs = pks.gui.viewers()
    out = dict(
        viewers=[dict(key=v.cell_key and v.cell_key[1], cell=v.cell_key and v.cell_key[0],
                      status=v.status,
                      state=v.widget.get_state() if v.widget is not None else None,
                      data=v.widget.current_data.name if v.widget is not None else None)
                 for v in vs],
        live_disp2d=sum(type(o) is _Disp2D for o in gc.get_objects()),
        saved_cells=sorted({k[0] for k in (_existing_state() or {}).get("cell_states", {})}),
    )
    with open("status.json", "w") as f:
        json.dump(out, f)
"""
CELL_A = "data.disp()"
CELL_B = "data.disp()"
PROBE = "probe()"
MUTATE = """\
_w = [v for v in pks.gui.viewers() if v.status == "active"][0].widget
_w.xhs[0].set_pos((-0.5, 5.0))
_w.show_DCs_checkboxes[2].setChecked(True)
_w.colorbar.setLevels(values=(10, 150))
del _w
"""

FIND = """(sel) => {
  const walk = (root, out) => {
    out.push(...root.querySelectorAll(sel));
    for (const el of root.querySelectorAll('*')) if (el.shadowRoot) walk(el.shadowRoot, out);
    return out;
  };
  return walk(document, []).filter(e => e.offsetParent !== null).length;
}"""


def check(cond, msg, results):
    results.append((bool(cond), msg))
    print(("PASS " if cond else "FAIL ") + msg, flush=True)


class Jupyter:
    def __init__(self, page, workdir):
        self.page, self.workdir = page, workdir

    def cell(self, i):
        return self.page.locator(".jp-Notebook .jp-Cell").nth(i)

    def run(self, i, wait=True):
        self.cell(i).locator(".jp-InputArea-editor").click()
        self.page.keyboard.press("Control+Enter")
        if wait:
            self.page.wait_for_selector(
                ".jp-Notebook-ExecutionIndicator[data-status='idle']",
                state="attached",
                timeout=60000,
            )
            time.sleep(1.0)

    def output(self, i):
        return self.cell(i).locator(".jp-OutputArea").first

    def delete(self, i):
        self.cell(i).locator(".jp-InputPrompt").click()
        self.page.keyboard.press("Escape")
        self.page.keyboard.press("d")
        self.page.keyboard.press("d")
        time.sleep(0.5)


class Marimo:
    def __init__(self, page, workdir):
        self.page, self.workdir = page, workdir
        self.names = {}

    def cell(self, i):
        return self.page.locator(f"[data-cell-name='{self.names[i]}']")

    def run(self, i, wait=True):
        cell = self.cell(i)
        cell.locator(".cm-content").click()
        self.page.keyboard.press("Control+Enter")
        if wait:
            time.sleep(2.5)

    def output(self, i):
        return self.cell(i).locator(".output-area").first

    def delete(self, i):
        raise NotImplementedError


def probe(fe, i_probe):
    path = fe.workdir / "status.json"
    path.unlink(missing_ok=True)
    fe.run(i_probe)
    for _ in range(100):
        if path.exists():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                pass
        time.sleep(0.1)
    fe.page.screenshot(path=str(fe.workdir / "timeout.png"), full_page=True)
    raise TimeoutError(f"probe did not write status.json; see {fe.workdir}/timeout.png")


def visible_count(locator, sel):
    return locator.evaluate(
        """(root, sel) => {
      const walk = (r, out) => {
        out.push(...r.querySelectorAll(sel));
        for (const el of r.querySelectorAll('*')) if (el.shadowRoot) walk(el.shadowRoot, out);
        return out;
      };
      return walk(root, []).filter(e => e.offsetParent !== null).length;
    }""",
        sel,
    )


def click_button(locator, text):
    locator.locator(f"button:text-is('{text}')").first.click()
    time.sleep(1.5)


def marimo_app(cells):
    src = "import marimo\n\napp = marimo.App()\n\n"
    for name, code in cells:
        body = "".join(f"    {ln}\n" for ln in code.splitlines())
        src += f"\n@app.cell\ndef {name}():\n{body}    return\n\n"
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontend", default="jupyter", choices=["jupyter", "marimo"])
    ap.add_argument("--shots", default=None)
    ap.add_argument(
        "--coexist",
        action="store_true",
        help="Jupyter only, run under xvfb-run: a native window and an inline viewer",
    )
    args = ap.parse_args()
    here = Path(__file__).resolve().parent
    shots = Path(args.shots) if args.shots else None
    workdir = Path(tempfile.mkdtemp(prefix="rfb_e2e_"))
    setup = f"import sys\nsys.path.insert(0, {str(here)!r})\n" + SETUP
    cells = [setup, CELL_A, CELL_B, PROBE, MUTATE]
    if args.coexist:
        cells[2] = "pks.opts.gui.max_viewers = 2\ndata2.disp(backend='qt')"
        cells[4] = "pks.gui.close_all()"
    port = free_port()
    env = dict(os.environ)
    if not args.coexist:
        env["QT_QPA_PLATFORM"] = "offscreen"
    if args.frontend == "jupyter":
        nb = {
            "cells": [
                {
                    "cell_type": "code",
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                    "source": c,
                }
                for c in cells
            ],
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        (workdir / "nb.ipynb").write_text(json.dumps(nb))
        cmd = [
            sys.executable,
            "-m",
            "jupyter",
            "lab",
            "--no-browser",
            f"--port={port}",
            "--IdentityProvider.token=t",
            f"--ServerApp.root_dir={workdir}",
        ]
        url = f"http://127.0.0.1:{port}/lab/tree/nb.ipynb?token=t"
    else:
        # mutate/probe are only run manually: make them not depend on A/B outputs
        names = ["setup", "cell_a", "cell_b", "probe_cell", "mutate"]
        marimo_cells = list(
            zip(
                names,
                [
                    setup,
                    "_v = " + CELL_A + "\n_v",
                    "_v = " + CELL_B + "\n_v",
                    PROBE,
                    MUTATE,
                ],
                strict=True,
            )
        )
        (workdir / "app.py").write_text(marimo_app(marimo_cells))
        cmd = [
            sys.executable,
            "-m",
            "marimo",
            "edit",
            "--headless",
            "--no-token",
            "-p",
            str(port),
            "app.py",
        ]
        url = f"http://127.0.0.1:{port}/"
    server = subprocess.Popen(
        cmd,
        cwd=workdir,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=open(workdir / "server.log", "w"),
    )
    results = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1400, "height": 1300})
            for _ in range(60):
                try:
                    page.goto(url)
                    break
                except Exception:
                    time.sleep(0.5)
            if args.frontend == "jupyter":
                fe = Jupyter(page, workdir)
                page.wait_for_selector(
                    ".jp-Notebook-ExecutionIndicator[data-status='idle']",
                    state="attached",
                    timeout=60000,
                )
                fe.run(0)
                fe.run(1)
            else:
                fe = Marimo(page, workdir)
                fe.names = dict(enumerate(names))
                page.wait_for_selector("[data-cell-name='cell_a']", timeout=60000)
                time.sleep(2)
                fe.run(0)  # may also run the dependent cells A and B
                time.sleep(4)
                fe.run(1)
            A, B, PROBE_I, MUT = 1, 2, 3, 4
            if args.coexist:
                coexist(fe, results, shots)
                browser.close()
                return finish(results)

            s = probe(fe, PROBE_I)
            active = [v for v in s["viewers"] if v["status"] == "active"]
            check(
                len(active) == 1
                and visible_count(fe.output(A), 'img[decoding="sync"]') == 1,
                "cell A shows one live viewer",
                results,
            )
            # marimo runs A then B on open, so A may already be suspended there
            fe.run(A)
            fe.run(MUT)
            s = probe(fe, PROBE_I)
            before = [v for v in s["viewers"] if v["status"] == "active"][0]["state"]
            check(before["visible"] == [True, True, True], "state mutated", results)

            # Re-run the same cell: replaced, state restored
            fe.run(A)
            s = probe(fe, PROBE_I)
            vs = s["viewers"]
            a_views = [
                v
                for v in vs
                if v["cell"] == [x for x in vs if x["status"] == "active"][0]["cell"]
            ]
            after = [v for v in vs if v["status"] == "active"][0]["state"]
            check(
                len(a_views) == 1 and s["live_disp2d"] == 1,
                f"re-run replaces the viewer (viewers={len(vs)}, live _Disp2D={s['live_disp2d']})",
                results,
            )
            check(
                after["visible"] == [True, True, True]
                and abs(after["crosshairs"][0][0] + 0.5) < 0.01
                and abs(after["levels"][0] - 10) < 1e-6,
                "re-run restores crosshair/visibility/levels",
                results,
            )
            check(
                visible_count(fe.output(A), 'img[decoding="sync"]') == 1,
                "re-run leaves exactly one live view in cell A",
                results,
            )
            if shots:
                fe.output(A).screenshot(path=str(shots / f"{args.frontend}_live.png"))

            # Identical code in another cell: new viewer, A suspended (limit 1)
            fe.run(B)
            s = probe(fe, PROBE_I)
            st = sorted((v["status"] for v in s["viewers"]))
            check(
                st == ["active", "suspended"] and s["live_disp2d"] == 1,
                f"cell B opens; cell A suspended (statuses {st})",
                results,
            )
            check(
                visible_count(fe.output(A), 'img[src^="data:image/png"]') == 1
                and visible_count(fe.output(A), 'img[decoding="sync"]') == 0,
                "cell A output shows the static image",
                results,
            )
            if shots:
                fe.output(A).screenshot(
                    path=str(shots / f"{args.frontend}_suspended.png")
                )

            # Re-run suspended A: interactive again, B suspended
            fe.run(A)
            s = probe(fe, PROBE_I)
            act = [v for v in s["viewers"] if v["status"] == "active"]
            check(
                len(s["viewers"]) == 2
                and len(act) == 1
                and act[0]["state"]["visible"] == [True, True, True],
                "re-running a suspended cell restores it (and suspends B)",
                results,
            )

            # Close button on B (suspended)
            click_button(fe.output(B), "Close")
            s = probe(fe, PROBE_I)
            check(
                len(s["viewers"]) == 1,
                "Close button removes B from the registry",
                results,
            )
            check(
                visible_count(fe.output(B), "img") == 0
                and visible_count(fe.output(B), "button") == 0,
                "Close button clears B's output",
                results,
            )
            if shots:
                fe.cell(B).screenshot(path=str(shots / f"{args.frontend}_closed.png"))

            # Suspend button on A
            click_button(fe.output(A), "Suspend")
            s = probe(fe, PROBE_I)
            check(
                [v["status"] for v in s["viewers"]] == ["suspended"]
                and s["live_disp2d"] == 0,
                "Suspend button releases the Qt viewer",
                results,
            )
            check(
                visible_count(fe.output(A), 'img[src^="data:image/png"]') == 1,
                "Suspend button leaves the static image",
                results,
            )

            if args.frontend == "jupyter":
                fe.delete(A)
                s = probe(fe, PROBE_I - 1)
                check(
                    s["viewers"] == [] and s["saved_cells"] == [],
                    f"deleting cell A releases its viewer and state ({s['saved_cells']})",
                    results,
                )
            browser.close()
    finally:
        server.terminate()
        server.wait(10)
    finish(results)


def finish(results):
    failed = [m for ok, m in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    sys.exit(1 if failed else 0)


def mapped_windows():
    out = subprocess.run(["xwininfo", "-root", "-tree"], capture_output=True, text=True)
    names = [ln for ln in out.stdout.splitlines() if "Display Panel" in ln]
    viewable = []
    for ln in names:
        wid = ln.split()[0]
        info = subprocess.run(["xwininfo", "-id", wid], capture_output=True, text=True)
        viewable.append("IsViewable" in info.stdout)
    return viewable


def coexist(fe, results, shots):
    """Inline viewer, then a native window in the same kernel (%gui qt6)."""
    A, NATIVE, PROBE_I, CLOSE_ALL = 1, 2, 3, 4
    img = fe.output(A).locator('img[decoding="sync"]').first
    fe.run(NATIVE)
    s = probe(fe, PROBE_I)
    check(
        [v["status"] for v in s["viewers"]] == ["active", "active"],
        "inline + native both live with max_viewers=2",
        results,
    )
    check(
        mapped_windows().count(True) == 1,
        f"only the native window is mapped (xwininfo viewable: {mapped_windows()})",
        results,
    )
    before = [v for v in s["viewers"] if v["cell"]][0]["state"]["view_range"]
    img.scroll_into_view_if_needed()
    box = img.bounding_box()
    x, y = box["x"] + box["width"] * 0.4, box["y"] + box["height"] * 0.35
    fe.page.mouse.move(x, y)
    fe.page.mouse.down()
    for i in range(1, 11):
        fe.page.mouse.move(x + 6 * i, y + 4 * i)
        time.sleep(0.03)
    fe.page.mouse.up()
    time.sleep(1)
    s = probe(fe, PROBE_I)
    after = [v for v in s["viewers"] if v["cell"]][0]["state"]["view_range"]
    check(after != before, "inline viewer still responds to browser drags", results)
    if shots:
        fe.output(A).screenshot(path=str(shots / "coexist_inline.png"))
    fe.run(CLOSE_ALL)
    s = probe(fe, PROBE_I)
    check(
        s["viewers"] == [] and mapped_windows().count(True) == 0,
        "close_all() closes the native window and the inline viewer",
        results,
    )


if __name__ == "__main__":
    main()
