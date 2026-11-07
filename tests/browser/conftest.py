"""Drive inline viewers in real JupyterLab and marimo sessions with headless Chromium.

These tests are opt-in: run them with ``pytest -m browser``. They need the
``browser-tests`` extra and a Playwright browser (``playwright install chromium``).

Each test writes a notebook (a setup cell, its own cells, and a probe cell), starts a
JupyterLab or ``marimo edit`` server for it, and drives it like a user. The probe cell
calls ``report(**state())``, where ``state()`` is defined by the test's setup code; the
kernel writes the values to ``status.json``, which the test reads.
"""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
# marimo does not allow star imports
HELPERS = "action_at, css, item_at, make_data, pks, report, viewer, widget_at"
IDLE = ".jp-Notebook-ExecutionIndicator[data-status='idle']"


def pytest_collection_modifyitems(config, items):
    """Skip browser tests unless they were selected with ``-m browser``."""
    if "browser" in (config.getoption("markexpr") or ""):
        return
    skip = pytest.mark.skip(reason="browser test: run with pytest -m browser")
    for item in items:
        if "browser" in item.keywords:
            item.add_marker(skip)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _marimo_app(cells):
    src = "import marimo\n\napp = marimo.App()\n"
    for name, code in cells:
        body = "".join(f"    {line}\n" for line in code.splitlines())
        src += f"\n\n@app.cell\ndef {name}():\n{body}    return\n"
    return src


class Notebook:
    """A notebook open in the browser. Cell 0 is the setup cell, the last the probe."""

    def __init__(self, frontend, page, workdir, n_cells):
        self.frontend, self.page, self.workdir = frontend, page, workdir
        self.probe_index = n_cells - 1

    def cell(self, i):
        if self.frontend == "jupyter":
            return self.page.locator(".jp-Notebook .jp-Cell").nth(i)
        return self.page.locator(f"[data-cell-name='c{i}']")

    def output(self, i):
        if self.frontend == "jupyter":
            return self.cell(i).locator(".jp-OutputArea").first
        return self.cell(i).locator(".output-area").first

    def run(self, i):
        """Run cell ``i`` and wait until the kernel is idle."""
        if self.frontend == "jupyter":
            self.cell(i).locator(".jp-InputArea-editor").click()
            self.page.keyboard.press("Control+Enter")
            time.sleep(0.5)
            self.page.wait_for_selector(IDLE, state="attached", timeout=180000)
            time.sleep(0.5)
        else:
            self.cell(i).locator(".cm-content").click()
            self.page.keyboard.press("Control+Enter")
            time.sleep(2.5)

    def probe(self):
        """Run the probe cell and return what it reported."""
        path = self.workdir / "status.json"
        path.unlink(missing_ok=True)
        self.run(self.probe_index)
        for _ in range(300):
            if path.exists():
                try:
                    return json.loads(path.read_text())
                except json.JSONDecodeError:
                    pass
            time.sleep(0.1)
        self.page.screenshot(path=str(self.workdir / "timeout.png"), full_page=True)
        raise TimeoutError(f"no status.json; see {self.workdir / 'timeout.png'}")

    def canvas(self, i):
        """Bounding box of the live viewer in cell ``i``."""
        img = self.output(i).locator('img[decoding="sync"]').first
        img.scroll_into_view_if_needed()
        return img.bounding_box()

    def point(self, i, xy):
        """Page position of ``xy`` (css px on the viewer's canvas in cell ``i``)."""
        box = self.canvas(i)
        return box["x"] + xy[0], box["y"] + xy[1]

    def click(self, i, xy, button="left", count=1):
        self.page.mouse.click(*self.point(i, xy), button=button, click_count=count)
        time.sleep(0.8)

    def hover(self, i, xy, wait=1.0):
        self.page.mouse.move(*self.point(i, xy), steps=5)
        time.sleep(wait)

    def drag(self, i, xy, delta, steps=10, pause=0.03):
        x, y = self.point(i, xy)
        self.page.mouse.move(x, y)
        self.page.mouse.down()
        for k in range(1, steps + 1):
            self.page.mouse.move(x + delta[0] * k / steps, y + delta[1] * k / steps)
            time.sleep(pause)
        self.page.mouse.up()

    def button(self, i, text):
        self.output(i).locator(f"button:text-is('{text}')").first.click()
        time.sleep(1.5)

    def visible_count(self, i, selector):
        """Visible elements matching ``selector`` in cell ``i``'s output."""
        return self.output(i).evaluate(
            """(root, sel) => {
              const walk = (r, out) => {
                out.push(...r.querySelectorAll(sel));
                for (const el of r.querySelectorAll('*')) {
                  if (el.shadowRoot) walk(el.shadowRoot, out)
                }
                return out
              }
              return walk(root, []).filter(e => e.offsetParent !== null).length
            }""",
            selector,
        )

    def delete(self, i):
        """Delete cell ``i`` (JupyterLab)."""
        self.cell(i).locator(".jp-InputPrompt").click()
        self.page.keyboard.press("Escape")
        self.page.keyboard.press("d")
        self.page.keyboard.press("d")
        time.sleep(0.5)
        self.probe_index -= 1


@pytest.fixture(scope="session")
def chromium():
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as p:
        browser = p.chromium.launch()
        yield browser
        browser.close()


@pytest.fixture
def notebook(chromium, tmp_path):
    """Open a notebook: ``notebook(frontend, setup, cells, viewport=(w, h))``.

    ``setup`` is appended to the setup cell, which imports :mod:`kernel_helpers`.
    The setup cell is run; the test runs its own cells.
    """
    servers, pages = [], []

    def open_notebook(frontend, setup, cells, viewport=(1500, 1300), qt_platform=None):
        pytest.importorskip("jupyter_rfb")
        pytest.importorskip("marimo" if frontend == "marimo" else "jupyterlab")
        setup = (
            f"import sys\nsys.path.insert(0, {str(HERE)!r})\n"
            f"from kernel_helpers import {HELPERS}\n{setup}"
        )
        sources = [setup, *cells, "report(**state())"]
        port = _free_port()
        if frontend == "jupyter":
            nb = {
                "cells": [
                    {
                        "cell_type": "code",
                        "metadata": {},
                        "execution_count": None,
                        "outputs": [],
                        "source": src,
                    }
                    for src in sources
                ],
                "metadata": {
                    "kernelspec": {"name": "python3", "display_name": "Python 3"}
                },
                "nbformat": 4,
                "nbformat_minor": 5,
            }
            (tmp_path / "nb.ipynb").write_text(json.dumps(nb))
            cmd = [
                sys.executable,
                "-m",
                "jupyter",
                "lab",
                "--no-browser",
                f"--port={port}",
                "--IdentityProvider.token=t",
                f"--ServerApp.root_dir={tmp_path}",
            ]
            url = f"http://127.0.0.1:{port}/lab/tree/nb.ipynb?token=t"
        else:
            named = [(f"c{i}", src) for i, src in enumerate(sources)]
            (tmp_path / "app.py").write_text(_marimo_app(named))
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
        env = dict(os.environ, QT_QPA_PLATFORM=qt_platform or "offscreen")
        if qt_platform == "xcb":
            env.pop("QT_QPA_PLATFORM")
        log = open(tmp_path / "server.log", "w")  # noqa: SIM115 (closed with the server)
        server = subprocess.Popen(
            cmd, cwd=tmp_path, env=env, stdout=subprocess.DEVNULL, stderr=log
        )
        servers.append((server, log))
        page = chromium.new_page(viewport={"width": viewport[0], "height": viewport[1]})
        pages.append(page)
        for _ in range(120):
            try:
                page.goto(url)
                break
            except Exception:
                time.sleep(0.5)
        nb = Notebook(frontend, page, tmp_path, len(sources))
        if frontend == "jupyter":
            page.wait_for_selector(IDLE, state="attached", timeout=120000)
            nb.run(0)
        else:
            page.wait_for_selector("[data-cell-name='c0']", timeout=120000)
            time.sleep(2)
            nb.run(0)  # marimo also runs the cells that depend on it
            time.sleep(4)
        return nb

    yield open_notebook
    for page in pages:
        page.close()
    for server, log in servers:
        server.terminate()
        try:
            server.wait(10)
        except subprocess.TimeoutExpired:
            server.kill()
        log.close()
