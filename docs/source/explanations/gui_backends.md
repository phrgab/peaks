(gui-backends)=
# Interactive viewers in notebooks

The interactive display panels (`data.disp()`) are PyQtGraph/Qt GUIs. They can be shown
in two ways:

| Backend | Where the viewer appears | Default |
|---|---|---|
| `"qt"` | A native pop-out window | Yes |
| `"notebook"` | Inline, in the JupyterLab or marimo cell output | No |

Both use the same viewers, with the same crosshairs, cuts, ROIs and keyboard controls.

Both run in the Python kernel. This is different from `data.iplot()`, which makes a
Bokeh plot drawn by the browser. Such plots are kept when a notebook is saved or
exported, but have none of the display panels' tools. See
{ref}`quick-interactive-plots`.

## Native windows (default)

```python
import peaks as pks

data.disp()
```

In Jupyter or IPython, Qt event-loop integration (`%gui qt6`) is enabled automatically.
In a script, `disp()` blocks until the window is closed.

## Inline notebook viewers

Install the optional dependencies once:

```bash
pip install "peaks-arpes[notebook]"
```

Then, in JupyterLab or marimo:

```python
pks.opts.gui.backend = "notebook"
data.disp()
```

or for a single call, `data.disp(backend="notebook")`. The viewer is the value returned
by `disp()`. Leave it as the last expression of the cell (or `viewer = data.disp()`
followed by `viewer`) to display it. In marimo there is no need to wrap it in
`mo.ui.anywidget`.

Click the viewer once before using the keyboard or the mouse wheel. The browser only
forwards those events to a focused widget. On a Mac, use Cmd where the viewers use
Control (e.g. Cmd + arrow keys to move crosshair 1), as in a native window.

Each inline viewer has buttons above it:

* **Suspend** keeps a static image of the viewer in the notebook and releases the Qt
  viewer and its memory. Re-run the cell to make it interactive again.
* **Close** releases everything and clears the output.
* **Help** shows the keyboard shortcuts (2D and 3D viewers), which native windows show
  from their menu bar.

The same is available from Python:

```python
viewer = data.disp(backend="notebook")
viewer.suspend()
viewer.close()

pks.gui.suspend_all()  # all live inline viewers; native windows are not affected
pks.gui.close_all()    # every peaks viewer, including native windows
pks.gui.viewers()      # open and suspended viewers, oldest first
```

`suspend()` and `close()` can be called repeatedly. Native windows cannot be suspended.

## The viewer limit

`pks.opts.gui.max_viewers` (default 1, `None` for no limit) is one limit on the number
of **live** Qt viewers, native and inline together. Suspended viewers do not count.

When a new viewer would exceed the limit, the oldest live inline viewer is suspended to
make room. Native windows are never suspended or closed automatically. If no inline
viewer can be suspended, `TooManyDisplayPanelsError` is raised.

| New viewer | Viewer already open (limit reached) | Result |
|---|---|---|
| inline | inline | oldest inline viewer suspended, new one opens |
| native | inline | oldest inline viewer suspended, new one opens |
| inline | native | `TooManyDisplayPanelsError` |
| native | native | `TooManyDisplayPanelsError` |

## Re-running a cell

Re-running a cell replaces the inline viewers it created before, even when the limit
would allow more. The old Qt viewer is destroyed first. The new viewer uses the newly
supplied data and restores the previous settings where they still fit:

* 2D: crosshair positions and visibility, integration widths, *All?* integration,
  mirror, axis ranges, colour map and levels, selected scan, lock options;
* 3D: cursor positions and widths, axis ranges, colour map and levels, mirror,
  rotation and alignment aids;
* 4D: active tab, primary crosshair, both 2D panels, ROI polygons and the ROI list.

Settings are skipped if they do not fit the new data, for example a crosshair outside
the new axis range or data with different dimensions. Settings are kept in memory for
the current kernel session only. They are discarded when the viewer is closed with
**Close**, `close()` or `pks.gui.close_all()`, or when the cell is deleted.

Cells are identified by the notebook frontend:

* **JupyterLab** sends a cell id with each execution. This also tells peaks when a cell
  is deleted. Cells containing identical code are distinct.
* **marimo**: the cell id and its re-run/delete hooks come from marimo's runtime. This
  uses marimo internals, so a future marimo release may need an update in peaks.
* Frontends that send no cell id (for example plain IPython) get no automatic
  replacement. The viewer limit still applies.

## Headless servers

On a Linux machine with no display, the notebook backend starts Qt with its
`offscreen` platform. The Qt platform is fixed when the Qt application is created, so
native windows are then unavailable for the rest of the kernel session. Changing
`pks.opts.gui.backend` does not change this. To choose explicitly, set
`QT_QPA_PLATFORM` (e.g. `offscreen`) before the first viewer is created. On a desktop
machine, native windows and inline viewers can be used in the same kernel.

## Limitations

* Menus, right-click menus and dialogs are separate windows and are not shown inline.
  The menu bar and plot context menus are disabled in notebook mode. Any dialog that a
  viewer opens is closed automatically, and its message is shown in the notebook, so
  it cannot block the kernel. *Clear all ROIs* in the 4D viewer asks for confirmation,
  so it does nothing inline; remove ROIs individually instead.
* **Copy** buttons copy to the browser clipboard, and the copied text is also printed
  below the viewer.
* Qt cursor shapes and tooltips are not shown.
* About 30 frames per second while dragging. The browser widget (jupyter_rfb) forwards
  at most one pointer move per 20 ms. Higher-DPI screens send larger frames
  (about 240 kB per frame at 2× scaling), which matters for remote kernels.
* Suspended viewers are kept as images only while the kernel runs. After saving and
  reopening a notebook, the outputs are not restored.
* Several simultaneous `marimo run` sessions are not supported, because Qt must run on
  one thread. Single-session `marimo run` and `marimo edit` work.
* In JupyterLab, clearing a cell's output does not notify the kernel, so that viewer
  stays live until it is suspended by the limit, closed, or its cell is re-run.
