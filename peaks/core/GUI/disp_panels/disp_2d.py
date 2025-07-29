"""Functions for the 2D interactive display panel."""

import sys
from functools import partial

import numpy as np
import pint
import pyperclip
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from peaks.core.GUI.GUI_utils import (
    CircularListWidget,
    Crosshair,
    KeyPressGraphicsLayoutWidget,
)
from peaks.core.GUI.GUI_utils.cursor_stats import _parse_norm_emission_cursor_stats
from peaks.core.metadata.metadata_methods import display_metadata
from peaks.core.process.tools import estimate_sym_point, sym


def _disp_2d(data, primary_dim, exclude_from_centering):
    """Display a 2D interactive display panel.

    Parameters
    ------------
    data : list or xarray.DataArray
         Either a single 2D :class:`xarray.DataArray` or a list of 2D :class:`xarray.DataArray` objects.

    primary_dim : str
        The primary dimension for the viewer, will be shown on the y-axis.

    exclude_from_centering : str or tuple of str or list of str or None
        The dimension to exclude from centering. Default is 'eV'.

    """
    global app  # Ensure the QApplication instance is not garbage collected
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    viewer = _Disp2D(data, primary_dim, exclude_from_centering)
    viewer.show()
    app.exec()


class _Disp2D(QtWidgets.QMainWindow):
    def __init__(self, data, primary_dim, exclude_from_centering):
        super().__init__()

        # Parse data
        if primary_dim:  # Set correct primary axis
            data = [array.transpose(primary_dim, ...) for array in data]
        self.data_arrays = [array.compute().pint.dequantify() for array in data]

        # Initialize some parameters
        self.init = False
        self.connected_plot_signals = []

        # Crosshair options
        self.num_xhs = 3
        self.start_active_xhs = 2
        self.xh_width_store = [None, None]  # Store for (dim0, dim1) DC widths
        self.xh_pos_store = [
            [None for i in range(self.num_xhs)],
            [None for i in range(self.num_xhs)],
        ]  # Store for (dim0, dim1) DC positions
        self.xh_visible_store = []
        self.DC_pens = [(238, 119, 51), (51, 187, 238), (0, 153, 136), (204, 51, 153)]
        self.xh_brushes = [
            (238, 119, 51, 70),
            (51, 187, 238, 70),
            (0, 153, 136, 70),
            (204, 51, 153, 70),
        ]

        # Set keys for the keyboard control
        def _get_key(param):
            return getattr(QtCore.Qt.Key, f"Key_{param}")

        self.key_modifiers_characters = {
            "move_csr2": "Control",
            "move_all": "Shift",
            "hide_all": "Space",
        }
        self.key_modifiers = {
            k: _get_key(v) for k, v in self.key_modifiers_characters.items()
        }
        self.show_hide_csr_key_characters = [str(i) for i in range(self.num_xhs)]
        self.show_hide_csr_keys = [
            _get_key(key) for key in self.show_hide_csr_key_characters
        ]
        self.show_hide_mirror_key_character = "M"
        self.show_hide_mirror_key = _get_key(self.show_hide_mirror_key_character)
        self.move_csr1_key_enabled = False
        self.move_all_key_enabled = False

        # Set options for dims to exclude from centering
        if isinstance(exclude_from_centering, str):
            self.exclude_from_centering = [exclude_from_centering]
        elif isinstance(exclude_from_centering, (tuple, list)):
            self.exclude_from_centering = list(exclude_from_centering)
        else:
            self.exclude_from_centering = []

        # Set up the GUI
        self._init_UI()  # Initialize the layout
        self._change_data()  # Initialize with data
        # Connect file change signals
        self.scans_list.currentRowChanged.connect(self._change_data)

        # Ensure the application quits when the window is closed
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.closeEvent = self._close_application

    def _close_application(self, event):
        """Close the application when the window is closed."""
        self.graphics_layout.close()
        app.quit()

    # ##############################
    # GUI layout
    # ##############################
    def _init_UI(self):
        self.setWindowTitle("Display Panel")
        self.setGeometry(100, 100, 650, 770)

        # Main window layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        self._build_plot_layout(layout)  # Build the basic plot layout
        self._build_controls_layout(layout)  # Build the control panel layout
        self._build_menu()  # Add the menu

    def _build_plot_layout(self, layout):
        # Create a GraphicsLayoutWidget
        self.graphics_layout = (
            KeyPressGraphicsLayoutWidget._KeyPressGraphicsLayoutWidget()
        )
        self.graphics_layout.viewport().setAttribute(
            QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, False
        )
        layout.addWidget(self.graphics_layout)

        self.image_plot = self.graphics_layout.addPlot(1, 0, 1, 1)
        self.DC_plots = []
        self.DC_plots.append(
            self.graphics_layout.addPlot(1, 1, 1, 1)
        )  # DC_plots[0] --> EDC for our conventional data order
        self.DC_plots[0].setYLink(self.image_plot)
        self.DC_plots[0].getAxis("left").hide()
        self.DC_plots[0].getAxis("right").show()
        self.DC_plots.append(
            self.graphics_layout.addPlot(2, 0, 1, 1)
        )  # DC_plots[1] --> MDC for our conventional data order
        self.DC_plots[1].setXLink(self.image_plot)
        self.graphics_layout.ci.layout.setRowStretchFactor(1, 3)
        self.graphics_layout.ci.layout.setColumnStretchFactor(0, 3)
        # Set margins to line up the plots
        self.DC_plots[0].getAxis("right").setWidth(60)
        self.DC_plots[1].getAxis("left").setWidth(40)
        self.image_plot.getAxis("left").setWidth(40)

    def _build_controls_layout(self, layout):
        # Left panel -------------------------------------
        bottom_panel_layout = QHBoxLayout()
        layout.addLayout(bottom_panel_layout)
        bottom_panel_layout_left = QVBoxLayout()
        bottom_panel_layout_left.setSpacing(5)
        bottom_panel_layout.addLayout(bottom_panel_layout_left)

        # Create a scroll area for cursor_stats
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedHeight(150)
        self.cursor_stats = QLabel()
        self.cursor_stats.setStyleSheet(
            "QLabel { background-color : black; color : white; }"
        )
        self.cursor_stats.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft
        )
        self.cursor_stats.setWordWrap(True)
        self.cursor_stats.setMaximumWidth(450)
        scroll_area.setWidget(self.cursor_stats)
        bottom_panel_layout_left.addWidget(scroll_area)

        # Cursor integration controls
        cursor_integrations = QHBoxLayout()
        bottom_panel_layout_left.addLayout(cursor_integrations)
        self.DC_groups = []
        self.DC_width_selectors = []
        self.DC_span_all_checkboxes = []
        DC_layouts = []
        for i in range(2):
            self.DC_groups.append(QtWidgets.QGroupBox(f"DC{i}"))
            self.DC_groups[i].setContentsMargins(5, 22, 5, 5)
            DC_layouts.append(QtWidgets.QHBoxLayout())
            DC_layouts[i].setContentsMargins(5, 0, 5, 0)
            self.DC_groups[i].setLayout(DC_layouts[i])
            cursor_integrations.addWidget(self.DC_groups[i])
            self.DC_width_selectors.append(QtWidgets.QDoubleSpinBox())
            self.DC_width_selectors[i].setDecimals(3)
            self.DC_width_selectors[i].setFixedWidth(75)
            DC_layouts[i].addWidget(self.DC_width_selectors[i])
            self.DC_span_all_checkboxes.append(QtWidgets.QCheckBox("All?"))
            DC_layouts[i].addWidget(self.DC_span_all_checkboxes[i])
        cursor_integrations.addStretch()

        # DC display controls
        DC_display_controls = QHBoxLayout()
        bottom_panel_layout_left.addLayout(DC_display_controls)
        show_DCs_group = QtWidgets.QGroupBox("Show DCs:")
        show_DCs_group.setContentsMargins(5, 22, 5, 5)
        show_DCs_layout = QtWidgets.QHBoxLayout()
        show_DCs_layout.setContentsMargins(5, 0, 5, 0)
        show_DCs_group.setLayout(show_DCs_layout)
        DC_display_controls.addWidget(show_DCs_group)
        self.show_DCs_checkboxes = []
        for i in range(self.num_xhs):
            self.show_DCs_checkboxes.append(
                QtWidgets.QCheckBox(
                    f"Csr{i}", checked=True if i < self.start_active_xhs else False
                )
            )
            show_DCs_layout.addWidget(self.show_DCs_checkboxes[i])
        self.show_mirror_checkbox = QtWidgets.QCheckBox("Mirror")
        show_DCs_layout.addWidget(self.show_mirror_checkbox)
        self.align_button = QtWidgets.QPushButton("Align")
        show_DCs_layout.addWidget(self.align_button)
        self.copy_button = QtWidgets.QPushButton("Copy")
        show_DCs_layout.addWidget(self.copy_button)

        DC_display_controls.addStretch()
        bottom_panel_layout_left.addStretch()

        # Right panel -------------------------------------
        bottom_panel_layout_right = QVBoxLayout()
        bottom_panel_layout_right.setSpacing(5)
        bottom_panel_layout.addLayout(bottom_panel_layout_right)

        # Scans list
        self.scans_list = CircularListWidget._CircularListWidget()
        self.scans_list.setFixedWidth(140)
        self.scans_list.setFixedHeight(180)
        self.scans_list.addItems([array.name for array in self.data_arrays])
        self.scans_list.setCurrentRow(0)
        bottom_panel_layout_right.addWidget(self.scans_list)

        # Add options for locking ranges on data cycle
        self.lock_opts = QtWidgets.QGroupBox("Lock?")
        self.lock_opts.setContentsMargins(5, 22, 5, 5)
        lock_opts_layout = QtWidgets.QHBoxLayout(self.lock_opts)
        lock_opts_layout.setContentsMargins(5, 0, 5, 0)
        self.cscale_lock = QtWidgets.QCheckBox("cScale")
        self.range_lock = QtWidgets.QCheckBox("Range", checked=True)
        lock_opts_layout.addWidget(self.cscale_lock)
        lock_opts_layout.addWidget(self.range_lock)
        bottom_panel_layout_right.addWidget(self.lock_opts)
        bottom_panel_layout_right.addStretch()

    def _build_menu(self):
        """Add a menu"""
        self.menu = self.menuBar().addMenu("Display Panel")
        self.shortcuts_action = QtGui.QAction("Help", self)

        self.help_text = f"""
                <table style='width:100%; border: 1px solid black; border-collapse: collapse;'>
                  <tr style='border: 1px solid black;'>
                    <th style='border: 1px solid black; padding: 5px; font-weight: normal;'>Action</th>
                    <th style='border: 1px solid black; padding: 5px; font-weight: normal;'>Key Binding</th>
                  </tr>
                  <tr>
                    <td style='padding: 5px; font-weight: normal;'>Move crosshair 0</td>
                    <td style='padding: 5px; font-weight: normal;'>Arrow keys</td>
                  </tr>
                  <tr>
                    <td style='padding: 5px; font-weight: normal;'>Move c_rosshair 1</td>
                    <td style='padding: 5px; font-weight: normal;'>{self.key_modifiers_characters["move_csr2"]} + Arrow keys</td>
                  </tr>
                  <tr>
                    <td style='padding: 5px; font-weight: normal;'>Move all crosshairs</td>
                    <td style='padding: 5px; font-weight: normal;'>{self.key_modifiers_characters["move_all"]} + Arrow keys</td>
                  </tr>
                  <tr>
                    <td style='padding: 5px; font-weight: normal;'>Hide all crosshairs</td>
                    <td style='padding: 5px; font-weight: normal;'>{self.key_modifiers_characters["hide_all"]}</td>
                  </tr>
                """

        for i in range(self.num_xhs):
            self.help_text += f"""
                  <tr>
                    <td style='padding: 5px; font-weight: normal;'>Enable/disable crosshair {i}</td>
                    <td style='padding: 5px; font-weight: normal;'>{self.show_hide_csr_key_characters[i]}</td>
                  </tr>
                """

        self.help_text += f"""
                  <tr>
                    <td style='padding: 5px; font-weight: normal;'>Enable/disable mirrored mode</td>
                    <td style='padding: 5px; font-weight: normal;'>{self.show_hide_mirror_key_character}</td>
                  </tr>
                </table>
                """

        self.shortcuts_action.triggered.connect(
            lambda: QtWidgets.QMessageBox.information(self, "Help", self.help_text)
        )
        self.menu.addAction(self.shortcuts_action)

    # ##############################
    # Data handling / plotting
    # ##############################
    def _set_data(self):
        """Set the current data array and extract some core parameters."""
        self.current_data = self.data_arrays[self.scans_list.currentRow()]
        self.dims = self.current_data.dims
        self.coords = [self.current_data.coords[dim].values for dim in self.dims]
        self.step_sizes = [coord[1] - coord[0] for coord in self.coords]
        self.ranges = [
            (min(coords) - self.step_sizes[i] / 2, max(coords) + self.step_sizes[i] / 2)
            for i, coords in enumerate(self.coords)
        ]
        self.data_span = [abs(range[1] - range[0]) for range in self.ranges]
        self.c_min = float(self.current_data.min())
        self.c_max = float(self.current_data.max())

        # Set the span range box options
        for i in range(2):
            self.DC_width_selectors[i].setRange(0, self.data_span[i])
            self.DC_width_selectors[i].setSingleStep(self.step_sizes[i])

        # Set centering dims
        self.centering_dims = [
            dim for dim in self.dims if dim not in self.exclude_from_centering
        ]

        # Read scan metadata
        self.metadata_text = "<span style='color:white'>"
        self.metadata_text += display_metadata(self.current_data, "html")
        self.metadata_text += "</span><br>"

    def _set_main_plot(self, cmap, c_range, xh_pos):
        # Set main image
        self.image_item = pg.ImageItem(self.current_data.values, axisOrder="row-major")
        self.image_plot.addItem(self.image_item)

        # Transform to match data coordinates
        self.image_item.setTransform(
            pg.QtGui.QTransform(
                self.step_sizes[1],
                0,
                0,
                0,
                self.step_sizes[0],
                0,
                self.ranges[1][0],
                self.ranges[0][0],
                1,
            )
        )

        # Add colour bar
        self.colorbar = pg.ColorBarItem(
            label=self.current_data.name,
            colorMap=cmap,
            colorMapMenu=True,
            limits=(self.c_min, self.c_max),
            rounding=min(abs(self.c_max - self.c_min) / 2000, 1),
            interactive=True,
            orientation="h",
            values=c_range,
        )
        self.colorbar.setImageItem(self.image_item, insert_in=self.image_plot)
        self.image_item.setColorMap(cmap)
        self.graphics_layout.addItem(self.colorbar, row=0, col=0)

        # Add crosshairs
        xh_target_kwargs = [
            {  # Crosshair1: orange
                "pen": pg.mkPen(color=(238, 119, 51)),
                "brush": pg.mkBrush(color=(238, 119, 51, 50)),
            },
            {  # Crosshair2: blue
                "pen": pg.mkPen(color=(51, 187, 238)),
                "brush": pg.mkBrush(color=(51, 187, 238, 50)),
            },
            {  # Crosshair3: green
                "pen": pg.mkPen(color=(0, 153, 136)),
                "brush": pg.mkBrush(color=(0, 153, 136, 50)),
            },
        ]
        xh_linear_region_kwargs = [
            {  # Crosshair1: orange
                "pen": pg.mkPen(color=(238, 119, 51)),
            },
            {  # Crosshair2: blue
                "pen": pg.mkPen(color=(51, 187, 238)),
            },
            {  # Crosshair3: green
                "pen": pg.mkPen(color=(0, 153, 136)),
            },
        ]
        self.xhs = []
        for i in range(self.num_xhs):
            self.xhs.append(
                Crosshair._Crosshair(
                    self.image_plot,
                    pos=xh_pos[i],
                    dim0_width=self.DC_width_selectors[0].value(),
                    dim1_width=self.DC_width_selectors[1].value(),
                    brush=self.xh_brushes[i % len(self.xh_brushes)],
                    bounds=[self.ranges[0], self.ranges[1]],
                    axisOrder="row-major",
                    target_item_kwargs=xh_target_kwargs[i % len(xh_target_kwargs)],
                    linear_region_item_kwargs=xh_linear_region_kwargs[
                        i % len(xh_linear_region_kwargs)
                    ],
                )
            )

            # Set visibility
            self.xhs[i].set_visible(self.show_DCs_checkboxes[i].isChecked())

    def _set_DC_plots(self):
        self.DC_plot_items = {}
        self.DC_plots_xhs = [[], []]
        for dim_no in range(2):
            dim = self.dims[dim_no]
            select_along_dim_no = (dim_no + 1) % 2
            self.DC_plot_items[dim_no] = []
            self.DC_plot_items[f"{dim_no}_m"] = []
            for i in range(self.num_xhs):
                DC = self._select_DC(
                    self.current_data,
                    select_along_dim_no,
                    (
                        self.xhs[i].get_dim1_span()
                        if select_along_dim_no == 1
                        else self.xhs[i].get_dim0_span()
                    ),
                )

                # Add to plots - get ordering correct
                if dim_no == 0:
                    a, b = DC.data, DC.coords[self.dims[dim_no]].values
                else:
                    a, b = DC.coords[self.dims[dim_no]].values, DC.data

                self.DC_plot_items[dim_no].append(
                    self.DC_plots[dim_no].plot(
                        a,
                        b,
                        pen=self.DC_pens[i % len(self.DC_pens)],
                    )
                )  # Main DC

                # Set visibility
                self.DC_plot_items[dim_no][i].setVisible(
                    self.show_DCs_checkboxes[i].isChecked()
                )

                # Add a mirror DC if dim in mirror group
                if dim in self.centering_dims:
                    try:
                        mirror_DC = sym(
                            DC, flipped=True, **{dim: self.xhs[i].get_pos()[dim_no]}
                        )
                        if dim_no == 0:
                            a, b = (
                                mirror_DC.data,
                                mirror_DC.coords[self.dims[dim_no]].values,
                            )
                        else:
                            a, b = (
                                mirror_DC.coords[self.dims[dim_no]].values,
                                mirror_DC.data,
                            )
                        self.DC_plot_items[f"{dim_no}_m"].append(
                            self.DC_plots[dim_no].plot(
                                a,
                                b,
                                pen=pg.mkPen(
                                    color=self.DC_pens[i % len(self.DC_pens)],
                                    style=QtCore.Qt.PenStyle.DashLine,
                                ),
                            )
                        )  # Mirrored DC
                        self.DC_plot_items[f"{dim_no}_m"][i].setVisible(
                            self.show_mirror_checkbox.isChecked()
                        )
                    except Exception:
                        pass

            # Add spans for crosshairs
            for i in range(self.num_xhs):
                self.DC_plots_xhs[dim_no].append(
                    pg.LinearRegionItem(
                        values=getattr(self.xhs[i], f"get_dim{dim_no}_span")(),
                        orientation="horizontal" if dim_no == 0 else "vertical",
                        pen=(0, 0, 0, 0),
                        brush=self.xh_brushes[i % len(self.xh_brushes)],
                        movable=False,
                    )
                )
                self.DC_plots[dim_no].addItem(self.DC_plots_xhs[dim_no][i])
                self.DC_plots_xhs[dim_no][i].setVisible(
                    self.show_DCs_checkboxes[i].isChecked()
                )  # Set visibility

        self.DC_plots[0].getViewBox().invertX(False)

    def _set_plot_ranges(self, previous_x_range, previous_y_range):
        # Get bounds of current data
        x_min, x_max = self.ranges[1]
        y_min, y_max = self.ranges[0]

        # Check if current view is within data limits
        if (
            previous_x_range[0] > x_min
            and previous_x_range[1] < x_max
            and self.range_lock.isChecked()
        ):
            x0, x1 = previous_x_range
        else:
            x0, x1 = x_min, x_max
        if (
            previous_y_range[0] > y_min
            and previous_y_range[1] < y_max
            and self.range_lock.isChecked()
        ):
            y0, y1 = previous_y_range
        else:
            y0, y1 = y_min, y_max

        # Set ranges
        self.DC_plots[1].getViewBox().setLimits(xMin=x_min, xMax=x_max)
        self.DC_plots[1].getViewBox().setXRange(x0, x1, padding=0.0)
        self.DC_plots[0].getViewBox().setLimits(yMin=y_min, yMax=y_max)
        self.DC_plots[0].getViewBox().setYRange(y0, y1, padding=0.0)
        self.image_plot.getViewBox().setLimits(
            xMin=x_min,
            xMax=x_max,
            yMin=y_min,
            yMax=y_max,
        )
        self.image_plot.getViewBox().setRange(
            xRange=(x0, x1), yRange=(y0, y1), padding=0.0
        )

    def _set_plot_labels(self):
        # Set labels
        for i in range(2):
            dim_units = self.current_data.coords[self.dims[i]].attrs.get("units")
            dim_label = f"{self.dims[i]} ({dim_units})" if dim_units else self.dims[i]
            self.DC_plots[i].setLabel("bottom" if i == 1 else "right", dim_label)
            self.DC_groups[i].setTitle(self.dims[i])

    def _select_DC(self, data, select_along_dim_no, span_range):
        """Select a DC from the data, handling averaging."""
        select_along_dim = self.dims[select_along_dim_no]
        if np.abs(span_range[1] - span_range[0]) < self.step_sizes[select_along_dim_no]:
            return data.sel({select_along_dim: np.mean(span_range)}, method="nearest")
        else:
            return data.sel(
                {select_along_dim: slice(span_range[0], span_range[1])}
            ).mean(select_along_dim)

    # ##############################
    # Data / plot updates
    # ##############################
    def _change_data(self):
        """Change the selected data array."""
        # Disconnect signals
        for signal in self.connected_plot_signals[:]:
            signal.disconnect()
            self.connected_plot_signals.remove(signal)

        # Set data
        self._set_data()

        # Define current view ranges and crosshair positions
        xh_pos = []  # Desired crosshair positions
        if self.init and self.range_lock.isChecked():
            x_range, y_range = self.image_plot.getViewBox().viewRange()
            for i in range(self.num_xhs):
                xh_pos.append(self.xhs[i].get_pos())
                if not self._check_crosshair_in_range(xh_pos[i]):
                    xh_pos[i] = self._init_crosshair_pos(i)
        else:
            y_range, x_range = self.ranges
            for i in range(self.num_xhs):
                xh_pos.append(self._init_crosshair_pos(i))
            for i in range(2):
                self.DC_width_selectors[i].setValue(self.data_span[i] / 200)

        # Set a colour map - once set on the graph, always keep the same one
        if not self.init:
            cmap = pg.colormap.get("Greys", source="matplotlib")
        else:
            cmap = self.image_item.getColorMap()
        # Get colour range
        if self.init and self.cscale_lock.isChecked():
            c_range = self.image_item.getLevels()
        else:
            c_range = (self.c_min, self.c_max)

        # Clear existing plots
        if self.init:
            self.image_plot.clear()
            for plot in self.DC_plots:
                plot.clear()
            self.graphics_layout.removeItem(self.colorbar)

        # Make plots
        self._set_main_plot(cmap, c_range, xh_pos)  # Make main plot
        self._set_DC_plots()  # Make DC plots
        self._set_plot_ranges(x_range, y_range)  # Set ranges
        self._set_plot_labels()  # Set labels
        self._update_cursor_stats_text()  # Update cursor stats

        # Connect signals
        self._connect_signals_crosshairs()
        self._connect_signals_DCspan_change()
        self._connect_signals_align()
        self._connect_key_press_signals()

        # Mark as initialized
        self.init = True

    def _update_DC(self, xh_no):
        """Update the DC plots when the crosshair is moved."""
        # Get the xh number and related plots
        xh = self.xhs[xh_no]

        for dim_no in range(2):
            # Plot to update
            plot = self.DC_plot_items[dim_no][xh_no]

            # Get the DC
            select_along_dim_no = (dim_no + 1) % 2
            DC = self._select_DC(
                self.current_data,
                select_along_dim_no,
                (xh.get_dim1_span() if select_along_dim_no == 1 else xh.get_dim0_span()),
            )

            # Update the plot
            if dim_no == 0:
                plot.setData(DC.data, self.coords[0])
            else:
                plot.setData(self.coords[1], DC.data)

            # Check if a mirrored DC exists and needs updating
            if (
                self.dims[dim_no] in self.centering_dims
                and self.show_mirror_checkbox.isChecked()
            ):
                plot_m = self.DC_plot_items[f"{dim_no}_m"][xh_no]
                mirror_DC = sym(
                    DC, flipped=True, **{self.dims[dim_no]: xh.get_pos()[dim_no]}
                )
                if dim_no == 0:
                    plot_m.setData(
                        mirror_DC.data, mirror_DC.coords[self.dims[dim_no]].values
                    )
                else:
                    plot_m.setData(
                        mirror_DC.coords[self.dims[dim_no]].values, mirror_DC.data
                    )

    def _update_DC_width(self, dim_no):
        """Update the DC markers and plots when the width is changed."""
        for i, xh in enumerate(self.xhs):
            getattr(xh, f"set_dim{dim_no}_width")(
                self.DC_width_selectors[dim_no].value()
            )
            self.DC_plots_xhs[dim_no][i].setRegion(
                getattr(xh, f"get_dim{dim_no}_span")()
            )
            self._update_DC(i)
        self._update_cursor_stats_text()

    def _update_DC_int(self, dim_no):
        """Update the DC plots when the integration checkbox is toggled."""
        if self.DC_span_all_checkboxes[dim_no].isChecked():
            # Store current DC position and width if not already stored
            if self.xh_width_store[dim_no] is None:
                self.xh_width_store[dim_no] = self.DC_width_selectors[dim_no].value()
                self.xh_pos_store[dim_no] = [xh.get_pos()[dim_no] for xh in self.xhs]

            # Set width box to full range
            self.DC_width_selectors[dim_no].setValue(
                self.DC_width_selectors[dim_no].maximum()
            )
            self.DC_width_selectors[dim_no].setDisabled(True)

            # Force crosshair to mid point
            mid_point = np.mean(self.ranges[dim_no])
            for xh in self.xhs:
                if dim_no == 0:
                    xh.set_lock_dim0(mid_point)
                else:
                    xh.set_lock_dim1(mid_point)
                xh.update_crosshair()
        else:
            # Set to original ranges and positions
            self.DC_width_selectors[dim_no].setDisabled(False)
            self.DC_width_selectors[dim_no].setValue(self.xh_width_store[dim_no])
            self.xh_width_store[dim_no] = None
            for i, xh in enumerate(self.xhs):
                # Unlock the cursor position
                if dim_no == 0:
                    xh.set_lock_dim0(None)
                else:
                    xh.set_lock_dim1(None)
                # Reset to stored position
                current_pos = xh.get_pos()
                pos0 = (
                    self.xh_pos_store[0][i]
                    if self.xh_pos_store[0][i] is not None
                    else current_pos[0]
                )
                pos1 = (
                    self.xh_pos_store[1][i]
                    if self.xh_pos_store[1][i] is not None
                    else current_pos[1]
                )
                xh.set_pos((pos0, pos1))
                self.xh_pos_store[dim_no][i] = None
                xh.update_crosshair()

    def _update_DC_crosshair_span(self, xh_no):
        """Follow the crosshair with the DC span."""
        xh = self.xhs[xh_no]
        for i in range(2):
            self.DC_plots_xhs[i][xh_no].setRegion(getattr(xh, f"get_dim{i}_span")())

    def _show_hide_DCs(self, xh_no):
        """Show or hide the DC plots."""
        show_DC = self.show_DCs_checkboxes[xh_no].isChecked()
        show_mirror = self.show_mirror_checkbox.isChecked()
        for key, plot_item_group in self.DC_plot_items.items():
            if isinstance(key, int):
                plot_item_group[xh_no].setVisible(show_DC)
            elif isinstance(key, str) and not plot_item_group == []:
                plot_item_group[xh_no].setVisible(show_mirror and show_DC)
        self.xhs[xh_no].set_visible(show_DC)
        self.DC_plots_xhs[0][xh_no].setVisible(show_DC)
        self.DC_plots_xhs[1][xh_no].setVisible(show_DC)
        self._update_cursor_stats_text()

    def _show_hide_mirror(self):
        for i in range(self.num_xhs):
            self._show_hide_DCs(i)

    def _align_data(self):
        """Estimate symmetry points in the data"""
        # Get the current view range
        x_range, y_range = self.image_plot.getViewBox().viewRange()
        data_to_centre = self.current_data.sel(
            {self.dims[0]: slice(*y_range), self.dims[1]: slice(*x_range)}
        )

        # Get the centre of the data
        centre = estimate_sym_point(data_to_centre, dims=self.centering_dims)

        # Activate mirror mode
        self.show_mirror_checkbox.setChecked(True)

        # Set crosshairs to the aligned position
        for xh in self.xhs:
            current_pos = xh.get_pos()
            dim0_pos = centre.get(self.dims[0], current_pos[0])
            dim1_pos = centre.get(self.dims[1], current_pos[1])
            xh.set_pos((dim0_pos, dim1_pos))

    def _update_cursor_stats_text(self):
        """Update the cursor stats."""

        cursor_colors = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in self.DC_pens]
        cursor_text = ""

        # Get data from active cursors
        cursor_pos = [
            xh.get_pos() if self.show_DCs_checkboxes[i].isChecked() else None
            for i, xh in enumerate(self.xhs)
        ]
        cursor_spans0 = [
            xh.get_dim0_span() if self.show_DCs_checkboxes[i].isChecked() else None
            for i, xh in enumerate(self.xhs)
        ]
        cursor_spans1 = [
            xh.get_dim1_span() if self.show_DCs_checkboxes[i].isChecked() else None
            for i, xh in enumerate(self.xhs)
        ]

        # Show current cursor positions and values (integrated over the crosshair selection)
        cursor_values = []
        for i in range(self.num_xhs):
            if cursor_pos[i] is None:
                cursor_values.append(None)
                cursor_text += "<br>"
            else:
                span0 = cursor_spans0[i]
                span1 = cursor_spans1[i]
                if abs(span0[1] - span0[0]) < self.step_sizes[0]:
                    data = self.current_data.sel(
                        {self.dims[0]: np.mean(span0)}, method="nearest"
                    )
                else:
                    data = self.current_data.sel(
                        {self.dims[0]: slice(span0[0], span0[1])}
                    ).mean(self.dims[0])
                if abs(span1[1] - span1[0]) < self.step_sizes[1]:
                    data = data.sel({self.dims[1]: np.mean(span1)}, method="nearest")
                else:
                    data = data.sel({self.dims[1]: slice(span1[0], span1[1])}).mean(
                        self.dims[1]
                    )
                cursor_values.append(data.mean().values)
                cursor_text += (
                    f"<span style='color:{cursor_colors[i % len(cursor_colors)]}'>"
                    f"Csr{i} {self.dims[0]}: {cursor_pos[i][0]:.3f} | "
                    f"{self.dims[1]}: {cursor_pos[i][1]:.3f} | "
                    f"Value: {cursor_values[i]:.3f}</span><br>"
                )

        # Show the delta between the first two cursors if active
        if cursor_pos[0] is not None and cursor_pos[1] is not None:
            cursor_text += (
                f"<span style='color:white'>"
                f"ΔCsr1-0: Δ{self.dims[0]}: {abs(cursor_pos[1][0] - cursor_pos[0][0]):.3f} | "
                f"Δ{self.dims[1]}: {abs(cursor_pos[1][1] - cursor_pos[0][1]):.3f}</span>"
            )
        else:
            cursor_text += "<br>"

        # Add normal emission
        if cursor_pos[0] is not None:
            norm_emission = self._get_norm_values()
            if norm_emission:
                cursor_text += "<br>"
                cursor_text += _parse_norm_emission_cursor_stats(
                    self.current_data, norm_emission
                )
        # Add metadata
        cursor_text += self.metadata_text
        # Set text
        self.cursor_stats.setText(cursor_text)

    def _get_norm_values(self):
        cursor_pos = [
            xh.get_pos() if self.show_DCs_checkboxes[i].isChecked() else None
            for i, xh in enumerate(self.xhs)
        ]

        try:
            norm_values = self.current_data.metadata.get_normal_emission_from_values(
                {
                    self.dims[0]: float(cursor_pos[0][0]),
                    self.dims[1]: float(cursor_pos[0][1]),
                },
            )
        except AttributeError:
            norm_values = {}
        return norm_values

    def _copy_norm_values(self):
        """Copy the normal emission values to the clipboard."""
        # Current norm values
        norm_values = self._get_norm_values()
        norm_values_to_return = {
            k: str(v) if isinstance(v, pint.Quantity) else v
            for k, v in norm_values.items()
        }
        pyperclip.copy(f".metadata.set_normal_emission({norm_values_to_return})")

    # ##############################
    # Signal connections
    # ##############################
    def _connect_signals_crosshairs(self):
        # Update when crosshair moves
        for i, xh_ in enumerate(self.xhs):
            signal = xh_.xh.sigPositionChanged
            signal.connect(partial(self._update_DC, i))  # Update DC plots
            signal.connect(partial(self._update_DC_crosshair_span, i))  # Follow DC span
            signal.connect(self._update_cursor_stats_text)  # Update label
            self.connected_plot_signals.append(signal)

        # Update when show/hide crosshair boxes toggled
        for i, checkbox in enumerate(self.show_DCs_checkboxes):
            checkbox.stateChanged.connect(partial(self._show_hide_DCs, i))
            self.connected_plot_signals.append(checkbox.stateChanged)

        # Update when show/hide mirror checkbox toggled
        signal = self.show_mirror_checkbox.stateChanged
        signal.connect(self._show_hide_mirror)
        self.connected_plot_signals.append(signal)

    def _connect_signals_DCspan_change(self):
        # Update when span ranges changed
        for i in range(2):
            signal = self.DC_width_selectors[i].valueChanged
            signal.connect(partial(self._update_DC_width, i))
            self.connected_plot_signals.append(signal)

        # Update with integrate all selection
        for i in range(2):
            signal = self.DC_span_all_checkboxes[i].stateChanged
            signal.connect(partial(self._update_DC_int, i))
            self.connected_plot_signals.append(signal)

    def _connect_signals_align(self):
        # Connect centre data button
        self.align_button.clicked.connect(self._align_data)
        self.connected_plot_signals.append(self.align_button.clicked)
        # Connect copy button
        self.copy_button.clicked.connect(self._copy_norm_values)

    def _connect_key_press_signals(self):
        signals = [self.graphics_layout.keyPressed, self.graphics_layout.keyReleased]
        fns = [self._key_press_event, self._key_release_event]
        for signal, fn in zip(signals, fns, strict=True):
            signal.connect(fn)
            self.connected_plot_signals.append(signal)

    def _key_press_event(self, event):
        # First deal with the modifiers
        if event.key() == self.key_modifiers["move_csr2"]:
            self.move_csr1_key_enabled = True
        elif event.key() == self.key_modifiers["move_all"]:
            self.move_all_key_enabled = True
        elif event.key() == self.key_modifiers["hide_all"]:
            # If this is the first press, store what xhs are currently visible and hide all
            if len(self.xh_visible_store) == 0:
                for i in range(self.num_xhs):
                    if self.show_DCs_checkboxes[i].isChecked():
                        self.xh_visible_store.append(self.xhs[i])
                        self.xhs[i].set_visible(False)
        elif event.key() in self.show_hide_csr_keys:
            # Show/hide cursors
            csr_no = self.show_hide_csr_keys.index(event.key())
            self.show_DCs_checkboxes[csr_no].setChecked(
                not self.show_DCs_checkboxes[csr_no].isChecked()
            )
        elif event.key() == self.show_hide_mirror_key:
            self.show_mirror_checkbox.setChecked(
                not self.show_mirror_checkbox.isChecked()
            )
        elif event.key() in [
            QtCore.Qt.Key.Key_Up,
            QtCore.Qt.Key.Key_Down,
            QtCore.Qt.Key.Key_Left,
            QtCore.Qt.Key.Key_Right,
        ]:
            if self.move_all_key_enabled:
                xhs = [
                    xh
                    for i, xh in enumerate(self.xhs)
                    if self.show_DCs_checkboxes[i].isChecked()
                ]
            elif self.move_csr1_key_enabled:
                xhs = [self.xhs[1]]
            else:
                xhs = [self.xhs[0]]
            for xh in xhs:
                current_pos = xh.get_pos()
                if event.key() == QtCore.Qt.Key.Key_Right:
                    xh.set_pos((current_pos[0], current_pos[1] + self.step_sizes[1]))
                elif event.key() == QtCore.Qt.Key.Key_Left:
                    xh.set_pos((current_pos[0], current_pos[1] - self.step_sizes[1]))
                elif event.key() == QtCore.Qt.Key.Key_Up:
                    xh.set_pos((current_pos[0] + self.step_sizes[0], current_pos[1]))
                elif event.key() == QtCore.Qt.Key.Key_Down:
                    xh.set_pos((current_pos[0] - self.step_sizes[0], current_pos[1]))

    def _key_release_event(self, event):
        if event.key() == self.key_modifiers["move_csr2"]:
            self.move_csr1_key_enabled = False
        elif event.key() == self.key_modifiers["move_all"]:
            self.move_all_key_enabled = False
        elif event.key() == self.key_modifiers["hide_all"]:
            # Restore xh to previous state
            for xh in self.xh_visible_store:
                xh.set_visible(True)
            self.xh_visible_store = []

    # ##############################
    # Helper functions
    # ##############################
    def _init_crosshair_pos(self, xh_no):
        """Initialize the crosshair position."""
        active_xh = [
            self.show_DCs_checkboxes[i].isChecked() for i in range(self.num_xhs)
        ]
        num_active_xh = sum(active_xh)
        csr_no_active = sum(active_xh[:xh_no])
        percentile = min((csr_no_active + 1) * 100 / (num_active_xh + 1), 95)
        return (
            np.percentile(self.ranges[0], percentile),
            np.percentile(self.ranges[1], percentile),
        )

    def _check_crosshair_in_range(self, pos):
        """Check if crosshair pos is within the coordinate range"""
        return (min(self.ranges[0]) <= pos[0] <= max(self.ranges[0])) and (
            min(self.ranges[1]) <= pos[1] <= max(self.ranges[1])
        )
