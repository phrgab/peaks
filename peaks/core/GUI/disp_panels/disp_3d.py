"""Functions for the 3D interactive display panel.

"""

import sys
from functools import partial
from PyQt6 import QtCore, QtWidgets, QtGui
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
)
import pyqtgraph as pg
import numpy as np


from peaks.core.GUI.GUI_utils import (
    Crosshair,
    KeyPressGraphicsLayoutWidget,
)
from peaks.core.fileIO.fileIO_opts import _BL_angles
from peaks.core.process.tools import sym, estimate_sym_point
from peaks.core.process.process import _estimate_EF


def _disp_3d(data, primary_dim, exclude_from_centering):
    """Display a 3D interactive display panel.

    Parameters
    ------------
    data : xarray.DataArray
         A single 3D :class:`xarray.DataArray`.

    primary_dim : str
        The primary dimension for the viewer, used to select the plane shown in the central panel.

    exclude_from_centering : str or tuple of str or list of str or None
        The dimension to exclude from centering. Default is 'eV'.

    """
    global app  # Ensure the QApplication instance is not garbage collected
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    viewer = _Disp3D(data, primary_dim, exclude_from_centering)
    viewer.show()
    app.exec()


class _Disp3D(QtWidgets.QMainWindow):
    def __init__(self, data, primary_dim, exclude_from_centering):
        super().__init__()

        # Parse data
        if primary_dim:  # UI functions assume primary dim is penultimate dim
            primary_dim_index = data.dims.index(primary_dim)
            if primary_dim_index != 1:
                dims = list(data.dims)
                dims.remove(primary_dim)
                dims.insert(1, primary_dim)
                data = data.transpose(*dims)
        self.data = data.compute()

        # Crosshair options
        self.DC_pen = (255, 0, 0)
        self.xh_brush = (255, 0, 0, 50)
        self.DC_xh_brush = (255, 0, 0, 70)

        # Set keys for the keyboard control
        def _get_key(param):
            return getattr(QtCore.Qt.Key, f"Key_{param}")

        self.key_modifiers_characters = {
            "hide_all": "Space",
            "move_primary_dim": "Shift",
            "show_hide_mirror": "M",
        }
        self.key_modifiers = {
            k: _get_key(v) for k, v in self.key_modifiers_characters.items()
        }

        # Set options for dims to exclude from centering
        if isinstance(exclude_from_centering, str):
            self.exclude_from_centering = [exclude_from_centering]
        elif isinstance(exclude_from_centering, (tuple, list)):
            self.exclude_from_centering = list(exclude_from_centering)
        else:
            self.exclude_from_centering = []

        self._init_data()  # Read some basic parameters from the data
        self._init_UI()  # Initialize the GUI layout
        self._set_data()  # Initialize with data

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
        self.setGeometry(100, 100, 1000, 700)

        # Main window layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout()
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

        # Set up the plots with the following number conventions
        #    #   --> indexes in array
        #   (..) --> dims displayed
        #        +----------+
        #        |  DC1 (2) |
        #        +----------+
        # +-----++----------++----------+
        # | DC0 ||  IMAGE 1 ||  IMAGE 2 |
        # | (0) ||   (0,2)  ||    (0,1) |
        # +-----++----------++----------+
        #        +----------+
        #        |  IMAGE 0 |
        #        |   (1,2)  |
        #        +----------+

        # Main image plots
        self.image_plots = []
        self.image_plots.append(self.graphics_layout.addPlot(2, 1, 1, 1))
        self.image_plots.append(self.graphics_layout.addPlot(1, 1, 1, 1))
        self.image_plots.append(self.graphics_layout.addPlot(1, 2, 1, 1))

        self.DC_plots = []
        self.DC_plots.append(self.graphics_layout.addPlot(1, 0, 1, 1))
        self.DC_plots.append(self.graphics_layout.addPlot(0, 1, 1, 1))

        self.DC_plots[0].setYLink(self.image_plots[1])
        self.DC_plots[1].setXLink(self.image_plots[1])
        self.image_plots[0].setXLink(self.image_plots[1])
        self.image_plots[2].setYLink(self.image_plots[1])
        for i in [0, 1]:
            self.DC_plots[i].getAxis("bottom").hide()
            self.DC_plots[i].getAxis("top").show()
        self.image_plots[2].getAxis("left").hide()
        self.image_plots[2].getAxis("right").show()
        for i in [1, 2]:
            self.image_plots[i].getAxis("bottom").hide()
            self.image_plots[i].getAxis("top").show()

        self.graphics_layout.ci.layout.setRowStretchFactor(1, 4)
        self.graphics_layout.ci.layout.setRowStretchFactor(2, 4)
        self.graphics_layout.ci.layout.setColumnStretchFactor(1, 4)
        self.graphics_layout.ci.layout.setColumnStretchFactor(2, 4)

        # Set margins to line up the plots
        self.DC_plots[0].getAxis("left").setWidth(40)
        self.DC_plots[0].getAxis("bottom").setWidth(50)
        self.DC_plots[1].getAxis("left").setWidth(60)
        self.image_plots[0].getAxis("left").setWidth(60)
        self.image_plots[1].getAxis("left").setWidth(60)
        self.image_plots[2].getAxis("right").setWidth(50)

    def _build_controls_layout(self, layout):
        # Right panel -------------------------------------
        right_panel_layout = QVBoxLayout()
        layout.addLayout(right_panel_layout)
        # Cursor stats etc.
        self.cursor_stats = QLabel()
        self.cursor_stats.setStyleSheet(
            "QLabel { background-color : black; color : white; }"
        )
        self.cursor_stats.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft
        )
        self.cursor_stats.setWordWrap(True)
        self.cursor_stats.setMaximumWidth(550)
        self.cursor_stats.setMinimumWidth(300)
        self.cursor_stats.setMinimumHeight(100)
        right_panel_layout.addWidget(self.cursor_stats)

        # xh positions and widths
        xh_group = QtWidgets.QGroupBox("Data selection")
        xh_group.setContentsMargins(5, 22, 5, 5)
        xh_layout = QtWidgets.QVBoxLayout()
        xh_layout.setContentsMargins(5, 0, 5, 0)
        xh_group.setLayout(xh_layout)
        right_panel_layout.addWidget(xh_group)
        # Add dimension labels positions and widths
        self.cursor_positions_selection = []
        self.cursor_widths_selection = []
        for i in range(3):
            # Make an HBox for each dimension
            hbox = QHBoxLayout()
            xh_layout.addLayout(hbox)
            label = QLabel(f"{self.dims[i]}")
            label.setFixedWidth(75)
            hbox.addWidget(label)
            # Make a selection box for cursor positions
            self.cursor_positions_selection.append(QtWidgets.QDoubleSpinBox())
            self.cursor_positions_selection[i].setRange(
                self.ranges[i][0], self.ranges[i][1]
            )
            self.cursor_positions_selection[i].setSingleStep(self.step_sizes[i])
            self.cursor_positions_selection[i].setDecimals(3)
            self.cursor_positions_selection[i].setFixedWidth(75)
            hbox.addWidget(self.cursor_positions_selection[i])
            # Make a selection box for cursor widths
            hbox.addWidget(QLabel("Δ:"))
            self.cursor_widths_selection.append(QtWidgets.QDoubleSpinBox())
            self.cursor_widths_selection[i].setRange(0, self.data_span[i])
            self.cursor_widths_selection[i].setSingleStep(self.step_sizes[i])
            self.cursor_widths_selection[i].setDecimals(3)
            self.cursor_widths_selection[i].setFixedWidth(75)
            hbox.addWidget(self.cursor_widths_selection[i])
            hbox.addStretch()
        self.mirror_checkbox = QtWidgets.QCheckBox("Mirror DCs?")
        xh_layout.addWidget(self.mirror_checkbox)

        right_panel_layout.addSpacing(10)

        # Alignment options
        align_group = QtWidgets.QGroupBox("Alignment")
        align_group.setContentsMargins(5, 22, 5, 5)
        align_layout = QtWidgets.QHBoxLayout()
        align_layout.setContentsMargins(5, 0, 5, 0)
        align_group.setLayout(align_layout)
        right_panel_layout.addWidget(align_group)
        # Add alignment buttons
        vbox = QVBoxLayout()
        align_layout.addLayout(vbox)
        hbox = QHBoxLayout()
        vbox.addLayout(hbox)
        label = QLabel("Rotation:")
        label.setFixedWidth(90)
        hbox.addWidget(label)
        self.rotation_selection = QtWidgets.QDoubleSpinBox()
        self.rotation_selection.setRange(-180, 180)
        self.rotation_selection.setSingleStep(1)
        self.rotation_selection.setDecimals(2)
        self.rotation_selection.setFixedWidth(75)
        hbox.addWidget(self.rotation_selection)
        hbox.addStretch()
        hbox = QHBoxLayout()
        vbox.addLayout(hbox)
        label = QLabel("Alignment aid:")
        label.setFixedWidth(85)
        hbox.addWidget(label)
        self.alignment_aid = QtWidgets.QComboBox()
        self.alignment_aid.addItems(
            ["None", "Square", "Hexagon", "Hexagon (r30)", "Circle"]
        )
        hbox.addWidget(self.alignment_aid)
        # Add a slider to set size of alignment aid
        self.alignment_aid_size = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.alignment_aid_size.setRange(0, 100)
        self.alignment_aid_size.setValue(50)
        self.alignment_aid_size.setSingleStep(1)
        self.alignment_aid_size.setFixedWidth(100)
        hbox.addWidget(self.alignment_aid_size)
        hbox.addStretch()

        hbox = QHBoxLayout()
        vbox.addLayout(hbox)
        self.align_button = QtWidgets.QPushButton("Align")
        hbox.addWidget(self.align_button)
        self.copy_button = QtWidgets.QPushButton("Copy")
        hbox.addWidget(self.copy_button)

        right_panel_layout.addSpacing(10)

        # Add colorbar
        colorbar_group = QtWidgets.QGroupBox("Image contrast")
        right_panel_layout.addWidget(colorbar_group)
        colorbar_container = QHBoxLayout()
        colorbar_group.setLayout(colorbar_container)
        self.colorbar_widget_container = pg.GraphicsLayoutWidget()
        self.colorbar_widget_container.setMaximumHeight(100)
        self.colorbar_widget_container.setMaximumWidth(400)
        colorbar_container.addWidget(self.colorbar_widget_container)

        right_panel_layout.addStretch()

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
                    <td style='padding: 5px; font-weight: normal;'>Move crosshair in main plot</td>
                    <td style='padding: 5px; font-weight: normal;'>Arrow keys</td>
                  </tr>
                  <tr>
                    <td style='padding: 5px; font-weight: normal;'>Change primary slice</td>
                    <td style='padding: 5px; font-weight: normal;'>{self.key_modifiers_characters['move_primary_dim']} 
                    + Up/Down arrow keys</td>
                  </tr>
                  <tr>
                    <td style='padding: 5px; font-weight: normal;'>Hide all crosshairs</td>
                    <td style='padding: 5px; font-weight: normal;'>{self.key_modifiers_characters['hide_all']}</td>
                  </tr>
                """

        self.help_text += f"""
                  <tr>
                    <td style='padding: 5px; font-weight: normal;'>Enable/disable mirrored mode</td>
                    <td style='padding: 5px; font-weight: normal;'>{self.key_modifiers_characters["show_hide_mirror"]}</td>
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
    def _init_data(self):
        """Extract some core parameters from the data."""
        self.dims = self.data.dims
        self.coords = [self.data.coords[dim].values for dim in self.dims]
        self.step_sizes = [coord[1] - coord[0] for coord in self.coords]
        self.ranges = [(min(coords), max(coords)) for coords in self.coords]
        self.data_span = [abs(range[1] - range[0]) for range in self.ranges]
        self.c_min = float(self.data.min())
        self.c_max = float(self.data.max())

        # Set centering dims
        self.centering_dims = [
            dim for dim in self.dims if dim not in self.exclude_from_centering
        ]

        # Attempt to read some metadata
        self.metadata_text = "<span style='color:white'>"
        if "beamline" in self.data.attrs:
            attrs = self.data.attrs
            loc = attrs.get("beamline", "default")
            self.current_data_loc_angle_conventions = _BL_angles.angles.get(loc, {})

            def _parse_meta(
                param_name_field,
                default_name,
                attr_name,
            ):
                try:
                    self.metadata_text += f"{self.current_data_loc_angle_conventions.get(param_name_field, default_name)}={attrs.get(attr_name, None):.3f} | "
                except (TypeError, ValueError):
                    pass

            param_name_fields = [
                "x1 name",
                "x2 name",
                "x3 name",
                "polar_name",
                "tilt_name",
                "azi_name",
                "defl_par_name",
                "defl_perp_name",
            ]
            attr_names = [
                "x1",
                "x2",
                "x3",
                "polar",
                "tilt",
                "azi",
                "defl_par",
                "defl_perp",
            ]
            for param_name_field, attr_name in zip(param_name_fields, attr_names):
                _parse_meta(
                    param_name_field,
                    attr_name,
                    attr_name,
                )

        self.metadata_text += "</span><br>"

        # Check if this is an ARPES analyser for automated normal emission parsing
        if "ana_slit_angle" in self.data.attrs:
            self.analyser_type = self.current_data_loc_angle_conventions.get(
                "ana_type", None
            )
        else:
            self.analyser_type = None

    def _set_main_plots(self):
        # Make arrays for plotitems and crosshairs
        self.image_items = []
        self.xhs = []

        # Populate these for each of the main plots
        for i in range(3):
            self.image_items.append(
                pg.ImageItem(self.images[i].values, axisOrder="row-major")
            )
            self.image_plots[i].addItem(self.image_items[i])

            # Transform to match data coordinates
            active_dim_nos = self._get_active_dim_nos(i)
            self.image_items[i].setTransform(
                pg.QtGui.QTransform(
                    self.step_sizes[active_dim_nos[1]],
                    0,
                    0,
                    0,
                    self.step_sizes[active_dim_nos[0]],
                    0,
                    self.coords[active_dim_nos[1]][0],
                    self.coords[active_dim_nos[0]][0],
                    1,
                )
            )

            # Crosshairs
            pos = tuple(
                [self.cursor_positions_selection[i].value() for i in active_dim_nos]
            )
            self.xhs.append(
                Crosshair._Crosshair(
                    self.image_plots[i],
                    pos=pos,
                    dim0_width=self.cursor_widths_selection[active_dim_nos[0]].value(),
                    dim1_width=self.cursor_widths_selection[active_dim_nos[1]].value(),
                    brush=self.xh_brush,
                    bounds=[
                        self.ranges[active_dim_nos[0]],
                        self.ranges[active_dim_nos[1]],
                    ],
                    axisOrder="row-major",
                )
            )

        # Add colour bar to main image
        self.cmap = pg.colormap.get("Greys", source="matplotlib")
        self.colorbar = pg.ColorBarItem(
            label=self.data.name,
            colorMap=self.cmap,
            colorMapMenu=True,
            limits=(self.c_min, self.c_max),
            interactive=True,
            orientation="h",
            values=(self.c_min, self.c_max),
        )
        self.colorbar.setImageItem(self.image_items)
        self.colorbar_widget_container.addItem(self.colorbar, row=0, col=0)

        # Flip axes as required
        self.image_plots[2].getViewBox().invertX(True)

    def _set_DC_plots(self):
        self.DC_plot_items = {}
        self.DC_plots_xhs = []
        for i in range(2):
            dim_no = [0, 2][i]  # Select the active dim for the relevant xh plot
            self.DC_plot_items[i] = []
            self.DC_plot_items[f"{i}_m"] = []
            DC = self._select_DC(dim_no)

            # Add to plots - get ordering correct
            if i == 0:
                a, b = DC.data, DC.coords[self.dims[dim_no]].values
            else:
                a, b = DC.coords[self.dims[dim_no]].values, DC.data

            self.DC_plot_items[i].append(
                self.DC_plots[i].plot(
                    a,
                    b,
                    pen=self.DC_pen,
                )
            )

            # Add spans for crosshairs
            self.DC_plots_xhs.append(
                pg.LinearRegionItem(
                    values=getattr(self.xhs[1], f"get_dim{i}_span")(),
                    orientation="horizontal" if i == 0 else "vertical",
                    pen=(0, 0, 0, 0),
                    brush=self.DC_xh_brush,
                    movable=False,
                )
            )
            self.DC_plots[i].addItem(self.DC_plots_xhs[i])

        self.DC_plots[0].getViewBox().invertX(True)

    def _set_plot_range_limits(self):
        # Set limits for image plots
        for i in range(3):
            active_dim_nos = self._get_active_dim_nos(i)
            xmin, xmax = self.ranges[active_dim_nos[1]]
            ymin, ymax = self.ranges[active_dim_nos[0]]
            self.image_plots[i].getViewBox().setLimits(
                xMin=xmin, xMax=xmax, yMin=ymin, yMax=ymax
            )

        # Set limits for DC plots
        ymin, ymax = self.ranges[0]
        xmin, xmax = self.ranges[2]
        self.DC_plots[0].getViewBox().setLimits(yMin=ymin, yMax=ymax)
        self.DC_plots[1].getViewBox().setLimits(xMin=xmin, xMax=xmax)

    def _set_plot_labels(self):
        # Extract labels
        dim_labels = []
        for i in range(3):
            dim_units = self.data.coords[self.dims[i]].attrs.get("units")
            dim_labels.append(
                f"{self.dims[i]} ({dim_units})" if dim_units else self.dims[i]
            )

        # Label the relevant plots
        self.DC_plots[0].setLabel("left", dim_labels[0])
        self.DC_plots[0].setLabel("top", " ")
        self.DC_plots[1].setLabel("top", dim_labels[2])
        self.image_plots[0].setLabel("bottom", dim_labels[2])
        self.image_plots[0].setLabel("left", dim_labels[1])
        self.image_plots[1].setLabel("top", dim_labels[2])
        self.image_plots[1].setLabel("left", dim_labels[0])
        self.image_plots[2].setLabel("top", dim_labels[1])
        self.image_plots[2].setLabel("right", dim_labels[0])

    def _select_DC(self, dim_no):
        """Select a DC along dim_no 0 from the data, handling averaging."""
        sum_over_dim_nos = [i for i in range(3) if i != dim_no]
        DC = self.data
        for i in sum_over_dim_nos:
            _pos = self.cursor_positions_selection[i].value()
            _width = self.cursor_widths_selection[i].value() / 2
            _range = slice(_pos - _width, _pos + _width)
            if _width < self.step_sizes[i]:
                DC = DC.sel({self.dims[i]: _pos}, method="nearest")
            else:
                DC = DC.sel({self.dims[i]: _range}).mean(self.dims[i])
        return DC

    # ##############################
    # Data / plot updates
    # ##############################
    def _set_data(self):
        """Set the data in the plots"""

        self.images = [None, None, None]

        # Initialise crosshair positions and widths - default to centre of data and 1/200th of range
        for i in range(3):
            self.cursor_positions_selection[i].setValue(sum(self.ranges[i]) / 2)
            self.cursor_widths_selection[i].setValue(self.data_span[i] / 200)

        # If eV in data, attempt to set a Fermi level and use this as initial xh pos
        if "eV" in self.dims:
            eV_dim = self.data.dims.index("eV")
            EF = _estimate_EF(self.data)
            if not EF:
                EF = sum(self.ranges[eV_dim]) / 2
            self.cursor_positions_selection[eV_dim].setValue(EF)

        # Make primary data slice
        self._set_slice(1)

        # Try to estimate a data centre position from this slice and update other xh positions
        centre = estimate_sym_point(self.images[1], dims=self.centering_dims)
        for dim, centre in centre.items():
            self.cursor_positions_selection[self.dims.index(dim)].setValue(centre)
        # Make the coresponding slices
        self._set_slice(0)
        self._set_slice(2)

        # Make plots
        self._set_main_plots()  # Make main plots
        self._set_DC_plots()  # Make DC plots
        self._set_plot_range_limits()  # Set plot range limits
        self._set_plot_labels()  # Set labels
        # self._update_cursor_stats_text()  # Update cursor stats
        #
        # # Connect signals
        # self._connect_signals_crosshairs()
        # self._connect_signals_DCspan_change()
        # self._connect_signals_align()
        # self._connect_key_press_signals()
        #
        # # Mark as initialized
        # self.init = True

    def _set_slice(self, dim_no):
        """Set a data slice based on the xh positions and widths"""
        _pos = self.cursor_positions_selection[dim_no].value()
        _width = self.cursor_widths_selection[dim_no].value() / 2
        _range = slice(_pos - _width, _pos + _width)
        self.images[dim_no] = self.data.sel({self.dims[dim_no]: _range}).mean(
            dim=self.dims[dim_no]
        )

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
                self.data,
                select_along_dim_no,
                (
                    xh.get_dim1_span()
                    if select_along_dim_no == 1
                    else xh.get_dim0_span()
                ),
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
                self.cursor_widths_selection[dim_no].value()
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
                self.xh_width_store[dim_no] = self.cursor_widths_selection[
                    dim_no
                ].value()
                self.xh_pos_store[dim_no] = [xh.get_pos()[dim_no] for xh in self.xhs]

            # Set width box to full range
            self.cursor_widths_selection[dim_no].setValue(
                self.cursor_widths_selection[dim_no].maximum()
            )
            self.cursor_widths_selection[dim_no].setDisabled(True)

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
            self.cursor_widths_selection[dim_no].setDisabled(False)
            self.cursor_widths_selection[dim_no].setValue(self.xh_width_store[dim_no])
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
        data_to_centre = self.data.sel(
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
                    data = self.data.sel(
                        {self.dims[0]: np.mean(span0)}, method="nearest"
                    )
                else:
                    data = self.data.sel(
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
        cursor_text += "<hr>"

        # Add metadata
        cursor_text += self.metadata_text
        cursor_text += "<br>"

        # Add normal emission
        if (
            self.analyser_type
            and "theta_par" in self.dims
            and cursor_pos[0] is not None
        ):
            cursor_text += self._get_norm_values(
                cursor_pos[0][self.dims.index("theta_par")]
            )

        self.cursor_stats.setText(cursor_text)

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
            signal = self.cursor_widths_selection[i].valueChanged
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

    def _connect_key_press_signals(self):
        signals = [self.graphics_layout.keyPressed, self.graphics_layout.keyReleased]
        fns = [self._key_press_event, self._key_release_event]
        for signal, fn in zip(signals, fns):
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
    def _get_active_dim_nos(self, i):
        return [dim_no for dim_no in range(3) if dim_no != i]

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
            (min(self.ranges[1]) <= pos[1] <= max(self.ranges[1]))
        )

    def _get_norm_values(self, theta_par):
        """Attempt to extract normal emission based on analyser configuration
        and metadata.

        Parameters
        ----------
        theta_par : float
            theta_par value for passing to the normal emission calculation.
        """

        r, g, b = self.DC_pens[0]
        cursor_text = f"<span style='color:#{r:02x}{g:02x}{b:02x}'>"
        if self.analyser_type == "I" or self.analyser_type == "Ip":  # Type I
            # Norm tilt
            tilt = self.data.attrs.get("tilt", 0) or 0
            defl_par = self.data.attrs.get("defl_par", 0) or 0
            norm_tilt = (
                (self.current_data_loc_angle_conventions.get("tilt", 1) * tilt)
                + (
                    self.current_data_loc_angle_conventions.get("defl_par", 1)
                    * defl_par
                )
                - (
                    self.current_data_loc_angle_conventions.get("theta_par", 1)
                    * theta_par
                )
            )
            cursor_text += f"Cursor 0: Normal emission [{self.current_data_loc_angle_conventions.get('tilt_name')}] = {norm_tilt:.3f}</span><br>"

        else:  # Type II
            # Norm polar
            polar = self.data.attrs.get("polar", 0) or 0
            defl_par = self.data.attrs.get("defl_par", 0) or 0
            norm_polar = (
                (self.current_data_loc_angle_conventions.get("polar", 1) * polar)
                + (
                    self.current_data_loc_angle_conventions.get("defl_par", 1)
                    * defl_par
                )
                - (
                    self.current_data_loc_angle_conventions.get("theta_par", 1)
                    * theta_par
                )
            )
            cursor_text += f"Cursor 0: Normal emission [{self.current_data_loc_angle_conventions.get('polar_name')}] = {norm_polar:.3f}</span><br>"

        return cursor_text
