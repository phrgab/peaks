"""Functions for the 3D interactive display panel."""

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
    Crosshair,
    KeyPressGraphicsLayoutWidget,
)
from peaks.core.GUI.GUI_utils.cursor_stats import _parse_norm_emission_cursor_stats
from peaks.core.metadata.metadata_methods import display_metadata
from peaks.core.process.tools import estimate_sym_point, sym


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

        # Read scan metadata
        self.metadata_text = "<span style='color:white'>"
        self.metadata_text += display_metadata(self.data, "html")
        self.metadata_text += "</span><br>"

        # Crosshair options
        self.DC_pen = (238, 119, 51)
        self.DC_mirror_pen = pg.mkPen(
            color=(51, 187, 238),
            style=QtCore.Qt.PenStyle.DashLine,
        )
        self.align_aid_pen = pg.mkPen(
            color=(204, 51, 153),
            style=QtCore.Qt.PenStyle.DotLine,
            width=2.5,
        )
        self.xh_brush = (238, 119, 51, 60)
        self.DC_xh_brush = (238, 119, 51, 80)

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
        self.move_primary_dim_key_enabled = False

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
        self.setGeometry(100, 100, 1200, 700)

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
        self.cursor_stats.setMinimumWidth(300)
        scroll_area.setWidget(self.cursor_stats)
        right_panel_layout.addWidget(scroll_area)

        # xh positions and widths
        xh_group = QtWidgets.QGroupBox("Data selection")
        xh_group.setContentsMargins(5, 22, 5, 5)
        xh_group.setMaximumWidth(450)
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
        align_group.setMaximumWidth(450)
        align_layout = QtWidgets.QHBoxLayout()
        align_layout.setContentsMargins(5, 0, 5, 0)
        align_group.setLayout(align_layout)
        right_panel_layout.addWidget(align_group)
        # Add alignment buttons
        vbox = QVBoxLayout()
        align_layout.addLayout(vbox)
        hbox = QHBoxLayout()
        hbox.setSpacing(5)
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
        self.rotation_delta = [-45, -30, 30, 45]
        self.rotation_delta_buttons = []
        for ang in self.rotation_delta:
            button_text = f"{'+' if ang > 0 else '-'}{abs(ang)}°"
            button = QtWidgets.QPushButton(button_text)
            button.setFixedWidth(40)
            self.rotation_delta_buttons.append(button)
            hbox.addWidget(button)
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
        self.alignment_aid_scaling = (self.ranges[2][1] - self.ranges[2][0]) / 150
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
        colorbar_group.setMaximumWidth(450)
        right_panel_layout.addWidget(colorbar_group)
        colorbar_container = QVBoxLayout()
        colorbar_group.setLayout(colorbar_container)
        self.colorbar_widget_container = pg.GraphicsLayoutWidget()
        self.colorbar_widget_container.viewport().setAttribute(
            QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, False
        )
        self.colorbar_widget_container.setMaximumHeight(100)
        self.colorbar_widget_container.setMaximumWidth(400)
        colorbar_container.addWidget(self.colorbar_widget_container)
        self.cscale_lock = QtWidgets.QCheckBox("cScale locked", checked=True)
        colorbar_container.addWidget(self.cscale_lock)

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
                    <td style='padding: 5px; font-weight: normal;'>{self.key_modifiers_characters["move_primary_dim"]} 
                    + Up/Down arrow keys</td>
                  </tr>
                  <tr>
                    <td style='padding: 5px; font-weight: normal;'>Hide all crosshairs</td>
                    <td style='padding: 5px; font-weight: normal;'>{self.key_modifiers_characters["hide_all"]}</td>
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
        self.ranges = [
            (min(coords) - self.step_sizes[i] / 2, max(coords) + self.step_sizes[i] / 2)
            for i, coords in enumerate(self.coords)
        ]
        self.data_span = [abs(range[1] - range[0]) for range in self.ranges]
        self.c_min = float(self.data.min())
        self.c_max = float(self.data.max())

        # Set centering dims
        self.centering_dims = [
            dim for dim in self.dims if dim not in self.exclude_from_centering
        ]

        # Attempt to read some metadata
        self.metadata_text = "<span style='color:white'>"
        self.metadata_text += display_metadata(self.data, "html")
        self.metadata_text += "</span><br>"

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
                    self.ranges[active_dim_nos[1]][0],
                    self.ranges[active_dim_nos[0]][0],
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
                    target_item_kwargs={
                        "pen": pg.mkPen(color=(238, 119, 51)),
                        "brush": pg.mkBrush(color=(238, 119, 51, 50)),
                    },
                    linear_region_item_kwargs={"pen": pg.mkPen(color=(238, 119, 51))},
                )
            )

        # Add alignment xh
        self.align_xhs = []
        for angle in [0, 90]:
            xh = pg.InfiniteLine(
                pos=(
                    self.cursor_positions_selection[2].value(),
                    self.cursor_positions_selection[0].value(),
                ),
                pen=self.align_aid_pen,
                angle=angle,
            )
            self.align_xhs.append(xh)
            self.image_plots[1].addItem(xh)
            xh.setVisible(False)

        # Add alignment aid
        dummy_centre = QtCore.QPointF(0, 0)
        d = self._create_alignment_shape(
            "Square",
            dummy_centre,
            self.alignment_aid_size.value() / 20,
            self.rotation_selection.value(),
        )
        self.alignment_aid_shape = pg.QtWidgets.QGraphicsPathItem(d)
        self.alignment_aid_shape.setPen(self.align_aid_pen)
        self.alignment_aid_shape.setVisible(False)
        self.image_plots[1].addItem(self.alignment_aid_shape)

        # Add colour bar to main image
        self.cmap = pg.colormap.get("Greys", source="matplotlib")
        self.colorbar = pg.ColorBarItem(
            label=self.data.name,
            colorMap=self.cmap,
            colorMapMenu=True,
            limits=(self.c_min, self.c_max),
            rounding=min(abs(self.c_max - self.c_min) / 2000, 1),
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
            if (
                self.data.attrs.get("scan_type") == "hv scan"
                and self.data.attrs.get("eV_type") == "kinetic"
            ):
                # Due to the particular data structure of an hv scan, take the central slice to estimate EF
                hv = self.data.hv.data
                middle_hv = hv[(len(hv) - 1) // 2]
                EF = self.data.disp_from_hv(middle_hv).estimate_EF()
            else:
                EF = self.data.estimate_EF()
            if isinstance(EF, (list, np.ndarray)):
                if any(EF):
                    EF = sum(self.ranges[eV_dim]) / 2
            else:
                EF = EF

            self.cursor_positions_selection[eV_dim].setValue(
                EF or sum(self.ranges[eV_dim]) / 2
            )

        # Make primary data slice
        self._set_slice(1)

        # Try to estimate data centre from this slice and update other xh positions
        try:
            centre = estimate_sym_point(self.images[1], dims=self.centering_dims)
            for dim, coord in centre.items():
                self.cursor_positions_selection[self.dims.index(dim)].setValue(coord)
        except Exception:
            pass
        # Make the coresponding slices
        self._set_slice(0)
        self._set_slice(2)

        # Make plots
        self._set_main_plots()  # Make main plots
        self._set_DC_plots()  # Make DC plots
        self._set_plot_range_limits()  # Set plot range limits
        self._set_plot_labels()  # Set labels
        self._update_cursor_stats_text()  # Update initial cursor stats

        # Connect signals
        self._connect_signals_crosshairs()
        self._connect_signals_cursor_boxes_change()
        self._connect_signals_xh_span_change()
        self._connect_signals_align()
        self._connect_signals_mirror_checkbox()
        self._connect_key_press_signals()

    def _set_slice(self, dim_no):
        """Set a data slice based on the xh positions and widths"""
        _pos = self.cursor_positions_selection[dim_no].value()
        _width = self.cursor_widths_selection[dim_no].value() / 2
        _range = slice(_pos - _width, _pos + _width)
        if _width < self.step_sizes[dim_no]:
            self.images[dim_no] = self.data.sel(
                {self.dims[dim_no]: _pos}, method="nearest"
            )
        else:
            self.images[dim_no] = self.data.sel({self.dims[dim_no]: _range}).mean(
                dim=self.dims[dim_no]
            )

    def _update_cursor_boxes(self, xh_no):
        """Update the cursor boxes when the crosshair is moved."""
        x, y = self.xhs[xh_no].get_pos()
        active_dim_nos = self._get_active_dim_nos(xh_no)
        self.cursor_positions_selection[active_dim_nos[0]].setValue(x)
        self.cursor_positions_selection[active_dim_nos[1]].setValue(y)

    def _update_xh_positions(self, dim_changed):
        """Update the crosshair positions when the cursor boxes are changed."""
        # Update the crosshair positions on the main plots
        for i in range(3):
            active_dim_nos = self._get_active_dim_nos(i)
            if dim_changed in active_dim_nos:
                xh = self.xhs[i]
                xh.set_pos(
                    (
                        self.cursor_positions_selection[active_dim_nos[0]].value(),
                        self.cursor_positions_selection[active_dim_nos[1]].value(),
                    )
                )

        if dim_changed in [0, 2]:
            # Update the alignment crosshairs
            for i, xh in enumerate(self.align_xhs):
                xh.setPos(
                    (
                        self.cursor_positions_selection[2].value(),
                        self.cursor_positions_selection[0].value(),
                    )
                )
                xh.setAngle(self.rotation_selection.value() + i * 90)
            if self.alignment_aid_shape.isVisible():
                self._update_align_shape()

        # Update the crosshair positions on the DCs
        if dim_changed == 0:
            xh = self.DC_plots_xhs[0]
            xh.setRegion(self.xhs[1].get_dim0_span())
        elif dim_changed == 2:
            xh = self.DC_plots_xhs[1]
            xh.setRegion(self.xhs[1].get_dim1_span())

    def _update_xh_widths(self, dim_changed):
        """Update the crosshair widths when the cursor boxes are changed."""
        # Update the crosshairs on the main plots
        for i in range(3):
            active_dim_nos = self._get_active_dim_nos(i)
            if dim_changed in active_dim_nos:
                xh = self.xhs[i]
                xh.set_dim0_width(
                    self.cursor_widths_selection[active_dim_nos[0]].value()
                )
                xh.set_dim1_width(
                    self.cursor_widths_selection[active_dim_nos[1]].value()
                )

        # Update the crosshairs on the DCs
        if dim_changed == 0:
            xh = self.DC_plots_xhs[0]
            xh.setRegion(self.xhs[1].get_dim0_span())
        elif dim_changed == 2:
            xh = self.DC_plots_xhs[1]
            xh.setRegion(self.xhs[1].get_dim1_span())

    def _update_align_xh_angle(self):
        """Update the alignment crosshair angles when the rotation is changed."""
        for i, xh in enumerate(self.align_xhs):
            xh.setAngle(self.rotation_selection.value() + i * 90)
            if self.rotation_selection.value() == 0:
                xh.setVisible(False)
            else:
                xh.setVisible(True)

        if self.alignment_aid_shape.isVisible():
            self._update_align_shape()

        self._update_cursor_stats_text()

    def _update_align_xh_angle_change_by_delta(self, delta):
        """Update the alignment crosshair angles when the rotation is changed by a button press."""
        self.rotation_selection.setValue(self.rotation_selection.value() + delta)

    def _update_align_shape(self):
        """Update the shape of the alignment aid."""
        if self.alignment_aid.currentText() == "None":
            self.alignment_aid_shape.setVisible(False)
        else:
            self.alignment_aid_shape.setVisible(True)
            shape = self.alignment_aid.currentText()
            size = self.alignment_aid_size.value() * self.alignment_aid_scaling
            angle = self.rotation_selection.value()
            center = QtCore.QPointF(
                self.cursor_positions_selection[2].value(),
                self.cursor_positions_selection[0].value(),
            )
            path = self._create_alignment_shape(shape, center, size, angle)
            self.alignment_aid_shape.setPath(path)

    def _update_slices(self, dim_no):
        """Update the data slices when the crosshair is moved."""
        c_levels = self.colorbar.levels()  # Get starting colorbar scale levels
        self._set_slice(dim_no)
        image_item = self.image_items[dim_no]
        image_item.setImage(self.images[dim_no].values)
        if self.cscale_lock.isChecked():
            self.colorbar.setLevels(c_levels)  # Set scale to original

    def _update_DC(self, dim_no):
        """Update the DC plots when the crosshair is moved."""
        if dim_no == 1 or self.mirror_checkbox.isChecked():  # Update both DCs
            DCs_to_update = [0, 1]
        elif dim_no == 0:  # Update only DC1
            DCs_to_update = [1]
        elif dim_no == 2:  # Update only DC0
            DCs_to_update = [0]

        for DC_no in DCs_to_update:
            plot = self.DC_plot_items[DC_no][0]

            # Get the DC
            dim_no = [0, 2][DC_no]  # Select the active dim for the relevant xh plot
            DC = self._select_DC(dim_no)

            # Update the plot - get ordering correct
            if DC_no == 0:
                a, b = DC.data, DC.coords[self.dims[dim_no]].values
            else:
                a, b = DC.coords[self.dims[dim_no]].values, DC.data

            plot.setData(a, b)

            # Check if a mirrored DC exists and needs updating
            if not len(self.DC_plot_items[f"{DC_no}_m"]) == 0:
                plot_m = self.DC_plot_items[f"{DC_no}_m"][0]
                mirror_DC = sym(
                    DC,
                    flipped=True,
                    **{
                        self.dims[dim_no]: self.cursor_positions_selection[
                            dim_no
                        ].value()
                    },
                )
                if DC_no == 0:
                    a, b = mirror_DC.data, mirror_DC.coords[self.dims[dim_no]].values
                else:
                    a, b = mirror_DC.coords[self.dims[dim_no]].values, mirror_DC.data
                plot_m.setData(a, b)

    def _show_hide_mirror(self):
        """Show or hide the mirrored DCs."""
        show_mirror = self.mirror_checkbox.isChecked()
        if not show_mirror:
            for i in range(2):
                if not len(self.DC_plot_items[f"{i}_m"]) == 0:
                    self.DC_plots[i].removeItem(self.DC_plot_items[f"{i}_m"][0])
                self.DC_plot_items[f"{i}_m"] = []
        else:
            for i in range(2):
                if len(self.DC_plot_items[f"{i}_m"]) == 0:
                    self.DC_plot_items[f"{i}_m"].append(
                        self.DC_plots[i].plot(
                            [0],
                            [0],
                            pen=self.DC_mirror_pen,
                        )
                    )

        self._update_DC(1)  # Update the mirror DCs

    def _align_data(self):
        """Estimate symmetry points in the data"""
        # Get the current view range
        x_range, y_range = self.image_plots[1].getViewBox().viewRange()
        data_to_centre = self.images[1].sel(
            {self.dims[0]: slice(*y_range), self.dims[2]: slice(*x_range)}
        )

        # Get the centre of the data
        centre = estimate_sym_point(data_to_centre, dims=self.centering_dims)

        # Activate mirror mode
        self.mirror_checkbox.setChecked(True)
        self._show_hide_mirror()

        # Set main crosshair to the aligned position
        for dim, value in centre.items():
            dim_no = self.dims.index(dim)
            self.cursor_positions_selection[dim_no].setValue(value)

    def _update_cursor_stats_text(self):
        """Update the cursor stats."""
        cursor_text = ""

        # Try and parse normal emission info
        norm_emission = self._get_norm_values()
        cursor_text += _parse_norm_emission_cursor_stats(self.data, norm_emission)
        # Add metadata
        cursor_text += self.metadata_text
        # Set text
        self.cursor_stats.setText(cursor_text)

    def _get_norm_values(self):
        positions = {
            self.dims[i]: self.cursor_positions_selection[i].value() for i in range(3)
        }
        positions["azi_offset"] = -self.rotation_selection.value()
        try:
            norm_emission = self.data.metadata.get_normal_emission_from_values(positions)
        except AttributeError:
            norm_emission = {}

        return norm_emission

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
        # Update display boxes when crosshairs move
        for i, xh_ in enumerate(self.xhs):
            signal = xh_.xh.sigPositionChanged
            signal.connect(partial(self._update_cursor_boxes, i))  # Update cursor boxes
        # Update alignment tool rotation changes
        self.rotation_selection.valueChanged.connect(self._update_align_xh_angle)
        for angle, button in zip(
            self.rotation_delta, self.rotation_delta_buttons, strict=True
        ):
            button.clicked.connect(
                partial(self._update_align_xh_angle_change_by_delta, angle)
            )
        # Update alignment tool shape changes
        self.alignment_aid.currentIndexChanged.connect(self._update_align_shape)
        self.alignment_aid_size.valueChanged.connect(self._update_align_shape)
        # Copy current normal emission values
        self.copy_button.clicked.connect(self._copy_norm_values)

    def _connect_signals_cursor_boxes_change(self):
        # Update when cursor positions change
        for i in range(3):
            signal = self.cursor_positions_selection[i].valueChanged
            signal.connect(partial(self._update_slices, i))
            signal.connect(partial(self._update_DC, i))
            signal.connect(partial(self._update_xh_positions, i))
            signal.connect(self._update_cursor_stats_text)

    def _connect_signals_xh_span_change(self):
        # Update when span ranges changed
        for i in range(3):
            signal = self.cursor_widths_selection[i].valueChanged
            signal.connect(partial(self._update_slices, i))
            signal.connect(partial(self._update_DC, i))
            signal.connect(partial(self._update_xh_widths, i))

    def _connect_signals_align(self):
        # Connect centre data button
        self.align_button.clicked.connect(self._align_data)

    def _connect_signals_mirror_checkbox(self):
        # Connect mirror checkbox
        self.mirror_checkbox.clicked.connect(self._show_hide_mirror)

    def _connect_key_press_signals(self):
        # Connect key press signals
        self.graphics_layout.keyPressed.connect(self._key_press_event)
        self.graphics_layout.keyReleased.connect(self._key_release_event)

    def _key_press_event(self, event):
        # First deal with the modifiers
        if event.key() == self.key_modifiers["move_primary_dim"]:
            self.move_primary_dim_key_enabled = True
        elif event.key() == self.key_modifiers["hide_all"]:
            for xh in self.xhs:
                xh.set_visible(False)
        elif event.key() == self.key_modifiers["show_hide_mirror"]:
            self.mirror_checkbox.setChecked(not self.mirror_checkbox.isChecked())
            self._show_hide_mirror()
        # Now deal with the arrow keys
        elif event.key() in [
            QtCore.Qt.Key.Key_Up,
            QtCore.Qt.Key.Key_Down,
            QtCore.Qt.Key.Key_Left,
            QtCore.Qt.Key.Key_Right,
        ]:
            if self.move_primary_dim_key_enabled:
                if event.key() == QtCore.Qt.Key.Key_Up:
                    self.cursor_positions_selection[1].stepUp()
                elif event.key() == QtCore.Qt.Key.Key_Down:
                    self.cursor_positions_selection[1].stepDown()
            else:
                if event.key() == QtCore.Qt.Key.Key_Up:
                    self.cursor_positions_selection[0].stepUp()
                elif event.key() == QtCore.Qt.Key.Key_Down:
                    self.cursor_positions_selection[0].stepDown()
                elif event.key() == QtCore.Qt.Key.Key_Left:
                    self.cursor_positions_selection[2].stepDown()
                elif event.key() == QtCore.Qt.Key.Key_Right:
                    self.cursor_positions_selection[2].stepUp()

    def _key_release_event(self, event):
        if event.key() == self.key_modifiers["move_primary_dim"]:
            self.move_primary_dim_key_enabled = False
        elif event.key() == self.key_modifiers["hide_all"]:
            for xh in self.xhs:
                xh.set_visible(True)

    # ##############################
    # Helper functions
    # ##############################
    def _get_active_dim_nos(self, i):
        return [dim_no for dim_no in range(3) if dim_no != i]

    def _create_alignment_shape(self, shape, center, size, angle):
        """Create a QPainterPath for the alignment aid."""
        path = QtGui.QPainterPath()

        if shape == "Square":
            points = [
                QtCore.QPointF(
                    center.x() + size * np.cos(np.radians(angle + i * 90 + 45)),
                    center.y() + size * np.sin(np.radians(angle + i * 90 + 45)),
                )
                for i in range(4)
            ]
            path.moveTo(points[0])
            for point in points[1:]:
                path.lineTo(point)
            path.lineTo(points[0])
        elif shape == "Hexagon" or shape == "Hexagon (r30)":
            if shape == "Hexagon (r30)":
                angle += 30
            points = [
                QtCore.QPointF(
                    center.x() + size * np.cos(np.radians(angle + i * 60)),
                    center.y() + size * np.sin(np.radians(angle + i * 60)),
                )
                for i in range(6)
            ]
            path.moveTo(points[0])
            for point in points[1:]:
                path.lineTo(point)
            path.lineTo(points[0])

        elif shape == "Circle":
            rect = QtCore.QRectF(
                center.x() - size, center.y() - size, 2 * size, 2 * size
            )
            path.moveTo(center.x() + size, center.y())
            path.arcTo(rect, 0, 360)

        return path
