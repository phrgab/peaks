"""Functions for the 4D interactive display panel."""

import sys
from functools import partial

import numpy as np
import pyperclip
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..GUI_utils import (
    Crosshair,
    KeyPressGraphicsLayoutWidget,
)
from .disp_2d import _Disp2D


def _disp_4d(data, primary_dim):
    """Display a 4D interactive display panel.

    Parameters
    ------------
    data : list or xarray.DataArray
         Either a single 2D :class:`xarray.DataArray` or a list of 2D :class:`xarray.DataArray` objects.

    primary_dim : tuple of str or list of str
        The primary dimensions for the viewer, will be shown on main data explorer map panel.

    """
    global app  # Ensure the QApplication instance is not garbage collected
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    viewer = _Disp4D(data, primary_dim)
    viewer.show()
    app.exec()


class _Disp4D(QtWidgets.QMainWindow):
    def __init__(self, data, primary_dim):
        super().__init__()

        # Parse data
        if primary_dim:  # Set correct primary axis
            data = data.transpose(primary_dim[0], primary_dim[1], ...)
        self.data = data

        # Set some options
        self.xh_brush = (255, 0, 255, 50)

        # Set up the GUI
        self._init_UI()  # Initialize the layout

    # ##############################
    # GUI layout
    # ##############################
    def _init_UI(self):
        self.setWindowTitle("Display Panel")
        self.setGeometry(100, 100, 1200, 800)

        # Main window layout
        central_widget = QTabWidget()
        self.setCentralWidget(central_widget)
        explorer_tab = QWidget()
        central_widget.addTab(explorer_tab, "Data Explorer")
        ROI_tab = QWidget()
        central_widget.addTab(ROI_tab, "Region of Interest")

        self._init_data()  # Read some basic parameters from the dataset
        self._build_explorer_tab(explorer_tab)
        self._build_roi_tab(ROI_tab)
        self._connect_signals()  # Connect signals
        # self._build_plot_layout(layout)  # Build the basic plot layout
        # self._build_controls_layout(layout)  # Build the control panel layout
        # self._build_menu()  # Add the menu

    def _build_explorer_tab(self, explorer_tab):
        """Build the data explorer tab layout."""
        layout = QHBoxLayout()
        explorer_tab.setLayout(layout)
        self._build_primary_dims_explorer_column(layout)
        # Add a disp2d panel for the dim2-dim3 data
        self.data_explorer_disp2d = self._build_data_explorer_2D_column(
            layout, self.data.isel({self.dims[0]: 0, self.dims[1]: 0})
        )

    def _build_roi_tab(self, roi_tab):
        """Build the region of interest tab layout."""
        layout = QHBoxLayout()
        roi_tab.setLayout(layout)
        self._build_primary_dims_roi_column(layout)
        # Add a disp2d panel for the dim2-dim3 data
        self.roi_disp2d = self._build_data_explorer_2D_column(
            layout, self.data_secondary_tot
        )
        # Add a ROI to the disp2d panel
        self._add_ROI_to_disp2d()

    def _build_primary_dims_explorer_column(self, layout):
        """Build left column - primary dims explorer"""
        left_column_container = QWidget()
        layout.addWidget(left_column_container)
        left_column_container.setMaximumWidth(500)
        left_column_layout = QVBoxLayout(left_column_container)

        _, _, self.primary_dims_xh, _, self.primary_dims_explorer_layout = (
            self._build_primary_dims_explorer_plot(left_column_layout)
        )
        self._build_primary_dims_explorer_controls(left_column_layout)

        left_column_layout.addStretch()

    def _build_primary_dims_roi_column(self, layout):
        """Build left column - primary dims ROI"""
        left_column_container = QWidget()
        layout.addWidget(left_column_container)
        left_column_container.setMaximumWidth(500)
        left_column_layout = QVBoxLayout(left_column_container)

        (
            self.primary_dims_roi_plot,
            self.primary_dims_roi_plot_image_item,
            xh,
            self.primary_dims_roi_plot_colorbar,
            _,
        ) = self._build_primary_dims_explorer_plot(left_column_layout)
        xh.set_visible(False)
        # Add a ROI
        self.roi_dim01 = pg.PolyLineROI(
            positions=self._set_ROI_default_pos(0, 1),
            pen=pg.mkPen((204, 51, 153), width=2.5),
            handlePen=pg.mkPen((204, 51, 153), width=2),
            closed=True,
        )
        self.primary_dims_roi_plot.addItem(self.roi_dim01)
        self._build_roi_controls(left_column_layout)

        left_column_layout.addStretch()

    def _build_primary_dims_explorer_plot(self, left_column_layout):
        # Create a GraphicsLayoutWidget
        primary_dims_explorer_layout = (
            KeyPressGraphicsLayoutWidget._KeyPressGraphicsLayoutWidget()
        )
        primary_dims_explorer_layout.setMaximumWidth(500)
        primary_dims_explorer_layout.viewport().setAttribute(
            QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, False
        )
        left_column_layout.addWidget(primary_dims_explorer_layout)
        primary_dims_explorer_plot = primary_dims_explorer_layout.addPlot(1, 0, 1, 1)
        # Add primary dims image total data to plot
        primary_dims_explorer_plot_image_item = pg.ImageItem(
            self.data_primary_tot.values, axisOrder="row-major"
        )
        primary_dims_explorer_plot.addItem(primary_dims_explorer_plot_image_item)

        # Transform to match data coordinates
        primary_dims_explorer_plot_image_item.setTransform(
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

        # Set view limits
        xmin, xmax = self.ranges[1]
        ymin, ymax = self.ranges[0]
        primary_dims_explorer_plot.getViewBox().setLimits(
            xMin=xmin, xMax=xmax, yMin=ymin, yMax=ymax
        )

        # Set axis labels
        dim_labels = []
        for i in range(2):
            dim_units = self.data.coords[self.dims[i]].attrs.get("units")
            dim_labels.append(
                f"{self.dims[i]} ({dim_units})" if dim_units else self.dims[i]
            )

        primary_dims_explorer_plot.setLabel("bottom", dim_labels[1])
        primary_dims_explorer_plot.setLabel("left", dim_labels[0])

        # Add a crosshair
        primary_dims_xh = Crosshair._Crosshair(
            primary_dims_explorer_plot,
            pos=(
                self.coords[0][0],
                self.coords[1][0],
            ),
            dim0_width=self.step_sizes[0],
            dim1_width=self.step_sizes[1],
            brush=self.xh_brush,
            bounds=[
                self.ranges[0],
                self.ranges[1],
            ],
            axisOrder="row-major",
        )

        # Add colour bar
        crange = (
            float(self.data_primary_tot.min()),
            float(self.data_primary_tot.max()),
        )
        cmap = pg.colormap.get("cividis", source="matplotlib")
        colorbar = pg.ColorBarItem(
            label=self.data.name,
            colorMap=cmap,
            colorMapMenu=True,
            limits=crange,
            rounding=min(abs(crange[1] - crange[0]) / 2000, 1),
            interactive=True,
            orientation="h",
            values=crange,
        )
        colorbar.setImageItem(
            primary_dims_explorer_plot_image_item,
            insert_in=primary_dims_explorer_plot,
        )
        primary_dims_explorer_plot_image_item.setColorMap(cmap)
        primary_dims_explorer_layout.addItem(colorbar, row=0, col=0)

        return (
            primary_dims_explorer_plot,
            primary_dims_explorer_plot_image_item,
            primary_dims_xh,
            colorbar,
            primary_dims_explorer_layout,
        )

    def _build_primary_dims_explorer_controls(self, left_column_layout):
        """Add controls to the primary dims explorer plot."""
        controls_layout = QVBoxLayout()
        left_column_layout.addLayout(controls_layout)
        self.dim01_sliders = []
        self.dim01_binning_boxes = []
        for i in range(2):
            hbox = QHBoxLayout()
            controls_layout.addLayout(hbox)
            label = QLabel(self.dims[i])
            hbox.addWidget(label)
            # Selected slice control
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
            slider.setRange(0, len(self.coords[i]) - 1)
            slider.setValue(0)
            hbox.addWidget(slider)
            self.dim01_sliders.append(slider)
            # Binning control
            label = QLabel("Bin:")
            hbox.addWidget(label)
            binning_box = QtWidgets.QDoubleSpinBox()
            binning_box.setRange(1, 50)
            binning_box.setSingleStep(1)
            binning_box.setValue(1)
            binning_box.setDecimals(0)
            hbox.addWidget(binning_box)
            self.dim01_binning_boxes.append(binning_box)

        # Add a text box to display the current slice
        self.primary_dims_cursor_stats = QLabel()
        self.primary_dims_cursor_stats.setStyleSheet(
            "QLabel { background-color : black; color : white; }"
        )
        self.primary_dims_cursor_stats.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft
        )
        self.primary_dims_cursor_stats.setWordWrap(True)
        self.primary_dims_cursor_stats.setMinimumHeight(60)
        controls_layout.addWidget(self.primary_dims_cursor_stats)
        self._update_primary_dim_cursor_stats()

    def _build_roi_controls(self, left_column_layout):
        """Add controls to the ROI tab."""
        controls_layout = QHBoxLayout()
        left_column_layout.addLayout(controls_layout)
        vbox = QVBoxLayout()
        controls_layout.addLayout(vbox)
        self.roi_list_box = QtWidgets.QListWidget()
        for key in self.ROI_store.keys():
            self.roi_list_box.addItem(key)
        self.roi_list_box.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.MultiSelection
        )
        vbox.addWidget(self.roi_list_box)
        self.clear_ROI_store_button = QtWidgets.QPushButton("Clear all ROIs")
        vbox.addWidget(self.clear_ROI_store_button)

        # Main buttons
        button_layout = QVBoxLayout()
        controls_layout.addLayout(button_layout)
        # Add buttons
        self.setROI_button = QtWidgets.QPushButton("Set ROI from plot")
        button_layout.addWidget(self.setROI_button)
        hbox = QHBoxLayout()
        button_layout.addLayout(hbox)
        hbox.addWidget(QLabel("|---->"))
        self.select_from_dim01_selector = QtWidgets.QRadioButton(
            f"From [{self.dims[0]}, {self.dims[1]}]"
        )
        self.select_from_dim01_selector.setChecked(True)
        hbox.addWidget(self.select_from_dim01_selector)
        hbox.addStretch()
        hbox = QHBoxLayout()
        button_layout.addLayout(hbox)
        hbox.addWidget(QLabel("<----|"))
        self.select_from_dim23_selector = QtWidgets.QRadioButton(
            f"From [{self.dims[2]}, {self.dims[3]}]"
        )
        hbox.addWidget(self.select_from_dim23_selector)
        hbox.addStretch()
        self.ROI_name_box = QtWidgets.QLineEdit()
        button_layout.addWidget(self.ROI_name_box)
        self.addROI_button = QtWidgets.QPushButton("Add to list")
        button_layout.addWidget(self.addROI_button)
        self.removeROI_button = QtWidgets.QPushButton("Remove from list")
        button_layout.addWidget(self.removeROI_button)
        self.setROI_from_list_button = QtWidgets.QPushButton("Set ROI from list")
        button_layout.addWidget(self.setROI_from_list_button)
        self.copy_ROI_store_button = QtWidgets.QPushButton("Copy ROI store")
        button_layout.addWidget(self.copy_ROI_store_button)
        self.resetplots_button = QtWidgets.QPushButton("Reset Plots")
        button_layout.addWidget(self.resetplots_button)
        button_layout.addStretch()

    def _build_data_explorer_2D_column(self, layout, data_to_plot):
        """Right column - data explorer utilising the disp2d panel"""
        right_column_container = QWidget()
        layout.addWidget(right_column_container)
        right_column = QVBoxLayout()
        layout.addLayout(right_column)
        # Add the 2D panel - initialise with the first slice
        data_explorer_disp2d = _Disp2D([data_to_plot], None, None)
        data_explorer_widget = data_explorer_disp2d.centralWidget()
        right_column.addWidget(data_explorer_widget)
        # Hide the unnecessary controls from disp2d
        data_explorer_disp2d.align_button.setVisible(False)
        data_explorer_disp2d.copy_button.setVisible(False)
        data_explorer_disp2d.scans_list.setVisible(False)

        return data_explorer_disp2d

    # ##############################
    # Signals
    # ##############################
    def _connect_signals(self):
        """Connect signals."""
        # Connect signals for primary dims explorer

        # Set sliders position based on xh
        signal = self.primary_dims_xh.xh.sigPositionChanged
        signal.connect(self._update_dim01_slider_position)

        for i in range(2):
            # Set binning for cursors
            signal = self.dim01_binning_boxes[i].valueChanged
            signal.connect(partial(self._update_dim01_binning, i))
            # Update plot when slider changes
            signal = self.dim01_sliders[i].valueChanged
            signal.connect(self._update_data_explorer_disp2d)
            # Move xh when slider dragged
            signal = self.dim01_sliders[i].sliderMoved
            signal.connect(self._update_primary_dim_xhs_from_sliders)

        # Set key press signals
        self.primary_dims_explorer_layout.keyPressed.connect(self._key_press_event)

        # Connect signals for ROI tab
        self.resetplots_button.clicked.connect(self._reset_roi_plots)
        self.setROI_button.clicked.connect(self._set_ROI_from_plot)
        self.addROI_button.clicked.connect(self._add_ROI_to_list)
        self.removeROI_button.clicked.connect(self._remove_selected_ROI_from_list)
        self.clear_ROI_store_button.clicked.connect(self._remove_all_ROIs_from_list)
        self.setROI_from_list_button.clicked.connect(self._set_ROI_from_list)
        self.roi_list_box.itemSelectionChanged.connect(self._select_ROI_from_list)
        self.copy_ROI_store_button.clicked.connect(
            lambda: pyperclip.copy(f".attrs['ROI']={str(self.ROI_store)}")
        )

    # ##############################
    # Data handling
    # ##############################
    def _init_data(self):
        """Extract some core parameters from the data."""
        self.dims = self.data.dims
        self.coords = [self.data.coords[dim].values for dim in self.dims]
        self.coarsened_coords = [coord for coord in self.coords[:2]]
        self.step_sizes = [coord[1] - coord[0] for coord in self.coords]
        self.ranges = [
            (min(coords) - self.step_sizes[i] / 2, max(coords) + self.step_sizes[i] / 2)
            for i, coords in enumerate(self.coords)
        ]
        self.data_span = [abs(range[1] - range[0]) for range in self.ranges]
        self.data_primary_tot = self.data.mean([self.dims[2], self.dims[3]]).compute()
        self.data_secondary_tot = self.data.mean([self.dims[0], self.dims[1]]).compute()

        # Check the real axis name if possible
        self.dim_labels = []
        for dim in self.dims:
            real_axis_name = None
            if hasattr(self.data.metadata, "manipulator"):
                if hasattr(self.data.metadata.manipulator, dim):
                    real_axis_name = getattr(
                        self.data.metadata.manipulator, dim
                    ).local_name
            self.dim_labels.append(
                f"{dim} [{real_axis_name}]" if real_axis_name else dim
            )

        # Make a ROI store
        self.ROI_store = self.data.attrs.get("ROI", {})
        self.display_ROIs_01 = []
        self.display_ROIs_23 = []

    def _update_primary_dim_cursor_stats(self):
        """Update the cursor stats display."""

        r, g, b = self.xh_brush[:3]
        cursor_text = f"<span style='color:#{r:02x}{g:02x}{b:02x}; font-size:15px'>"
        for i, pos in enumerate(self.primary_dims_xh.get_pos()):
            cursor_text += f"&nbsp;&nbsp;{self.dim_labels[i]}:&nbsp;&nbsp; {pos:.2f}<br>"
        cursor_text += "</span>"
        self.primary_dims_cursor_stats.setText(cursor_text)

    def _update_dim01_slider_position(self):
        """Update the slider positions based on the crosshair position."""
        pos = self.primary_dims_xh.get_pos()
        for i in range(2):
            index = self._find_nearest_sorted(self.coarsened_coords[i], pos[i])
            self.dim01_sliders[i].setValue(index)
        self._update_primary_dim_cursor_stats()

    def _update_dim01_binning(self, dim_no):
        """Update the dim0 and dim1 binning."""
        binning = int(self.dim01_binning_boxes[dim_no].value())
        dim = self.dims[dim_no]
        # Update the xh
        getattr(self.primary_dims_xh, f"set_dim{dim_no}_width")(
            self.step_sizes[dim_no] * binning
        )
        # Make a new rolling coordinate
        self.coarsened_coords[dim_no] = (
            self.data[self.dims[dim_no]]
            .rolling(dim={dim: binning}, center=True)
            .mean()
            .dropna(dim)
            .values
        )
        self.dim01_sliders[dim_no].setRange(0, len(self.coarsened_coords[dim_no]) - 1)
        self._update_data_explorer_disp2d()

    def _update_data_explorer_disp2d(self):
        """Update the 2D data explorer plot."""
        # st = time.perf_counter()
        # Find the indices to slice over
        pos = self.primary_dims_xh.get_pos()
        indices = [
            self._find_nearest_n_indexes(
                self.coords[i], pos[i], int(self.dim01_binning_boxes[i].value())
            )
            for i in range(2)
        ]
        # Slice (and if needed average) the data
        data_2d = self.data.data[indices[0], indices[1], :, :]
        if data_2d.ndim == 4:
            data_2d = data_2d.mean((0, 1))
        elif data_2d.ndim == 3:
            data_2d = data_2d.mean(0)

        if not isinstance(data_2d, np.ndarray):
            data_2d = data_2d.compute()

        self.data_explorer_disp2d.data_arrays[0].data = data_2d
        self.data_explorer_disp2d._change_data()

        # print(f"Time to update 2D plot: {time.perf_counter() - st}")

    def _update_primary_dim_xhs_from_sliders(self):
        """Update the primary dims crosshairs from the sliders."""
        # Disconnect signal to avoid circular updates
        signal = self.primary_dims_xh.xh.sigPositionChanged
        signal.disconnect(self._update_dim01_slider_position)

        pos = []
        for i in range(2):
            index = self.dim01_sliders[i].value()
            pos.append(self.coarsened_coords[i][index])
        self.primary_dims_xh.set_pos((pos[0], pos[1]))
        self._update_primary_dim_cursor_stats()

        # Reconnect signal
        signal.connect(self._update_dim01_slider_position)

    def _key_press_event(self, event):
        """Handle key press events."""
        if event.key() in [
            QtCore.Qt.Key.Key_Up,
            QtCore.Qt.Key.Key_Down,
            QtCore.Qt.Key.Key_Left,
            QtCore.Qt.Key.Key_Right,
        ]:
            if event.key() == QtCore.Qt.Key.Key_Up:
                self.dim01_sliders[0].setValue(self.dim01_sliders[0].value() + 1)
            elif event.key() == QtCore.Qt.Key.Key_Down:
                self.dim01_sliders[0].setValue(self.dim01_sliders[0].value() - 1)
            elif event.key() == QtCore.Qt.Key.Key_Left:
                self.dim01_sliders[1].setValue(self.dim01_sliders[1].value() - 1)
            elif event.key() == QtCore.Qt.Key.Key_Right:
                self.dim01_sliders[1].setValue(self.dim01_sliders[1].value() + 1)
            pos = []
            for i in range(2):
                index = self.dim01_sliders[i].value()
                pos.append(self.coarsened_coords[i][index])
            self.primary_dims_xh.set_pos((pos[0], pos[1]))
            self._update_primary_dim_cursor_stats()

    def _reset_roi_plots(self):
        """Reset the ROI plots to the default."""
        self._update_dim01_roi_plot(self.data_primary_tot.values, reset_ROI=True)
        self._update_dim23_roi_plot(self.data_secondary_tot.values, resest_ROI=True)

    def _add_ROI_to_disp2d(self, ROI_points=None):
        """Add the ROI to the right plot."""
        if ROI_points is None:
            ROI_points = self._set_ROI_default_pos(2, 3)
        self.roi_dim23 = pg.PolyLineROI(
            positions=ROI_points,
            pen=pg.mkPen((204, 51, 153), width=2.5),
            handlePen=pg.mkPen((204, 51, 153), width=2),
            closed=True,
        )
        self.roi_disp2d.image_plot.addItem(self.roi_dim23)

    def _get_ROI_from_plot(self):
        # Get the ROI points
        if self.select_from_dim01_selector.isChecked():
            pts_scene = self.roi_dim01.getSceneHandlePositions()
            points = [self.roi_dim01.mapSceneToParent(pt[1]) for pt in pts_scene]
            dims = [self.dims[0], self.dims[1]]
        else:
            pts_scene = self.roi_dim23.getSceneHandlePositions()
            points = [self.roi_dim23.mapSceneToParent(pt[1]) for pt in pts_scene]
            dims = [self.dims[2], self.dims[3]]
        valuesA = [i.y() for i in points]
        valuesB = [i.x() for i in points]
        return {dims[0]: valuesA, dims[1]: valuesB}

    def _set_ROI_from_plot(self):
        """Set the ROI on the data and apply to the plots."""
        ROI_dict = self._get_ROI_from_plot()
        new_data_to_plot = self.data.mask_data(ROI_dict).values
        if self.dims[0] in ROI_dict.keys():
            self._update_dim23_roi_plot(new_data_to_plot)
            self._set_ROI_outline(0, ROI_dict)
        else:
            self._update_dim01_roi_plot(new_data_to_plot)
            self._set_ROI_outline(1, ROI_dict)

    def _set_ROI_from_list(self):
        """Apply the ROI to the data and plots based on list selection"""
        selected_items = self.roi_list_box.selectedItems()
        if len(selected_items) == 0:
            return
        elif len(selected_items) == 1:
            ROI_dict_A = self.ROI_store[selected_items[0].text()]
            # Check this ROI compatible with plot
            if set(ROI_dict_A.keys()) != set(self.dims[:2]) and set(
                ROI_dict_A.keys()
            ) != set(self.dims[2:]):
                selected_items[0].setSelected(False)
                QtWidgets.QMessageBox.warning(
                    self,
                    "ROI cannot be displayed",
                    "ROI definition incompatible with current view.",
                    QtWidgets.QMessageBox.StandardButton.Ok,
                    QtWidgets.QMessageBox.StandardButton.Ok,
                )
                return
            new_data_to_plot = self.data.mask_data(ROI_dict_A).values
        elif len(selected_items) == 2:
            ROI_dict_A = self.ROI_store[selected_items[0].text()]
            ROI_dict_B = self.ROI_store[selected_items[1].text()]
            # Check these ROIs compatible with plot
            if (
                set(ROI_dict_A.keys()) != set(self.dims[:2])
                and set(ROI_dict_A.keys()) != set(self.dims[2:])
                or set(ROI_dict_B.keys()) != set(ROI_dict_A.keys())
            ):
                for item in selected_items:
                    item.setSelected(False)
                QtWidgets.QMessageBox.warning(
                    self,
                    "ROI cannot be displayed",
                    "ROI definitions incompatible with each other or current view.",
                    QtWidgets.QMessageBox.StandardButton.Ok,
                    QtWidgets.QMessageBox.StandardButton.Ok,
                )
                return
            new_data_to_plot = (
                self.data.mask_data(ROI_dict_A) - self.data.mask_data(ROI_dict_B)
            ).values
        if self.dims[0] in ROI_dict_A.keys():
            self._update_dim23_roi_plot(new_data_to_plot)
            if len(selected_items) == 1:
                self._set_ROI_outline(0, ROI_dict_A)
            else:
                self._set_ROI_outline(0, ROI_dict_A, ROI_dict_B)
        else:
            self._update_dim01_roi_plot(new_data_to_plot)
            if len(selected_items) == 1:
                self._set_ROI_outline(1, ROI_dict_A)
            else:
                self._set_ROI_outline(1, ROI_dict_A, ROI_dict_B)

    def _set_ROI_outline(self, plot_no, ROI_dict_A, ROI_dict_B=None):
        """Set the ROI outline on the plot when a ROI region is activated."""
        if plot_no == 0:
            for i in self.display_ROIs_01:
                self.primary_dims_roi_plot.removeItem(i)
            self.display_ROIs_01 = []
        else:
            for i in self.display_ROIs_23:
                self.roi_disp2d.image_plot.removeItem(i)
            self.display_ROIs_23 = []

        if ROI_dict_B:
            ROIs = [ROI_dict_A, ROI_dict_B]
        else:
            ROIs = [ROI_dict_A]
        dimA = self.dims[0] if plot_no == 0 else self.dims[2]
        dimB = self.dims[1] if plot_no == 0 else self.dims[3]
        for ROI in ROIs:
            dimA_data = ROI[dimA]
            dimA_data.extend([dimA_data[0]])
            dimB_data = ROI[dimB]
            dimB_data.extend([dimB_data[0]])

            if plot_no == 0:
                line_plot = pg.PlotDataItem(dimB_data, dimA_data, pen="g")
                self.primary_dims_roi_plot.addItem(line_plot)
                self.display_ROIs_01.append(line_plot)
            else:
                line_plot = pg.PlotDataItem(dimB_data, dimA_data, pen="g")
                self.roi_disp2d.image_plot.addItem(line_plot)
                self.display_ROIs_23.append(line_plot)

    def _update_dim01_roi_plot(self, data, reset_ROI=False):
        """Update the primary dims ROI plot.

        Parameters
        ------------
        data : np.ndarray
            The data to display.
        """
        # Reset the data on the plots
        self.primary_dims_roi_plot_image_item.setImage(data)
        # Reset colour scales
        crange = (float(data.min()), float(data.max()))
        self.primary_dims_roi_plot_colorbar.hi_lim = crange[1]
        self.primary_dims_roi_plot_colorbar.lo_lim = crange[0]
        self.primary_dims_roi_plot_colorbar.setLevels(crange)

        if reset_ROI:
            # Reset the ROI on the left plot
            for i in self.display_ROIs_01:
                self.primary_dims_roi_plot.removeItem(i)
            self.display_ROIs_01 = []
            self.roi_dim01.setSize(1)
            self.roi_dim01.setPos([0, 0])
            self.roi_dim01.setAngle(0)
            self.roi_dim01.setPoints(self._set_ROI_default_pos(0, 1))

    def _update_dim23_roi_plot(self, data, resest_ROI=False):
        """Update the secondary dims ROI plot.

        Parameters
        ------------
        data : np.ndarray
            The data to display.
        """
        if not resest_ROI:
            pts_scene = self.roi_dim23.getSceneHandlePositions()
            ROI_points = [self.roi_dim23.mapSceneToParent(pt[1]) for pt in pts_scene]
        self.roi_disp2d.data_arrays[0].data = data
        self.roi_disp2d._change_data()

        # Re-add the ROI to the right plot
        if resest_ROI:
            self._add_ROI_to_disp2d()
        else:
            self._add_ROI_to_disp2d(ROI_points)

    def _add_ROI_to_list(self):
        """Add the current ROI to the list."""

        name = self.ROI_name_box.text()
        # Ensure name unique
        names = [
            self.roi_list_box.item(i).text() for i in range(self.roi_list_box.count())
        ]
        if not name or name in names:
            i = 1
            while f"{name}_({i})" in names:
                i += 1
            name = f"{name}_({i})"
        # Add the name to the list
        self.roi_list_box.addItem(name)
        # Add it to ROI store
        self.ROI_store[name] = self._get_ROI_from_plot()

    def _remove_selected_ROI_from_list(self):
        """Remove the selected ROI(s) from the list."""
        selected_items = self.roi_list_box.selectedItems()
        if not selected_items:
            return

        for item in selected_items:
            del self.ROI_store[item.text()]
            self.roi_list_box.takeItem(self.roi_list_box.row(item))

    def _remove_all_ROIs_from_list(self):
        """Remove all ROIs from the list."""
        items = [self.roi_list_box.item(i) for i in range(self.roi_list_box.count())]

        # Create a confirmation dialog
        reply = QtWidgets.QMessageBox.question(
            self,
            "Confirm Deletion",
            "Are you sure you want to delete all ROI(s)?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )

        # If the user confirms, proceed with deletion
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            for item in items:
                del self.ROI_store[item.text()]
                self.roi_list_box.takeItem(self.roi_list_box.row(item))

    def _select_ROI_from_list(self):
        selected_items = self.roi_list_box.selectedItems()
        modifiers = QtWidgets.QApplication.keyboardModifiers()

        if modifiers == QtCore.Qt.KeyboardModifier.ShiftModifier:
            if len(selected_items) > 2:
                self.roi_list_box.blockSignals(True)
                for item in selected_items[:-2]:
                    item.setSelected(False)
                self.roi_list_box.blockSignals(False)
        else:
            if len(selected_items) > 1:
                self.roi_list_box.blockSignals(True)
                for item in selected_items[:-1]:
                    item.setSelected(False)
                self.roi_list_box.blockSignals(False)

    # ##############################
    # Helper functions
    # ##############################
    def _find_nearest_sorted(self, array, value):
        """Find the index of the nearest point to a given value in an ordered NumPy array.

        Parameters
        ------------
        array : np.ndarray
            The input ordered array.
        value : float
            The value to find the nearest neighbor to.

        Returns
        ------------
        int
            The index of the nearest point to the given value.
        """
        array = np.asarray(array)
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (
            idx == len(array)
            or np.abs(value - array[idx - 1]) < np.abs(value - array[idx])
        ):
            return idx - 1
        else:
            return idx

    def _find_nearest_n_indexes(self, array, value, n):
        """
        Find the indexes of the N nearest points to a given value in an ordered NumPy array.

        Parameters
        ----------
        array : np.ndarray
            The input ordered array.
        value : float
            The value to find the nearest neighbors to.
        n : int
            The number of nearest neighbors to find.

        Returns
        -------
        np.ndarray
            The indexes of the nearest N points to the given value.
        """
        idx = np.searchsorted(array, value)
        left = np.clip(idx - n, 0, len(array))
        right = np.clip(idx + n, 0, len(array))
        window = array[left:right]
        sorted_indexes = np.argsort(np.abs(window - value))
        nearest_indexes = np.arange(left, right)[sorted_indexes[:n]]
        start, end = nearest_indexes.min(), nearest_indexes.max() + 1
        return slice(start, end)

    def _set_ROI_default_pos(self, dim0, dim1):
        return [
            [
                np.percentile(self.ranges[dim1], 45),
                np.percentile(self.ranges[dim0], 45),
            ],
            [
                np.percentile(self.ranges[dim1], 55),
                np.percentile(self.ranges[dim0], 45),
            ],
            [
                np.percentile(self.ranges[dim1], 55),
                np.percentile(self.ranges[dim0], 55),
            ],
            [
                np.percentile(self.ranges[dim1], 45),
                np.percentile(self.ranges[dim0], 55),
            ],
        ]
