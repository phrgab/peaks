# -*- coding: utf-8 -*-
# 4d ARPES display for spatial map data inspection and ROI analysis
import time

from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import numpy as np
from matplotlib import cm
from functools import partial
import inspect
import pyperclip

from peaks.core.display import ROI_select
from peaks.core.fileIO.fileIO_opts import BL_angles
from peaks.core.GUI.pyqt_CursorItem import CursorItem
from peaks.core.GUI.pyqt_ZoomItem import ZoomItem
from peaks.core.GUI.pyqt_PlotWidgetKP import PlotWidgetKP
from peaks.core.GUI.pyqt_ResizeItem import ResizeItem
from peaks.core.utils.misc import cell_above

# Class to create 4d display panel object
class Ui_disp4d_Panel(object):

    # Run when AlignFS Panel object is created
    def __init__(self, disp4d_panel):
        # Store the framework of the app
        self.disp4d_panel = disp4d_panel

        # Retrieve user preferences
        self.mySetup()

        # Initialize the GUI
        self.setupUi()

        # Resize GUI elements
        self.GUI_resize()

        # GUI internal objects initialization
        self.GUI_internal()

    ############################################################################

    # User preferences
    def mySetup(self):
        ### General:

        # Set pyqtgraph background color
        pg.setConfigOptions(background='k')

        # Define default Colormap
        self.colormap = self.get_mpl_colormap('Greys')

        # Initial padding in percentage of total range
        self.pd = 0.05

        # Default cursor position (offset from center in percentage of total range)
        self.offcs = [-0.05, +0.05, -0.15, +0.15]

        ### Cursors properties:

        # Size
        self.csr_sizes = [15, 15, 15, 15]

        # Symbol
        self.csr_symbols = [['+'], ['+'], ['+'], ['+']]

        # Brush
        self.csr_brushs = ['g', 'r', (255, 131, 0), (153, 90, 233)]

        # Text
        self.csr_texts = ["1", "2", "", ""]

        # Text color
        self.csr_textcs = ['g', 'r', (255, 131, 0), (153, 90, 233)]

        # Crosshair state
        # [] -> none
        # [0] -> vertical
        # [1] -> horizontal
        # [2] -> both
        self.csrs = [2, 2, 2, 2]

        # Crosshair color
        self.csrcs = ['g', 'r', (255, 131, 0), (153, 90, 233)]

        # Integration-crosshair spanx
        self.ispanx = 0
        self.ispanx2 = 0

        # Integration-crosshair spany
        self.ispany = 0
        self.ispany2 = 0

        # Integration-crosshair state
        # [] -> none
        # [0] -> vertical
        # [1] -> horizontal
        # [2] -> both
        self.icsrs = [2, 2, 2, 2]

        # Integration-crosshair color
        self.icsrcs = [(0, 255, 0), (255, 0, 0), (255, 131, 0), (153, 90, 233)]

        # Integration-crosshair brush (RGBA)
        self.ibrushs = [(0, 255, 0, 50), (255, 0, 0, 50), (255, 131, 0, 50), (153, 90, 233, 50)]

        # Allow/deny vertical/horizontal cursor movements
        self.h = True
        self.v = True

        ### Keyboard commands:

        # Activate arrows movement key

        # (Cursor 1 moves using the arrows only)
        self.key2 = QtCore.Qt.Key.Key_Shift  # (For some reason the shift key does not auto-repeat when pressed)
        self.key3 = QtCore.Qt.Key.Key_Y
        self.key4 = QtCore.Qt.Key.Key_X
        self.groupkey = QtCore.Qt.Key.Key_C

        # Show/hide cursor key
        self.shkeys = [QtCore.Qt.Key.Key_1, QtCore.Qt.Key.Key_2, QtCore.Qt.Key.Key_3, QtCore.Qt.Key.Key_4]

        # Switch crosshair state key
        self.switchkeys = [QtCore.Qt.Key.Key_Q, QtCore.Qt.Key.Key_W, QtCore.Qt.Key.Key_E, QtCore.Qt.Key.Key_R]

        # Show/hide DCs key
        self.DCkeys = [QtCore.Qt.Key.Key_A, QtCore.Qt.Key.Key_S, QtCore.Qt.Key.Key_D, QtCore.Qt.Key.Key_F]

        # Switch integration-crosshair state key
        self.iswitchkeys = [QtCore.Qt.Key.Key_5, QtCore.Qt.Key.Key_6, QtCore.Qt.Key.Key_7, QtCore.Qt.Key.Key_8]

        # Switch integration-crosshair fade key
        self.fadekey = QtCore.Qt.Key.Key_G

        # Hide/show all active objects key
        self.allkey = QtCore.Qt.Key.Key_Space

        # Reset cursors key
        self.resetkey = QtCore.Qt.Key.Key_Return

        ### Label properties:

        # Label color
        self.labelcs = ['g', 'g', 'g', 'r', 'y', 'y', (255, 131, 0)]
        self.labelcs2 = ['r', 'r', 'g', 'r', 'y', 'y', (255, 131, 0)]

        ### Side plots properties

        # Plot color
        self.sidecs = [(0, 255, 150), (255, 0, 150), (255, 131, 150), (153, 150, 233)]

        ### Whole range integration properties

        # Plot color
        self.peniMDC = (0, 255, 255)
        self.peniEDC = (255, 255, 0)

        # Region brush
        self.brushiX = (255, 255, 0, 50)
        self.brushiY = (0, 255, 255, 50)

        # Region pen
        self.peniX = (255, 255, 0)
        self.peniY = (0, 255, 255)

        ### CursorStats format

        # Font size
        self.fontsz = "12"

        # Digits after decimal point
        self.digits = "3"

        # drc color (in label6)
        self.drcc = "cyan"

        ### Half-span/step digits after decimal point (dimension units mode)
        self.spandec = 4

        ### Create cursors dictionaries (do not edit)
        self.dictsMain = []
        self.dictsMain2 = []
        self.dictsMap = []
        for i in range(4):
            self.dictsMain.append(
                {'dtype': float, 'size': self.csr_sizes[i], 'symbol': self.csr_symbols[i],
                 'brush': self.csr_brushs[i],
                 'text': self.csr_texts[i], 'textc': self.csr_textcs[i], 'pxMode': True, 'csr': self.csrs[i],
                 'csrc': self.csrcs[i], 'ispanx': self.ispanx, 'ispany': self.ispany, 'icsr': self.icsrs[i],
                 'icsrc': self.icsrcs[i], 'ibrush': self.ibrushs[i], 'h': self.h, 'v': self.v})
            self.dictsMain2.append(
                {'dtype': float, 'size': self.csr_sizes[i], 'symbol': self.csr_symbols[i],
                 'brush': self.csr_brushs[i],
                 'text': self.csr_texts[i], 'textc': self.csr_textcs[i], 'pxMode': True, 'csr': self.csrs[i],
                 'csrc': self.csrcs[i], 'ispanx': self.ispanx2, 'ispany': self.ispany2, 'icsr': self.icsrs[i],
                 'icsrc': self.icsrcs[i], 'ibrush': self.ibrushs[i], 'h': self.h, 'v': self.v})
        i=0
        self.dictsMap.append(
            {'dtype': float, 'size': self.csr_sizes[i], 'symbol': self.csr_symbols[i],
             'brush': self.csr_brushs[i],
             'text': self.csr_texts[i], 'textc': self.csr_textcs[i], 'pxMode': True, 'csr': self.csrs[i],
             'csrc': self.csrcs[i], 'ispanx': self.ispanx, 'ispany': self.ispany, 'icsr': self.icsrs[i],
             'icsrc': self.icsrcs[i], 'ibrush': self.ibrushs[i], 'h': self.h, 'v': self.v})


        # ROI cursor formats
        self.ROI_map_pen = pg.mkPen(color=(255,0,0))
        self.ROI_disp_pen = pg.mkPen(color=(255, 0, 0))

    ############################################################################
    # Initialize the GUI
    def setupUi(self):

        ### Define general GUI elements and geometry:

        # Specify main window properties
        self.disp4d_panel.setObjectName("disp4d_panel")
        self.disp4d_panel.setEnabled(True)
        self.disp4d_panel.resize(1180, 653)

        # Define central widget within main window
        self.centralwidget = QtWidgets.QWidget(self.disp4d_panel)
        self.centralwidget.setMouseTracking(False)
        self.centralwidget.setAutoFillBackground(False)
        self.centralwidget.setObjectName("centralwidget")
        # self.CursorMode = QtWidgets.QLabel(self.centralwidget)
        # self.CursorMode.setGeometry(QtCore.QRect(1010, 590, 171, 16))
        # self.CursorMode.setObjectName("CursorMode")

        # Set central widget background color
        self.centralwidget.setAutoFillBackground(False)
        p = self.centralwidget.palette()
        p.setColor(self.centralwidget.backgroundRole(), QtCore.Qt.GlobalColor.white) # Input color here
        self.centralwidget.setPalette(p)

        # Set main tab
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 1181, 591))
        self.tabWidget.setObjectName("tabWidget")

        #######################################
        #######################################
        # Map explorer tab
        self.map = QtWidgets.QWidget()
        self.map.setObjectName("map")
        self.tabWidget.addTab(self.map, "")

        # Define Plot frames
        # Dispersion
        self.MainPlot = PlotWidgetKP(self.map)
        self.MainPlot.setGeometry(QtCore.QRect(470, 0, 351, 381))
        self.MainPlot.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))
        self.MainPlot.setObjectName("MainPlot")

        # MDC
        self.MDCPlot = pg.PlotWidget(self.map)
        self.MDCPlot.setGeometry(QtCore.QRect(470, 380, 351, 161))
        self.MDCPlot.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))
        self.MDCPlot.setObjectName("MDCPlot")

        # EDC
        self.EDCPlot = pg.PlotWidget(self.map)
        self.EDCPlot.setGeometry(QtCore.QRect(820, 0, 191, 381))
        self.EDCPlot.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))
        self.EDCPlot.setObjectName("EDCPlot")

        # Spatial map
        self.MapPlot = PlotWidgetKP(self.map)
        self.MapPlot.setGeometry(QtCore.QRect(20, 90, 321, 281))
        self.MapPlot.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))
        self.MapPlot.setObjectName("MapPlot")

        #######################################
        # Define CursorStats frame
        # CursorStats must not be in contact with Plots:
        # (to avoid random artefacts in the cursor stats labels when moving the cursors)
        self.CursorStats = pg.GraphicsLayoutWidget(self.map)
        self.CursorStats.setGeometry(QtCore.QRect(20, 380, 371, 161))
        self.CursorStats.setObjectName("CursorStats")
        self.CursorStats.ci.setSpacing(0)  # Set to zero the spacing between the cells in the GraphicsLayout
        self.CursorStats.ci.setBorder(color='k')  # Color the borders between the cells in the GraphicsLayout (useful to see the cell limits)

        #######################################
        # Define the map position selections
        self.x1_select = QtWidgets.QSlider(self.map)
        self.x1_select.setGeometry(QtCore.QRect(50, 20, 251, 22))
        self.x1_select.setMaximum(100)
        self.x1_select.setProperty("value", 0)
        self.x1_select.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.x1_select.setInvertedAppearance(False)
        self.x1_select.setInvertedControls(False)
        self.x1_select.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.x1_select.setObjectName("x1_select")

        self.x2_select = QtWidgets.QSlider(self.map)
        self.x2_select.setGeometry(QtCore.QRect(50, 50, 251, 22))
        self.x2_select.setMaximum(100)
        self.x2_select.setProperty("value", 0)
        self.x2_select.setSliderPosition(0)
        self.x2_select.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.x2_select.setInvertedAppearance(False)
        self.x2_select.setInvertedControls(False)
        self.x2_select.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.x2_select.setObjectName("x2_select")

        # Labels
        self.Label_xsel1 = QtWidgets.QLabel(self.map)
        self.Label_xsel1.setGeometry(QtCore.QRect(20, 20, 21, 16))
        self.Label_xsel1.setObjectName("Label_xsel1")
        self.Label_xsel2 = QtWidgets.QLabel(self.map)
        self.Label_xsel2.setGeometry(QtCore.QRect(20, 50, 21, 16))
        self.Label_xsel2.setObjectName("Label_xsel1_2")

        # Binning selection
        self.Bin_x1 = QtWidgets.QDoubleSpinBox(self.map)
        self.Bin_x1.setGeometry(QtCore.QRect(320, 20, 61, 24))
        self.Bin_x1.setLocale(QtCore.QLocale(QtCore.QLocale.Language.English, QtCore.QLocale.Country.UnitedStates))
        self.Bin_x1.setObjectName("Bin_x1")

        self.Bin_x2 = QtWidgets.QDoubleSpinBox(self.map)
        self.Bin_x2.setGeometry(QtCore.QRect(320, 50, 61, 24))
        self.Bin_x2.setLocale(QtCore.QLocale(QtCore.QLocale.Language.English, QtCore.QLocale.Country.UnitedStates))
        self.Bin_x2.setObjectName("Bin_x1")

        self.Bin_label = QtWidgets.QLabel(self.map)
        self.Bin_label.setGeometry(QtCore.QRect(330, 0, 60, 16))
        self.Bin_label.setObjectName("Bin_label")

        #######################################
        # DC selection options
        # DC labels
        self.LabelX_1 = QtWidgets.QLabel(self.map)
        self.LabelX_1.setGeometry(QtCore.QRect(860, 505, 60, 16))
        self.LabelX_1.setObjectName("LabelX_1")
        self.LabelY_1 = QtWidgets.QLabel(self.map)
        self.LabelY_1.setGeometry(QtCore.QRect(860, 535, 60, 16))
        self.LabelY_1.setObjectName("LabelY_1")

        # Integrate DCs
        self.IntX_sel1 = QtWidgets.QCheckBox(self.map)
        self.IntX_sel1.setGeometry(QtCore.QRect(1100, 500, 51, 21))
        self.IntX_sel1.setObjectName("IntX_sel1")
        self.IntY_sel1 = QtWidgets.QCheckBox(self.map)
        self.IntY_sel1.setGeometry(QtCore.QRect(1100, 530, 51, 21))
        self.IntY_sel1.setObjectName("IntY_sel1")

        # Integrate over finite span
        self.SpanX_1 = QtWidgets.QDoubleSpinBox(self.map)
        self.SpanX_1.setGeometry(QtCore.QRect(930, 500, 91, 24))
        self.SpanX_1.setLocale(QtCore.QLocale(QtCore.QLocale.Language.English, QtCore.QLocale.Country.UnitedStates))
        self.SpanX_1.setObjectName("SpanX_1")
        self.SpanY_1 = QtWidgets.QDoubleSpinBox(self.map)
        self.SpanY_1.setGeometry(QtCore.QRect(930, 530, 91, 24))
        self.SpanY_1.setLocale(QtCore.QLocale(QtCore.QLocale.Language.English, QtCore.QLocale.Country.UnitedStates))
        self.SpanY_1.setObjectName("SpanY_1")
        self.SpanY_1.setDecimals(3)
        self.LabelHSpan_1 = QtWidgets.QLabel(self.map)
        self.LabelHSpan_1.setGeometry(QtCore.QRect(940, 480, 71, 16))
        self.LabelHSpan_1.setObjectName("LabelHSpan_1")

        # Delta step sizes
        self.DeltaX_1 = QtWidgets.QLineEdit(self.map)
        self.DeltaX_1.setGeometry(QtCore.QRect(1030, 500, 61, 24))
        self.DeltaX_1.setObjectName("DeltaX_1")
        self.DeltaY_1 = QtWidgets.QLineEdit(self.map)
        self.DeltaY_1.setGeometry(QtCore.QRect(1030, 530, 61, 24))
        self.DeltaY_1.setObjectName("DeltaY_1")
        self.LabelStep_1 = QtWidgets.QLabel(self.map)
        self.LabelStep_1.setGeometry(QtCore.QRect(1040, 480, 60, 16))
        self.LabelStep_1.setObjectName("LabelStep_1")

        # Bilateral link
        self.Bilateral_1 = QtWidgets.QCheckBox(self.map)
        self.Bilateral_1.setGeometry(QtCore.QRect(1020, 380, 101, 20))
        self.Bilateral_1.setChecked(True)
        self.Bilateral_1.setObjectName("Bilateral_1")


        #######################################
        # Contrast
        # Low cutoff slider
        self.LCutOff_1 = QtWidgets.QSlider(self.map)
        self.LCutOff_1.setGeometry(QtCore.QRect(840, 450, 251, 22))
        self.LCutOff_1.setMaximum(100)
        self.LCutOff_1.setProperty("value", 0)
        self.LCutOff_1.setSliderPosition(0)
        self.LCutOff_1.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.LCutOff_1.setInvertedAppearance(False)
        self.LCutOff_1.setInvertedControls(False)
        self.LCutOff_1.setObjectName("LCutOff_1")

        # High cutoff slider
        self.HCutOff_1 = QtWidgets.QSlider(self.map)
        self.HCutOff_1.setGeometry(QtCore.QRect(840, 420, 251, 22))
        self.HCutOff_1.setMaximum(100)
        self.HCutOff_1.setProperty("value", 100)
        self.HCutOff_1.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.HCutOff_1.setInvertedAppearance(False)
        self.HCutOff_1.setInvertedControls(False)
        self.HCutOff_1.setObjectName("HCutOff_1")

        # Invert cmap
        self.cmap_inv_sel1 = QtWidgets.QCheckBox(self.map)
        self.cmap_inv_sel1.setGeometry(QtCore.QRect(1100, 420, 51, 21))
        self.cmap_inv_sel1.setObjectName("cmap_inv_sel1")

        # Select cmap
        self.Cmaps_1 = QtWidgets.QComboBox(self.map)
        self.Cmaps_1.setGeometry(QtCore.QRect(850, 390, 151, 26))
        self.Cmaps_1.setEditable(False)
        self.Cmaps_1.setCurrentText("Greys")
        self.Cmaps_1.setObjectName("Cmaps")
        self.Cmaps_1.addItems(["Greys", "Blues", "viridis", "plasma", "inferno", "magma", "cividis", "bone", "spring", "summer", "autumn", "winter", "cool", "jet", "ocean", "RdBu", "BrBG", "PuOr", "coolwarm", "bwr", "seismic"])  # Choose your favourites

        # Contrast multiplier
        self.cont_mult_lab = QtWidgets.QLabel(self.map)
        self.cont_mult_lab.setGeometry(QtCore.QRect(1100, 450, 21, 16))
        self.cont_mult_lab.setObjectName("x")
        self.cont_mult = QtWidgets.QLineEdit(self.map)
        self.cont_mult.setGeometry(QtCore.QRect(1120, 450, 41, 24))
        self.cont_mult.setObjectName("cont_mult")
        self.cont_mult.setText('1')

        #######################################
        #######################################
        # ROI analysis tab
        self.ROI = QtWidgets.QWidget()
        self.ROI.setObjectName("ROI")
        self.tabWidget.addTab(self.ROI, "")

        # ROI list
        self.ROI_list = QtWidgets.QListWidget(self.ROI)
        self.ROI_list.setGeometry(QtCore.QRect(1020, 30, 131, 351))
        self.ROI_list.setObjectName("ROI_list")
        self.ROI_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
        self.ROI_list_label = QtWidgets.QLabel(self.ROI)
        self.ROI_list_label.setGeometry(QtCore.QRect(1020, 10, 71, 16))
        self.ROI_list_label.setObjectName("ROI_list_label")
        self.delROI = QtWidgets.QPushButton(self.ROI)
        self.delROI.setGeometry(QtCore.QRect(1100, 0, 51, 32))
        self.delROI.setObjectName("delROI")

        # ROI setting objects
        self.setROI = QtWidgets.QPushButton(self.ROI)
        self.setROI.setGeometry(QtCore.QRect(350, 110, 113, 32))
        self.setROI.setObjectName("setROI")
        self.ROI_set_name = QtWidgets.QLineEdit(self.ROI)
        self.ROI_set_name.setGeometry(QtCore.QRect(360, 250, 91, 24))
        self.ROI_set_name.setObjectName("ROI_set_name")
        self.saveROI = QtWidgets.QPushButton(self.ROI)
        self.saveROI.setGeometry(QtCore.QRect(350, 280, 113, 32))
        self.saveROI.setObjectName("saveROI")
        self.resetMap = QtWidgets.QPushButton(self.ROI)
        self.resetMap.setGeometry(QtCore.QRect(350, 310, 113, 32))
        self.resetMap.setObjectName("resetMap")
        self.resetDisp = QtWidgets.QPushButton(self.ROI)
        self.resetDisp.setGeometry(QtCore.QRect(350, 340, 113, 32))
        self.resetDisp.setObjectName("resetDisp")
        self.ROI_from_map = QtWidgets.QRadioButton(self.ROI)
        self.ROI_from_map.setGeometry(QtCore.QRect(360, 160, 100, 20))
        self.ROI_from_map.setObjectName("ROI_from_map")
        self.ROI_from_disp = QtWidgets.QRadioButton(self.ROI)
        self.ROI_from_disp.setGeometry(QtCore.QRect(360, 200, 100, 20))
        self.ROI_from_disp.setObjectName("ROI_from_disp")
        self.ROI_from_disp.setChecked(True)
        self.ROI_select_label = QtWidgets.QLabel(self.ROI)
        self.ROI_select_label.setGeometry(QtCore.QRect(360, 140, 101, 16))
        self.ROI_select_label.setObjectName("ROI_select_label")
        self.ROI_select_label_2 = QtWidgets.QLabel(self.ROI)
        self.ROI_select_label_2.setGeometry(QtCore.QRect(360, 220, 101, 16))
        self.ROI_select_label_2.setObjectName("ROI_select_label_2")

        # Define Plot frames
        # Dispersion
        self.MainPlot_2 = PlotWidgetKP(self.ROI)
        self.MainPlot_2.setGeometry(QtCore.QRect(470, 0, 351, 381))
        self.MainPlot_2.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))
        self.MainPlot_2.setObjectName("MainPlot_2")

        # MDC
        self.MDCPlot_2 = pg.PlotWidget(self.ROI)
        self.MDCPlot_2.setGeometry(QtCore.QRect(470, 380, 351, 161))
        self.MDCPlot_2.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))
        self.MDCPlot_2.setObjectName("MDCPlot_2")

        # EDC
        self.EDCPlot_2 = pg.PlotWidget(self.ROI)
        self.EDCPlot_2.setGeometry(QtCore.QRect(820, 0, 191, 381))
        self.EDCPlot_2.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))
        self.EDCPlot_2.setObjectName("EDCPlot_2")

        # Spatial map
        self.MapPlot_2 = PlotWidgetKP(self.ROI)
        self.MapPlot_2.setGeometry(QtCore.QRect(20, 90, 321, 281))
        self.MapPlot_2.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))
        self.MapPlot_2.setObjectName("MapPlot_2")

        #######################################
        # Define CursorStats frame
        # CursorStats must not be in contact with Plots:
        # (to avoid random artefacts in the cursor stats labels when moving the cursors)
        self.CursorStats_3 = pg.GraphicsLayoutWidget(self.ROI)
        self.CursorStats_3.setGeometry(QtCore.QRect(20, 380, 371, 161))
        self.CursorStats_3.setObjectName("CursorStats_3")
        self.CursorStats_3.ci.setSpacing(0)  # Set to zero the spacing between the cells in the GraphicsLayout
        self.CursorStats_3.ci.setBorder(color='k')  # Color the borders between the cells in the GraphicsLayout (useful to see the cell limits)

        #######################################
        # DC selection options
        # DC labels
        self.LabelX_3 = QtWidgets.QLabel(self.ROI)
        self.LabelX_3.setGeometry(QtCore.QRect(860, 505, 60, 16))
        self.LabelX_3.setObjectName("LabelX_3")
        self.LabelY_3 = QtWidgets.QLabel(self.ROI)
        self.LabelY_3.setGeometry(QtCore.QRect(860, 535, 60, 16))
        self.LabelY_3.setObjectName("LabelY_3")

        # Integrate DCs
        self.IntX_sel3 = QtWidgets.QCheckBox(self.ROI)
        self.IntX_sel3.setGeometry(QtCore.QRect(1100, 500, 51, 21))
        self.IntX_sel3.setObjectName("IntX_sel3")
        self.IntY_sel3 = QtWidgets.QCheckBox(self.ROI)
        self.IntY_sel3.setGeometry(QtCore.QRect(1100, 530, 51, 21))
        self.IntY_sel3.setObjectName("IntY_sel3")

        # Integrate over finite span
        self.SpanX_3 = QtWidgets.QDoubleSpinBox(self.ROI)
        self.SpanX_3.setGeometry(QtCore.QRect(930, 500, 91, 24))
        self.SpanX_3.setLocale(QtCore.QLocale(QtCore.QLocale.Language.English, QtCore.QLocale.Country.UnitedStates))
        self.SpanX_3.setObjectName("SpanX_3")
        self.SpanY_3 = QtWidgets.QDoubleSpinBox(self.ROI)
        self.SpanY_3.setGeometry(QtCore.QRect(930, 530, 91, 24))
        self.SpanY_3.setLocale(QtCore.QLocale(QtCore.QLocale.Language.English, QtCore.QLocale.Country.UnitedStates))
        self.SpanY_3.setObjectName("SpanY_3")
        self.SpanY_3.setDecimals(3)
        self.LabelHSpan_3 = QtWidgets.QLabel(self.ROI)
        self.LabelHSpan_3.setGeometry(QtCore.QRect(940, 480, 71, 16))
        self.LabelHSpan_3.setObjectName("LabelHSpan_3")

        # Delta step size
        self.DeltaX_3 = QtWidgets.QLineEdit(self.ROI)
        self.DeltaX_3.setGeometry(QtCore.QRect(1030, 500, 61, 24))
        self.DeltaX_3.setObjectName("DeltaX_3")
        self.DeltaY_3 = QtWidgets.QLineEdit(self.ROI)
        self.DeltaY_3.setGeometry(QtCore.QRect(1030, 530, 61, 24))
        self.DeltaY_3.setObjectName("DeltaY_3")
        self.LabelStep_3 = QtWidgets.QLabel(self.ROI)
        self.LabelStep_3.setGeometry(QtCore.QRect(1040, 480, 60, 16))
        self.LabelStep_3.setObjectName("LabelStep_3")


        # Bilateral link
        self.Bilateral_3 = QtWidgets.QCheckBox(self.ROI)
        self.Bilateral_3.setGeometry(QtCore.QRect(1020, 380, 101, 20))
        self.Bilateral_3.setChecked(True)
        self.Bilateral_3.setObjectName("Bilateral_3")

        #######################################
        # Contrast - dispersion
        # Low cutoff slider
        self.LCutOff_3 = QtWidgets.QSlider(self.ROI)
        self.LCutOff_3.setGeometry(QtCore.QRect(840, 450, 251, 22))
        self.LCutOff_3.setMaximum(100)
        self.LCutOff_3.setProperty("value", 0)
        self.LCutOff_3.setSliderPosition(0)
        self.LCutOff_3.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.LCutOff_3.setInvertedAppearance(False)
        self.LCutOff_3.setInvertedControls(False)
        self.LCutOff_3.setObjectName("LCutOff_3")

        # High cutoff slider
        self.HCutOff_3 = QtWidgets.QSlider(self.ROI)
        self.HCutOff_3.setGeometry(QtCore.QRect(840, 420, 251, 22))
        self.HCutOff_3.setMaximum(100)
        self.HCutOff_3.setProperty("value", 100)
        self.HCutOff_3.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.HCutOff_3.setInvertedAppearance(False)
        self.HCutOff_3.setInvertedControls(False)
        self.HCutOff_3.setObjectName("HCutOff_3")

        # Invert cmap
        self.cmap_inv_sel3 = QtWidgets.QCheckBox(self.ROI)
        self.cmap_inv_sel3.setGeometry(QtCore.QRect(1100, 420, 51, 21))
        self.cmap_inv_sel3.setObjectName("cmap_inv_sel3")

        # Select cmap
        self.Cmaps_3 = QtWidgets.QComboBox(self.ROI)
        self.Cmaps_3.setGeometry(QtCore.QRect(850, 390, 151, 26))
        self.Cmaps_3.setEditable(False)
        self.Cmaps_3.setCurrentText("Greys")
        self.Cmaps_3.setObjectName("Cmaps")
        self.Cmaps_3.addItems(["Greys", "Blues", "viridis", "plasma", "inferno", "magma", "cividis", "bone", "spring", "summer", "autumn", "winter", "cool", "jet", "ocean", "RdBu", "BrBG", "PuOr", "coolwarm", "bwr", "seismic"])  # Choose your favourites

        # Contrast - map
        # Low cutoff slider
        self.LCutOff_4 = QtWidgets.QSlider(self.ROI)
        self.LCutOff_4.setGeometry(QtCore.QRect(50, 60, 251, 22))
        self.LCutOff_4.setMaximum(100)
        self.LCutOff_4.setProperty("value", 0)
        self.LCutOff_4.setSliderPosition(0)
        self.LCutOff_4.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.LCutOff_4.setInvertedAppearance(False)
        self.LCutOff_4.setInvertedControls(False)
        self.LCutOff_4.setObjectName("LCutOff_4")

        # High cutoff slider
        self.HCutOff_4 = QtWidgets.QSlider(self.ROI)
        self.HCutOff_4.setGeometry(QtCore.QRect(50, 30, 251, 22))
        self.HCutOff_4.setMaximum(100)
        self.HCutOff_4.setProperty("value", 100)
        self.HCutOff_4.setTracking(True)
        self.HCutOff_4.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.HCutOff_4.setInvertedAppearance(False)
        self.HCutOff_4.setInvertedControls(False)
        self.HCutOff_4.setTickPosition(QtWidgets.QSlider.TickPosition.NoTicks)
        self.HCutOff_4.setObjectName("HCutOff_4")

        # Invert cmap
        self.cmap_inv_sel4 = QtWidgets.QCheckBox(self.ROI)
        self.cmap_inv_sel4.setGeometry(QtCore.QRect(310, 30, 51, 21))
        self.cmap_inv_sel4.setObjectName("cmap_inv_sel4")

        # Select cmap
        self.Cmaps_4 = QtWidgets.QComboBox(self.ROI)
        self.Cmaps_4.setGeometry(QtCore.QRect(60, 0, 151, 26))
        self.Cmaps_4.setEditable(False)
        self.Cmaps_4.setCurrentText("Greys")
        self.Cmaps_4.setObjectName("Cmaps")
        self.Cmaps_4.addItems(["Greys", "Blues", "viridis", "plasma", "inferno", "magma", "cividis", "bone", "spring", "summer", "autumn", "winter", "cool", "jet", "ocean", "RdBu", "BrBG", "PuOr", "coolwarm", "bwr", "seismic"])  # Choose your favourites



        self.close = QtWidgets.QPushButton(self.centralwidget)
        self.close.setGeometry(QtCore.QRect(10, 590, 113, 32))
        self.close.setObjectName("close")
        self.save_and_exit = QtWidgets.QPushButton(self.centralwidget)
        self.save_and_exit.setGeometry(QtCore.QRect(130, 590, 113, 32))
        self.save_and_exit.setObjectName("save_and_exit")

        # Define other GUI elements
        self.disp4d_panel.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(self.disp4d_panel)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1180, 22))
        self.menubar.setObjectName("menubar")
        self.disp4d_panel.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self.disp4d_panel)
        self.statusbar.setObjectName("statusbar")
        self.disp4d_panel.setStatusBar(self.statusbar)

        # Initialize widget's labels
        self.retranslateUi()
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(self.disp4d_panel)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.disp4d_panel.setWindowTitle(_translate("disp4d_panel", "Spatial map inspector and ROI analysis"))
        # self.CursorMode.setText(_translate("disp4d_panel", "Cursor mode: Normal"))
        self.Label_xsel1.setText(_translate("disp4d_panel", "x1"))
        self.Label_xsel2.setText(_translate("disp4d_panel", "x2"))
        self.Bin_label.setText(_translate("disp4d_panel", "Bin"))
        self.IntX_sel1.setText(_translate("disp4d_panel", "Int."))
        self.LabelX_1.setText(_translate("disp4d_panel", 'ang_scale'))
        self.LabelHSpan_1.setText(_translate("disp4d_panel", "Half span"))
        self.Bilateral_1.setText(_translate("disp4d_panel", "Bilateral Link"))
        self.LabelStep_1.setText(_translate("disp4d_panel", "Step"))
        self.IntY_sel1.setText(_translate("disp4d_panel", "Int."))
        self.cmap_inv_sel1.setText(_translate("disp4d_panel", "Inv."))
        self.cont_mult_lab.setText(_translate("disp4d_panel", "x"))
        self.LabelY_1.setText(_translate("disp4d_panel", "eV"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.map), _translate("disp4d_panel", "Map Explorer"))
        self.ROI_list_label.setText(_translate("disp4d_panel", "ROI Select"))
        self.LabelHSpan_3.setText(_translate("disp4d_panel", "Half span"))
        self.Bilateral_3.setText(_translate("disp4d_panel", "Bilateral Link"))
        self.IntY_sel3.setText(_translate("disp4d_panel", "Int."))
        self.cmap_inv_sel3.setText(_translate("disp4d_panel", "Inv."))
        self.LabelX_3.setText(_translate("disp4d_panel", "TextLabel"))
        self.IntX_sel3.setText(_translate("disp4d_panel", "Int."))
        self.LabelStep_3.setText(_translate("disp4d_panel", "Step"))
        self.LabelY_3.setText(_translate("disp4d_panel", "eV"))
        self.cmap_inv_sel4.setText(_translate("disp4d_panel", "Inv."))
        self.setROI.setText(_translate("disp4d_panel", "Set ROI"))
        self.resetMap.setText(_translate("disp4d_panel", "Reset Map"))
        self.resetDisp.setText(_translate("disp4d_panel", "Reset Disp"))
        self.saveROI.setText(_translate("disp4d_panel", "Save ROI"))
        self.delROI.setText(_translate("disp4d_panel", "Del."))
        self.ROI_from_map.setText(_translate("disp4d_panel", "From map"))
        self.ROI_from_disp.setText(_translate("disp4d_panel", "From disp"))
        self.ROI_select_label.setText(_translate("disp4d_panel", "------------->>"))
        self.ROI_select_label_2.setText(_translate("disp4d_panel", "<<-------------"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.ROI), _translate("disp4d_panel", "ROI analysis"))
        self.close.setText(_translate("disp4d_panel", "Close"))
        self.save_and_exit.setText(_translate("disp4d_panel", "Save and Exit"))

    ############################################################################
    # Resize GUI elements
    # (See pyqt_ResizeItem.py)
    def GUI_resize(self):
        ResizeItem(self.disp4d_panel, self.tabWidget)
        # ResizeItem(self.disp4d_panel, self.CursorMode)
        ResizeItem(self.disp4d_panel, self.MainPlot)
        ResizeItem(self.disp4d_panel, self.MDCPlot)
        ResizeItem(self.disp4d_panel, self.EDCPlot)
        ResizeItem(self.disp4d_panel, self.MapPlot)
        ResizeItem(self.disp4d_panel, self.CursorStats)
        ResizeItem(self.disp4d_panel, self.x1_select)
        ResizeItem(self.disp4d_panel, self.x2_select)
        ResizeItem(self.disp4d_panel, self.Label_xsel1)
        ResizeItem(self.disp4d_panel, self.Label_xsel2)
        ResizeItem(self.disp4d_panel, self.Bin_x1)
        ResizeItem(self.disp4d_panel, self.Bin_x2)
        ResizeItem(self.disp4d_panel, self.Bin_label)
        ResizeItem(self.disp4d_panel, self.LabelX_1)
        ResizeItem(self.disp4d_panel, self.LabelY_1)
        ResizeItem(self.disp4d_panel, self.IntX_sel1)
        ResizeItem(self.disp4d_panel, self.IntY_sel1)
        ResizeItem(self.disp4d_panel, self.SpanX_1)
        ResizeItem(self.disp4d_panel, self.SpanY_1)
        ResizeItem(self.disp4d_panel, self.LabelHSpan_1)
        ResizeItem(self.disp4d_panel, self.DeltaX_1)
        ResizeItem(self.disp4d_panel, self.DeltaY_1)
        ResizeItem(self.disp4d_panel, self.LabelStep_1)
        ResizeItem(self.disp4d_panel, self.Bilateral_1)
        ResizeItem(self.disp4d_panel, self.HCutOff_1)
        ResizeItem(self.disp4d_panel, self.LCutOff_1)
        ResizeItem(self.disp4d_panel, self.cont_mult_lab)
        ResizeItem(self.disp4d_panel, self.cont_mult)
        ResizeItem(self.disp4d_panel, self.cmap_inv_sel1)
        ResizeItem(self.disp4d_panel, self.Cmaps_1)
        ResizeItem(self.disp4d_panel, self.ROI_list)
        ResizeItem(self.disp4d_panel, self.ROI_list_label)
        ResizeItem(self.disp4d_panel, self.setROI)
        ResizeItem(self.disp4d_panel, self.saveROI)
        ResizeItem(self.disp4d_panel, self.ROI_set_name)
        ResizeItem(self.disp4d_panel, self.resetMap)
        ResizeItem(self.disp4d_panel, self.resetDisp)
        ResizeItem(self.disp4d_panel, self.delROI)
        ResizeItem(self.disp4d_panel, self.ROI_from_map)
        ResizeItem(self.disp4d_panel, self.ROI_from_disp)
        ResizeItem(self.disp4d_panel, self.ROI_select_label)
        ResizeItem(self.disp4d_panel, self.ROI_select_label_2)
        ResizeItem(self.disp4d_panel, self.MainPlot_2)
        ResizeItem(self.disp4d_panel, self.MDCPlot_2)
        ResizeItem(self.disp4d_panel, self.EDCPlot_2)
        ResizeItem(self.disp4d_panel, self.MapPlot_2)
        ResizeItem(self.disp4d_panel, self.CursorStats_3)
        ResizeItem(self.disp4d_panel, self.LabelX_3)
        ResizeItem(self.disp4d_panel, self.LabelY_3)
        ResizeItem(self.disp4d_panel, self.IntX_sel3)
        ResizeItem(self.disp4d_panel, self.IntY_sel3)
        ResizeItem(self.disp4d_panel, self.SpanX_3)
        ResizeItem(self.disp4d_panel, self.SpanY_3)
        ResizeItem(self.disp4d_panel, self.LabelHSpan_3)
        ResizeItem(self.disp4d_panel, self.DeltaX_3)
        ResizeItem(self.disp4d_panel, self.DeltaY_3)
        ResizeItem(self.disp4d_panel, self.LabelStep_3)
        ResizeItem(self.disp4d_panel, self.Bilateral_3)
        ResizeItem(self.disp4d_panel, self.HCutOff_3)
        ResizeItem(self.disp4d_panel, self.LCutOff_3)
        ResizeItem(self.disp4d_panel, self.cmap_inv_sel3)
        ResizeItem(self.disp4d_panel, self.Cmaps_3)
        ResizeItem(self.disp4d_panel, self.HCutOff_4)
        ResizeItem(self.disp4d_panel, self.LCutOff_4)
        ResizeItem(self.disp4d_panel, self.cmap_inv_sel4)
        ResizeItem(self.disp4d_panel, self.Cmaps_4)
        ResizeItem(self.disp4d_panel, self.close)
        ResizeItem(self.disp4d_panel, self.save_and_exit)

    ############################################################################

    def GUI_internal(self):
        # Zoom object definition at plots
        # (See pyqt_ZoomItem.py)
        self.ZoomMain = ZoomItem(self.MainPlot)
        self.ZoomMDC = ZoomItem(self.MDCPlot)
        self.ZoomEDC = ZoomItem(self.EDCPlot)
        self.ZoomMap = ZoomItem(self.MapPlot)
        self.ZoomMain_2 = ZoomItem(self.MainPlot_2)
        self.ZoomMDC_2 = ZoomItem(self.MDCPlot_2)
        self.ZoomEDC_2 = ZoomItem(self.EDCPlot_2)
        self.ZoomMap_2 = ZoomItem(self.MapPlot_2)

        # Create image objects
        self.ImageMain = pg.ImageItem()
        self.ImageMap = pg.ImageItem()
        self.ImageMain_2 = pg.ImageItem()
        self.ImageMap_2 = pg.ImageItem()

        # Set the initial colormap
        self.ImageMain.setLookupTable(self.colormap)
        self.ImageMap.setLookupTable(self.colormap)
        self.ImageMain_2.setLookupTable(self.colormap)
        self.ImageMap_2.setLookupTable(self.colormap)

        # Create cursor stats labels (force their height using setFixedHeight inherited to LabelItem from GraphicsWidget)
        self.labels = []
        self.labels2 = []

        for i in range(7):
            label = pg.LabelItem(justify='left', color=self.labelcs[i])
            label.setFixedHeight(20)
            self.labels.append(label)
            label2 = pg.LabelItem(justify='left', color=self.labelcs2[i])
            label2.setFixedHeight(20)
            self.labels2.append(label2)

    # Retrieve input data
    def GUI_files(self, data):
        # To extract the name of the wave in the Notebook
        def retrieve_name(var):
            callers_local_vars = inspect.currentframe().f_back.f_back.f_back.f_back.f_back.f_locals.items()
            return [var_name for var_name, var_val in callers_local_vars if var_val is var]


        # Store the input data in the Panel
        self.data = data.fillna(0)

        # Wave name as defined in the Notebook
        self.wname = retrieve_name(data)

        # Define the Panel data from the first input file
        self.GUI_setdata()

        # Construct the Panel GUI initial content
        self.GUI_initial()

        # Create auxiliary signal objects
        self.GUI_auxiliary()

        # Initialize Panel GUI elements updates
        self.GUI_connect(None)

    # Set GUI data
    def GUI_setdata(self):
        # Reindex array to ensure it has increasing angles with array pixel and check angle/k scale
        for i in self.data.dims:
            if self.data[i].data[1] - self.data[i].data[0] < 0:  # Array currently has decreasing order
                self.data = self.data.reindex({i: self.data[i][::-1]})
            if i != 'eV' and i != 'x1' and i != 'x2':
                self.kdim = i


        # Check the data has the correct axis ordering
        self.data = self.data.transpose('x1', 'x2', self.kdim, 'eV')

        # Convert the input data to attributes of the Panel
        # Dimension scales
        self.x1 = self.data.x1.data
        self.x1_step = self.x1[1] - self.x1[0]
        if 'units' in self.data.coords['x1'].attrs:
            self.x1uts=self.data.coords['x1'].attrs['units']
        else:
            self.x1uts=''
        self.x2 = self.data.x2.data
        self.x2_step = self.x2[1] - self.x2[0]
        if 'units' in self.data.coords['x2'].attrs:
            self.x2uts=self.data.coords['x2'].attrs['units']
        else:
            self.x2uts=''
        self.E = self.data.eV.data
        if 'units' in self.data.coords['eV'].attrs:
            self.Euts=self.data.coords['eV'].attrs['units']
        else:
            self.Euts=''
        self.Estep = self.E[1] - self.E[0]
        self.k = self.data[self.kdim].data
        if 'units' in self.data.coords[self.kdim].attrs:
            self.kuts=self.data.coords[self.kdim].attrs['units']
        else:
            self.kuts=''
        self.kstep = self.k[1] - self.k[0]

        # Intensity cubes
        self.Int = self.data.data  # Full cube
        self.Int_int_Ek = np.sum(self.Int, axis=(2,3)) / (self.data.shape[2] * self.data.shape[3])  # integrated in energy and angle
        self.Int_int_sp = np.sum(self.Int, axis=(0,1)) / (self.data.shape[0] * self.data.shape[1])  # integrated across the map

        self.Idim = 'counts'

        self.I_max = np.max(self.Int)
        self.I_min = np.min(self.Int)
        self.I_max_int_Ek = np.max(self.Int_int_Ek)
        self.I_min_int_Ek = np.min(self.Int_int_Ek)
        self.I_max_int_sp = np.max(self.Int_int_sp)
        self.I_min_int_sp = np.min(self.Int_int_sp)

        # Get beamline conventions
        if self.data.attrs['beamline'] in BL_angles.angles:
            self.BL_conventions = BL_angles.angles[self.data.attrs['beamline']]
        else:
            self.BL_conventions = BL_angles.angles['default']

        # Check if any ROIs already exist and load if not, else create a blank dictionary
        try:
            if self.data.ROI is not None:
                self.ROI_store = self.data.ROI
                for i in self.data.ROI:
                    self.ROI_list.addItem(i)
            else:
                self.ROI_store = {}  # Dictionary to store stored ROIs
        except:
            self.ROI_store = {}  # Dictionary to store stored ROIs


    ############################################################################

    # Construct Panel GUI initial content (dependant on loaded data)
    def GUI_initial(self):

        ### Set the axis labels

        # Map
        self.MapPlot.setLabel('left', text='x2', units=self.x2uts, unitPrefix=None)
        self.MapPlot.setLabel('bottom', text='x1', units=self.x1uts, unitPrefix=None)

        # Dispersion
        self.MainPlot.setLabel('left', text='eV', units=self.Euts, unitPrefix=None)
        self.MainPlot.setLabel('bottom', text=self.kdim, units=self.kuts, unitPrefix=None)

        # MDC
        self.MDCPlot.setLabel('left', text='Intensity', units=None, unitPrefix=None)
        self.MDCPlot.setLabel('bottom', text=self.kdim, units=self.kuts, unitPrefix=None)

        # EDC
        self.EDCPlot.setLabel('left', text='eV', units=None, unitPrefix=None)
        self.EDCPlot.setLabel('bottom', text='Intensity', units=None, unitPrefix=None)

        # Map2
        self.MapPlot_2.setLabel('left', text='x2', units=self.x2uts, unitPrefix=None)
        self.MapPlot_2.setLabel('bottom', text='x1', units=self.x1uts, unitPrefix=None)

        # Dispersion2
        self.MainPlot_2.setLabel('left', text='eV', units=self.Euts, unitPrefix=None)
        self.MainPlot_2.setLabel('bottom', text=self.kdim, units=self.kuts, unitPrefix=None)

        # MDC2
        self.MDCPlot_2.setLabel('left', text='Intensity', units=None, unitPrefix=None)
        self.MDCPlot_2.setLabel('bottom', text=self.kdim, units=self.kuts, unitPrefix=None)

        # EDC2
        self.EDCPlot_2.setLabel('left', text='eV', units=None, unitPrefix=None)
        self.EDCPlot_2.setLabel('bottom', text='Intensity', units=None, unitPrefix=None)

        ########################################################################

        ### MainPlot image initialization:

        # Add image objects
        self.MainPlot.addItem(self.ImageMain)
        self.MapPlot.addItem(self.ImageMap)
        self.MainPlot_2.addItem(self.ImageMain_2)
        self.MapPlot_2.addItem(self.ImageMap_2)

        # Create limits rectangle for the data in MainPlot
        self.ImageMain_Rectangle = QtCore.QRectF(self.k[0], self.E[-1], self.k[-1] - self.k[0], self.E[0] - self.E[-1])
        # And for map plots (NB being careful with offsets to centre the pixel at the relevant map coords
        self.ImageMap_Rectangle =  QtCore.QRectF(self.x1[0] - (self.x1_step/2), self.x2[-1] + (self.x2_step/2), self.x1[-1] - self.x1[0] + self.x1_step, self.x2[0] - self.x2[-1] - self.x2_step)

        # Set initial slices
        self.Int_slice = self.Int[0, 0, :, :]
        self.I_max_int_slice = np.max(self.Int_slice)
        self.I_min_int_slice = np.min(self.Int_slice)
        self.Int_slice2 = self.Int_int_sp
        self.I_max_int_slice2 = np.max(self.Int_slice2)
        self.I_min_int_slice2 = np.min(self.Int_slice2)
        self.Int_map2 = self.Int_int_Ek
        self.I_max_int_map2 = np.max(self.Int_map2)
        self.I_min_int_map2 = np.min(self.Int_map2)

        # Set the initial images
        self.ImageMain.setImage(np.flip(self.Int_slice, 1))
        self.ImageMain_2.setImage(np.flip(self.Int_slice2, 1))
        self.ImageMap.setImage(np.flip(self.Int_int_Ek, 1))
        self.ImageMap_2.setImage(np.flip(self.Int_map2, 1))

        # Set the contrast
        self.contrast1()
        self.contrast3()
        self.contrast4()

        # Scale 2D data
        self.ImageMain.setRect(self.ImageMain_Rectangle)
        self.ImageMain_2.setRect(self.ImageMain_Rectangle)
        self.ImageMap.setRect(self.ImageMap_Rectangle)
        self.ImageMap_2.setRect(self.ImageMap_Rectangle)


        ########################################################################

        # Set corresponding widgets

        # x_select
        self.x1_select.setMinimum(0)
        self.x1_select.setMaximum(self.data.shape[0]-1)
        self.x1_select.setSingleStep(1)
        self.x1_select.setTickInterval(1)
        self.x1_select.setValue(0)
        self.x2_select.setMinimum(0)
        self.x2_select.setMaximum(self.data.shape[1]-1)
        self.x2_select.setSingleStep(1)
        self.x2_select.setTickInterval(1)
        self.x2_select.setValue(0)

        # Binning
        self.Bin_x1.setSingleStep(1)
        self.Bin_x1.setDecimals(0)
        self.Bin_x1.setValue(1)
        self.Bin_x1.setMinimum(1)
        self.Bin_x1.setMaximum(self.data.shape[0])
        self.Bin_x2.setSingleStep(1)
        self.Bin_x2.setDecimals(0)
        self.Bin_x2.setValue(1)
        self.Bin_x2.setMinimum(1)
        self.Bin_x2.setMaximum(self.data.shape[1])


        ########################################################################

        ### Limit zoom and panning of MainPlot and CutPlots:

        # Define zoom/panning limits
        self.x1min_lim = self.x1[0] - (self.x1_step/2) - (self.x1[-1] - self.x1[0]) * self.pd
        self.x1max_lim = self.x1[-1] + (self.x1_step/2)+ (self.x1[-1] - self.x1[0]) * self.pd
        self.x2min_lim = self.x2[0] - (self.x2_step/2) - (self.x2[-1] - self.x2[0]) * self.pd
        self.x2max_lim = self.x2[-1] + (self.x2_step/2) + (self.x2[-1] - self.x2[0]) * self.pd
        self.kmin_lim = self.k[0] - (self.k[-1] - self.k[0]) * self.pd
        self.kmax_lim = self.k[-1] + (self.k[-1] - self.k[0]) * self.pd
        self.Emin_lim = self.E[0] - (self.E[-1] - self.E[0]) * self.pd
        self.Emax_lim = self.E[-1] + (self.E[-1] - self.E[0]) * self.pd

        # Set zoom/panning limits of plots
        self.MainPlot.getViewBox().setLimits(xMin=self.kmin_lim, xMax=self.kmax_lim, yMin=self.Emin_lim, yMax=self.Emax_lim)
        self.MainPlot_2.getViewBox().setLimits(xMin=self.kmin_lim, xMax=self.kmax_lim, yMin=self.Emin_lim, yMax=self.Emax_lim)
        self.MapPlot.getViewBox().setLimits(xMin=self.x1min_lim, xMax=self.x1max_lim, yMin=self.x2min_lim, yMax=self.x2max_lim)
        self.MapPlot_2.getViewBox().setLimits(xMin=self.x1min_lim, xMax=self.x1max_lim, yMin=self.x2min_lim, yMax=self.x2max_lim)

        # MDCPlot intensity minimum and maximum are set to Imin and Imax of data
        self.MDCPlot.getViewBox().setLimits(xMin=self.kmin_lim, xMax=self.kmax_lim, yMin=self.I_min, yMax=self.I_max)
        self.MDCPlot_2.getViewBox().setLimits(xMin=self.kmin_lim, xMax=self.kmax_lim, yMin=self.I_min, yMax=self.I_max)

        # EDCPlot intensity minimum and maximum are set to Imin and Imax of data
        self.EDCPlot.getViewBox().setLimits(xMin=self.I_min, xMax=self.I_max, yMin=self.Emin_lim, yMax=self.Emax_lim)
        self.EDCPlot_2.getViewBox().setLimits(xMin=self.I_min, xMax=self.I_max, yMin=self.Emin_lim, yMax=self.Emax_lim)

        ########################################################################

        ### Set the initial range of plots:

        # Set the initial range of plots
        self.MainPlot.getViewBox().setXRange(self.kmin_lim, self.kmax_lim, padding=0)
        self.MainPlot.getViewBox().setYRange(self.Emin_lim, self.Emax_lim, padding=0)
        self.MainPlot_2.getViewBox().setXRange(self.kmin_lim, self.kmax_lim, padding=0)
        self.MainPlot_2.getViewBox().setYRange(self.Emin_lim, self.Emax_lim, padding=0)
        self.MapPlot.getViewBox().setXRange(self.x1min_lim, self.x1max_lim, padding=0)
        self.MapPlot.getViewBox().setYRange(self.x2min_lim, self.x2max_lim, padding=0)
        self.MapPlot_2.getViewBox().setXRange(self.x1min_lim, self.x1max_lim, padding=0)
        self.MapPlot_2.getViewBox().setYRange(self.x2min_lim, self.x2max_lim, padding=0)

        # Set the initial range of MDCPlot equal to MainPlot
        self.MDCPlot.getViewBox().setXRange(self.kmin_lim, self.kmax_lim, padding=0)
        self.MDCPlot_2.getViewBox().setXRange(self.kmin_lim, self.kmax_lim, padding=0)

        # Set the initial range of MDCPlot intensity equal to the I limits of the data in MainPlot
        self.MDCPlot.setYRange(self.I_min, self.I_max, padding=0)
        self.MDCPlot_2.setYRange(self.I_min, self.I_max, padding=0)

        # Set the initial range of EDCPlot equal to MainPlot
        self.EDCPlot.getViewBox().setYRange(self.Emin_lim, self.Emax_lim, padding=0)
        self.EDCPlot_2.getViewBox().setYRange(self.Emin_lim, self.Emax_lim, padding=0)

        # Set the initial range of MDCPlot intensity equal to the I limits of the data in MainPlot
        self.EDCPlot.setXRange(self.I_min, self.I_max, padding=0)
        self.EDCPlot_2.setXRange(self.I_min, self.I_max, padding=0)

        ########################################################################
        # Add ROIs on tab2
        # Make some positions for starting ROI
        map2_rng = self.MapPlot_2.viewRange()
        main2_rng = self.MainPlot_2.viewRange()

        ROI_map_x1 = self.dcp(map2_rng, 1, 0)
        ROI_map_x2 = self.dcp(map2_rng, 2, 0)
        ROI_map_y1 = self.dcp(map2_rng, 1, 1)
        ROI_map_y2 = self.dcp(map2_rng, 2, 1)

        ROI_disp_x1 = main2_rng[0][1] - 0.1 * (main2_rng[0][1]-main2_rng[0][0])
        ROI_disp_x2 = main2_rng[0][1] - 0.05 * (main2_rng[0][1]-main2_rng[0][0])
        ROI_disp_y1 = main2_rng[1][1] - 0.05 * (main2_rng[1][1]-main2_rng[1][0])
        ROI_disp_y2 = main2_rng[1][1] - 0.1 * (main2_rng[1][1]-main2_rng[1][0])

        # Define ROIs
        self.current_ROI = {}  # Dictionary to store current ROI
        self.current_ROI_map_name = 'None'  # Name of current ROI for the map
        self.current_ROI_disp_name = 'None'  # Name of current ROI for the dispersion
        self.ROI_map = pg.PolyLineROI([[ROI_map_x1, ROI_map_y1], [ROI_map_x1, ROI_map_y2], [ROI_map_x2, ROI_map_y2], [ROI_map_x2, ROI_map_y1]],
                                      closed=True, movable=False, pen=self.ROI_map_pen)
        self.ROI_disp = pg.PolyLineROI([[ROI_disp_x1, ROI_disp_y1], [ROI_disp_x1, ROI_disp_y2], [ROI_disp_x2, ROI_disp_y2], [ROI_disp_x2, ROI_disp_y1]],
                                       closed=True, movable=False, pen=self.ROI_disp_pen)

        # Add these to the plots
        self.MapPlot_2.addItem(self.ROI_map)
        self.MainPlot_2.addItem(self.ROI_disp)

        ########################################################################

        ### Set the initial autorange policy:

        # Disable auto range for plots
        self.MainPlot.disableAutoRange()
        self.MainPlot_2.disableAutoRange()
        self.MapPlot.disableAutoRange()
        self.MapPlot_2.disableAutoRange()

        # Enable auto range for DCPlot
        self.MDCPlot.enableAutoRange()
        self.MDCPlot_2.enableAutoRange()
        self.EDCPlot.enableAutoRange()
        self.EDCPlot_2.enableAutoRange()


        ########################################################################

        ### Internal GUI elements setup:

        # Delete/create the cursors in Mainplot
        # (See pyqt_CursorItem.py)
        # If the cursors are created during the call to GUI_internal and then they are reinitialized
        # using self.cursors[i].__init__() in GUI_files, the performance of zooming/panning in self.MainPlot
        # decreases slowly everytime a new file is selected
        # (For some reason) the same happens to the 'hide/show all' functionality
        # This behaviour seems not to be related with the accumulation of spurious objects, but rather with
        # some kind of system saturation when invoking pg.GraphItem.__init__(self) too many times in pyqt_CursorItem.py
        # Deleting/creating the cursors everytime a new file is selected fixes this issue at no cost
        self.MainCursors = []
        self.Main2Cursors = []

        for i in range(4):
            self.MainCursors.append(CursorItem())
            self.Main2Cursors.append(CursorItem())
        self.MapCursors = CursorItem()


        # Define the zoom/cursor objects space
        self.spaceMain = [[self.k[0], self.E[0]], [self.k[-1], self.E[-1]]]
        self.spaceMDC = [[self.k[0], self.I_min], [self.k[-1], self.I_max]]
        self.spaceEDC = [[self.I_min, self.E[0]], [self.I_max, self.E[-1]]]
        self.spaceMap = [[self.x1[0], self.x2[0]], [self.x1[-1], self.x2[-1]]]
        self.spaceMain_2 = [[self.k[0], self.E[0]], [self.k[-1], self.E[-1]]]
        self.spaceMDC_2 = [[self.k[0], self.I_min], [self.k[-1], self.I_max]]
        self.spaceEDC_2 = [[self.I_min, self.E[0]], [self.I_max, self.E[-1]]]
        self.spaceMap_2 = [[self.x1[0], self.x2[0]], [self.x1[-1], self.x2[-1]]]

        # Set the space for the zoom objects
        self.ZoomMain.setSpace(self.spaceMain)
        self.ZoomMDC.setSpace(self.spaceMDC)
        self.ZoomEDC.setSpace(self.spaceEDC)
        self.ZoomMap.setSpace(self.spaceMap)
        self.ZoomMain_2.setSpace(self.spaceMain_2)
        self.ZoomMDC_2.setSpace(self.spaceMDC_2)
        self.ZoomEDC_2.setSpace(self.spaceEDC_2)
        self.ZoomMap_2.setSpace(self.spaceMap_2)

        # Set the space for the cursors
        for i in range(4):
            self.MainCursors[i].setSpace(self.spaceMain)
            self.Main2Cursors[i].setSpace(self.spaceMain_2)
        self.MapCursors.setSpace(self.spaceMap)

        # Add the leader cursors in MainPlot
        for i in range(2):
            self.MainPlot.addItem(self.MainCursors[i])
            self.MainPlot_2.addItem(self.Main2Cursors[i])
        self.MapPlot.addItem(self.MapCursors)

        # Get the initial MainPlot range
        self.rng1 = self.MainPlot.viewRange()
        self.rng2 = self.MainPlot_2.viewRange()

        # Set the initial cursors configuration
        for i in range(4):
            self.MainCursors[i].setData(pos=np.array([[self.dcp(self.rng1, i + 1, 0), self.dcp(self.rng1, i + 1, 1)]]), **self.dictsMain[i])
            self.Main2Cursors[i].setData(pos=np.array([[self.dcp(self.rng2, i + 1, 0), self.dcp(self.rng2, i + 1, 1)]]), **self.dictsMain2[i])
        self.MapCursors.setData(pos=[[self.x1[0], self.x2[0]]], **self.dictsMap[0])

        # Create infinite vertical lines in MDCPlots (vertical side cursors)
        self.scsrvsMain = []
        self.scsrvsMain2 = []
        for i in range(4):
            self.scsrvsMain.append(pg.InfiniteLine(pos=self.MainCursors[i].data['pos'][0][0], angle=90, pen=self.csrcs[i]))
            self.scsrvsMain2.append(pg.InfiniteLine(pos=self.Main2Cursors[i].data['pos'][0][0], angle=90, pen=self.csrcs[i]))

        # Add the leader vertical side cursors in MainPlots
        for i in range(2):
            self.MDCPlot.addItem(self.scsrvsMain[i])
            self.MDCPlot_2.addItem(self.scsrvsMain2[i])

        # Create infinite horizontal lines in EDCPlots (horizontal side cursors)
        self.scsrhsMain = []
        self.scsrhsMain2 = []
        for i in range(4):
            self.scsrhsMain.append(pg.InfiniteLine(pos=self.MainCursors[i].data['pos'][0][1], angle=0, pen=self.csrcs[i]))
            self.scsrhsMain2.append(pg.InfiniteLine(pos=self.Main2Cursors[i].data['pos'][0][1], angle=0, pen=self.csrcs[i]))

        # Add the leader horizontal side cursors in MainPlots
        for i in range(2):
            self.EDCPlot.addItem(self.scsrhsMain[i])
            self.EDCPlot_2.addItem(self.scsrhsMain2[i])

        # Set up widgets for DCs
        self.SpanX_1.setSingleStep(self.kstep)
        self.SpanY_1.setSingleStep(self.E[1] - self.E[0])
        self.SpanX_3.setSingleStep(self.kstep)
        self.SpanY_3.setSingleStep(self.E[1] - self.E[0])
        self.DeltaX_1.setText(str(np.round(self.kstep,3)))
        self.DeltaY_1.setText(str(np.round(self.Estep,3)))
        self.DeltaX_3.setText(str(np.round(self.kstep,3)))
        self.DeltaY_3.setText(str(np.round(self.Estep,3)))
        self.LabelX_1.setText(self.kdim)
        self.LabelX_3.setText(self.kdim)



        # Add the cursor stats labels in CursorStats
        self.CursorStats.addItem(self.labels[0], row=0, col=0)  # Row 1 - x1 info
        self.CursorStats.addItem(self.labels[1], row=1, col=0)  # Row 2 - x2 info
        self.CursorStats.addItem(self.labels[2], row=3, col=0)  # Row 3 - dispersion cursor info
        self.CursorStats.addItem(self.labels[3], row=4, col=0)  # Row 4 - dispersion cursor info
        self.CursorStats.addItem(self.labels[4],row=5,col=0) # Delta dx dy
        self.CursorStats.addItem(self.labels[5],row=6,col=0) # Delta dz dr dc
        self.CursorStats.addItem(self.labels[6], row=2, col=0)  # Separator
        #
        ### Define the cursor stats format

        # String particles (dependant on user preferences)
        str1 = "<span style='font-size:" + self.fontsz + "pt'>"
        str2 = "%0." + self.digits + "f"
        str4 = "<span style='color:" + self.drcc + "'>"
        #
        # # Full strings
        self.format_csr1 = str1 + 'x1: ' + str2 + ' (Manipulator axis: %s)'
        self.format_csr2 = str1 + 'x2: ' + str2 + ' (Manipulator axis: %s)'
        self.format_csr3 = str1+"[%i,%i] %s="+str2+", %s="+str2+", %s=%i" # Cursors
        self.format_dxy = str1 + "d(%s)=" + str2 + ", d(%s)=" + str2  # Delta dx dy
        self.format_dzrc = str1 + "d(%s)=" + "%i" + " &nbsp;&nbsp;" + str4 + " d[%i,%i]"  # Delta dz dr dc


        # Set the initial cursor stats labels info (cursors)
        # Set label text
        self.update_labels([0, 1, 2, 3])
        self.labels[6].setText('----------------------------------------------------')

        # Add the cursor stats labels in CursorStats
        self.CursorStats.addItem(self.labels[0], row=0, col=0)  # Row 1 - x1 info
        self.CursorStats.addItem(self.labels[1], row=1, col=0)  # Row 2 - x2 info
        self.CursorStats.addItem(self.labels[2], row=3, col=0)  # Row 3 - dispersion cursor info
        self.CursorStats.addItem(self.labels[3], row=4, col=0)  # Row 4 - dispersion cursor info
        self.CursorStats.addItem(self.labels[4], row=5, col=0)  # Delta dx dy
        self.CursorStats.addItem(self.labels[5], row=6, col=0)  # Delta dz dr dc
        self.CursorStats.addItem(self.labels[6], row=2, col=0)  # Separator


        ### Define the cursor stats format for tab 2

        # String particles (dependant on user preferences)
        str1 = "<span style='font-size:" + self.fontsz + "pt'>"
        str2 = "%0." + self.digits + "f"
        str4 = "<span style='color:" + self.drcc + "'>"
        #
        # # Full strings
        self.format2_csr1 = str1 + 'Map ROI selection: %s'
        self.format2_csr2 = str1 + 'Dispersion ROI selection: %s'
        self.format2_csr3 = str1 + "[%i,%i] %s=" + str2 + ", %s=" + str2 + ", %s=%i"  # Cursors
        self.format2_dxy = str1 + "d(%s)=" + str2 + ", d(%s)=" + str2  # Delta dx dy
        self.format2_dzrc = str1 + "d(%s)=" + "%i" + " &nbsp;&nbsp;" + str4 + " d[%i,%i]"  # Delta dz dr dc

        # Set the initial cursor stats labels info (cursors)
        # Set label text
        self.update_labels2([0, 1, 2, 3])
        self.labels2[6].setText('----------------------------------------------------')

        # Add the cursor stats labels in CursorStats2
        self.CursorStats_3.addItem(self.labels2[0], row=0, col=0)  # Row 1 - x1 info
        self.CursorStats_3.addItem(self.labels2[1], row=1, col=0)  # Row 2 - x2 info
        self.CursorStats_3.addItem(self.labels2[2], row=3, col=0)  # Row 3 - dispersion cursor info
        self.CursorStats_3.addItem(self.labels2[3], row=4, col=0)  # Row 4 - dispersion cursor info
        self.CursorStats_3.addItem(self.labels2[4], row=5, col=0)  # Delta dx dy
        self.CursorStats_3.addItem(self.labels2[5], row=6, col=0)  # Delta dz dr dc
        self.CursorStats_3.addItem(self.labels2[6], row=2, col=0)  # Separator
        #

        # Set the initial side plots
        self.MDCs = []
        self.MDC2s = []
        self.EDCs = []
        self.EDC2s = []


        for i in range(4):
            if i < 2:
                self.MDCs.append(self.MDCPlot.plot(self.k, self.Int_slice[:, self.find_nearest(self.E,self.MainCursors[i].data['pos'][0][1])], pen=self.sidecs[i]))
                self.MDC2s.append(self.MDCPlot_2.plot(self.k, self.Int_int_sp[:, self.find_nearest(self.E,self.Main2Cursors[i].data['pos'][0][1])], pen=self.sidecs[i]))
                self.EDCs.append(self.EDCPlot.plot(self.Int_slice[self.find_nearest(self.k, self.MainCursors[i].data['pos'][0][0]), :], self.E, pen=self.sidecs[i]))
                self.EDC2s.append(self.EDCPlot_2.plot(self.Int_int_sp[self.find_nearest(self.k, self.Main2Cursors[i].data['pos'][0][0]), :], self.E, pen=self.sidecs[i]))
            else:
                self.MDCs.append(None)
                self.MDC2s.append(None)
                self.EDCs.append(None)
                self.EDC2s.append(None)


        # Initialize the signals array
        # self.cns[0] = None, such that the signal count starts at 1
        self.cns = [None] + [False] * self.GUI_connect([0])

    # Create auxiliary signal objects
    def GUI_auxiliary(self):

        # Lambda functions can be used to pass arguments to a connected function
        # Partials can be used to pass arguments to a connected function while also handling event objects

        # ----------------------------------------------------------------------#

        # Update MDCs
        self.updateMainMDCs = [None] * 4
        self.updateMainMDCs[0] = lambda: self.updateMainMDC(1)  # Signal object 7
        self.updateMainMDCs[1] = lambda: self.updateMainMDC(2)  # Signal object 8
        self.updateMainMDCs[2] = lambda: self.updateMainMDC(3)  # Signal object 9
        self.updateMainMDCs[3] = lambda: self.updateMainMDC(4)  # Signal object 10
        self.updateMain2MDCs = [None] * 4
        self.updateMain2MDCs[0] = lambda: self.updateMain2MDC(1)  # Signal object 35
        self.updateMain2MDCs[1] = lambda: self.updateMain2MDC(2)  # Signal object 36
        self.updateMain2MDCs[2] = lambda: self.updateMain2MDC(3)  # Signal object 37
        self.updateMain2MDCs[3] = lambda: self.updateMain2MDC(4)  # Signal object 38

        # Update EDCs
        self.updateMainEDCs = [None] * 4
        self.updateMainEDCs[0] = lambda: self.updateMainEDC(1)  # Signal object 11
        self.updateMainEDCs[1] = lambda: self.updateMainEDC(2)  # Signal object 12
        self.updateMainEDCs[2] = lambda: self.updateMainEDC(3)  # Signal object 13
        self.updateMainEDCs[3] = lambda: self.updateMainEDC(4)  # Signal object 14
        self.updateMain2EDCs = [None] * 4
        self.updateMain2EDCs[0] = lambda: self.updateMain2EDC(1)  # Signal object 39
        self.updateMain2EDCs[1] = lambda: self.updateMain2EDC(2)  # Signal object 40
        self.updateMain2EDCs[2] = lambda: self.updateMain2EDC(3)  # Signal object 41
        self.updateMain2EDCs[3] = lambda: self.updateMain2EDC(4)  # Signal object 42

        # Update all visible MDCs
        self.updateMainDCs_selM = partial(self.updateMainDCs, 1)  # Signal object 18
        self.updateMainDCs_selE = partial(self.updateMainDCs, 2)  # Signal object 17
        self.updateMain2DCs_selM = partial(self.updateMain2DCs, 1)  # Signal object 45
        self.updateMain2DCs_selE = partial(self.updateMain2DCs, 2)  # Signal object 46

        # ----------------------------------------------------------------------#

        # Signals when allX checkbox is checked/unchecked
        self.integrateallx1_sig = partial(self.integrateallx1, [11, 12, 13, 14, 17])  # Signal object 19

        # Signals when allY checkbox is checked/unchecked
        self.integrateally1_sig = partial(self.integrateally1, [7, 8, 9, 10, 18])  # Signal object 20

        # Signals when allX checkbox is checked/unchecked on tab2
        self.integrateallx3_sig = partial(self.integrateallx3, [39, 40, 41, 42, 45])  # Signal object 49

        # Signals when allY checkbox is checked/unchecked on tab2
        self.integrateally3_sig = partial(self.integrateally3, [35, 36, 37, 38, 46])  # Signal object 50

        # ----------------------------------------------------------------------#
        # Update side cursors
        self.update_scsrs = [None] * 4
        self.update_scsrs[0] = lambda: self.update_scsr(1)  # Signal object 23
        self.update_scsrs[1] = lambda: self.update_scsr(2)  # Signal object 24
        self.update_scsrs[2] = lambda: self.update_scsr(3)  # Signal object 25
        self.update_scsrs[3] = lambda: self.update_scsr(4)  # Signal object 26
        # And for MainPlot2
        self.update_scsrs2 = [None] * 4
        self.update_scsrs2[0] = lambda: self.update_scsr2(1)  # Signal object 23
        self.update_scsrs2[1] = lambda: self.update_scsr2(2)  # Signal object 24
        self.update_scsrs2[2] = lambda: self.update_scsr2(3)  # Signal object 25
        self.update_scsrs2[3] = lambda: self.update_scsr2(4)  # Signal object 26

        # ----------------------------------------------------------------------#

        # # Signals to show all cursor objects
        # self.show_csr_sig = partial(self.show_csr, self.allkey)  # Signal object 35 - 38
        #
        # # Signals to hide all cursor objects
        # self.hide_csr_sig = partial(self.hide_csr, self.allkey)  # Signal object 39 - 43

        # ----------------------------------------------------------------------#

        # ----------------------------------------------------------------------#
    ############################################################################

    # Connect Display Panel GUI elements updates
    # If cn = None, connect all the signals
    # If cn = [0], return the total number of signals
    # If cn is a list of positive integers, connect all the signals in the list
    def GUI_connect(self, cn):

        # Keep track of the connections using booleans:
        # There are no methods to determine if an object is connected to a function
        # Keeping track of the connections let us avoid issues such as:
        # Already disconnected errors
        # Descrease in performance due to duplicated connections

        # ----------------------------------------------------------------------#
        # Update slice from x1 select
        i = 1
        if not cn or i in cn:
            if self.cns[i] == False:
                self.x1_select.valueChanged.connect(self.slice_select)
                self.cns[i] = True

        # Update slice from x2 select
        i = 2
        if not cn or i in cn:
            if self.cns[i] == False:
                self.x2_select.valueChanged.connect(self.slice_select)
                self.cns[i] = True

        # ----------------------------------------------------------------------#
        # Signals when self.Bin_x1 is changed
        i = 3
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Bin_x1.valueChanged.connect(self.update_bin)
                self.cns[i] = True

        # Signals when self.Bin_x2 is changed
        i = 4
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Bin_x2.valueChanged.connect(self.update_bin)
                self.cns[i] = True

        # ----------------------------------------------------------------------#
        # Signals when cursor dragged on MapPlot
        i = 5
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MapCursors.scatter.sigPlotChanged.connect(self.map_csr_move)
                self.cns[i] = True

        # Signals when arrow keys moved on MapPlot
        i = 6
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MapPlot.sigKeyPress.connect(self.map_arrowmove)
                self.cns[i] = True

        # ----------------------------------------------------------------------#
        # Update main DCs
        # Update MDCs
        i = 7
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainCursors[0].scatter.sigPlotChanged.connect(self.updateMainMDCs[0])
                self.cns[i] = True

        i = 8
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainCursors[1].scatter.sigPlotChanged.connect(self.updateMainMDCs[1])
                self.cns[i] = True

        i = 9
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainCursors[2].scatter.sigPlotChanged.connect(self.updateMainMDCs[2])
                self.cns[i] = True

        i = 10
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainCursors[3].scatter.sigPlotChanged.connect(self.updateMainMDCs[3])
                self.cns[i] = True

        # Update EDCs
        i = 11
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainCursors[0].scatter.sigPlotChanged.connect(self.updateMainEDCs[0])
                self.cns[i] = True

        i = 12
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainCursors[1].scatter.sigPlotChanged.connect(self.updateMainEDCs[1])
                self.cns[i] = True

        i = 13
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainCursors[2].scatter.sigPlotChanged.connect(self.updateMainEDCs[2])
                self.cns[i] = True

        i = 14
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainCursors[3].scatter.sigPlotChanged.connect(self.updateMainEDCs[3])
                self.cns[i] = True


        # ----------------------------------------------------------------------#
        # Signals when half-span range is changed
        i = 15
        if not cn or i in cn:
            if self.cns[i] == False:
                self.SpanX_1.valueChanged.connect(self.updateispanMain)
                self.cns[i] = True

        i = 16
        if not cn or i in cn:
            if self.cns[i] == False:
                self.SpanY_1.valueChanged.connect(self.updateispanMain)
                self.cns[i] = True

        i = 17
        if not cn or i in cn:
            if self.cns[i] == False:
                self.SpanX_1.valueChanged.connect(self.updateMainDCs_selE)
                self.cns[i] = True

        i = 18
        if not cn or i in cn:
            if self.cns[i] == False:
                self.SpanY_1.valueChanged.connect(self.updateMainDCs_selM)
                self.cns[i] = True

        # ----------------------------------------------------------------------#
        # Signals when half-span step is changed
        i = 19
        if not cn or i in cn:
            if self.cns[i] == False:
                self.DeltaX_1.editingFinished.connect(self.update_DeltaX_1)
                self.cns[i] = True

        i = 20
        if not cn or i in cn:
            if self.cns[i] == False:
                self.DeltaY_1.editingFinished.connect(self.update_DeltaY_1)
                self.cns[i] = True

        # ----------------------------------------------------------------------#

        # Signals when allX checkbox is checked/unchecked
        i = 21
        if not cn or i in cn:
            if self.cns[i] == False:
                self.IntX_sel1.stateChanged.connect(self.integrateallx1_sig)
                self.cns[i] = True

        # Signals when allY checkbox is checked/unchecked
        i = 22
        if not cn or i in cn:
            if self.cns[i] == False:
                self.IntY_sel1.stateChanged.connect(self.integrateally1_sig)
                self.cns[i] = True

        # ----------------------------------------------------------------------#

        # Update side cursors
        i = 23
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainCursors[0].scatter.sigPlotChanged.connect(self.update_scsrs[0])
                self.cns[i] = True

        i = 24
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainCursors[1].scatter.sigPlotChanged.connect(self.update_scsrs[1])
                self.cns[i] = True

        i = 25
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainCursors[2].scatter.sigPlotChanged.connect(self.update_scsrs[2])
                self.cns[i] = True

        i = 26
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainCursors[3].scatter.sigPlotChanged.connect(self.update_scsrs[3])
                self.cns[i] = True

        # ----------------------------------------------------------------------#

        # Signals when the HCutOff_1 slider is moved
        i = 27
        if not cn or i in cn:
            if self.cns[i] == False:
                self.HCutOff_1.valueChanged.connect(self.contrast1)
                self.cns[i] = True

        # Signals when the LCutOff_1 slider is moved
        i = 28
        if not cn or i in cn:
            if self.cns[i] == False:
                self.LCutOff_1.valueChanged.connect(self.contrast1)
                self.cns[i] = True

        # Signals when the cmap+inv_sel1 checkbox is checked/unchecked
        i = 29
        if not cn or i in cn:
            if self.cns[i] == False:
                self.cmap_inv_sel1.clicked.connect(self.invert1)
                self.cns[i] = True

        # Signals when an option in Cmaps_1 is selected
        i = 30
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Cmaps_1.activated.connect(self.cmap_select1)
                self.cns[i] = True

        # ----------------------------------------------------------------------#

        # Signals when the MainPlot range is changed
        i = 31
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainPlot.sigRangeChanged.connect(self.updatesides)
                self.cns[i] = True

        # Signals when the MDCPlot range is changed
        i = 32
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MDCPlot.sigRangeChanged.connect(self.updatemainfromMDC)
                self.cns[i] = True

        # Signals when the EDCPlot range is changed
        i = 33
        if not cn or i in cn:
            if self.cns[i] == False:
                self.EDCPlot.sigRangeChanged.connect(self.updatemainfromEDC)
                self.cns[i] = True

        # Signals if Bilateral state is changed
        i = 34
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Bilateral_1.stateChanged.connect(self.bilateral_1)
                self.cns[i] = True

        # ----------------------------------------------------------------------#
        # Update main2 DCs
        # Update MDCs
        i = 35
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Main2Cursors[0].scatter.sigPlotChanged.connect(self.updateMain2MDCs[0])
                self.cns[i] = True

        i = 36
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Main2Cursors[1].scatter.sigPlotChanged.connect(self.updateMain2MDCs[1])
                self.cns[i] = True

        i = 37
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Main2Cursors[2].scatter.sigPlotChanged.connect(self.updateMain2MDCs[2])
                self.cns[i] = True

        i = 38
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Main2Cursors[3].scatter.sigPlotChanged.connect(self.updateMain2MDCs[3])
                self.cns[i] = True

        # Update EDCs
        i = 39
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Main2Cursors[0].scatter.sigPlotChanged.connect(self.updateMain2EDCs[0])
                self.cns[i] = True

        i = 40
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Main2Cursors[1].scatter.sigPlotChanged.connect(self.updateMain2EDCs[1])
                self.cns[i] = True

        i = 41
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Main2Cursors[2].scatter.sigPlotChanged.connect(self.updateMain2EDCs[2])
                self.cns[i] = True

        i = 42
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Main2Cursors[3].scatter.sigPlotChanged.connect(self.updateMain2EDCs[3])
                self.cns[i] = True

        # ----------------------------------------------------------------------#
        # Signals when half-span range is changed for plot 2
        i = 43
        if not cn or i in cn:
            if self.cns[i] == False:
                self.SpanX_3.valueChanged.connect(self.updateispanMain2)
                self.cns[i] = True

        i = 44
        if not cn or i in cn:
            if self.cns[i] == False:
                self.SpanY_3.valueChanged.connect(self.updateispanMain2)
                self.cns[i] = True

        i = 45
        if not cn or i in cn:
            if self.cns[i] == False:
                self.SpanX_3.valueChanged.connect(self.updateMain2DCs_selE)
                self.cns[i] = True

        i = 46
        if not cn or i in cn:
            if self.cns[i] == False:
                self.SpanY_3.valueChanged.connect(self.updateMain2DCs_selM)
                self.cns[i] = True


        # ----------------------------------------------------------------------#
        # Signals when half-span step is changed
        i = 47
        if not cn or i in cn:
            if self.cns[i] == False:
                self.DeltaX_3.editingFinished.connect(self.update_DeltaX_3)
                self.cns[i] = True

        i = 48
        if not cn or i in cn:
            if self.cns[i] == False:
                self.DeltaY_3.editingFinished.connect(self.update_DeltaY_3)
                self.cns[i] = True

        # ----------------------------------------------------------------------#

        # Signals when allX checkbox is checked/unchecked on tab2
        i = 49
        if not cn or i in cn:
            if self.cns[i] == False:
                self.IntX_sel3.stateChanged.connect(self.integrateallx3_sig)
                self.cns[i] = True

        # Signals when allY checkbox is checked/unchecked on tab2
        i = 50
        if not cn or i in cn:
            if self.cns[i] == False:
                self.IntY_sel3.stateChanged.connect(self.integrateally3_sig)
                self.cns[i] = True

        # ----------------------------------------------------------------------#

        # Update side cursors on MainPlot2
        i = 51
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Main2Cursors[0].scatter.sigPlotChanged.connect(self.update_scsrs2[0])
                self.cns[i] = True

        i = 52
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Main2Cursors[1].scatter.sigPlotChanged.connect(self.update_scsrs2[1])
                self.cns[i] = True

        i = 53
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Main2Cursors[2].scatter.sigPlotChanged.connect(self.update_scsrs2[2])
                self.cns[i] = True

        i = 54
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Main2Cursors[3].scatter.sigPlotChanged.connect(self.update_scsrs2[3])
                self.cns[i] = True
        # ----------------------------------------------------------------------#

        # Signals when the HCutOff_3 slider is moved
        i = 55
        if not cn or i in cn:
            if self.cns[i] == False:
                self.HCutOff_3.valueChanged.connect(self.contrast3)
                self.cns[i] = True

        # Signals when the LCutOff_3 slider is moved
        i = 56
        if not cn or i in cn:
            if self.cns[i] == False:
                self.LCutOff_3.valueChanged.connect(self.contrast3)
                self.cns[i] = True

        # Signals when the cmap+inv_sel3 checkbox is checked/unchecked
        i = 57
        if not cn or i in cn:
            if self.cns[i] == False:
                self.cmap_inv_sel3.clicked.connect(self.invert3)
                self.cns[i] = True

        # Signals when an option in Cmaps_1 is selected
        i = 58
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Cmaps_3.activated.connect(self.cmap_select3)
                self.cns[i] = True

        # Signals when the HCutOff_4 slider is moved
        i = 59
        if not cn or i in cn:
            if self.cns[i] == False:
                self.HCutOff_4.valueChanged.connect(self.contrast4)
                self.cns[i] = True

        # Signals when the LCutOff_4 slider is moved
        i = 60
        if not cn or i in cn:
            if self.cns[i] == False:
                self.LCutOff_4.valueChanged.connect(self.contrast4)
                self.cns[i] = True

        # Signals when the cmap+inv_sel4 checkbox is checked/unchecked
        i = 61
        if not cn or i in cn:
            if self.cns[i] == False:
                self.cmap_inv_sel4.clicked.connect(self.invert4)
                self.cns[i] = True

        # Signals when an option in Cmaps_4 is selected
        i = 62
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Cmaps_4.activated.connect(self.cmap_select4)
                self.cns[i] = True

        # ----------------------------------------------------------------------#

        # Signals when the MainPlot_2 range is changed
        i = 63
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainPlot_2.sigRangeChanged.connect(self.updatesides2)
                self.cns[i] = True

        # Signals when the MDCPlot_2 range is changed
        i = 64
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MDCPlot_2.sigRangeChanged.connect(self.updatemainfromMDC2)
                self.cns[i] = True

        # Signals when the EDCPlot_2 range is changed
        i = 65
        if not cn or i in cn:
            if self.cns[i] == False:
                self.EDCPlot_2.sigRangeChanged.connect(self.updatemainfromEDC2)
                self.cns[i] = True

        # Signals if Bilateral_3 state is changed
        i = 66
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Bilateral_3.stateChanged.connect(self.bilateral_3)
                self.cns[i] = True

        # ----------------------------------------------------------------------#
        # On click of ROI from map radio
        i = 67
        if not cn or i in cn:
            if self.cns[i] == False:
                self.ROI_from_map.clicked.connect(self.setROIfromMap)
                self.cns[i] = True

        # On click of ROI from disp radio
        i = 68
        if not cn or i in cn:
            if self.cns[i] == False:
                self.ROI_from_disp.clicked.connect(self.setROIfromDisp)
                self.cns[i] = True

        # ----------------------------------------------------------------------#
        # On Set ROI click
        i = 69
        if not cn or i in cn:
            if self.cns[i] == False:
                self.setROI.clicked.connect(self.applyROI)
                self.cns[i] = True

        # On Reset Map click
        i = 70
        if not cn or i in cn:
            if self.cns[i] == False:
                self.resetMap.clicked.connect(self.resetMapROI)
                self.cns[i] = True

        # On Reset Map click
        i = 71
        if not cn or i in cn:
            if self.cns[i] == False:
                self.resetDisp.clicked.connect(self.resetDispROI)
                self.cns[i] = True

        # On Save ROI click
        i = 72
        if not cn or i in cn:
            if self.cns[i] == False:
                self.saveROI.clicked.connect(self.add_ROI_to_list)
                self.cns[i] = True

        # On ROI selection in list
        i = 73
        if not cn or i in cn:
            if self.cns[i] == False:
                self.ROI_list.itemSelectionChanged.connect(self.restore_ROI)
                self.cns[i] = True

        # ----------------------------------------------------------------------#
        # On save and exit press
        i = 74
        if not cn or i in cn:
            if self.cns[i] == False:
                self.save_and_exit.clicked.connect(self.save_on_quit)
                self.cns[i] = True

        # On exit press
        i = 75
        if not cn or i in cn:
            if self.cns[i] == False:
                self.close.clicked.connect(self.close_app)
                self.cns[i] = True

        # ----------------------------------------------------------------------#
        # On delROI press
        i = 76
        if not cn or i in cn:
            if self.cns[i] == False:
                self.delROI.clicked.connect(self.deleteROI)
                self.cns[i] = True
        # ----------------------------------------------------------------------#
        # # Add ROI point in MapPlot_2 on mouse click
        # i = 67
        # if not cn or i in cn:
        #     if self.cns[i] == False:
        #         self.MapPlot_2.scene().sigMouseClicked.connect(self.ROI_map_mouse_click)
        #         self.cns[i] = True
        #
        # # Drag ROI point in MapPlot_2
        # i = 68
        # if not cn or i in cn:
        #     if self.cns[i] == False:
        #         self.MapPlot_2.scene().sigMouseMoved.connect(self.ROI_map_mouse_drag)
        #         self.cns[i] = True

        # ----------------------------------------------------------------------#

        # # Signals to show all cursor objects on Map explorer tab
        # i = 35
        # if not cn or i in cn:
        #     if self.cns[i] == False:
        #         self.MapPlot.sigKeyRelease.connect(self.show_csr_sig)
        #         self.cns[i] = True
        #
        # i = 36
        # if not cn or i in cn:
        #     if self.cns[i] == False:
        #         self.MainPlot.sigKeyRelease.connect(self.show_csr_sig)
        #         self.cns[i] = True
        #
        # i = 37
        # if not cn or i in cn:
        #     if self.cns[i] == False:
        #         self.MDCPlot.sigKeyRelease.connect(self.show_csr_sig)
        #         self.cns[i] = True
        #
        # i = 38
        # if not cn or i in cn:
        #     if self.cns[i] == False:
        #         self.EDCPlot.sigKeyRelease.connect(self.show_csr_sig)
        #         self.cns[i] = True
        #
        #
        # # Signals to hide all cursor objects on map explorer tab
        # i = 39
        # if not cn or i in cn:
        #     if self.cns[i] == False:
        #         self.MapPlot.sigKeyPress.connect(self.hide_csr_sig)
        #         self.cns[i] = True
        #
        # i = 40
        # if not cn or i in cn:
        #     if self.cns[i] == False:
        #         self.MainPlot.sigKeyPress.connect(self.hide_csr_sig)
        #         self.cns[i] = True
        #
        # i = 41
        # if not cn or i in cn:
        #     if self.cns[i] == False:
        #         self.EDCPlot.sigKeyPress.connect(self.hide_csr_sig)
        #         self.cns[i] = True
        #
        # i = 42
        # if not cn or i in cn:
        #     if self.cns[i] == False:
        #         self.MDCPlot.sigKeyPress.connect(self.hide_csr_sig)
        #         self.cns[i] = True
        # ----------------------------------------------------------------------#

        # Return the total number of signals
        if cn == [0]:
            return i

    ############################################################################

    # Disconnect Display Panel GUI elements updates
    # If cn = None, disconnect all the signals
    # If cn is a list of positive integers, disconnect all the signals in the list
    def GUI_disconnect(self, cn):

        # If an object is already disconnected an exception is raised and ignored
        # (Disconnecting signals using for loops does not work)

        # ----------------------------------------------------------------------#
        # Update slice from x1 select
        i = 1
        if not cn or i in cn:
            try:
                self.x1_select.valueChanged.disconnect(self.slice_select)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # Update slice from x2 select
        i = 2
        if not cn or i in cn:
            try:
                self.x2_select.valueChanged.disconnect(self.slice_select)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # ----------------------------------------------------------------------#
        # Signals when self.Bin_x1 is changed
        i = 3
        if not cn or i in cn:
            try:
                self.Bin_x1.valueChanged.disconnect(self.upadte_bin)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # Signals when self.Bin_x2 is changed
        i = 4
        if not cn or i in cn:
            try:
                self.Bin_x2.valueChanged.disconnect(self.upadte_bin)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # ----------------------------------------------------------------------#
        # Signals when cursor dragged on MapPlot
        i = 5
        if not cn or i in cn:
            try:
                self.MapCursors.scatter.sigPlotChanged.disconnect(self.map_csr_move)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # Signals when arrow keys moved on MapPlot
        i = 6
        if not cn or i in cn:
            try:
                self.MapPlot.sigKeyPress.disconnect(self.map_arrowmove)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # ----------------------------------------------------------------------#
        # Update Main DCs
        # Update MDCs
        i = 7
        if not cn or i in cn:
            try:
                self.MainCursors[0].scatter.sigPlotChanged.disconnect(self.updateMainMDCs[0])
            except:
                print(i)
            finally:
                self.cns[i] = False

        i = 8
        if not cn or i in cn:
            try:
                self.MainCursors[1].scatter.sigPlotChanged.disconnect(self.updateMainMDCs[1])
            except:
                print(i)
            finally:
                self.cns[i] = False

        i = 9
        if not cn or i in cn:
            try:
                self.MainCursors[2].scatter.sigPlotChanged.disconnect(self.updateMainMDCs[2])
            except:
                print(i)
            finally:
                self.cns[i] = False

        i = 10
        if not cn or i in cn:
            try:
                self.MainCursors[3].scatter.sigPlotChanged.disconnect(self.updateMainMDCs[3])
            except:
                print(i)
            finally:
                self.cns[i] = False

        # Update EDCs
        i = 11
        if not cn or i in cn:
            try:
                self.MainCursors[0].scatter.sigPlotChanged.disconnect(self.updateMainEDCs[0])
            except:
                print(i)
            finally:
                self.cns[i] = False

        i = 12
        if not cn or i in cn:
            try:
                self.MainCursors[1].scatter.sigPlotChanged.disconnect(self.updateMainEDCs[1])
            except:
                print(i)
            finally:
                self.cns[i] = False

        i = 13
        if not cn or i in cn:
            try:
                self.MainCursors[2].scatter.sigPlotChanged.disconnect(self.updateMainEDCs[2])
            except:
                print(i)
            finally:
                self.cns[i] = False

        i = 14
        if not cn or i in cn:
            try:
                self.MainCursors[3].scatter.sigPlotChanged.disconnect(self.updateMainEDCs[3])
            except:
                print(i)
            finally:
                self.cns[i] = False

        # ----------------------------------------------------------------------#
        # Signals when half-span range is changed
        i = 15
        if not cn or i in cn:
            try:
                self.SpanX_1.valueChanged.disconnect(self.updateispanMain)
            except:
                print(i)
            finally:
                self.cns[i] = False

        i = 16
        if not cn or i in cn:
            try:
                self.SpanY_1.valueChanged.disconnect(self.updateispanMain)
            except:
                print(i)
            finally:
                self.cns[i] = False

        i = 17
        if not cn or i in cn:
            try:
                self.SpanX_1.valueChanged.disconnect(self.updateMainDCs_selE)
            except:
                print(i)
            finally:
                self.cns[i] = False

        i = 18
        if not cn or i in cn:
            try:
                self.SpanY_1.valueChanged.disconnect(self.updateMainDCs_selM)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # ----------------------------------------------------------------------#
        # Signals when half-span step is changed
        i = 19
        if not cn or i in cn:
            try:
                self.DeltaX_1.editingFinished.disconnect(self.update_DeltaX_1)
            except:
                print(i)
            finally:
                self.cns[i] = False

        i = 20
        if not cn or i in cn:
            try:
                self.DeltaY_1.editingFinished.disconnect(self.update_DeltaY_1)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # ----------------------------------------------------------------------#

        # Signals when allX checkbox is checked/unchecked
        i = 21
        if not cn or i in cn:
            try:
                self.IntX_sel1.stateChanged.disconnect(self.integrateallx1_sig)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # Signals when allY checkbox is checked/unchecked
        i = 22
        if not cn or i in cn:
            try:
                self.IntY_sel1.stateChanged.disconnect(self.integrateally1_sig)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # ----------------------------------------------------------------------#

        # Update side cursors
        i = 23
        if not cn or i in cn:
            try:
                self.MainCursors[0].scatter.sigPlotChanged.disconnect(self.update_scsrs[0])
            except:
                print(i)
            finally:
                self.cns[i] = False

        i = 24
        if not cn or i in cn:
            try:
                self.MainCursors[1].scatter.sigPlotChanged.disconnect(self.update_scsrs[1])
            except:
                print(i)
            finally:
                self.cns[i] = False

        i = 25
        if not cn or i in cn:
            try:
                self.MainCursors[2].scatter.sigPlotChanged.disconnect(self.update_scsrs[2])
            except:
                print(i)
            finally:
                self.cns[i] = False

        i = 26
        if not cn or i in cn:
            try:
                self.MainCursors[3].scatter.sigPlotChanged.disconnect(self.update_scsrs[3])
            except:
                print(i)
            finally:
                self.cns[i] = False

        # ----------------------------------------------------------------------#
        # Signals when the HCutOff_1 slider is moved
        i = 27
        if not cn or i in cn:
            try:
                self.HCutOff_1.valueChanged.disconnect(self.contrast1)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # Signals when the LCutOff_1 slider is moved
        i = 28
        if not cn or i in cn:
            try:
                self.LCutOff_1.valueChanged.disconnect(self.contrast1)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # Signals when the cmap+inv_sel1 checkbox is checked/unchecked
        i = 29
        if not cn or i in cn:
            try:
                self.cmap_inv_sel1.clicked.disconnect(self.invert1)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # Signals when an option in Cmaps_1 is selected
        i = 30
        if not cn or i in cn:
            try:
                self.Cmaps_1.activated.disconnect(self.cmap_select1)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # ----------------------------------------------------------------------#

        # Signals when the MainPlot range is changed
        i = 31
        if not cn or i in cn:
            try:
                self.MainPlot.sigRangeChanged.disconnect(self.updatesides)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # Signals when the MDCPlot range is changed
        i = 32
        if not cn or i in cn:
            try:
                self.MDCPlot.sigRangeChanged.disconnect(self.updatemainfromMDC)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # Signals when the EDCPlot range is changed
        i = 33
        if not cn or i in cn:
            try:
                self.EDCPlot.sigRangeChanged.disconnect(self.updatemainfromEDC)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # Signals if Bilateral state is changed
        i = 34
        if not cn or i in cn:
            try:
                self.Bilateral_1.stateChanged.disconnect(self.bilateral_1)
            except:
                print(i)
            finally:
                self.cns[i] = False


        # ----------------------------------------------------------------------#
        # Update Main2 DCs
        # Update MDCs
        i = 35
        if not cn or i in cn:
            try:
                self.Main2Cursors[0].scatter.sigPlotChanged.disconnect(self.updateMain2MDCs[0])
            except:
                print(i)
            finally:
                self.cns[i] = False

        i = 36
        if not cn or i in cn:
            try:
                self.Main2Cursors[1].scatter.sigPlotChanged.disconnect(self.updateMain2MDCs[1])
            except:
                print(i)
            finally:
                self.cns[i] = False

        i = 37
        if not cn or i in cn:
            try:
                self.Main2Cursors[2].scatter.sigPlotChanged.disconnect(self.updateMain2MDCs[2])
            except:
                print(i)
            finally:
                self.cns[i] = False

        i = 38
        if not cn or i in cn:
            try:
                self.Main2Cursors[3].scatter.sigPlotChanged.disconnect(self.updateMain2MDCs[3])
            except:
                print(i)
            finally:
                self.cns[i] = False

        # Update EDCs
        i = 39
        if not cn or i in cn:
            try:
                self.Main2Cursors[0].scatter.sigPlotChanged.disconnect(self.updateMain2EDCs[0])
            except:
                print(i)
            finally:
                self.cns[i] = False

        i = 40
        if not cn or i in cn:
            try:
                self.Main2Cursors[1].scatter.sigPlotChanged.disconnect(self.updateMain2EDCs[1])
            except:
                print(i)
            finally:
                self.cns[i] = False

        i = 41
        if not cn or i in cn:
            try:
                self.Main2Cursors[2].scatter.sigPlotChanged.disconnect(self.updateMain2EDCs[2])
            except:
                print(i)
            finally:
                self.cns[i] = False

        i = 42
        if not cn or i in cn:
            try:
                self.Main2Cursors[3].scatter.sigPlotChanged.disconnect(self.updateMain2EDCs[3])
            except:
                print(i)
            finally:
                self.cns[i] = False

        # ----------------------------------------------------------------------#
        # Signals when half-span range is changed
        i = 43
        if not cn or i in cn:
            try:
                self.SpanX_3.valueChanged.disconnect(self.updateispanMain2)
            except:
                print(i)
            finally:
                self.cns[i] = False

        i = 44
        if not cn or i in cn:
            try:
                self.SpanY_3.valueChanged.disconnect(self.updateispanMain2)
            except:
                print(i)
            finally:
                self.cns[i] = False

        i = 45
        if not cn or i in cn:
            try:
                self.SpanX_3.valueChanged.disconnect(self.updateMain2DCs_selE)
            except:
                print(i)
            finally:
                self.cns[i] = False

        i = 46
        if not cn or i in cn:
            try:
                self.SpanY_3.valueChanged.disconnect(self.updateMain2DCs_selM)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # ----------------------------------------------------------------------#
        # Signals when half-span step is changed
        i = 47
        if not cn or i in cn:
            try:
                self.DeltaX_3.editingFinished.disconnect(self.update_DeltaX_3)
            except:
                print(i)
            finally:
                self.cns[i] = False

        i = 48
        if not cn or i in cn:
            try:
                self.DeltaY_3.editingFinished.disconnect(self.update_DeltaY_3)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # ----------------------------------------------------------------------#

        # Signals when allX checkbox is checked/unchecked on tab2
        i = 49
        if not cn or i in cn:
            try:
                self.IntX_sel3.stateChanged.disconnect(self.integrateallx3_sig)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # Signals when allY checkbox is checked/unchecked on tab2
        i = 50
        if not cn or i in cn:
            try:
                self.IntY_sel3.stateChanged.disconnect(self.integrateally3_sig)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # ----------------------------------------------------------------------#

        # Update side cursors on MainPlot2
        i = 51
        if not cn or i in cn:
            try:
                self.Main2Cursors[0].scatter.sigPlotChanged.disconnect(self.update_scsrs2[0])
            except:
                print(i)
            finally:
                self.cns[i] = False

        i = 52
        if not cn or i in cn:
            try:
                self.Main2Cursors[1].scatter.sigPlotChanged.disconnect(self.update_scsrs2[1])
            except:
                print(i)
            finally:
                self.cns[i] = False

        i = 53
        if not cn or i in cn:
            try:
                self.Main2Cursors[2].scatter.sigPlotChanged.disconnect(self.update_scsrs2[2])
            except:
                print(i)
            finally:
                self.cns[i] = False

        i = 54
        if not cn or i in cn:
            try:
                self.Main2Cursors[3].scatter.sigPlotChanged.disconnect(self.update_scsrs2[3])
            except:
                print(i)
            finally:
                self.cns[i] = False

        # ----------------------------------------------------------------------#
        # Signals when the HCutOff_3 slider is moved
        i = 55
        if not cn or i in cn:
            try:
                self.HCutOff_3.valueChanged.disconnect(self.contrast3)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # Signals when the LCutOff_3 slider is moved
        i = 56
        if not cn or i in cn:
            try:
                self.LCutOff_3.valueChanged.disconnect(self.contrast3)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # Signals when the cmap+inv_sel1 checkbox is checked/unchecked
        i = 57
        if not cn or i in cn:
            try:
                self.cmap_inv_sel3.clicked.disconnect(self.invert3)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # Signals when an option in Cmaps_3 is selected
        i = 58
        if not cn or i in cn:
            try:
                self.Cmaps_3.activated.disconnect(self.cmap_select3)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # Signals when the HCutOff_4 slider is moved
        i = 59
        if not cn or i in cn:
            try:
                self.HCutOff_4.valueChanged.disconnect(self.contrast4)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # Signals when the LCutOff_4 slider is moved
        i = 60
        if not cn or i in cn:
            try:
                self.LCutOff_4.valueChanged.disconnect(self.contrast4)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # Signals when the cmap+inv_sel4 checkbox is checked/unchecked
        i = 61
        if not cn or i in cn:
            try:
                self.cmap_inv_sel4.clicked.disconnect(self.invert4)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # Signals when an option in Cmaps_4 is selected
        i = 62
        if not cn or i in cn:
            try:
                self.Cmaps_4.activated.disconnect(self.cmap_select4)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # ----------------------------------------------------------------------#

        # Signals when the MainPlot_2 range is changed
        i = 63
        if not cn or i in cn:
            try:
                self.MainPlot_2.sigRangeChanged.disconnect(self.updatesides2)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # Signals when the MDCPlot_2 range is changed
        i = 64
        if not cn or i in cn:
            try:
                self.MDCPlot_2.sigRangeChanged.disconnect(self.updatemainfromMDC2)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # Signals when the EDCPlot_2 range is changed
        i = 65
        if not cn or i in cn:
            try:
                self.EDCPlot_2.sigRangeChanged.disconnect(self.updatemainfromEDC2)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # Signals if Bilateral_3 state is changed
        i = 66
        if not cn or i in cn:
            try:
                self.Bilateral_3.stateChanged.disconnect(self.bilateral_3)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # ----------------------------------------------------------------------#
        # On click of ROI from map radio
        i = 67
        if not cn or i in cn:
            try:
                self.ROI_from_map.clicked.disconnect(self.setROIfromMap)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # On click of ROI from disp radio
        i = 68
        if not cn or i in cn:
            try:
                self.ROI_from_disp.clicked.disconnect(self.setROIfromDisp)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # ----------------------------------------------------------------------#
        # On Set ROI click
        i = 69
        if not cn or i in cn:
            try:
                self.setROI.clicked.disconnect(self.applyROI)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # On Reset Map click
        i = 70
        if not cn or i in cn:
            try:
                self.resetMap.clicked.disconnect(self.resetMapROI)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # On Reset Map click
        i = 71
        if not cn or i in cn:
            try:
                self.resetDisp.clicked.disconnect(self.resetDispROI)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # On Save ROI click
        i = 72
        if not cn or i in cn:
            try:
                self.saveROI.clicked.disconnect(self.add_ROI_to_list)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # On ROI selection in list
        i = 73
        if not cn or i in cn:
            try:
                self.ROI_list.itemSelectionChanged.disconnect(self.restore_ROI)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # ----------------------------------------------------------------------#
        # On save and exit press
        i = 74
        if not cn or i in cn:
            try:
                self.save_and_exit.clicked.disconnect(self.close_app)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # On exit press
        i = 75
        if not cn or i in cn:
            try:
                self.close.clicked.disconnect(self.save_on_quit)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # ----------------------------------------------------------------------#
        # On delROI press
        i = 76
        if not cn or i in cn:
            try:
                self.delROI.clicked.disconnect(self.deleteROI)
            except:
                print(i)
            finally:
                self.cns[i] = False


        # ----------------------------------------------------------------------#
        # # Add ROI point in MapPlot_2 on mouse click
        # i = 67
        # if not cn or i in cn:
        #     try:
        #         self.MapPlot_2.scene().sigMouseClicked.disconnect(self.ROI_map_mouse_click)
        #     except:
        #         print(i)
        #     finally:
        #         self.cns[i] = False
        #
        # # Drag ROI point in MapPlot_2
        # i = 68
        # if not cn or i in cn:
        #     try:
        #         self.MapPlot_2.scene().sigMouseMoved.disconnect(self.ROI_map_mouse_drag)
        #     except:
        #         print(i)
        #     finally:
        #         self.cns[i] = False

        # ----------------------------------------------------------------------#

        # # Signals to show all cursor objects on Map explorer tab
        # i = 35
        # if not cn or i in cn:
        #     try:
        #         self.MapPlot.sigKeyRelease.disconnect(self.show_csr_sig)
        #     except:
        #         print(i)
        #     finally:
        #         self.cns[i] = False
        #
        # i = 36
        # if not cn or i in cn:
        #     try:
        #         self.MainPlot.sigKeyRelease.disconnect(self.show_csr_sig)
        #     except:
        #         print(i)
        #     finally:
        #         self.cns[i] = False
        #
        # i = 37
        # if not cn or i in cn:
        #     try:
        #         self.MDCPlot.sigKeyRelease.disconnect(self.show_csr_sig)
        #     except:
        #         print(i)
        #     finally:
        #         self.cns[i] = False
        #
        # i = 38
        # if not cn or i in cn:
        #     try:
        #         self.EDCPlot.sigKeyRelease.disconnect(self.show_csr_sig)
        #     except:
        #         print(i)
        #     finally:
        #         self.cns[i] = False
        #
        #
        # # Signals to hide all cursor objects on map explorer tab
        # i = 39
        # if not cn or i in cn:
        #     try:
        #         self.MapPlot.sigKeyPress.disconnect(self.hide_csr_sig)
        #     except:
        #         print(i)
        #     finally:
        #         self.cns[i] = False
        #
        # i = 40
        # if not cn or i in cn:
        #     try:
        #         self.MainPlot.sigKeyPress.disconnect(self.hide_csr_sig)
        #     except:
        #         print(i)
        #     finally:
        #         self.cns[i] = False
        #
        # i = 41
        # if not cn or i in cn:
        #     try:
        #         self.EDCPlot.sigKeyPress.disconnect(self.hide_csr_sig)
        #     except:
        #         print(i)
        #     finally:
        #         self.cns[i] = False
        #
        # i = 42
        # if not cn or i in cn:
        #     try:
        #         self.MDCPlot.sigKeyPress.disconnect(self.hide_csr_sig)
        #     except:
        #         print(i)
        #     finally:
        #         self.cns[i] = False

        # ----------------------------------------------------------------------#


    ############################################################################
    # Internal methods begin
    ############################################################################

    ### GUI internal methods (spatial map panel functionalities)
    # Find index of element in array closest to value
    # (An elegant function used to map values into indexes with high efficiency)
    def find_nearest(self, array, value):

        # Find the array length
        n = len(array)

        l = 0  # Initialize lower index
        u = n - 1  # Initialize upper index

        # Do while the difference between the upper limit and the lower limit is more than 1 unit
        while u - l > 1:

            m = u + l >> 1  # Compute midpoint with a bitshift

            if value >= array[m]:
                l = m  # Update the lower limit
            else:
                u = m  # Update the upper limit

        # Return index closest to value
        if value == array[0]:
            return 0
        elif value == array[n - 1]:
            return n - 1
        else:
            return l

    # Select slice to display
    def slice_select(self):
        # Take off any existing DC integration and store for later
        self.IntX_sel1_temp = False
        self.IntY_sel1_temp = False
        if self.IntX_sel1.isChecked():
            self.IntX_sel1_temp = True
            self.IntX_sel1.setChecked(False)
        if self.IntY_sel1.isChecked():
            self.IntY_sel1_temp = True
            self.IntY_sel1.setChecked(False)

        # Get positions
        x1i = self.x1_select.value()
        x2i = self.x2_select.value()

        # Check binning
        bin_x1 = int(self.Bin_x1.value())
        bin_x2 = int(self.Bin_x2.value())

        # Extract relevant slice from array
        if bin_x1 == 1 and bin_x2 == 1:  # No binning, take single slice
            self.Int_slice = self.Int[x1i, x2i, :, :]
            # Set cursor
            self.MapCursors.setData(pos=[[self.x1[x1i], self.x2[x2i]]], **self.dictsMap[0])
        elif bin_x1 != 1 and bin_x2 == 1:  # Binning in x1 only
            self.Int_slice = np.sum(self.Int[slice(x1i, x1i + bin_x1), x2i, :, :], 0)/bin_x1
            # Set cursor
            self.MapCursors.setData(pos=[[np.mean(self.x1[slice(x1i, x1i + bin_x1)]), self.x2[x2i]]], **self.dictsMap[0])
        elif bin_x1 == 1 and bin_x2 != 1:  # Binning in x2 only
            self.Int_slice = np.sum(self.Int[x1i, slice(x2i, x2i + bin_x2), :, :], 0)/bin_x2
            # Set cursor
            self.MapCursors.setData(pos=[[self.x1[x1i], np.mean(self.x2[slice(x2i, x2i + bin_x2)])]], **self.dictsMap[0])
        else:  # Binning in both axes
            self.Int_slice = np.sum(self.Int[slice(x1i, x1i + bin_x1), slice(x2i, x2i + bin_x2),
                                            :, :], (0,1)) / (bin_x1*bin_x2)
            # Set cursor
            self.MapCursors.setData(pos=[[np.mean(self.x1[slice(x1i, x1i + bin_x1)]), np.mean(self.x2[slice(x2i, x2i + bin_x2)])]], **self.dictsMap[0])

        # Update cursorstats labels
        self.update_labels([0, 1, 2, 3])

        # Get intensity limits
        self.I_max_int_slice = np.max(self.Int_slice)
        self.I_min_int_slice = np.min(self.Int_slice)

        # Set plot
        self.ImageMain.setImage(np.flip(self.Int_slice,1))

        # Update DCs
        self.updateMainDCs()

        # Set any integrations back
        if self.IntX_sel1_temp == True:
            self.IntX_sel1.setChecked(True)
        if self.IntY_sel1_temp == True:
            self.IntY_sel1.setChecked(True)

        # Update contrast
        self.contrast1()

    # Trigger updates on all visible DCs on MainPlot
    def updateMainDCs(self, sel=0):
        # Update DCs
        for i in range(4):
            if self.MainCursors[i] in self.MainPlot.getViewBox().allChildren():
                if sel==0:
                    self.updateMainMDC(i + 1)
                    self.updateMainEDC(i + 1)
                elif sel==1:
                    self.updateMainMDC(i + 1)
                elif sel==2:
                    self.updateMainEDC(i + 1)

    # Trigger updates on all visible DCs on MainPlot
    def updateMain2DCs(self, sel=0):
        # Update DCs
        for i in range(4):
            if self.Main2Cursors[i] in self.MainPlot_2.getViewBox().allChildren():
                if sel==0:
                    self.updateMain2MDC(i + 1)
                    self.updateMain2EDC(i + 1)
                elif sel==1:
                    self.updateMain2MDC(i + 1)
                elif sel==2:
                    self.updateMain2EDC(i + 1)


    # Update the side MDC plots for a cursor
    def updateMainMDC(self, csr_num):

        # Remove current MDC plot
        self.MDCPlot.removeItem(self.MDCs[csr_num - 1])

        # If the y-integration is zero plot the MDC at the crosshair y-position
        if self.ispany == 0:
            self.MDCs[csr_num - 1] = self.MDCPlot.plot(self.k, self.Int_slice[:, self.find_nearest(self.E, self.MainCursors[
                csr_num - 1].data['pos'][0][1])], pen=self.sidecs[csr_num - 1])

        # If the y-integration is non-zero perform the integration
        else:

            # Calculate y-indexes of the integration-crosshair boundaries
            a = self.find_nearest(self.E, self.MainCursors[csr_num - 1].data['pos'][0][1] - self.ispany)
            b = self.find_nearest(self.E, self.MainCursors[csr_num - 1].data['pos'][0][1] + self.ispany)

            # If the indexes are equal plot the MDC at the crosshair y-position
            # (This happens if the integration value is smaller than the length of one data pixel)
            if a == b:
                self.MDCs[csr_num - 1] = self.MDCPlot.plot(self.k, self.Int_slice[:, self.find_nearest(self.E, self.MainCursors[
                    csr_num - 1].data['pos'][0][1])], pen=self.sidecs[csr_num - 1])

            # If the indexes are not equal average the MDCs within the integration-crosshair
            else:
                self.MDCs[csr_num - 1] = self.MDCPlot.plot(self.k, np.sum(self.Int_slice[:, a:b + 1], axis=1) / (b - a + 1),
                                                           pen=self.sidecs[csr_num - 1])

    # Update the side MDC2 plots for a cursor
    def updateMain2MDC(self, csr_num):

        # Remove current MDC plot
        self.MDCPlot_2.removeItem(self.MDC2s[csr_num - 1])

        # If the y-integration is zero plot the MDC at the crosshair y-position
        if self.ispany2 == 0:
            self.MDC2s[csr_num - 1] = self.MDCPlot_2.plot(self.k, self.Int_slice2[:, self.find_nearest(self.E, self.Main2Cursors[
                csr_num - 1].data['pos'][0][1])], pen=self.sidecs[csr_num - 1])

        # If the y-integration is non-zero perform the integration
        else:

            # Calculate y-indexes of the integration-crosshair boundaries
            a = self.find_nearest(self.E, self.Main2Cursors[csr_num - 1].data['pos'][0][1] - self.ispany2)
            b = self.find_nearest(self.E, self.Main2Cursors[csr_num - 1].data['pos'][0][1] + self.ispany2)

            # If the indexes are equal plot the MDC at the crosshair y-position
            # (This happens if the integration value is smaller than the length of one data pixel)
            if a == b:
                self.MDC2s[csr_num - 1] = self.MDCPlot_2.plot(self.k, self.Int_slice2[:, self.find_nearest(self.E, self.Main2Cursors[
                    csr_num - 1].data['pos'][0][1])], pen=self.sidecs[csr_num - 1])

            # If the indexes are not equal average the MDCs within the integration-crosshair
            else:
                self.MDC2s[csr_num - 1] = self.MDCPlot_2.plot(self.k, np.sum(self.Int_slice2[:, a:b + 1], axis=1) / (b - a + 1),
                                                           pen=self.sidecs[csr_num - 1])

    # Update the side EDC plots for a cursor
    def updateMainEDC(self, csr_num):

        # Remove current EDC plot
        self.EDCPlot.removeItem(self.EDCs[csr_num - 1])

        # If the x-integration is zero plot the EDC at the crosshair x-position
        if self.ispanx == 0:
            self.EDCs[csr_num - 1] = self.EDCPlot.plot(
                self.Int_slice[self.find_nearest(self.k, self.MainCursors[csr_num - 1].data['pos'][0][0]), :], self.E,
                pen=self.sidecs[csr_num - 1])

        # If the x-integration is non-zero perform the integration
        else:

            # Calculate x-indexes of the integration-crosshair boundaries
            a = self.find_nearest(self.k, self.MainCursors[csr_num - 1].data['pos'][0][0] - self.ispanx)
            b = self.find_nearest(self.k, self.MainCursors[csr_num - 1].data['pos'][0][0] + self.ispanx)

            # If the indexes are equal plot the EDC at the crosshair x-position
            # (This happens if the integration value is smaller than the length of one data pixel)
            if a == b:
                self.EDCs[csr_num - 1] = self.EDCPlot.plot(
                    self.Int_slice[self.find_nearest(self.k, self.MainCursors[csr_num - 1].data['pos'][0][0]), :], self.E,
                    pen=self.sidecs[csr_num - 1])

            # If the indexes are not equal average the EDCs within the integration-crosshair
            else:
                self.EDCs[csr_num - 1] = self.EDCPlot.plot(np.sum(self.Int_slice[a:b + 1, :], axis=0) / (b - a + 1), self.E, pen=self.sidecs[csr_num - 1])

    def updateMain2EDC(self, csr_num):

        # Remove current EDC plot
        self.EDCPlot_2.removeItem(self.EDC2s[csr_num - 1])

        # If the x-integration is zero plot the EDC at the crosshair x-position
        if self.ispanx2 == 0:
            self.EDC2s[csr_num - 1] = self.EDCPlot_2.plot(
                self.Int_slice2[self.find_nearest(self.k, self.Main2Cursors[csr_num - 1].data['pos'][0][0]), :], self.E,
                pen=self.sidecs[csr_num - 1])

        # If the x-integration is non-zero perform the integration
        else:

            # Calculate x-indexes of the integration-crosshair boundaries
            a = self.find_nearest(self.k, self.Main2Cursors[csr_num - 1].data['pos'][0][0] - self.ispanx2)
            b = self.find_nearest(self.k, self.Main2Cursors[csr_num - 1].data['pos'][0][0] + self.ispanx2)

            # If the indexes are equal plot the EDC at the crosshair x-position
            # (This happens if the integration value is smaller than the length of one data pixel)
            if a == b:
                self.EDC2s[csr_num - 1] = self.EDCPlot_2.plot(
                    self.Int_slice2[self.find_nearest(self.k, self.Main2Cursors[csr_num - 1].data['pos'][0][0]), :], self.E,
                    pen=self.sidecs[csr_num - 1])

            # If the indexes are not equal average the EDCs within the integration-crosshair
            else:
                self.EDC2s[csr_num - 1] = self.EDCPlot_2.plot(np.sum(self.Int_slice2[a:b + 1, :], axis=0) / (b - a + 1), self.E, pen=self.sidecs[csr_num - 1])

    # Update the positions of the side cursors
    def update_scsr(self, csr_num):
        self.scsrvsMain[csr_num - 1].setValue(self.MainCursors[csr_num - 1].data['pos'][0][0])  # Vertical
        self.scsrhsMain[csr_num - 1].setValue(self.MainCursors[csr_num - 1].data['pos'][0][1])  # Horizontal

        # Update cursorstats
        self.update_labels([csr_num + 1])

    # Update the positions of the side cursors on MainPlot2
    def update_scsr2(self, csr_num):
        self.scsrvsMain2[csr_num - 1].setValue(self.Main2Cursors[csr_num - 1].data['pos'][0][0])  # Vertical
        self.scsrhsMain2[csr_num - 1].setValue(self.Main2Cursors[csr_num - 1].data['pos'][0][1])  # Horizontal

        # Update cursorstats
        self.update_labels2([csr_num + 1])

    # Update binning for slice select
    def update_bin(self):
        # Check binning
        bin_x1 = int(self.Bin_x1.value())
        bin_x2 = int(self.Bin_x2.value())

        # Reset slider range
        self.x1_select.setMaximum(len(self.x1) - bin_x1)
        self.x2_select.setMaximum(len(self.x2) - bin_x2)

        # Set the span for the crosshair
        if bin_x1 == 1 and bin_x2 == 1:  # If both binning factors are 1, then don't want any width to x-hair
            self.ispanx_map = 0
            self.ispany_map = 0
        else:
            self.ispanx_map = (self.x1_step * bin_x1) / 2
            self.ispany_map = (self.x2_step * bin_x2) / 2

        # Write these to dictionary
        self.dictsMap[0]['ispanx'] = self.ispanx_map
        self.dictsMap[0]['ispany'] = self.ispany_map

        # Update slices
        self.slice_select()

    # When cursor is moved on MapPlot
    def map_csr_move(self):
        # Disconnect slider move widget
        self.GUI_disconnect([1,2])

        # Get cursor position discritised onto finite grid (ignoring binning)
        x1csr = self.find_nearest(self.x1, self.MapCursors.data['pos'][0][0])
        x2csr = self.find_nearest(self.x2, self.MapCursors.data['pos'][0][1])

        # Update cursor positions
        self.x1_select.setValue(min([x1csr,self.x1_select.maximum()]))
        self.x2_select.setValue(min([x2csr, self.x2_select.maximum()]))

        # Get positions from sliders
        x1i = self.x1_select.value()
        x2i = self.x2_select.value()

        # Check binning
        bin_x1 = int(self.Bin_x1.value())
        bin_x2 = int(self.Bin_x2.value())

        # Extract relevant slice from array
        if bin_x1 == 1 and bin_x2 == 1:  # No binning, take single slice
            self.Int_slice = self.Int[x1i, x2i, :, :]
        elif bin_x1 != 1 and bin_x2 == 1:  # Binning in x1 only
            self.Int_slice = np.sum(self.Int[slice(x1i, x1i + bin_x1), x2i, :, :], 0) / bin_x1
        elif bin_x1 == 1 and bin_x2 != 1:  # Binning in x2 only
            self.Int_slice = np.sum(self.Int[x1i, slice(x2i, x2i + bin_x2), :, :], 0) / bin_x2
        else:  # Binning in both axes
            self.Int_slice = np.sum(self.Int[slice(x1i, x1i + bin_x1), slice(x2i, x2i + bin_x2),
                                            :, :], (0, 1)) / (bin_x1 * bin_x2)

        # Set plot
        self.ImageMain.setImage(np.flip(self.Int_slice,1))

        # Update contrast
        self.contrast1()

        # Update DCs
        self.updateMainDCs()

        # Update cursorstats labels
        self.update_labels([0, 1, 2, 3])

        # Connect slider widgets
        self.GUI_connect([1,2])

    # Cursor arrow movements
    def map_arrowmove(self, evt):

        # Current positions of sliders
        x1_pos = self.x1_select.value()
        x2_pos = self.x2_select.value()

        # Do the cursor moves by moving the sliders and everything follows from there
        # Move cursor data-row up
        if evt.key() == QtCore.Qt.Key.Key_Up and x2_pos < self.x2_select.maximum():
            self.x2_select.setValue(x2_pos + 1)

        # Move cursor data-row down
        if evt.key() == QtCore.Qt.Key.Key_Down and x2_pos > self.x2_select.minimum():
            self.x2_select.setValue(x2_pos - 1)

        # Move cursor data-row right
        if evt.key() == QtCore.Qt.Key.Key_Right and x1_pos < self.x1_select.maximum():
            self.x1_select.setValue(x1_pos + 1)

        # Move cursor data-row left
        if evt.key() == QtCore.Qt.Key.Key_Left and x1_pos > self.x1_select.minimum():
            self.x1_select.setValue(x1_pos - 1)

    # Update ispan ranges
    def updateispanMain(self):
        self.ispanx = self.SpanX_1.value()/2
        self.ispany = self.SpanY_1.value()/2

        # Update cursors
        for i, item in enumerate(self.MainCursors):
            # Update the cursor dictionary
            self.dictsMain[i]['ispany'] = self.ispany
            self.dictsMain[i]['ispanx'] = self.ispanx

            # Update the integration-crosshairs
            item.ispany = self.dictsMain[i]['ispany']
            item.ispanx = self.dictsMain[i]['ispanx']

            # Refresh the cursor if in MainPlot
            if item in self.MainPlot.getViewBox().allChildren():
                # Set the integration-crosshair
                item.setICrosshair()

                # Refresh the integration-crosshair position
                item.updateGraph()

    # Update ispan2 ranges
    def updateispanMain2(self):
        self.ispanx2 = self.SpanX_3.value()/2
        self.ispany2 = self.SpanY_3.value()/2

        # Update cursors
        for i, item in enumerate(self.Main2Cursors):
            # Update the cursor dictionary
            self.dictsMain2[i]['ispany'] = self.ispany2
            self.dictsMain2[i]['ispanx'] = self.ispanx2

            # Update the integration-crosshairs
            item.ispany2 = self.dictsMain2[i]['ispany']
            item.ispanx2 = self.dictsMain2[i]['ispanx']

            # Refresh the cursor if in MainPlot
            if item in self.MainPlot_2.getViewBox().allChildren():
                item.setData(pos=item.data['pos'], **self.dictsMain2[i])


    # Update half-span step size
    # DeltaX_1
    def update_DeltaX_1(self):
        self.SpanX_1.setSingleStep(float(self.DeltaX_1.text()))

    # DeltaY_1
    def update_DeltaY_1(self):
        self.SpanY_1.setSingleStep(float(self.DeltaY_1.text()))

    # DeltaX_3
    def update_DeltaX_3(self):
        self.SpanX_3.setSingleStep(float(self.DeltaX_3.text()))

    # DeltaY_1
    def update_DeltaY_3(self):
        self.SpanY_3.setSingleStep(float(self.DeltaY_3.text()))

    # Integrate whole x range
    def integrateallx1(self, cn):  # ----> Non-trivial cns

        if self.IntX_sel1.isChecked():

            self.tempEDC = []
            self.tempispanx = []

            # Remove and store current EDCs
            for i, item in enumerate(self.EDCs):

                if item in self.EDCPlot.getViewBox().allChildren():
                    self.EDCPlot.removeItem(item)
                    self.tempEDC.append(i)

            # Disconnect EDC updates with the cursors
            # Disconnect hide/show EDCs objects
            self.GUI_disconnect(cn)

            # Disable x integration-crosshairs
            self.tempispanx = self.SpanX_1.value()
            self.SpanX_1.setValue(0)
            self.SpanX_1.setEnabled(False)
            self.DeltaX_1.setEnabled(False)

            # Plot the integrated EDC covering the whole x range
            self.iX = self.EDCPlot.plot(np.sum(self.Int_slice[0:len(self.k), :], axis=0) / (len(self.k)), self.E,
                                        pen=self.peniEDC)

            # Create a linear region covering the whole x range over MainPlot
            self.iregionX = pg.LinearRegionItem(values=(self.kmin_lim, self.kmax_lim), orientation='vertical',
                                                brush=self.brushiX, pen=self.peniX, movable=False)
            self.MainPlot.addItem(self.iregionX)

        else:

            # Restore the EDCs
            for i in self.tempEDC:
                self.EDCPlot.addItem(self.EDCs[i])

            # Connect EDC updates with the cursors
            # Connect hide/show EDCs objects
            self.GUI_connect(cn)

            # Enable x integration-crosshairs
            self.SpanX_1.setValue(self.tempispanx)
            self.SpanX_1.setEnabled(True)
            self.DeltaX_1.setEnabled(True)

            # Remove the integrated EDC and the linear region
            self.EDCPlot.removeItem(self.iX)
            self.MainPlot.removeItem(self.iregionX)

    # Integrate whole y range
    def integrateally1(self, cn):  # ----> Non-trivial cns

        if self.IntY_sel1.isChecked():

            self.tempMDC = []
            self.tempispany = []

            # Remove and store current MDCs
            for i, item in enumerate(self.MDCs):

                if item in self.MDCPlot.getViewBox().allChildren():
                    self.MDCPlot.removeItem(item)
                    self.tempMDC.append(i)

            # Disconnect MDC updates with the cursors
            # Disconnect hide/show MDCs objects
            self.GUI_disconnect(cn)

            # Disable y integration-crosshairs
            self.tempispany = self.SpanY_1.value()
            self.SpanY_1.setValue(0)
            self.SpanY_1.setEnabled(False)
            self.DeltaY_1.setEnabled(False)

            # Plot the integrated MDC covering the whole y range
            self.iY = self.MDCPlot.plot(self.k, np.sum(self.Int_slice[:, 0:len(self.E)], axis=1) / (len(self.E)),
                                        pen=self.peniMDC)

            # Create a linear region covering the whole y range over MainPlot
            self.iregionY = pg.LinearRegionItem(values=(self.Emin_lim, self.Emax_lim), orientation='horizontal',
                                                brush=self.brushiY, pen=self.peniY, movable=False)
            self.MainPlot.addItem(self.iregionY)

        else:

            # Restore the MDCs
            for i in self.tempMDC:
                self.MDCPlot.addItem(self.MDCs[i])

            # Connect MDC updates with the cursors
            # Connect hide/show MDCs objects
            self.GUI_connect(cn)

            # Enable y integration-crosshairs
            self.SpanY_1.setValue(self.tempispany)
            self.SpanY_1.setEnabled(True)
            self.DeltaY_1.setEnabled(True)

            # Remove the integrated MDC and the linear region
            self.MDCPlot.removeItem(self.iY)
            self.MainPlot.removeItem(self.iregionY)

    # Integrate whole x range
    def integrateallx3(self, cn):  # ----> Non-trivial cns

        if self.IntX_sel3.isChecked():

            self.tempEDC2 = []
            self.tempispanx2 = []

            # Remove and store current EDCs
            for i, item in enumerate(self.EDC2s):

                if item in self.EDCPlot_2.getViewBox().allChildren():
                    self.EDCPlot_2.removeItem(item)
                    self.tempEDC2.append(i)

            # Disconnect EDC updates with the cursors
            # Disconnect hide/show EDCs objects
            self.GUI_disconnect(cn)

            # Disable x integration-crosshairs
            self.tempispanx2 = self.SpanX_3.value()
            self.SpanX_3.setValue(0)
            self.SpanX_3.setEnabled(False)
            self.DeltaX_3.setEnabled(False)

            # Plot the integrated EDC covering the whole x range
            self.iX2 = self.EDCPlot_2.plot(np.sum(self.Int_slice2[0:len(self.k), :], axis=0) / (len(self.k)), self.E,
                                        pen=self.peniEDC)

            # Create a linear region covering the whole x range over MainPlot
            self.iregionX2 = pg.LinearRegionItem(values=(self.kmin_lim, self.kmax_lim), orientation='vertical',
                                                brush=self.brushiX, pen=self.peniX, movable=False)
            self.MainPlot_2.addItem(self.iregionX2)

        else:

            # Restore the EDCs
            for i in self.tempEDC2:
                self.EDCPlot_2.addItem(self.EDC2s[i])

            # Connect EDC updates with the cursors
            # Connect hide/show EDCs objects
            self.GUI_connect(cn)

            # Enable x integration-crosshairs
            self.SpanX_3.setValue(self.tempispanx)
            self.SpanX_3.setEnabled(True)
            self.DeltaX_3.setEnabled(True)

            # Remove the integrated EDC and the linear region
            self.EDCPlot_2.removeItem(self.iX2)
            self.MainPlot_2.removeItem(self.iregionX2)

    # Integrate whole y range
    def integrateally3(self, cn):  # ----> Non-trivial cns

        if self.IntY_sel3.isChecked():

            self.tempMDC2 = []
            self.tempispany2 = []

            # Remove and store current MDCs
            for i, item in enumerate(self.MDC2s):

                if item in self.MDCPlot_2.getViewBox().allChildren():
                    self.MDCPlot_2.removeItem(item)
                    self.tempMDC2.append(i)

            # Disconnect MDC updates with the cursors
            # Disconnect hide/show MDCs objects
            self.GUI_disconnect(cn)

            # Disable y integration-crosshairs
            self.tempispany2 = self.SpanY_3.value()
            self.SpanY_3.setValue(0)
            self.SpanY_3.setEnabled(False)
            self.DeltaY_3.setEnabled(False)

            # Plot the integrated MDC covering the whole y range
            self.iY2 = self.MDCPlot_2.plot(self.k, np.sum(self.Int_slice2[:, 0:len(self.E)], axis=1) / (len(self.E)),
                                        pen=self.peniMDC)

            # Create a linear region covering the whole y range over MainPlot
            self.iregionY2 = pg.LinearRegionItem(values=(self.Emin_lim, self.Emax_lim), orientation='horizontal',
                                                brush=self.brushiY, pen=self.peniY, movable=False)
            self.MainPlot_2.addItem(self.iregionY2)

        else:

            # Restore the MDCs
            for i in self.tempMDC2:
                self.MDCPlot_2.addItem(self.MDC2s[i])

            # Connect MDC updates with the cursors
            # Connect hide/show MDCs objects
            self.GUI_connect(cn)

            # Enable y integration-crosshairs
            self.SpanY_3.setValue(self.tempispany)
            self.SpanY_3.setEnabled(True)
            self.DeltaY_3.setEnabled(True)

            # Remove the integrated MDC and the linear region
            self.MDCPlot_2.removeItem(self.iY2)
            self.MainPlot_2.removeItem(self.iregionY2)

    # Link range:  MainPlot -> EDCPlot, MDCPlot (unilateral)
    def updatesides(self):

        self.MDCPlot.sigRangeChanged.disconnect(self.updatemainfromMDC)
        self.EDCPlot.sigRangeChanged.disconnect(self.updatemainfromEDC)

        rng = self.MainPlot.viewRange()

        self.MDCPlot.setXRange(rng[0][0], rng[0][1], padding=0)
        self.EDCPlot.setYRange(rng[1][0], rng[1][1], padding=0)

        self.MDCPlot.sigRangeChanged.connect(self.updatemainfromMDC)
        self.EDCPlot.sigRangeChanged.connect(self.updatemainfromEDC)

    # Link range:  MainPlot <-> EDCPlot, MDCPlot (bilateral)
    # Can also be achieved using:
    # MainPlotItem.setXLink(MDCPlotItem)
    # MainPlotItem.setYLink(EDCPlotItem)
    # But it's kinda buggy...
    def updatemainfromMDC(self):

        if self.Bilateral_1.isChecked():
            self.MainPlot.sigRangeChanged.disconnect(self.updatesides)

            rng = self.MDCPlot.viewRange()
            self.MainPlot.setXRange(rng[0][0], rng[0][1], padding=0)

            self.MainPlot.sigRangeChanged.connect(self.updatesides)

    def updatemainfromEDC(self):

        if self.Bilateral_1.isChecked():
            self.MainPlot.sigRangeChanged.disconnect(self.updatesides)

            rng = self.EDCPlot.viewRange()
            self.MainPlot.setYRange(rng[1][0], rng[1][1], padding=0)

            self.MainPlot.sigRangeChanged.connect(self.updatesides)

    # Set the range of MDCPlot and EDCPlot to MainPlot when Bilateral is checked
    def bilateral_1(self):

        if self.Bilateral_1.isChecked():
            self.MDCPlot.sigRangeChanged.disconnect(self.updatemainfromMDC)
            self.EDCPlot.sigRangeChanged.disconnect(self.updatemainfromEDC)

            rng = self.MainPlot.viewRange()
            self.MDCPlot.setXRange(rng[0][0], rng[0][1], padding=0)
            self.EDCPlot.setYRange(rng[1][0], rng[1][1], padding=0)

            self.MDCPlot.sigRangeChanged.connect(self.updatemainfromMDC)
            self.EDCPlot.sigRangeChanged.connect(self.updatemainfromEDC)


    # Link range:  MainPlot_2 -> EDCPlot_2, MDCPlot_2 (unilateral)
    def updatesides2(self):

        self.MDCPlot_2.sigRangeChanged.disconnect(self.updatemainfromMDC2)
        self.EDCPlot_2.sigRangeChanged.disconnect(self.updatemainfromEDC2)

        rng = self.MainPlot_2.viewRange()

        self.MDCPlot_2.setXRange(rng[0][0], rng[0][1], padding=0)
        self.EDCPlot_2.setYRange(rng[1][0], rng[1][1], padding=0)

        self.MDCPlot_2.sigRangeChanged.connect(self.updatemainfromMDC2)
        self.EDCPlot_2.sigRangeChanged.connect(self.updatemainfromEDC2)

    # Link range:  MainPlot <-> EDCPlot, MDCPlot (bilateral)
    # Can also be achieved using:
    # MainPlotItem.setXLink(MDCPlotItem)
    # MainPlotItem.setYLink(EDCPlotItem)
    # But it's kinda buggy...
    def updatemainfromMDC2(self):

        if self.Bilateral_3.isChecked():
            self.MainPlot_2.sigRangeChanged.disconnect(self.updatesides2)

            rng = self.MDCPlot_2.viewRange()
            self.MainPlot_2.setXRange(rng[0][0], rng[0][1], padding=0)

            self.MainPlot_2.sigRangeChanged.connect(self.updatesides2)

    def updatemainfromEDC2(self):

        if self.Bilateral_3.isChecked():
            self.MainPlot_2.sigRangeChanged.disconnect(self.updatesides2)

            rng = self.EDCPlot_2.viewRange()
            self.MainPlot_2.setYRange(rng[1][0], rng[1][1], padding=0)

            self.MainPlot_2.sigRangeChanged.connect(self.updatesides2)

    # Set the range of MDCPlot and EDCPlot to MainPlot when Bilateral is checked
    def bilateral_3(self):

        if self.Bilateral_3.isChecked():
            self.MDCPlot_2.sigRangeChanged.disconnect(self.updatemainfromMDC2)
            self.EDCPlot_2.sigRangeChanged.disconnect(self.updatemainfromEDC2)

            rng = self.MainPlot_2.viewRange()
            self.MDCPlot_2.setXRange(rng[0][0], rng[0][1], padding=0)
            self.EDCPlot_2.setYRange(rng[1][0], rng[1][1], padding=0)

            self.MDCPlot_2.sigRangeChanged.connect(self.updatemainfromMDC2)
            self.EDCPlot_2.sigRangeChanged.connect(self.updatemainfromEDC2)



    # Get default cursor position
    def dcp(self, rng, csr_num, i):
        return (rng[i][0] + rng[i][1]) / 2 - (rng[i][1] - rng[i][0]) * self.offcs[csr_num - 1]

    # Set the levels in the image according to the positions of HCutOff and LCutOff
    def contrast1(self):
        self.ImageMain.setLevels([(self.LCutOff_1.value() / 100) * (self.I_max_int_slice - self.I_min_int_slice) * float(self.cont_mult.text()) + self.I_min_int_slice,
                              (self.HCutOff_1.value() / 100) * (self.I_max_int_slice - self.I_min_int_slice) * float(self.cont_mult.text()) + self.I_min_int_slice])
        self.ImageMap.setLevels([(self.LCutOff_1.value() / 100) * (self.I_max_int_Ek - self.I_min_int_Ek) + self.I_min_int_Ek,
                                  (self.HCutOff_1.value() / 100) * (self.I_max_int_Ek - self.I_min_int_Ek) + self.I_min_int_Ek])

    def contrast3(self):
        self.ImageMain_2.setLevels([(self.LCutOff_3.value() / 100) * (self.I_max_int_slice2 - self.I_min_int_slice2) + self.I_min_int_slice2,
                                  (self.HCutOff_3.value() / 100) * (self.I_max_int_slice2 - self.I_min_int_slice2) + self.I_min_int_slice2])

    def contrast4(self):
        self.ImageMap_2.setLevels([(self.LCutOff_4.value() / 100) * (self.I_max_int_map2 - self.I_min_int_map2) + self.I_min_int_map2,
                                  (self.HCutOff_4.value() / 100) * (self.I_max_int_map2 - self.I_min_int_map2) + self.I_min_int_map2])

    # Invert contrast sliders
    def invert1(self):

        if self.cmap_inv_sel1.isChecked():
            self.HCutOff_1.setSliderPosition(0)
            self.LCutOff_1.setSliderPosition(100)

        else:
            self.HCutOff_1.setSliderPosition(100)
            self.LCutOff_1.setSliderPosition(0)

    # Invert contrast sliders3
    def invert3(self):

        if self.cmap_inv_sel3.isChecked():
            self.HCutOff_3.setSliderPosition(0)
            self.LCutOff_3.setSliderPosition(100)

        else:
            self.HCutOff_3.setSliderPosition(100)
            self.LCutOff_3.setSliderPosition(0)

    # Invert contrast sliders4
    def invert4(self):

        if self.cmap_inv_sel4.isChecked():
            self.HCutOff_4.setSliderPosition(0)
            self.LCutOff_4.setSliderPosition(100)

        else:
            self.HCutOff_4.setSliderPosition(100)
            self.LCutOff_4.setSliderPosition(0)

    # Select colormap from a menu
    def cmap_select1(self):
        # Retrieve selected colormap
        self.colormap_1 = self.get_mpl_colormap(self.Cmaps_1.currentText())

        # Apply selected colormap
        self.ImageMain.setLookupTable(self.colormap_1)
        self.ImageMap.setLookupTable(self.colormap_1)

    # Select colormap from a menu
    def cmap_select3(self):
        # Retrieve selected colormap
        self.colormap_3 = self.get_mpl_colormap(self.Cmaps_3.currentText())

        # Apply selected colormap
        self.ImageMain_2.setLookupTable(self.colormap_3)
        # Select colormap from a menu

    def cmap_select4(self):
        # Retrieve selected colormap
        self.colormap_4 = self.get_mpl_colormap(self.Cmaps_4.currentText())

        # Apply selected colormap
        self.ImageMap_2.setLookupTable(self.colormap_4)

    # Retrieve colormap from matplotlib
    def get_mpl_colormap(self, cmap):
        # Get the colormap from matplotlib
        colormap = cm.get_cmap(cmap)
        colormap._init()

        # Convert the matplotlib colormap from 0-1 to 0-255 for PyQt5
        # Ignore the last 3 rows of the colormap._lut:
        # colormap._lut has shape (N+3,4) where the first N rows are the RGBA color representations
        # The last 3 rows deal with the treatment of out-of-range and masked values
        # They need to be ignored to avoid the generation of artifacts possibly related to floating-point errors
        lut = (colormap._lut * 255)[1:-3][:]

        return lut


    # -------------------------------------------------------------- #
    # Update cursorstats labels
    def update_labels(self, lcn):
        if 0 in lcn:
            self.labels[0].setText(self.format_csr1 % (self.MapCursors.data['pos'][0][0], self.BL_conventions['x1 name']))
        if 1 in lcn:
            self.labels[1].setText(self.format_csr2 % (self.MapCursors.data['pos'][0][1], self.BL_conventions['x2 name']))
        if 2 in lcn or 3 in lcn:
            k1 = self.MainCursors[0].data['pos'][0][0]
            E1 = self.MainCursors[0].data['pos'][0][1]
            k2 = self.MainCursors[1].data['pos'][0][0]
            E2 = self.MainCursors[1].data['pos'][0][1]
            ik1 = self.find_nearest(self.k, self.MainCursors[0].data['pos'][0][0])
            iE1 = self.find_nearest(self.E, self.MainCursors[0].data['pos'][0][1])
            ik2 = self.find_nearest(self.k, self.MainCursors[1].data['pos'][0][0])
            iE2 = self.find_nearest(self.E, self.MainCursors[1].data['pos'][0][1])

            # Update cursor1 info
            if 2 in lcn:
                # Set label text
                self.labels[2].setText(self.format_csr3 % (ik1, iE1, self.kdim, k1, 'eV', E1, 'counts', self.Int_slice[ik1, iE1]))
            # Update cursor2 info
            if 3 in lcn:
                # Set label text
                self.labels[3].setText(self.format_csr3 % (ik2, iE2, self.kdim, k2, 'eV', E2, 'counts', self.Int_slice[ik2, iE2]))

            # Update delta info
            self.labels[4].setText(self.format_dxy % (self.kdim, k2-k1, 'eV', E2-E1))
            self.labels[5].setText(self.format_dzrc % ('counts', self.Int_slice[ik2, iE2] - self.Int_slice[ik1, iE1], ik2 - ik1, iE2 - iE1))

    # Update cursorstats2 labels
    def update_labels2(self, lcn):
        if 0 in lcn:
            self.labels2[0].setText(self.format2_csr1 % (self.current_ROI_map_name))
            pass
        if 1 in lcn:
            self.labels2[1].setText(self.format2_csr2 % (self.current_ROI_disp_name))
            pass
        if 2 in lcn or 3 in lcn:
            k1 = self.Main2Cursors[0].data['pos'][0][0]
            E1 = self.Main2Cursors[0].data['pos'][0][1]
            k2 = self.Main2Cursors[1].data['pos'][0][0]
            E2 = self.Main2Cursors[1].data['pos'][0][1]
            ik1 = self.find_nearest(self.k, self.Main2Cursors[0].data['pos'][0][0])
            iE1 = self.find_nearest(self.E, self.Main2Cursors[0].data['pos'][0][1])
            ik2 = self.find_nearest(self.k, self.Main2Cursors[1].data['pos'][0][0])
            iE2 = self.find_nearest(self.E, self.Main2Cursors[1].data['pos'][0][1])

            # Update cursor1 info
            if 2 in lcn:
                # Set label text
                self.labels2[2].setText(self.format2_csr3 % (ik1, iE1, self.kdim, k1, 'eV', E1, 'counts', self.Int_slice2[ik1, iE1]))
            # Update cursor2 info
            if 3 in lcn:
                # Set label text
                self.labels2[3].setText(self.format2_csr3 % (ik2, iE2, self.kdim, k2, 'eV', E2, 'counts', self.Int_slice2[ik2, iE2]))

            # Update delta info
            self.labels2[4].setText(self.format2_dxy % (self.kdim, k2-k1, 'eV', E2-E1))
            self.labels2[5].setText(self.format2_dzrc % ('counts', self.Int_slice2[ik2, iE2] - self.Int_slice2[ik1, iE1], ik2 - ik1, iE2 - iE1))


    # -------------------------------------------------------------- #
    # Set ROI from map option
    def setROIfromMap(self):
        self.ROI_from_map.setChecked(True)
        self.ROI_from_disp.setChecked(False)

    # Set ROI from dispersion option
    def setROIfromDisp(self):
        self.ROI_from_map.setChecked(False)
        self.ROI_from_disp.setChecked(True)

    # Apply ROI on button click
    def applyROI(self):
        if self.ROI_from_map.isChecked() == True:  # Apply ROI from map
            ROI_map_handles = self.ROI_map.getLocalHandlePositions()
            x1_list = []
            x2_list = []
            for i in ROI_map_handles:
                x1_list.append(i[1].x())
                x2_list.append(i[1].y())

            self.current_ROI = {'x1': x1_list, 'x2': x2_list}

            # Do the ROI selection
            self.Int_slice2 = ROI_select(self.data, self.current_ROI).data
            self.current_ROI_disp_name = 'Current'

            # Refresh the display, DCs etc.
            self.refresh_Main2_slice()


        elif self.ROI_from_disp.isChecked() == True:  # Apply ROI from dispersion
            ROI_disp_handles = self.ROI_disp.getLocalHandlePositions()
            k_list = []
            E_list = []
            for i in ROI_disp_handles:
                k_list.append(i[1].x())
                E_list.append(i[1].y())

            self.current_ROI = {'eV': E_list, self.kdim: k_list}

            # Do the ROI selection
            self.Int_map2 = ROI_select(self.data, self.current_ROI).data
            self.current_ROI_map_name = 'Current'

            # Update the display, DCs, etc.
            self.refresh_Map2_slice()


    # Reset ROI on button click
    def resetMapROI(self):
        # Disconnect on selection change widget
        self.GUI_disconnect([73])

        # Reset to the original integrated map
        self.Int_map2 = self.Int_int_Ek

        # Set current ROI label
        self.current_ROI_map_name = 'None'

        # Reset ROI selection widget
        self.ROI_list.clearSelection()

        # Update the display, DCs, etc.
        self.refresh_Map2_slice()

        # Reconnect on selection change widget
        self.GUI_connect([73])

    def resetDispROI(self):
        # Disconnect on selection change widget
        self.GUI_disconnect([73])

        # Reset to the original integrated slice
        self.Int_slice2 = self.Int_int_sp

        # Set current ROI label
        self.current_ROI_disp_name = 'None'

        # Reset ROI selection widget
        self.ROI_list.clearSelection()

        # Update the display, DCs, etc.
        self.refresh_Main2_slice()

        # Reconnect on selection change widget
        self.GUI_connect([73])

    def refresh_Main2_slice(self):
        # Update the plot
        # Take off any existing DC integration and store for later
        self.IntX_sel3_temp = False
        self.IntY_sel3_temp = False
        if self.IntX_sel3.isChecked():
            self.IntX_sel3_temp = True
            self.IntX_sel3.setChecked(False)
        if self.IntY_sel3.isChecked():
            self.IntY_sel3_temp = True
            self.IntY_sel3.setChecked(False)

        # Get intensity limits
        self.I_max_int_slice2 = np.max(self.Int_slice2)
        self.I_min_int_slice2 = np.min(self.Int_slice2)

        # Set plot
        self.ImageMain_2.setImage(np.flip(self.Int_slice2, 1))

        # Update DCs
        self.updateMain2DCs()

        # Set any integrations back
        if self.IntX_sel3_temp == True:
            self.IntX_sel3.setChecked(True)
        if self.IntY_sel3_temp == True:
            self.IntY_sel3.setChecked(True)

        # Update contrast
        self.contrast3()

        # Update labels
        self.update_labels2([0, 1, 2, 3])

    def refresh_Map2_slice(self):
        # Update the plot
        # Get intensity limits
        self.I_max_int_map2 = np.max(self.Int_map2)
        self.I_min_int_map2 = np.min(self.Int_map2)

        # Set plot
        self.ImageMap_2.setImage(np.flip(self.Int_map2, 1))

        # Update contrast
        self.contrast4()

        # Update labels
        self.update_labels2([0, 1, 2, 3])

    def add_ROI_to_list(self):

        # Disconnect on selection change widget
        self.GUI_disconnect([73])

        # Check if a ROI name specified
        if self.ROI_set_name.text() == '':
            ROI_name_base = 'ROI'
        else:
            ROI_name_base = self.ROI_set_name.text()

        if ROI_name_base != 'ROI' and ROI_name_base not in self.ROI_store:
            ROI_name = ROI_name_base
        else:
            num = 1
            ROI_name = ROI_name_base + '_' + str(num)
            while ROI_name in self.ROI_store:
                num+=1
                ROI_name = ROI_name_base + '_' + str(num)

        # Set the ROI (even if already set)
        self.applyROI()

        # Get new label
        if self.ROI_from_map.isChecked() == True:
            self.current_ROI_disp_name = ROI_name
        else:
            self.current_ROI_map_name = ROI_name

        # Refresh the display, DCs etc.
        self.refresh_Main2_slice()

        # Add ROI to list
        self.ROI_store[ROI_name] = self.current_ROI

        # Add ROI name to list box
        self.ROI_list.clearSelection()
        self.ROI_list.addItem(ROI_name)

        # Reconnect on selection change widget
        self.GUI_connect([73])

    def deleteROI(self):

        # Get selected ROI list items
        selected_ROIs = self.ROI_list.selectedItems()

        if len(selected_ROIs) == 1:  # Only single item selected
            selected_ROI_name = selected_ROIs[0].text()

            # Remove ROI from the store
            self.ROI_store.pop(selected_ROI_name)

            # Re-populate ROI list
            self.ROI_list.clearSelection()
            self.ROI_list.clear()
            for i in self.ROI_store:
                self.ROI_list.addItem(i)

        # Reset map to show no ROI selection
        self.resetDispROI()
        self.resetMapROI()


    # Show ROI on selection
    def restore_ROI(self):
        # Get selected ROI list items
        selected_ROIs = self.ROI_list.selectedItems()

        if len(selected_ROIs) == 1:  # Only single item selected
            selected_ROI_name = selected_ROIs[0].text()

            self.current_ROI = self.ROI_store[selected_ROI_name]

            if 'x1' in self.current_ROI:  # Map-based ROI
                # Do the ROI selection
                self.Int_slice2 = ROI_select(self.data, self.current_ROI).data
                self.current_ROI_disp_name = selected_ROI_name

                # Refresh the display, DCs etc.
                self.refresh_Main2_slice()

                # Get the original ROI points in format to pass to pyqtgraph ROI
                ROI_points = []
                for i,j in zip(self.current_ROI['x1'],self.current_ROI['x2']):
                    ROI_points.append((i,j))

                # Set displayed ROI to these points
                self.ROI_map.setPoints(ROI_points)


            else:  # Dispersion-based ROI
                # Do the ROI selection
                self.Int_map2 = ROI_select(self.data, self.current_ROI).data
                self.current_ROI_map_name = selected_ROI_name

                # Update the display, DCs, etc.
                self.refresh_Map2_slice()

                # Get the original ROI points in format to pass to pyqtgraph ROI
                ROI_points = []
                for i, j in zip(self.current_ROI[self.kdim], self.current_ROI['eV']):
                    ROI_points.append((i, j))

                # Set displayed ROI to these points
                self.ROI_disp.setPoints(ROI_points)

        elif len(selected_ROIs) == 2:  # Two items selected
            selected_ROI_name1 = selected_ROIs[0].text()
            selected_ROI_name2 = selected_ROIs[1].text()

            ROI1 = self.ROI_store[selected_ROI_name1]
            ROI2 = self.ROI_store[selected_ROI_name2]

            if 'x1' in ROI1 and 'x1' in ROI2:  # Map-based ROI
                # Do the ROI selection
                self.Int_slice2 = ROI_select(self.data, ROI1).data - ROI_select(self.data, ROI2).data
                self.current_ROI_disp_name = selected_ROI_name1 + '-' + selected_ROI_name2

                # Refresh the display, DCs etc.
                self.refresh_Main2_slice()

                #ToDo figure out ROI displays for multi-selection

            elif 'eV' in ROI1 and 'eV' in ROI2:  # Dispersion-based ROI
                # Do the ROI selection
                self.Int_map2 = ROI_select(self.data, ROI1).data - ROI_select(self.data, ROI2).data
                self.current_ROI_map_name = selected_ROI_name1 + '-' + selected_ROI_name2

                # Update the display, DCs, etc.
                self.refresh_Map2_slice()

            else:
                #print('Selection of ROIs invalid for subtraction mode')
                self.resetDispROI()
                self.resetMapROI()
                self.ROI_list.clearSelection()

        elif len(selected_ROIs) == 0:  # No items selected, reset the ROIs
            self.resetDispROI()
            self.resetMapROI()


        else:
            #print('Selection of ROIs invalid for subtraction mode')
            self.resetDispROI()
            self.resetMapROI()
            self.ROI_list.clearSelection()

    # -------------------------------------------------------------- #
    # Save ROI data on exit
    def save_on_quit(self):
        # Old hack to copy to the clipboard until we figure out the proper onexit versions here
        pyperclip.copy(str(self.ROI_store))

        try:
            # Write the relevant ROI into a new cell in the Jupyter notebook
            text = str(self.wname[0])+'.attrs[\'ROI\']='+str(self.ROI_store)
            cell_above(text)
        except:
            print('ROI region definition copied to the clipboard. Set \"a.attrs[\'ROI\'] = <<INSERT COPIED TEXT>>\" to add to attributes of DataArray `a`')

        time.sleep(0.5)

        # Close the panel
        self.disp4d_panel.close()

    # Close the app
    def close_app(self):
        time.sleep(0.5)
        self.disp4d_panel.close()










    # # ROI definition functions
    # # Define by click
    # def ROI_map_mouse_click(self, ev):
    #     pos = ev.scenePos()
    #     if (self.MapPlot_2.sceneBoundingRect().contains(pos)):
    #         # If it was the left mouse button, add a point
    #         if ev.button() == QtCore.Qt.LeftButton:
    #             # Get clicked point in axis location
    #             mousePoint = self.MapPlot_2.getViewBox().mapSceneToView(pos)
    #
    #             # Get current axis ranges
    #             rng = self.MapPlot_2.viewRange()
    #
    #             # Only add or remove point if click is within current axis limits
    #             if mousePoint.x() > rng[0][0] and mousePoint.x() < rng[0][1] and mousePoint.y() > rng[1][0] and mousePoint.y() < rng[1][1]:
    #                 # If clicking on an existing point, delete it
    #                 del_index = []
    #                 try:
    #                     # Iterate through the lists and check if an existing point is present
    #                     threshold = 0.02  # Delete points if within 2% of axis range of existing points
    #
    #                     # Get indices of point to delete
    #                     for count, ij in enumerate(zip(self.ROI_map_points.getData()[0],self.ROI_map_points.getData()[1])):
    #                         if abs(mousePoint.x() - ij[0])/(self.x1max_lim-self.x1min_lim) < threshold and abs(mousePoint.y() - ij[1])/(self.x2max_lim-self.x2min_lim) < threshold:
    #                             del_index.append(count)
    #
    #                 except:
    #                     pass
    #
    #                 # Delete point if single point identified
    #                 # Don't do anything if clicked point is close to 2 or more points
    #                 # Add point if no point identified to delete
    #
    #                 if len(del_index) == 0:  # add point
    #                     # add clicked point to scatter item
    #                     self.ROI_map_points.addPoints([mousePoint.x()], [mousePoint.y()])
    #
    #                 elif len(del_index) == 1:  # delete point
    #                     # Determine new points
    #                     new_points_x = np.delete(self.ROI_map_points.getData()[0], del_index)
    #                     new_points_y = np.delete(self.ROI_map_points.getData()[1], del_index)
    #
    #                     # Reset scatter plot array
    #                     self.ROI_map_points.setData(x=new_points_x, y=new_points_y)
    #
    #                 else:
    #                     return
    #
    #                 # Update the stored ROI
    #                 self.ROI_map = {'x1': self.ROI_map_points.getData()[0], 'x2': self.ROI_map_points.getData()[1]}
    #
    #                 # Add lines to the plot, closing the loop on the ROI
    #                 self.ROI_map_line.setData(x=np.append(self.ROI_map['x1'], self.ROI_map['x1'][0]),
    #                                           y=np.append(self.ROI_map['x2'], self.ROI_map['x2'][0]))
    #
    #         # If other mouse button pressed, ignore
    #         else:
    #             ev.ignore()
    #             return
    #
    # # Adjust ROI by mouse drag
    # def ROI_map_mouse_drag(self, ev):
    #     #ToDo look at replacing this with a polyROI defulat in pyqtgraph
    #
    #     # Return the mouse button that generated the click event
    #     # If it was not the mouse left button ignore the event
    #     if ev.button() != QtCore.Qt.LeftButton:
    #         ev.ignore()
    #         return
    #
    #     # Run if a mouse drag is initiated (if the mouse moves before the click above is released)
    #     if ev.isStart():
    #
    #         # Get the position of the cursor when the button was first pressed
    #         pos = ev.buttonDownPos()
    #
    #         # Find the point at the cursor when the button was first pressed
    #         # If the point was not under the cursor when the button was first pressed ignore the event
    #         if len(self.ROI_map_points.pointsAt(pos)) == 0:
    #             ev.ignore()
    #             return
    #
    #         # Get the drag point
    #         self.dragPoint = self.ROI_map_points.pointsAt(pos)
    #         print(dragPoint)
    #
    #         # # Get the distance between the original point position and the cursor when the button was first pressed
    #         # self.dragOffset = self.data['pos'][0] - pos
    #
    #     # Run when the click is released after the drag
    #     elif ev.isFinish():
    #
    #         # Clear the drag point
    #         self.dragPoint = None
    #         return
    #
    #     # Run if there is no drag, ignore the event
    #     else:
    #         if self.dragPoint is None:
    #             ev.ignore()
    #             return


#
#
# if __name__ == "__main__":
#     import sys
#     app = QtWidgets.QApplication(sys.argv)
#     DisplayPanel = QtWidgets.QMainWindow()
#     ui = Ui_DisplayPanel()
#     ui.setupUi(DisplayPanel)
#     DisplayPanel.show()
#     sys.exit(app.exec_())
