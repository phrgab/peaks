# -*- coding: utf-8 -*-

# Dependencies
from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import numpy as np
from matplotlib import cm
from skimage.registration import phase_cross_correlation
from functools import partial
import inspect
import pyperclip

from peaks.core.GUI.pyqt_CursorItem import CursorItem
from peaks.core.GUI.pyqt_ZoomItem import ZoomItem
from peaks.core.GUI.pyqt_PlotWidgetKP import PlotWidgetKP
from peaks.core.GUI.pyqt_ResizeItem import ResizeItem
from peaks.core.GUI.align.feedfromLoaders import feedfromLoaders
from peaks.core.utils.misc import cell_above

# Class to create a Fermi surface alignment panel object
class Ui_AlignFS_panel(object):

    # Run when AlignFS Panel object is created
    def __init__(self, AlignFS_panel):

        # Store the framework of the app
        self.AlignFS_panel = AlignFS_panel

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
        self.csr_texts = ["", "", "", ""]

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

        # Integration-crosshair spany
        self.ispany = 0

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
        self.labelcs = ['g', 'g', (255, 131, 0), 'y', 'y']

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
        self.dicts = []
        for i in range(4):
            self.dicts.append(
                {'dtype': float, 'size': self.csr_sizes[i], 'symbol': self.csr_symbols[i], 'brush': self.csr_brushs[i],
                 'text': self.csr_texts[i], 'textc': self.csr_textcs[i], 'pxMode': True, 'csr': self.csrs[i],
                 'csrc': self.csrcs[i], 'ispanx': self.ispanx, 'ispany': self.ispany, 'icsr': self.icsrs[i],
                 'icsrc': self.icsrcs[i], 'ibrush': self.ibrushs[i], 'h': self.h, 'v': self.v})

    ############################################################################
    # Initialize the GUI
    def setupUi(self):

        ### Define general GUI elements and geometry:

        # Specify main window properties
        self.AlignFS_panel.setObjectName("AlignFS_panel")
        #self.AlignFS_panel.setEnabled(True)
        self.AlignFS_panel.resize(999, 675)

        # Define central widget within main window
        self.centralwidget = QtWidgets.QWidget(self.AlignFS_panel)
        self.centralwidget.setMouseTracking(False)
        self.centralwidget.setObjectName("centralwidget")

        # Set central widget background color
        self.centralwidget.setAutoFillBackground(False)
        p = self.centralwidget.palette()
        p.setColor(self.centralwidget.backgroundRole(), QtCore.Qt.GlobalColor.white) # Input color here
        self.centralwidget.setPalette(p)

        # Define Plot frames
        # Constant energy cut
        self.Plot2 = PlotWidgetKP(self.centralwidget)
        self.Plot2.setGeometry(QtCore.QRect(30, 10, 401, 341))
        self.Plot2.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))
        self.Plot2.setObjectName("Plot2")

        # Dispersion along mapping angle frame
        self.Plot1 = PlotWidgetKP(self.centralwidget)
        self.Plot1.setGeometry(QtCore.QRect(430, 10, 271, 341))
        self.Plot1.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))
        self.Plot1.setObjectName("Plot1")

        # Dispersion along detector angle frame
        self.Plot0 = PlotWidgetKP(self.centralwidget)
        self.Plot0.setGeometry(QtCore.QRect(700, 10, 271, 341))
        self.Plot0.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))
        self.Plot0.setObjectName("Plot0")

        # Define CursorStats frame
        # CursorStats must not be in contact with Plots:
        # (to avoid random artefacts in the cursor stats labels when moving the cursors)
        self.CursorStats = pg.GraphicsLayoutWidget(self.centralwidget)  # The associated graphics layout is 'self.CursorStats.ci'
        self.CursorStats.setGeometry(QtCore.QRect(30, 355, 541, 120))
        self.CursorStats.setObjectName("CursorStats")
        self.CursorStats.ci.setSpacing(0)  # Set to zero the spacing between the cells in the GraphicsLayout
        self.CursorStats.ci.setBorder(color='k')  # Color the borders between the cells in the GraphicsLayout (useful to see the cell limits)

        # Define High cutoff slider
        self.HCutOff = QtWidgets.QSlider(self.centralwidget)
        self.HCutOff.setGeometry(QtCore.QRect(650, 400, 251, 22))
        self.HCutOff.setMaximum(100)
        self.HCutOff.setProperty("value", 90)
        self.HCutOff.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.HCutOff.setInvertedAppearance(False)
        self.HCutOff.setInvertedControls(False)
        self.HCutOff.setObjectName("HCutOff")

        # Define Low cutoff slider
        self.LCutOff = QtWidgets.QSlider(self.centralwidget)
        self.LCutOff.setGeometry(QtCore.QRect(650, 430, 251, 22))
        self.LCutOff.setMaximum(100)
        self.LCutOff.setProperty("value", 0)
        self.LCutOff.setSliderPosition(0)
        self.LCutOff.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.LCutOff.setInvertedAppearance(False)
        self.LCutOff.setInvertedControls(False)
        self.LCutOff.setObjectName("LCutOff")

        # Define Invert contrast checkbox
        self.Invert = QtWidgets.QCheckBox(self.centralwidget)
        self.Invert.setGeometry(QtCore.QRect(910, 420, 21, 21))
        self.Invert.setObjectName("Invert")

        # Define Colormap dropdown menu
        self.Cmaps = QtWidgets.QComboBox(self.centralwidget)
        self.Cmaps.setGeometry(QtCore.QRect(610, 370, 151, 26))
        self.Cmaps.setEditable(False)
        self.Cmaps.setCurrentText("Greys")
        self.Cmaps.setObjectName("Cmaps")
        self.Cmaps.addItems(["Greys", "Blues", "viridis", "plasma", "inferno", "magma", "cividis", "bone", "spring", "summer", "autumn", "winter", "cool", "jet", "ocean", "RdBu", "BrBG", "PuOr", "coolwarm", "bwr", "seismic"])  # Choose your favourites

        # Set energy labels
        self.Label_En = QtWidgets.QLabel(self.centralwidget)
        self.Label_En.setGeometry(QtCore.QRect(30, 495, 60, 16))
        self.Label_En.setObjectName("Label_En")
        self.Label_dE = QtWidgets.QLabel(self.centralwidget)
        self.Label_dE.setGeometry(QtCore.QRect(30, 525, 60, 16))
        self.Label_dE.setObjectName("Label_dE")

        # Set energy selection slider
        self.En_select = QtWidgets.QSlider(self.centralwidget)
        self.En_select.setGeometry(QtCore.QRect(100, 490, 161, 22))
        self.En_select.setMaximum(100)
        self.En_select.setProperty("value", 100)
        self.En_select.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.En_select.setInvertedAppearance(False)
        self.En_select.setInvertedControls(False)
        self.En_select.setObjectName("En_select")

        # Set energy integration selection
        self.dE_select = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.dE_select.setGeometry(QtCore.QRect(100, 520, 91, 24))
        self.dE_select.setLocale(QtCore.QLocale(QtCore.QLocale.Language.English, QtCore.QLocale.Country.UnitedStates))
        self.dE_select.setObjectName("dE_select")
        self.dE_select.setDecimals(3)
        self.dE_select.setSingleStep(0.001)

        # Define cut integration span
        self.Span_DC = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.Span_DC.setGeometry(QtCore.QRect(100, 550, 91, 24))
        self.Span_DC.setObjectName("Span_DC")
        self.Span_DC.setLocale(QtCore.QLocale(QtCore.QLocale.Language.English, QtCore.QLocale.Country.UnitedStates))
        self.Span_DC.setSingleStep(0.1)

        # Define rotation angle for crosshairs
        self.Rot_select = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.Rot_select.setGeometry(QtCore.QRect(100, 580, 91, 24))
        self.Rot_select.setLocale(QtCore.QLocale(QtCore.QLocale.Language.English, QtCore.QLocale.Country.UnitedStates))
        self.Rot_select.setObjectName("Rot_select")
        self.Rot_select.setSingleStep(1)
        self.Rot_select.setRange(-89,89)

        # Set setp size for rot_select change
        self.Delta_Rot = QtWidgets.QLineEdit(self.centralwidget)
        self.Delta_Rot.setGeometry(QtCore.QRect(200, 580, 61, 24))
        self.Delta_Rot.setObjectName("Delta_Rot")
        self.Delta_Rot.setText('1')

        # Define step labels
        self.LabelStep = QtWidgets.QLabel(self.centralwidget)
        self.LabelStep.setGeometry(QtCore.QRect(210, 610, 60, 16))
        self.LabelStep.setObjectName("LabelStep")
        self.LabelStep2 = QtWidgets.QLabel(self.centralwidget)
        self.LabelStep2.setGeometry(QtCore.QRect(520, 590, 60, 16))
        self.LabelStep2.setObjectName("LabelStep")

        # Define DC integration label
        self.Label_DC = QtWidgets.QLabel(self.centralwidget)
        self.Label_DC.setGeometry(QtCore.QRect(30, 555, 60, 16))
        self.Label_DC.setObjectName("Label_DC")

        # Define rotation angle label
        self.Label_Rot = QtWidgets.QLabel(self.centralwidget)
        self.Label_Rot.setGeometry(QtCore.QRect(30, 585, 60, 16))
        self.Label_Rot.setObjectName("Label_Rot")

        # Define centering label
        self.Cen_label = QtWidgets.QLabel(self.centralwidget)
        self.Cen_label.setGeometry(QtCore.QRect(330, 480, 171, 16))
        self.Cen_label.setObjectName("Cen_label")

        # Define x-center label
        self.Label_X_cen = QtWidgets.QLabel(self.centralwidget)
        self.Label_X_cen.setGeometry(QtCore.QRect(329, 530, 71, 20))
        self.Label_X_cen.setObjectName("Label_X_cen")

        # Define y-center label
        self.Label_Y_cen = QtWidgets.QLabel(self.centralwidget)
        self.Label_Y_cen.setGeometry(QtCore.QRect(329, 560, 71, 20))
        self.Label_Y_cen.setObjectName("Label_Y_cen")

        # Define x-center angle
        self.X_cen = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.X_cen.setGeometry(QtCore.QRect(410, 530, 91, 24))
        self.X_cen.setLocale(QtCore.QLocale(QtCore.QLocale.Language.English, QtCore.QLocale.Country.UnitedStates))
        self.X_cen.setObjectName("X_cen")
        self.X_cen.setSingleStep(0.1)

        # Define y-center angle
        self.Y_cen = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.Y_cen.setGeometry(QtCore.QRect(410, 560, 91, 24))
        self.Y_cen.setLocale(QtCore.QLocale(QtCore.QLocale.Language.English, QtCore.QLocale.Country.UnitedStates))
        self.Y_cen.setObjectName("Y_cen")
        self.Y_cen.setSingleStep(0.1)

        # Define x-center step size
        self.Delta_X_cen = QtWidgets.QLineEdit(self.centralwidget)
        self.Delta_X_cen.setGeometry(QtCore.QRect(510, 530, 61, 24))
        self.Delta_X_cen.setObjectName("Delta_X_cen")

        # Define y-center step size
        self.Delta_Y_cen = QtWidgets.QLineEdit(self.centralwidget)
        self.Delta_Y_cen.setGeometry(QtCore.QRect(510, 560, 61, 24))
        self.Delta_Y_cen.setObjectName("Delta_Y_cen")

        # Define autocentering button
        self.Auto_cen = QtWidgets.QPushButton(self.centralwidget)
        self.Auto_cen.setGeometry(QtCore.QRect(430, 490, 113, 32))
        self.Auto_cen.setObjectName("Auto_cen")

        # Define alignment label
        self.Label_align = QtWidgets.QLabel(self.centralwidget)
        self.Label_align.setGeometry(QtCore.QRect(610, 480, 171, 16))
        self.Label_align.setObjectName("Label_align")

        # Define alignment tool shape selection dropdown
        self.align_tool_shape = QtWidgets.QComboBox(self.centralwidget)
        self.align_tool_shape.setGeometry(QtCore.QRect(640, 510, 121, 26))
        self.align_tool_shape.setEditable(False)
        self.align_tool_shape.setCurrentText("None")
        self.align_tool_shape.setObjectName("align_tool_shape")
        self.align_tool_shape.addItems(["None", "Square", "Hexagon", "Hex_r90"])

        # Define alignment tool size slider
        self.align_tool_size = QtWidgets.QSlider(self.centralwidget)
        self.align_tool_size.setGeometry(QtCore.QRect(780, 510, 111, 22))
        self.align_tool_size.setMaximum(100)
        self.align_tool_size.setProperty("value", 50)
        self.align_tool_size.setSliderPosition(50)
        self.align_tool_size.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.align_tool_size.setInvertedAppearance(False)
        self.align_tool_size.setInvertedControls(False)
        self.align_tool_size.setObjectName("align_tool_size")

        # Define set normal attributes button
        self.set_norm = QtWidgets.QPushButton(self.centralwidget)
        self.set_norm.setGeometry(QtCore.QRect(880, 600, 113, 32))
        self.set_norm.setObjectName("Set and Exit")

        # Define exit button
        self.exit = QtWidgets.QPushButton(self.centralwidget)
        self.exit.setGeometry(QtCore.QRect(770, 600, 113, 32))
        self.exit.setObjectName("Close")

        # Define other GUI elements
        self.AlignFS_panel.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(self.AlignFS_panel)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 999, 22))
        self.menubar.setObjectName("menubar")
        self.AlignFS_panel.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self.AlignFS_panel)
        self.statusbar.setObjectName("statusbar")
        self.AlignFS_panel.setStatusBar(self.statusbar)

        # Initialize widget's labels
        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self.AlignFS_panel)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.AlignFS_panel.setWindowTitle(_translate("AlignFS_panel", "Align FS"))
        self.LabelStep.setText(_translate("AlignFS_panel", "Step"))
        self.Label_DC.setText(_translate("AlignFS_panel", "DC int."))
        self.Label_Rot.setText(_translate("AlignFS_panel", "Rot angle"))
        self.Cen_label.setText(_translate("AlignFS_panel", "Centering:"))
        self.Label_X_cen.setText(_translate("AlignFS_panel", "theta_par"))
        self.Label_Y_cen.setText(_translate("AlignFS_panel", "map_angle"))
        self.Delta_X_cen.setText(_translate("AlignFS_panel", "0.1"))
        self.Delta_Y_cen.setText(_translate("AlignFS_panel", "0.1"))
        self.Auto_cen.setText(_translate("AlignFS_panel", "Auto"))
        self.Label_En.setText(_translate("AlignFS_panel", "Energy"))
        self.Label_dE.setText(_translate("AlignFS_panel", "dE"))
        self.set_norm.setText(_translate("AlignFS_panel", "Set and Exit"))
        self.exit.setText(_translate("AlignFS_panel", "Close"))
        self.Label_align.setText(_translate("AlignFS_panel", "Alignment aid:"))

    ############################################################################
    # Resize GUI elements
    # (See pyqt_ResizeItem.py)
    def GUI_resize(self):
        ResizeItem(self.AlignFS_panel, self.Plot2)
        ResizeItem(self.AlignFS_panel, self.Plot1)
        ResizeItem(self.AlignFS_panel, self.Plot0)
        ResizeItem(self.AlignFS_panel, self.CursorStats)
        ResizeItem(self.AlignFS_panel, self.HCutOff)
        ResizeItem(self.AlignFS_panel, self.LCutOff)
        ResizeItem(self.AlignFS_panel, self.Invert)
        ResizeItem(self.AlignFS_panel, self.Cmaps)
        ResizeItem(self.AlignFS_panel, self.Label_En)
        ResizeItem(self.AlignFS_panel, self.Label_dE)
        ResizeItem(self.AlignFS_panel, self.En_select)
        ResizeItem(self.AlignFS_panel, self.dE_select)
        ResizeItem(self.AlignFS_panel, self.Span_DC)
        ResizeItem(self.AlignFS_panel, self.Rot_select)
        ResizeItem(self.AlignFS_panel, self.Delta_Rot)
        ResizeItem(self.AlignFS_panel, self.LabelStep)
        ResizeItem(self.AlignFS_panel, self.Label_DC)
        ResizeItem(self.AlignFS_panel, self.Label_Rot)
        ResizeItem(self.AlignFS_panel, self.Cen_label)
        ResizeItem(self.AlignFS_panel, self.Label_X_cen)
        ResizeItem(self.AlignFS_panel, self.Label_Y_cen)
        ResizeItem(self.AlignFS_panel, self.X_cen)
        ResizeItem(self.AlignFS_panel, self.Y_cen)
        ResizeItem(self.AlignFS_panel, self.Delta_X_cen)
        ResizeItem(self.AlignFS_panel, self.Delta_Y_cen)
        ResizeItem(self.AlignFS_panel, self.Auto_cen)
        ResizeItem(self.AlignFS_panel, self.Label_align)
        ResizeItem(self.AlignFS_panel, self.align_tool_shape)
        ResizeItem(self.AlignFS_panel, self.align_tool_size)
        ResizeItem(self.AlignFS_panel, self.set_norm)
        ResizeItem(self.AlignFS_panel, self.exit)


    ############################################################################

    def GUI_internal(self):
        # Zoom object definition at plots
        # (See pyqt_ZoomItem.py)
        self.Zoom2 = ZoomItem(self.Plot2)
        self.Zoom1 = ZoomItem(self.Plot1)
        self.Zoom0 = ZoomItem(self.Plot0)

        # Create image objects
        self.Image0 = pg.ImageItem()
        self.Image1 = pg.ImageItem()
        self.Image2 = pg.ImageItem()

        # Set the initial colormap
        self.Image0.setLookupTable(self.colormap)
        self.Image1.setLookupTable(self.colormap)
        self.Image2.setLookupTable(self.colormap)

        # Create cursor stats labels (force their height using setFixedHeight inherited to LabelItem from GraphicsWidget)
        self.labels = []
        for i in range(5):
            label = pg.LabelItem(justify='left', color=self.labelcs[i])
            label.setFixedHeight(20)
            self.labels.append(label)

    # Retrieve input data
    def GUI_files(self, data):
        # To extract the name of the wave in the Notebook
        def retrieve_name(var):
            callers_local_vars = inspect.currentframe().f_back.f_back.f_back.f_back.f_back.f_back.f_locals.items()
            return [var_name for var_name, var_val in callers_local_vars if var_val is var]

        # Store the input data in the Panel
        self.data = data.fillna(0)

        # Wave name as defined in the Notebook
        self.wname = retrieve_name(self.data)

        # Define the Panel data from the first input file
        self.GUI_setdata(**feedfromLoaders(self.data))

        # Construct the Panel GUI initial content
        self.GUI_initial()

        # Create auxiliary signal objects
        self.GUI_auxiliary()

        # Initialize Panel GUI elements updates
        self.GUI_connect(None)

    ############################################################################

    # Set GUI data
    def GUI_setdata(self, x, y, z, Int, xdim, ydim, zdim, Idim, xuts, yuts, zuts, Iuts, EF, angles, BL_conv):
        # Convert the input data to attributes of the Panel
        self.x = x
        self.y = y
        self.z = z
        self.Int = Int

        self.xdim = xdim
        self.ydim = ydim
        self.zdim = zdim
        self.Idim = Idim

        self.xuts = xuts
        self.yuts = yuts
        self.zuts = zuts
        self.Iuts = Iuts

        # Extract x parameters
        self.x_min = min(self.x)
        self.x_max = max(self.x)
        self.x_size = len(self.x)
        self.x_delta = x[1] - x[0]

        # Extract y parameters
        self.y_min = min(self.y)
        self.y_max = max(self.y)
        self.y_size = len(self.y)
        self.y_delta = y[1] - y[0]

        # Extract z parameters
        self.z_min = np.min(self.z)
        self.z_max = np.max(self.z)
        self.z_size = len(self.z)
        self.z_delta = z[1] - z[0]

        # Extract intensity parameters
        self.I_min = np.min(self.Int)
        self.I_max = np.max(self.Int)

        # Extract Fermi energy
        self.EF = EF

        # Extract relevant angles
        # Polar
        if self.xdim != 'polar' and self.ydim != 'polar':
            self.polar = angles['polar']
        else:
            self.polar = np.NaN
        # Tilt
        if self.xdim != 'tilt' and self.ydim != 'tilt':
            self.tilt = angles['tilt']
        else:
            self.tilt = np.NaN
        # Azi
        self.azi = angles['azi']

        # Deflector angles (dictionary entry zero if they don't exist)
        if self.xdim != 'defl_perp' and self.ydim != 'defl_perp':
            self.defl_perp = angles['defl_perp']
        else:
            self.defl_perp = np.NaN

        self.defl_par = angles['defl_par']  # This one should never be the mapping angle

        # Ana_polar
        if self.xdim != 'ana_polar' and self.ydim != 'ana_polar':
            self.ana_polar = angles['ana_polar']
        else:
            self.ana_polar = np.NaN

        self.ana_type = angles['ana_type']

        self.BL_conv = BL_conv


    ############################################################################

    # Construct Panel GUI initial content (dependant on loaded data)
    def GUI_initial(self):

        ### Set the axis labels

        # Energy slice
        self.Plot2.setLabel('left', text=self.ydim, units=self.yuts, unitPrefix=None)
        self.Plot2.setLabel('bottom', text=self.xdim, units=self.xuts, unitPrefix=None)

        # Dispersion along detector angle
        self.Plot1.setLabel('left', text=self.zdim, units=self.zuts, unitPrefix=None)
        self.Plot1.setLabel('bottom', text=self.xdim, units=self.xuts, unitPrefix=None)

        # Dispersion along mapping angle
        self.Plot0.setLabel('left', text=self.zdim, units=self.zuts, unitPrefix=None)
        self.Plot0.setLabel('bottom', text=self.ydim, units=self.yuts, unitPrefix=None)

        ########################################################################

        ### Set the integration crosshairs labels

        self.Label_X_cen.setText(self.xdim)
        self.Label_Y_cen.setText(self.ydim)

        ########################################################################

        ### MainPlot image initialization:

        # Add image objects
        self.Plot2.addItem(self.Image2)
        self.Plot1.addItem(self.Image1)
        self.Plot0.addItem(self.Image0)

        # Create limits rectangle for the data in MainPlot
        self.Image2Rectangle = QtCore.QRectF(self.x_min, self.y_max, self.x_max - self.x_min, self.y_min - self.y_max)
        self.Image1Rectangle = QtCore.QRectF(self.x_min, self.z_max, self.x_max - self.x_min, self.z_min - self.z_max)
        self.Image0Rectangle = QtCore.QRectF(self.y_min, self.z_max, self.y_max - self.y_min, self.z_min - self.z_max)

        # Plot 2D data
        self.sliceData = [0,0,0]  # List to store all the slices
        # Energy of desired slice - default to determined EF
        self.En = self.EF
        # Extract energy slice integrated over En +/- 5 meV (axis 2 for energy slice)
        self.get_slice(self.find_nearest(self.z,self.En-0.005), self.find_nearest(self.z,self.En+0.005), 2)
        self.Image2.setImage(self.sliceData[2])
        #ToDo Need to check theta_par sign for map

        # Run an auto-alignment for this slice
        self.auto_align()
        # Extract the normal emissions
        self.get_norm_values()

        # Determine slice for angles at +/- 0.2 deg from estimated auto-alignment as a default
        self.get_slice(self.find_nearest(self.x, self.x_off - 0.1), self.find_nearest(self.x, self.x_off + 0.1), 0)
        self.get_slice(self.find_nearest(self.y, self.y_off - 0.1), self.find_nearest(self.y, self.y_off + 0.1), 1)
        self.Image0.setImage(self.sliceData[0])
        self.Image1.setImage(self.sliceData[1])

        # Set the contrast
        self.contrast()

        # Scale 2D data
        self.Image0.setRect(self.Image0Rectangle)
        self.Image1.setRect(self.Image1Rectangle)
        self.Image2.setRect(self.Image2Rectangle)

        ########################################################################

        # Set corresponding widgets for these
        # Energy
        self.En_select.setMaximum(self.z_size - 1)
        self.En_select.setValue(int(self.z_size*(self.En-self.z_min)/(self.z_max-self.z_min)))

        # dE
        self.dE_select.setValue(0.01)

        # DC int
        self.Span_DC.setValue(0.2)

        # Centering
        self.X_cen.setRange(self.x_min,self.x_max)
        self.X_cen.setValue(self.x_off)
        self.X_cen.setSingleStep(self.x_delta)
        self.Delta_X_cen.setText(str(np.round(self.x_delta,3)))
        self.Y_cen.setRange(self.y_min, self.y_max)
        self.Y_cen.setValue(self.y_off)
        self.Y_cen.setSingleStep(self.y_delta)
        self.Delta_Y_cen.setText(str(np.round(self.y_delta,3)))

        ########################################################################

        ### Limit zoom and panning of MainPlot and CutPlots:

        # Define zoom/panning limits
        self.xmin_lim = self.x_min - (self.x_max - self.x_min) * self.pd
        self.xmax_lim = self.x_max + (self.x_max - self.x_min) * self.pd
        self.ymin_lim = self.y_min - (self.y_max - self.y_min) * self.pd
        self.ymax_lim = self.y_max + (self.y_max - self.y_min) * self.pd
        self.zmin_lim = self.z_min - (self.z_max - self.z_min) * self.pd
        self.zmax_lim = self.z_max + (self.z_max - self.z_min) * self.pd

        # Set zoom/panning limits of plots
        self.Plot2.getViewBox().setLimits(xMin=self.xmin_lim, xMax=self.xmax_lim, yMin=self.ymin_lim, yMax=self.ymax_lim)
        self.Plot1.getViewBox().setLimits(xMin=self.xmin_lim, xMax=self.xmax_lim, yMin=self.zmin_lim, yMax=self.zmax_lim)
        self.Plot0.getViewBox().setLimits(xMin=self.ymin_lim, xMax=self.ymax_lim, yMin=self.zmin_lim, yMax=self.zmax_lim)

        ########################################################################

        ### Set the initial range of plots:

        # Set the initial range of plots
        self.Plot2.getViewBox().setXRange(self.xmin_lim, self.xmax_lim, padding=0)
        self.Plot2.getViewBox().setYRange(self.ymin_lim, self.ymax_lim, padding=0)
        self.Plot1.getViewBox().setXRange(self.xmin_lim, self.xmax_lim, padding=0)
        self.Plot1.getViewBox().setYRange(self.zmin_lim, self.zmax_lim, padding=0)
        self.Plot0.getViewBox().setXRange(self.ymin_lim, self.ymax_lim, padding=0)
        self.Plot0.getViewBox().setYRange(self.zmin_lim, self.zmax_lim, padding=0)

        ########################################################################

        ### Set the initial autorange policy:

        # Disable auto range for plots
        self.Plot2.disableAutoRange()
        self.Plot1.disableAutoRange()
        self.Plot0.disableAutoRange()

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
        self.cursors = CursorItem()

        # Define the zoom/cursor objects space
        self.space2 = [[self.x_min, self.y_min], [self.x_max, self.y_max]]
        self.space1 = [[self.x_min, self.z_min], [self.x_max, self.z_max]]
        self.space0 = [[self.y_min, self.z_min], [self.y_max, self.z_max]]

        # Set the space for the zoom objects
        self.Zoom2.setSpace(self.space2)
        self.Zoom1.setSpace(self.space1)
        self.Zoom0.setSpace(self.space0)

        # Set the space for the cursors
        self.cursors.setSpace(self.space2)

        # Add the leader cursors in MainPlot
        self.Plot2.addItem(self.cursors)

        # Set the initial cursors configuration
        self.cursors.setData(pos=np.array([[self.x_off, self.y_off]]), **self.dicts[0])

        # Create normal emission line in plot1
        self.scsrvs1 = pg.InfiniteLine(pos=self.cursors.data['pos'][0][0], angle=90, pen=self.csrcs[0])
        self.Plot1.addItem(self.scsrvs1)

        # Create normal emission line in plot
        self.scsrvs0 = pg.InfiniteLine(pos=self.cursors.data['pos'][0][1], angle=90, pen=self.csrcs[0])
        self.Plot0.addItem(self.scsrvs0)

        # Add energy markers in cuts
        self.scsrh1 = pg.InfiniteLine(pos=self.En, angle=0, pen=self.csrcs[0])
        self.scsrh0 = pg.InfiniteLine(pos=self.En, angle=0, pen=self.csrcs[0])
        self.Plot1.addItem(self.scsrh1)
        self.Plot0.addItem(self.scsrh0)

        # Add the cursor stats labels in CursorStats
        self.CursorStats.addItem(self.labels[0], row=0, col=0)  # Row 1 - offset info
        self.CursorStats.addItem(self.labels[1], row=1, col=0)  # Row 2 - energy info
        self.CursorStats.addItem(self.labels[2], row=2, col=0)  # Row 3 - peaks normal emissions
        self.CursorStats.addItem(self.labels[3], row=3, col=0)  # Row 4 - current manipulator info
        self.CursorStats.addItem(self.labels[4], row=4, col=0)  # Row 5 - norm emision info

        ### Define the cursor stats format

        # String particles (dependant on user preferences)
        str1 = "<span style='font-size:" + self.fontsz + "pt'> Map angular offsets: "
        str2 = "%0." + self.digits + "f"
        str3 = "<span style='font-size:" + self.fontsz + "pt'> Current slice @ E = "
        str4 = "<span style='font-size:" + self.fontsz + "pt'> peaks norm emission: "
        str5 = "<span style='font-size:" + self.fontsz + "pt'> Current manip angles: "
        str6 = "<span style='font-size:" + self.fontsz + "pt'> Normal emission: "


        # Full strings
        self.format_csr = str1 + "%s = " + str2 + "; %s = " + str2 + "; azimuth = " + str2
        self.format_csr2 = str3 + str2 + " +/- " + str2 + ' eV'
        self.format_csr3 = str4 + "%s = " + str2 + "; %s = " + str2 + "; %s = " + str2
        self.format_csr4 = str5 + "%s = " + str2 + "; %s = " + str2 + "; %s = " + str2
        self.format_csr5 = str6 + "%s = " + str2 + "; %s = " + str2 + "; %s = " + str2


        # Set the initial cursor stats labels info (cursors)
        # Set label text
        self.update_labels([0,1,2,3,4])

        # Initialize the signals array
        # self.cns[0] = None, such that the signal count starts at 1
        self.cns = [None] + [False] * self.GUI_connect([0])



    ############################################################################
    # Border between static elements and dynamic elements
    ############################################################################

    ############################################################################

    def GUI_auxiliary(self):
        # ----------------------------------------------------------------------#

        # Signals to show all cursor objects
        self.show_csr_sig = partial(self.show_csr, self.allkey)  # Signal object 20

        # Signals to hide all cursor objects
        self.hide_csr_sig = partial(self.hide_csr, self.allkey)  # Signal object 21

        # ----------------------------------------------------------------------#

        # Individual cursor arrow movement
        self.csrs_mov = [None] * 3
        self.csrs_mov[0] = partial(self.arrowmove, 0)  # Signal object 26
        self.csrs_mov[1] = partial(self.arrowmove, 1)  # Signal object 27
        self.csrs_mov[2] = partial(self.arrowmove, 2)  # Signal object 28


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

        # Signals when the HCutOff slider is moved
        i = 1
        if not cn or i in cn:
            if self.cns[i] == False:
                self.HCutOff.valueChanged.connect(self.contrast)
                self.cns[i] = True

        # Signals when the LCutOff slider is moved
        i = 2
        if not cn or i in cn:
            if self.cns[i] == False:
                self.LCutOff.valueChanged.connect(self.contrast)
                self.cns[i] = True

        # Signals when the Invert checkbox is checked/unchecked
        i = 3
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Invert.clicked.connect(self.invert)
                self.cns[i] = True

        # Signals when an option in Cmaps is selected
        i = 4
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Cmaps.activated.connect(self.cmap_select)
                self.cns[i] = True

        # ----------------------------------------------------------------------#
        # Signals when the En_select slider is moved
        i = 5
        if not cn or i in cn:
            if self.cns[i] == False:
                self.En_select.valueChanged.connect(self.update_Eslice)
                self.cns[i] = True

        # Signals when the dE_select value is changed
        i = 6
        if not cn or i in cn:
            if self.cns[i] == False:
                self.dE_select.valueChanged.connect(self.update_Eslice)
                self.cns[i] = True

        # ----------------------------------------------------------------------#
        # Signals when the Plot2 range is changed
        i = 7
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Plot2.sigRangeChanged.connect(self.updaterange_from_plot2)
                self.cns[i] = True

        # Signals when the Plot1 range is changed
        i = 8
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Plot1.sigRangeChanged.connect(self.updaterange_from_plot1)
                self.cns[i] = True

        # Signals when the Plot0 range is changed
        i = 9
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Plot0.sigRangeChanged.connect(self.updaterange_from_plot0)
                self.cns[i] = True

        # ----------------------------------------------------------------------#
        # Signals when the rotation selection value is changed
        i = 10
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Rot_select.valueChanged.connect(self.rot_update)
                self.cns[i] = True

        # Signals when enter is pressed over self.Delta_Rot or when self.Delta_Rot loses focus
        i = 11
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Delta_Rot.editingFinished.connect(self.rot_step_update)
                self.cns[i] = True

        # ----------------------------------------------------------------------#
        # Signals when the mapping_angle selection value is changed
        i = 12
        if not cn or i in cn:
            if self.cns[i] == False:
                self.X_cen.valueChanged.connect(self.update_cursor_pos)
                self.cns[i] = True

        # Signals when enter is pressed over self.Delta_X_cen or when self.Delta_X_cen loses focus
        i = 13
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Delta_X_cen.editingFinished.connect(self.X_cen_step_update)
                self.cns[i] = True

        # Signals when the theta_par selection value is changed
        i = 14
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Y_cen.valueChanged.connect(self.update_cursor_pos)
                self.cns[i] = True

        # Signals when enter is pressed over self.Delta_Y_cen or when self.Delta_Y_cen loses focus
        i = 15
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Delta_Y_cen.editingFinished.connect(self.Y_cen_step_update)
                self.cns[i] = True

        # ----------------------------------------------------------------------#

        # Update when cursor is dragged
        i = 16
        if not cn or i in cn:
            if self.cns[i] == False:
                self.cursors.scatter.sigPlotChanged.connect(self.update_from_cursor)
                self.cns[i] = True

        # ----------------------------------------------------------------------#

        # Change alignment shape
        i = 17
        if not cn or i in cn:
            if self.cns[i] == False:
                self.align_tool_shape.activated.connect(self.update_align_shape)
                self.cns[i] = True

        # Change alignment shape size
        i = 18
        if not cn or i in cn:
            if self.cns[i] == False:
                self.align_tool_size.valueChanged.connect(self.update_align_shape)
                self.cns[i] = True

        # ----------------------------------------------------------------------#
        # Run autoalignment on self.Auto_cen click
        i = 19
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Auto_cen.clicked.connect(self.refresh_auto_align)
                self.cns[i] = True

        # ----------------------------------------------------------------------#
        # Signals to show all cursor objects
        i = 20
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Plot2.sigKeyRelease.connect(self.show_csr_sig)
                self.cns[i] = True

        i = 21
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Plot1.sigKeyRelease.connect(self.show_csr_sig)
                self.cns[i] = True

        i = 22
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Plot0.sigKeyRelease.connect(self.show_csr_sig)
                self.cns[i] = True

        # Signals to hide all cursor objects
        i = 23
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Plot2.sigKeyPress.connect(self.hide_csr_sig)
                self.cns[i] = True

        i = 24
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Plot1.sigKeyPress.connect(self.hide_csr_sig)
                self.cns[i] = True

        i = 25
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Plot0.sigKeyPress.connect(self.hide_csr_sig)
                self.cns[i] = True

        # ----------------------------------------------------------------------#

        # Individual cursor arrow movements
        i = 26
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Plot0.sigKeyPress.connect(self.csrs_mov[0])
                self.cns[i] = True

        i = 27
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Plot1.sigKeyPress.connect(self.csrs_mov[1])
                self.cns[i] = True

        i = 28
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Plot2.sigKeyPress.connect(self.csrs_mov[2])
                self.cns[i] = True

        # ----------------------------------------------------------------------#

        # Close the app on button press
        i = 29
        if not cn or i in cn:
            if self.cns[i] == False:
                self.exit.clicked.connect(self.close_app)
                self.cns[i] = True

        # Set normal emission and close the app on button press
        i = 30
        if not cn or i in cn:
            if self.cns[i] == False:
                self.set_norm.clicked.connect(self.set_and_close_app)
                self.cns[i] = True

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

        # Signals when the HCutOff slider is moved
        i = 1
        if not cn or i in cn:
            try:
                self.HCutOff.valueChanged.disconnect(self.contrast)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # Signals when the LCutOff slider is moved
        i = 2
        if not cn or i in cn:
            try:
                self.LCutOff.valueChanged.disconnect(self.contrast)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # Signals when the Invert checkbox is checked/unchecked
        i = 3
        if not cn or i in cn:
            try:
                self.Invert.clicked.disconnect(self.invert)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # Signals when an option in Cmaps is selected
        i = 4
        if not cn or i in cn:
            try:
                self.Cmaps.activated.disconnect(self.cmap_select)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # ----------------------------------------------------------------------#
        # Signals when the En_select slider is moved
        i = 5
        if not cn or i in cn:
            try:
                self.En_select.valueChanged.disconnect(self.update_Eslice)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # Signals when the dE_select value is changed
        i = 6
        if not cn or i in cn:
            try:
                self.dE_select.valueChanged.disconnect(self.update_Eslice)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # ----------------------------------------------------------------------#
        # Signals when the Plot2 range is changed
        i = 7
        if not cn or i in cn:
            try:
                self.Plot2.sigRangeChanged.disconnect(self.updaterange_from_plot2)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # Signals when the Plot1 range is changed
        i = 8
        if not cn or i in cn:
            try:
                self.Plot1.sigRangeChanged.disconnect(self.updaterange_from_plot1)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # Signals when the Plot0 range is changed
        i = 9
        if not cn or i in cn:
            try:
                self.Plot0.sigRangeChanged.disconnect(self.updaterange_from_plot0)
            except:
                print(i)
            finally:
                self.cns[i] = False


        # ----------------------------------------------------------------------#
        # Signals when the Rot_select value is changed
        i = 10
        if not cn or i in cn:
            try:
                self.Rot_select.valueChanged.disconnect(self.rot_update)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # Signals when enter is pressed over self.Delta_Rot or when self.Delta_Rot loses focus
        i = 11
        if not cn or i in cn:
            try:
                self.Delta_Rot.editingFinished.disconnect(self.rot_step_update)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # ----------------------------------------------------------------------#
        # Signals when the mapping_angle selection value is changed
        i = 12
        if not cn or i in cn:
            try:
                self.X_cen.valueChanged.disconnect(self.update_cursor_pos)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # Signals when enter is pressed over self.Delta_X_cen or when self.Delta_X_cen loses focus
        i = 13
        if not cn or i in cn:
            try:
                self.Delta_X_cen.editingFinished.disconnect(self.X_cen_step_update)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # Signals when the theta_par selection value is changed
        i = 14
        if not cn or i in cn:
            try:
                self.Y_cen.valueChanged.disconnect(self.update_cursor_pos)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # Signals when enter is pressed over self.Delta_Y_cen or when self.Delta_Y_cen loses focus
        i = 15
        if not cn or i in cn:
            try:
                self.Delta_Y_cen.editingFinished.disconnect(self.Y_cen_step_update)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # ----------------------------------------------------------------------#

        # Update when cursor is dragged
        i = 16
        if not cn or i in cn:
            try:
                self.cursors.scatter.sigPlotChanged.disconnect(self.update_from_cursor)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # ----------------------------------------------------------------------#

        # Change alignment shape
        i = 17
        if not cn or i in cn:
            try:
                self.align_tool_shape.activated.disconnect(self.update_align_shape)
            except:
                print(i)
            finally:
                self.cns[i] = False

        i = 18
        if not cn or i in cn:
            try:
                self.align_tool_size.valueChanged.disconnect(self.update_align_shape)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # ----------------------------------------------------------------------#
        # Run autoalignment on self.Auto_cen click
        i = 19
        if not cn or i in cn:
            try:
                self.Auto_cen.clicked.disconnect(self.refresh_auto_align)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # ----------------------------------------------------------------------#
        # Signals to show all cursor objects
        i = 20
        if not cn or i in cn:
            try:
                self.Plot2.sigKeyRelease.disconnect(self.show_csr_sig)
            except:
                print(i)
            finally:
                self.cns[i] = False

        i = 21
        if not cn or i in cn:
            try:
                self.Plot1.sigKeyRelease.disconnect(self.show_csr_sig)
            except:
                print(i)
            finally:
                self.cns[i] = False

        i = 22
        if not cn or i in cn:
            try:
                self.Plot0.sigKeyRelease.disconnect(self.show_csr_sig)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # Signals to hide all cursor objects
        i = 23
        if not cn or i in cn:
            try:
                self.Plot2.sigKeyPress.disconnect(self.hide_csr_sig)
            except:
                print(i)
            finally:
                self.cns[i] = False

        i = 24
        if not cn or i in cn:
            try:
                self.Plot1.sigKeyPress.disconnect(self.hide_csr_sig)
            except:
                print(i)
            finally:
                self.cns[i] = False

        i = 25
        if not cn or i in cn:
            try:
                self.Plot0.sigKeyPress.disconnect(self.hide_csr_sig)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # ----------------------------------------------------------------------#

        # Individual cursor arrow movement
        i = 26
        if not cn or i in cn:
            try:
                self.Plot0.sigKeyPress.disconnect(self.csrs_mov[0])
            except:
                print(i)
            finally:
                self.cns[i] = False

        i = 27
        if not cn or i in cn:
            try:
                self.Plot1.sigKeyPress.disconnect(self.csrs_mov[1])
            except:
                print(i)
            finally:
                self.cns[i] = False

        i = 28
        if not cn or i in cn:
            try:
                self.Plot0.sigKeyPress.disconnect(self.csrs_mov[2])
            except:
                print(i)
            finally:
                self.cns[i] = False

        # ----------------------------------------------------------------------#

        # Close the app on button press
        i = 29
        if not cn or i in cn:
            try:
                self.exit.clicked.disconnect(self.close_app)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # Set norm and close the app on button press
        i = 30
        if not cn or i in cn:
            try:
                self.set_norm.clicked.disconnect(self.set_and_close_app)
            except:
                print(i)
            finally:
                self.cns[i] = False

    ############################################################################
    # Internal methods begin
    ############################################################################

    ### GUI internal methods (FS align panel functionalities)

    # Extract slice from 3D array along given axis (sliceData[0]: cut1, sliceData[1]: cut2, sliceData[2]: Energy slice
    def get_slice(self, i0, i1, ax):
        if i0 == i1:
            if ax == 2:
                self.sliceData[ax] = np.flip(self.Int.take(indices=i0, axis=ax),1)
            else:
                self.sliceData[ax] = np.flip(self.Int.take(indices=i0, axis=ax),1)
        else:
            if ax == 2:
                self.sliceData[ax] = np.flip(np.mean(self.Int.take(indices=range(min([i0,i1]),max([i0,i1])), axis=ax), axis=ax),1)
            else:
                self.sliceData[ax] = np.flip(np.mean(self.Int.take(indices=range(min([i0, i1]), max([i0, i1])), axis=ax), axis=ax),1)

    # Update the energy slice
    def update_Eslice(self):
        # Remove current plot
        #self.Plot2.removeItem(self.Image2)

        # Get new energy limits
        Emin = self.z[self.En_select.value()] - self.dE_select.value()/2
        Emin_i = self.find_nearest(self.z, Emin)
        Emax = self.z[self.En_select.value()] + self.dE_select.value()/2
        Emax_i = self.find_nearest(self.z, Emax)

        # Select corresponding slice
        self.get_slice(Emin_i, Emax_i, 2)

        # Set plot
        self.Image2.setImage(self.sliceData[2])

        # Update contrast
        self.contrast()

        # Update horizontal lines
        self.scsrh1.setPos((Emax + Emin)/2)
        self.scsrh0.setPos((Emax + Emin)/2)

        # Update labels
        self.update_labels([1])
        #self.labels[1].setText(self.format_csr2 % (self.z[self.En_select.value()], self.dE_select.value() / 2))

    # Update the angular slices
    def update_ang_slice(self):

        # Get new energy limits
        xmin = self.X_cen.value() - self.Span_DC.value() / 2
        xmin_i = self.find_nearest(self.x, xmin)
        xmax = self.X_cen.value() + self.Span_DC.value() / 2
        xmax_i = self.find_nearest(self.x, xmax)
        ymin = self.Y_cen.value() - self.Span_DC.value() / 2
        ymin_i = self.find_nearest(self.y, ymin)
        ymax = self.Y_cen.value() + self.Span_DC.value() / 2
        ymax_i = self.find_nearest(self.y, ymax)

        # Select corresponding slices
        self.get_slice(xmin_i, xmax_i, 0)
        self.get_slice(ymin_i, ymax_i, 1)

        # Set plots
        self.Image0.setImage(self.sliceData[0])
        self.Image1.setImage(self.sliceData[1])

        # Update contrast
        self.contrast()


    # Update rotation angle
    def rot_update(self):

        # Set the rotation of the cursors
        self.cursors.setData(pos=[[self.cursors.data['pos'][0][0],self.cursors.data['pos'][0][1]]], angle=-self.Rot_select.value(), **self.dicts[0])

        # Update the labels
        self.update_labels([0, 2, 3, 4])
        # self.labels[0].setText(self.format_csr % (self.xdim, self.cursors.data['pos'][0][0], self.ydim, self.cursors.data['pos'][0][1], self.Rot_select.value()))
        # self.labels[2].setText(self.format_csr5 % ('norm_polar', self.norm_polar, 'norm_tilt', self.norm_tilt, 'norm_azi', self.norm_azi))
        # self.labels[3].setText(self.format_csr4 % (self.BL_conv['polar_name'], self.polar, self.BL_conv['tilt_name'], self.tilt, self.BL_conv['azi_name'], self.azi)), self.get_norm_values()
        # self.labels[4].setText(self.format_csr5 % (self.BL_conv['polar_name'], self.x_off, self.BL_conv['tilt_name'], self.y_off, self.BL_conv['azi_name'], self.Rot_select.value()))

    # Update step size for rotation angle selector box
    def rot_step_update(self):
        # set the step size for self.Rot_select
        self.Rot_select.setSingleStep(float(self.Delta_Rot.text()))


    # Update cursor position from numerical boxes
    def update_cursor_pos(self):
        # Disconnect other cursor updates
        self.cursors.scatter.sigPlotChanged.disconnect(self.update_from_cursor)

        # Set the position of the cursors on slice2
        self.cursors.setData(pos=[[self.X_cen.value(), self.Y_cen.value()]], angle=-self.Rot_select.value(), **self.dicts[0])

        # Set the positions of the normal emission lines on the other slices
        self.scsrvs0.setPos(pos=self.cursors.data['pos'][0][1])
        self.scsrvs1.setPos(pos=self.cursors.data['pos'][0][0])

        # Update the slices
        self.update_ang_slice()

        # Update the labels
        self.update_labels([0,2,3,4])
        #self.labels[0].setText(self.format_csr % (self.xdim, self.cursors.data['pos'][0][0], self.ydim, self.cursors.data['pos'][0][1], self.Rot_select.value()))

        # Update alignment shape marker
        self.update_align_shape()

        # Reconnect other cursor updates
        self.cursors.scatter.sigPlotChanged.connect(self.update_from_cursor)


    # Update from cursor drag
    def update_from_cursor(self):
        # Disconnect other cursor updates
        self.X_cen.valueChanged.disconnect(self.update_cursor_pos)
        self.Y_cen.valueChanged.disconnect(self.update_cursor_pos)

        # From drag of cursor on slice2, set the positions of the normal emission lines on the other slices
        self.scsrvs0.setPos(pos=self.cursors.data['pos'][0][1])
        self.scsrvs1.setPos(pos=self.cursors.data['pos'][0][0])

        # Update the dialog boxes
        self.X_cen.setValue(self.cursors.data['pos'][0][0])
        self.Y_cen.setValue(self.cursors.data['pos'][0][1])

        # Update the slices
        self.update_ang_slice()

        # Update the labels
        self.update_labels([0, 2, 3, 4])
        #self.labels[0].setText(self.format_csr % (self.xdim, self.cursors.data['pos'][0][0], self.ydim, self.cursors.data['pos'][0][1], self.Rot_select.value()))

        # Update alignment shape marker
        self.update_align_shape()

        # Reconnect other cursor updates
        self.X_cen.valueChanged.connect(self.update_cursor_pos)
        self.Y_cen.valueChanged.connect(self.update_cursor_pos)


    # Cursor arrow movements
    def arrowmove(self, plt_num, evt):

        if plt_num == 2:  # Move on energy slice
            # Current positions
            x_pos = self.cursors.data['pos'][0][0]
            y_pos = self.cursors.data['pos'][0][1]

            # Move cursor data-row up
            if evt.key() == QtCore.Qt.Key.Key_Up and y_pos < self.y_max:
                # Get new position
                y_pos_new = self.y[self.find_nearest(self.y, y_pos) + 1]
                # Set the position of the cursors on slice2
                self.cursors.setData(pos=[[x_pos, y_pos_new]], angle=-self.Rot_select.value(), **self.dicts[0])

            # Move cursor data-row down
            if evt.key() == QtCore.Qt.Key.Key_Down and y_pos > self.y_min:
                # Get new position
                y_pos_new = self.y[self.find_nearest(self.y, y_pos) - 1]
                # Set the position of the cursors on slice2
                self.cursors.setData(pos=[[x_pos, y_pos_new]], angle=-self.Rot_select.value(), **self.dicts[0])

            # Move cursor data-row left
            if evt.key() == QtCore.Qt.Key.Key_Left and x_pos > self.x_min:
                # Get new position
                x_pos_new = self.x[self.find_nearest(self.x, x_pos) - 1]
                # Set the position of the cursors on slice2
                self.cursors.setData(pos=[[x_pos_new, y_pos]], angle=-self.Rot_select.value(), **self.dicts[0])

            # Move cursor data-row right
            if evt.key() == QtCore.Qt.Key.Key_Right and x_pos < self.x_max:
                # Get new position
                x_pos_new = self.x[self.find_nearest(self.x, x_pos) + 1]
                # Set the position of the cursors on slice2
                self.cursors.setData(pos=[[x_pos_new, y_pos]], angle=-self.Rot_select.value(), **self.dicts[0])

        if plt_num == 1:  # Move on slice1 cut
            # Current positions
            ang_pos = self.cursors.data['pos'][0][0]
            z_pos = self.En_select.value()

            # Move energy slice up
            if evt.key() == QtCore.Qt.Key.Key_Up and z_pos < self.En_select.maximum():
                # Move the energy selection widget, everything follows from there
                self.En_select.setValue(z_pos + 1)

            # Move energy slice down
            if evt.key() == QtCore.Qt.Key.Key_Down and z_pos > self.En_select.minimum():
                # Move the energy selection widget, everything follows from there
                self.En_select.setValue(z_pos - 1)

            # Move cursor data-row left
            if evt.key() == QtCore.Qt.Key.Key_Left and ang_pos > self.x_min:
                # Get new position
                ang_pos_new = self.x[self.find_nearest(self.x, ang_pos) - 1]
                # Set the position of the cursors on slice2, triggering other changes from there
                self.cursors.setData(pos=[[ang_pos_new, self.cursors.data['pos'][0][1]]], angle=-self.Rot_select.value(), **self.dicts[0])

            # Move cursor data-row right
            if evt.key() == QtCore.Qt.Key.Key_Right and ang_pos < self.x_max:
                # Get new position
                ang_pos_new = self.x[self.find_nearest(self.x, ang_pos) + 1]
                # Set the position of the cursors on slice2, triggering other changes from there
                self.cursors.setData(pos=[[ang_pos_new, self.cursors.data['pos'][0][1]]],angle=-self.Rot_select.value(), **self.dicts[0])

        if plt_num == 0:  # Move on slice0 cut
            # Current positions
            ang_pos = self.cursors.data['pos'][0][1]
            z_pos = self.En_select.value()

            # Move energy slice up
            if evt.key() == QtCore.Qt.Key.Key_Up and z_pos < self.En_select.maximum():
                # Move the energy selection widget, everything follows from there
                self.En_select.setValue(z_pos + 1)

            # Move energy slice down
            if evt.key() == QtCore.Qt.Key.Key_Down and z_pos > self.En_select.minimum():
                # Move the energy selection widget, everything follows from there
                self.En_select.setValue(z_pos - 1)

            # Move cursor data-row left
            if evt.key() == QtCore.Qt.Key.Key_Left and ang_pos > self.y_min:
                # Get new position
                ang_pos_new = self.y[self.find_nearest(self.y, ang_pos) - 1]
                # Set the position of the cursors on slice2, triggering other changes from there
                self.cursors.setData(pos=[[self.cursors.data['pos'][0][0], ang_pos_new]], angle=-self.Rot_select.value(), **self.dicts[0])

            # Move cursor data-row right
            if evt.key() == QtCore.Qt.Key.Key_Right and ang_pos < self.y_max:
                # Get new position
                ang_pos_new = self.y[self.find_nearest(self.y, ang_pos) + 1]
                # Set the position of the cursors on slice2, triggering other changes from there
                self.cursors.setData(pos=[[self.cursors.data['pos'][0][0], ang_pos_new]],angle=-self.Rot_select.value(), **self.dicts[0])



    # Update step size for rotation angle selector box
    def X_cen_step_update(self):
        # set the step size for self.X_cen
        self.X_cen.setSingleStep(float(self.Delta_X_cen.text()))

    # Update step size for rotation angle selector box
    def Y_cen_step_update(self):
        # set the step size for self.Y_cen
        self.Y_cen.setSingleStep(float(self.Delta_Y_cen.text()))

    # Link range: plot2 -> plot1, plot0
    def updaterange_from_plot2(self):
        # Disconnect other auto-range updates
        self.Plot0.sigRangeChanged.disconnect(self.updaterange_from_plot0)
        self.Plot1.sigRangeChanged.disconnect(self.updaterange_from_plot1)

        # Determine angular range from plot2
        rng = self.Plot2.viewRange()

        # Set relevant ranges to plots 1 and 0
        self.Plot0.setXRange(rng[1][0],rng[1][1], padding=0)
        self.Plot1.setXRange(rng[0][0], rng[0][1], padding=0)

        # Reconnect other auto-range methods
        self.Plot0.sigRangeChanged.connect(self.updaterange_from_plot0)
        self.Plot1.sigRangeChanged.connect(self.updaterange_from_plot1)

    # Link range: plot1 -> plot2, plot0
    def updaterange_from_plot1(self):
        # Disconnect other auto-range updates
        self.Plot0.sigRangeChanged.disconnect(self.updaterange_from_plot0)
        self.Plot2.sigRangeChanged.disconnect(self.updaterange_from_plot2)

        # Determine angular range from plot2
        rng = self.Plot1.viewRange()

        # Set relevant ranges to plots 1 and 0
        self.Plot0.setYRange(rng[1][0], rng[1][1], padding=0)
        self.Plot2.setXRange(rng[0][0], rng[0][1], padding=0)

        # Reconnect other auto-range methods
        self.Plot0.sigRangeChanged.connect(self.updaterange_from_plot0)
        self.Plot2.sigRangeChanged.connect(self.updaterange_from_plot2)

    # Link range: plot0 -> plot2, plot1
    def updaterange_from_plot0(self):
        # Disconnect other auto-range updates
        self.Plot1.sigRangeChanged.disconnect(self.updaterange_from_plot1)
        self.Plot2.sigRangeChanged.disconnect(self.updaterange_from_plot2)

        # Determine angular range from plot2
        rng = self.Plot0.viewRange()

        # Set relevant ranges to plots 1 and 0
        self.Plot1.setYRange(rng[1][0], rng[1][1], padding=0)
        self.Plot2.setYRange(rng[0][0], rng[0][1], padding=0)

        # Reconnect other auto-range methods
        self.Plot1.sigRangeChanged.connect(self.updaterange_from_plot1)
        self.Plot2.sigRangeChanged.connect(self.updaterange_from_plot2)

    def update_align_shape(self):
        # Remove existing shape if it is on the plot
        try:
            self.Plot2.removeItem(self.align_aid)
        except:
            pass

        # Define origin
        ox = self.cursors.data['pos'][0][0]
        oy = self.cursors.data['pos'][0][1]

        # Define representative size for shape
        size = max([self.x_max - self.x_min, self.y_max - self.y_min]) / 2

        # Rotation angle in radians (define as negative for clockwise rotation)
        rot_rad = -np.radians(self.Rot_select.value())

        # Define the new shape
        # Square
        if self.align_tool_shape.currentText() == 'Square':
            # Define vertices of square
            x1 = ox + size * self.align_tool_size.value()/100
            x2 = ox - size * self.align_tool_size.value()/100
            y1 = oy + size * self.align_tool_size.value()/100
            y2 = oy - size * self.align_tool_size.value()/100

            # Make co-ordinate lists
            x_non_rot = np.asarray([x1,x1,x2,x2,x1])
            y_non_rot = np.asarray([y1,y2,y2,y1,y1])

        # Hexagon
        else:
            # Define vertices
            hex_r = size * self.align_tool_size.value() / 100
            x1 = ox + hex_r
            x2 = ox - hex_r
            y1 = oy + (hex_r/np.sqrt(3))
            y2 = oy + (2*hex_r/np.sqrt(3))
            y3 = oy - (hex_r/np.sqrt(3))
            y4 = oy - (2*hex_r/np.sqrt(3))

            # Make co-ordinate lists
            x_non_rot = np.asarray([x1, x1, ox, x2, x2, ox, x1])
            y_non_rot = np.asarray([y1, y3, y4, y3, y1, y2, y1])

        # Rotated hexagon
        if self.align_tool_shape.currentText() == 'Hex_r90':
            rot_rad += np.radians(90)

        if self.align_tool_shape.currentText() != 'None':
            if rot_rad == 0:  # No rotation required
                # Define shape as a path
                self.align_aid_path = pg.arrayToQPath(x_non_rot, y_non_rot)
            else:  # Rotation required
                x_rot = []
                y_rot = []

                # Do required rotation
                for i,j in zip(x_non_rot,y_non_rot):
                    x_rot.append(ox + np.cos(rot_rad) * (i - ox) - np.sin(rot_rad) * (j - oy))
                    y_rot.append(oy + np.sin(rot_rad) * (i - ox) + np.cos(rot_rad) * (j - oy))

                # Turn rotation lists into numpy arrays (needed for pg.arrayToQPath
                x_rot = np.asarray(x_rot)
                y_rot = np.asarray(y_rot)

                # Define shape as a path
                self.align_aid_path = pg.arrayToQPath(x_rot, y_rot)

            # Make a graphics object
            self.align_aid = pg.QtGui.QGraphicsPathItem(self.align_aid_path)
            # Set line style
            self.align_aid.setPen(pg.mkPen('r', width=1))
            # Add to plot
            self.Plot2.addItem(self.align_aid)


    # Estimate the normal emission from cross-correlation analysis of the map and a 180deg rotated version
    def auto_align(self):

        # Original data (in original array order, not flipped for display)
        orig_data = np.flip(self.sliceData[2],1)

        # Get a version of the slice rotated by 180 deg
        rotData = np.rot90(orig_data, 2)  # np.rot90 applied two times

        # Work out offset with subpixel precision
        self.shift, error, diffphase = phase_cross_correlation(orig_data, rotData, upsample_factor=100)
        self.x_mid = np.median(self.x)  # Centre of mapping angle scale
        self.y_mid = np.median(self.y)  # Centre of theta_par scale

        # Work out estimated normal emissions (note map was rotated around centre point of array, not angles=0 so need to take care of that offset too
        self.x_off = np.round(self.x_mid + ((self.shift[0] / 2) * self.x_delta), 3)
        self.y_off = np.round(self.y_mid + ((self.shift[1] / 2) * self.y_delta), 3)

        # ToDo Need to get actual angles out of this too

    def refresh_auto_align(self):

        # Determine current angular range from plot2
        rng = self.Plot2.viewRange()

        # Get indices of range from this
        x0 = self.find_nearest(self.x, rng[0][0])
        x1 = self.find_nearest(self.x, rng[0][1])
        y0 = self.find_nearest(self.y, rng[1][0])
        y1 = self.find_nearest(self.y, rng[1][1])

        # Arrays over reduced range
        x_red = self.x[x0:x1]
        y_red = self.y[y0:y1]
        slice_red = np.flip(self.sliceData[2],1)[x0:x1,y0:y1]  # Flipped back to original array order, not flipped version for display
        slice_red_rot = np.rot90(slice_red, 2)  # np.rot90 applied two times

        # Work out offset with subpixel precision
        self.shift, error, diffphase = phase_cross_correlation(slice_red, slice_red_rot, upsample_factor=100)
        x_red_mid = np.median(x_red)  # Centre of mapping angle scale reduced range
        y_red_mid = np.median(y_red)  # Centre of theta_par scale reduced range

        # Work out estimated normal emissions (note map was rotated around centre point of array, not angles=0 so need to take care of that offset too
        self.x_off = np.round(x_red_mid + ((self.shift[0] / 2) * self.x_delta), 3)
        self.y_off = np.round(y_red_mid + ((self.shift[1] / 2) * self.y_delta), 3)

        # ToDo Need to get actual angles out of this too

        # Set the position of the cursors on slice2
        self.cursors.setData(pos=[[self.x_off, self.y_off]], angle=-self.Rot_select.value(),**self.dicts[0])

    # Extract normal emission from determined offsets, including current manip values and with beamline specific sign changes
    def get_norm_values(self):
        # Extract normal emission based on analyser configuration
        # self.norm_x is peaks convention
        # UPDATE 13/2/22 - I think now peaks convention is defunct, this should now be same as manipulator normals
        # self.manip_norm_x is angle to set on the manipulator for normal emission
        if self.ana_type == 'I' or self.ana_type == 'Ip':  # Type I, with or without deflector
            # Norm tilt
            self.norm_tilt = self.tilt + self.defl_par - self.Y_cen.value()
            self.manip_norm_tilt = (self.BL_conv['tilt'] * self.tilt) + (self.BL_conv['defl_perp'] * self.defl_par) - (self.BL_conv['theta_par'] * self.Y_cen.value())

            # Norm polar
            if np.isnan(self.polar):
                self.norm_polar = self.X_cen.value() + self.defl_perp + self.ana_polar
                self.manip_norm_polar = self.X_cen.value() + (self.BL_conv['defl_perp'] * self.defl_perp) + (self.BL_conv['ana_polar'] * self.ana_polar)
            elif np.isnan(self.defl_perp):
                self.norm_polar = self.polar + self.X_cen.value() + self.ana_polar
                self.manip_norm_polar = (self.BL_conv['polar'] * self.polar) + (self.BL_conv['defl_perp'] * self.X_cen.value()) + (self.BL_conv['ana_polar'] * self.ana_polar)
            elif np.isnan(self.ana_polar):
                self.norm_polar = self.polar + self.defl_perp + self.X_cen.value()
                self.manip_norm_polar = (self.BL_conv['polar'] * self.polar) + (self.BL_conv['defl_perp'] * self.defl_perp) + (self.BL_conv['ana_polar'] * self.X_cen.value())

        else:  # Type II, with or without deflector
            # Norm polar
            self.norm_polar = self.polar + self.defl_par - self.Y_cen.value()
            self.manip_norm_polar = (self.BL_conv['polar'] * self.polar) + (self.BL_conv['defl_par'] * self.defl_par) - (self.BL_conv['theta_par'] * self.Y_cen.value())

            # Norm tilt
            if np.isnan(self.tilt):
                self.norm_tilt = self.X_cen.value() + self.defl_perp
                self.manip_norm_tilt = self.X_cen.value() + (self.BL_conv['defl_perp'] * self.defl_perp)
            elif np.isnan(self.defl_perp):
                self.norm_tilt = self.tilt + self.X_cen.value()
                self.manip_norm_tilt = (self.tilt * self.BL_conv['tilt']) + (self.BL_conv['defl_perp'] * self.X_cen.value())

        # Norm azi
        self.norm_azi = self.azi + self.Rot_select.value()
        self.manip_norm_azi = self.azi + (self.Rot_select.value()) * self.BL_conv['azi']


    def update_labels(self, lcn):
        if 0 in lcn:
            self.labels[0].setText(self.format_csr % (self.xdim, self.X_cen.value(), self.ydim, self.Y_cen.value(), self.Rot_select.value()))
        if 1 in lcn:
            self.labels[1].setText(self.format_csr2 % (self.z[self.En_select.value()], self.dE_select.value() / 2))
        if 2 in lcn:
            self.get_norm_values()  # Update normal emission
            # Update 13/2/22 - change these to match manipulator co-ordinates, I think that is now correct
            #self.labels[2].setText(self.format_csr3 % ('norm_polar', self.norm_polar, 'norm_tilt', self.norm_tilt, 'norm_azi', self.norm_azi))
            self.labels[2].setText(self.format_csr3 % ('norm_polar', self.manip_norm_polar, 'norm_tilt', self.manip_norm_tilt, 'norm_azi', self.manip_norm_azi))
        if 3 in lcn:
            self.labels[3].setText(self.format_csr4 % (self.BL_conv['polar_name'], self.polar, self.BL_conv['tilt_name'], self.tilt, self.BL_conv['azi_name'], self.azi))
        if 4 in lcn:
            self.labels[4].setText(self.format_csr5 % (self.BL_conv['polar_name'], self.manip_norm_polar, self.BL_conv['tilt_name'], self.manip_norm_tilt, self.BL_conv['azi_name'], self.manip_norm_azi))

    # Hide all cursor objects
    def hide_csr(self,key,evt):
        # Trigger if key is pressed
        if evt.key() == key and not evt.isAutoRepeat():

            # Hide cursors
            self.Plot2.removeItem(self.cursors)
            self.Plot1.removeItem(self.scsrvs1)
            self.Plot1.removeItem(self.scsrh1)
            self.Plot0.removeItem(self.scsrvs0)
            self.Plot0.removeItem(self.scsrh0)

            # Hide alignment aid
            try:
                self.Plot2.removeItem(self.align_aid)
            except:
                pass

            # # Disconnect the signals for cursor arrow movements
            # self.GUI_disconnect(0)

    # Show all cursor objects
    def show_csr(self, key, evt):
        # Trigger if key is pressed
        if evt.key() == key and not evt.isAutoRepeat():

            # Hide cursors
            self.Plot2.addItem(self.cursors)
            self.Plot1.addItem(self.scsrvs1)
            self.Plot1.addItem(self.scsrh1)
            self.Plot0.addItem(self.scsrvs0)
            self.Plot0.addItem(self.scsrh0)

            # Show alignment aid
            try:
                self.Plot2.addItem(self.align_aid)
            except:
                pass

            # # Connect the signals for cursor arrow movements
            # self.GUI_connect(0)

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

    # Set the levels in the image according to the positions of HCutOff and LCutOff
    def contrast(self):
        self.Image0.setLevels([(self.LCutOff.value() / 100) * (self.I_max - self.I_min) + self.I_min,
                              (self.HCutOff.value() / 100) * (self.I_max - self.I_min) + self.I_min])
        self.Image1.setLevels([(self.LCutOff.value() / 100) * (self.I_max - self.I_min) + self.I_min,
                               (self.HCutOff.value() / 100) * (self.I_max - self.I_min) + self.I_min])
        self.Image2.setLevels([(self.LCutOff.value() / 100) * (self.I_max - self.I_min) + self.I_min,
                               (self.HCutOff.value() / 100) * (self.I_max - self.I_min) + self.I_min])

    # Invert contrast sliders
    def invert(self):

        if self.Invert.isChecked():
            self.HCutOff.setSliderPosition(0)
            self.LCutOff.setSliderPosition(100)

        else:
            self.HCutOff.setSliderPosition(100)
            self.LCutOff.setSliderPosition(0)

    # Select colormap from a menu
    def cmap_select(self):
        # Retrieve selected colormap
        self.colormap = self.get_mpl_colormap(self.Cmaps.currentText())

        # Apply selected colormap
        self.Image0.setLookupTable(self.colormap)
        self.Image1.setLookupTable(self.colormap)
        self.Image2.setLookupTable(self.colormap)

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

    # Close the app
    def close_app(self):
        self.AlignFS_panel.close()

    # Set normal emission and close the app
    def set_and_close_app(self):
        try:
            # Write the relevant normal emission settings into a new cell in the Jupyter notebook
            text = str(self.wname[0]) + '.attrs[\'norm_polar\']=' + str(self.norm_polar) + '\n' + str(self.wname[0]) + '.attrs[\'norm_tilt\']=' + str(self.norm_tilt) + '\n' + str(self.wname[0]) + '.attrs[\'norm_azi\']=' + str(self.norm_azi)
            cell_above(text)
        except:
            # Copy it to the clipboard
            pyperclip.copy('.attrs[\'norm_polar\']=' + str(self.norm_polar) + '\n' + '.attrs[\'norm_tilt\']=' + str(self.norm_tilt) + '\n' + '.attrs[\'norm_azi\']=' + str(self.norm_azi))

            print('Normal emission data copied to the clipboard')

        # Close the panel
        self.AlignFS_panel.close()

    ################################################################################


# # Runner
# if __name__ == "__main__":
#     import sys
#
#     app = QtCore.QCoreApplication.instance()
#     if app is None:
#         app = QtWidgets.QApplication(sys.argv)
#
#     ui = Ui_AlignFS_panel(QtWidgets.QMainWindow())
#     ui.AlignFS_panel.show()
#     sys.exit(app.exec_())

