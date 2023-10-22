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

# Class to create a dispersion alignment panel object
class Ui_Aligndisp_panel(object):

    # Run when AlignFS Panel object is created
    def __init__(self, Aligndisp_panel):

        # Store the framework of the app
        self.Aligndisp_panel = Aligndisp_panel

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
        self.Aligndisp_panel.setObjectName("Aligndisp_panel")
        self.Aligndisp_panel.setEnabled(True)
        self.Aligndisp_panel.resize(746, 523)

        # Define central widget within main window
        self.centralwidget = QtWidgets.QWidget(self.Aligndisp_panel)
        self.centralwidget.setMouseTracking(False)
        self.centralwidget.setAutoFillBackground(False)
        self.centralwidget.setObjectName("centralwidget")

        # Set central widget background color
        self.centralwidget.setAutoFillBackground(False)
        p = self.centralwidget.palette()
        p.setColor(self.centralwidget.backgroundRole(), QtCore.Qt.GlobalColor.white) # Input color here
        self.centralwidget.setPalette(p)

        # Define Plot frames
        # Dispersion
        self.MainPlot = PlotWidgetKP(self.centralwidget)
        self.MainPlot.setGeometry(QtCore.QRect(30, 10, 401, 291))
        self.MainPlot.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))
        self.MainPlot.setObjectName("MainPlot")

        # MDC
        self.MDCPlot = PlotWidgetKP(self.centralwidget)
        self.MDCPlot.setGeometry(QtCore.QRect(30, 300, 401, 171))
        self.MDCPlot.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))
        self.MDCPlot.setObjectName("MDCPlot")

        # Define CursorStats frame
        # CursorStats must not be in contact with Plots:
        # (to avoid random artefacts in the cursor stats labels when moving the cursors)
        self.CursorStats = pg.GraphicsLayoutWidget(self.centralwidget)  # The associated graphics layout is 'self.CursorStats.ci'
        self.CursorStats.setGeometry(QtCore.QRect(440, 140, 301, 111))
        self.CursorStats.setObjectName("CursorStats")
        self.CursorStats.ci.setSpacing(0)  # Set to zero the spacing between the cells in the GraphicsLayout
        self.CursorStats.ci.setBorder(color='k')  # Color the borders between the cells in the GraphicsLayout (useful to see the cell limits)

        # Define High cutoff slider
        self.HCutOff = QtWidgets.QSlider(self.centralwidget)
        self.HCutOff.setGeometry(QtCore.QRect(440, 360, 251, 22))
        self.HCutOff.setMaximum(100)
        self.HCutOff.setProperty("value", 90)
        self.HCutOff.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.HCutOff.setInvertedAppearance(False)
        self.HCutOff.setInvertedControls(False)
        self.HCutOff.setObjectName("HCutOff")

        # Define Low cutoff slider
        self.LCutOff = QtWidgets.QSlider(self.centralwidget)
        self.LCutOff.setGeometry(QtCore.QRect(440, 390, 251, 22))
        self.LCutOff.setMaximum(100)
        self.LCutOff.setProperty("value", 0)
        self.LCutOff.setSliderPosition(0)
        self.LCutOff.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.LCutOff.setInvertedAppearance(False)
        self.LCutOff.setInvertedControls(False)
        self.LCutOff.setObjectName("LCutOff")

        # Define Invert contrast checkbox
        self.Invert = QtWidgets.QCheckBox(self.centralwidget)
        self.Invert.setGeometry(QtCore.QRect(700, 380, 21, 21))
        self.Invert.setObjectName("Invert")

        # Define Colormap dropdown menu
        self.Cmaps = QtWidgets.QComboBox(self.centralwidget)
        self.Cmaps.setGeometry(QtCore.QRect(440, 320, 151, 26))
        self.Cmaps.setEditable(False)
        self.Cmaps.setCurrentText("Greys")
        self.Cmaps.setObjectName("Cmaps")
        self.Cmaps.addItems(["Greys", "Blues", "viridis", "plasma", "inferno", "magma", "cividis", "bone", "spring", "summer", "autumn", "winter", "cool", "jet", "ocean", "RdBu", "BrBG", "PuOr", "coolwarm", "bwr", "seismic"])  # Choose your favourites

        # Set labels
        self.Label_dE = QtWidgets.QLabel(self.centralwidget)
        self.Label_dE.setGeometry(QtCore.QRect(460, 20, 60, 16))
        self.Label_dE.setObjectName("Label_dE")
        self.Cen_label = QtWidgets.QLabel(self.centralwidget)
        self.Cen_label.setGeometry(QtCore.QRect(440, 70, 171, 16))
        self.Cen_label.setObjectName("Cen_label")
        self.Label_X_cen = QtWidgets.QLabel(self.centralwidget)
        self.Label_X_cen.setGeometry(QtCore.QRect(439, 100, 71, 20))
        self.Label_X_cen.setObjectName("Label_X_cen")

        # Set energy integration selection
        self.dE_select = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.dE_select.setGeometry(QtCore.QRect(560, 20, 91, 24))
        self.dE_select.setLocale(QtCore.QLocale(QtCore.QLocale.Language.English, QtCore.QLocale.Country.UnitedStates))
        self.dE_select.setObjectName("dE_select")
        self.dE_select.setDecimals(3)
        self.dE_select.setSingleStep(0.001)

        # Define x-center angle
        self.X_cen = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.X_cen.setGeometry(QtCore.QRect(520, 100, 91, 24))
        self.X_cen.setLocale(QtCore.QLocale(QtCore.QLocale.Language.English, QtCore.QLocale.Country.UnitedStates))
        self.X_cen.setObjectName("X_cen")
        self.X_cen.setSingleStep(0.1)

        # Define x-center step size
        self.Delta_X_cen = QtWidgets.QLineEdit(self.centralwidget)
        self.Delta_X_cen.setGeometry(QtCore.QRect(620, 100, 61, 24))
        self.Delta_X_cen.setObjectName("Delta_X_cen")

        # Define autocentering button
        self.Auto_cen = QtWidgets.QPushButton(self.centralwidget)
        self.Auto_cen.setGeometry(QtCore.QRect(550, 60, 113, 32))
        self.Auto_cen.setObjectName("Auto_cen")

        # Define set normal attributes button
        self.set_norm = QtWidgets.QPushButton(self.centralwidget)
        self.set_norm.setGeometry(QtCore.QRect(630, 440, 113, 32))
        self.set_norm.setObjectName("Set and Exit")

        # Define exit button
        self.exit = QtWidgets.QPushButton(self.centralwidget)
        self.exit.setGeometry(QtCore.QRect(520, 440, 113, 32))
        self.exit.setObjectName("Close")

        # Define other GUI elements
        self.Aligndisp_panel.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(self.Aligndisp_panel)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 746, 22))
        self.menubar.setObjectName("menubar")
        self.Aligndisp_panel.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self.Aligndisp_panel)
        self.statusbar.setObjectName("statusbar")
        self.Aligndisp_panel.setStatusBar(self.statusbar)

        # Initialize widget's labels
        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self.Aligndisp_panel)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.Aligndisp_panel.setWindowTitle(_translate("Aligndisp_panel", "Align dispersion"))
        self.Cen_label.setText(_translate("Aligndisp_panel", "Centering:"))
        self.Label_X_cen.setText(_translate("Aligndisp_panel", "theta_par"))
        self.Delta_X_cen.setText(_translate("Aligndisp_panel", "0.1"))
        self.Auto_cen.setText(_translate("Aligndisp_panel", "Auto"))
        self.Label_dE.setText(_translate("Aligndisp_panel", "dE"))
        self.set_norm.setText(_translate("Aligndisp_panel", "Set and Exit"))
        self.exit.setText(_translate("Aligndisp_panel", "Close"))

    ############################################################################
    # Resize GUI elements
    # (See pyqt_ResizeItem.py)
    def GUI_resize(self):
        ResizeItem(self.Aligndisp_panel, self.MainPlot)
        ResizeItem(self.Aligndisp_panel, self.MDCPlot)
        ResizeItem(self.Aligndisp_panel, self.CursorStats)
        ResizeItem(self.Aligndisp_panel, self.HCutOff)
        ResizeItem(self.Aligndisp_panel, self.LCutOff)
        ResizeItem(self.Aligndisp_panel, self.Invert)
        ResizeItem(self.Aligndisp_panel, self.Cmaps)
        ResizeItem(self.Aligndisp_panel, self.Label_dE)
        ResizeItem(self.Aligndisp_panel, self.dE_select)
        ResizeItem(self.Aligndisp_panel, self.Cen_label)
        ResizeItem(self.Aligndisp_panel, self.Label_X_cen)
        ResizeItem(self.Aligndisp_panel, self.X_cen)
        ResizeItem(self.Aligndisp_panel, self.Delta_X_cen)
        ResizeItem(self.Aligndisp_panel, self.Auto_cen)
        ResizeItem(self.Aligndisp_panel, self.set_norm)
        ResizeItem(self.Aligndisp_panel, self.exit)


    ############################################################################

    def GUI_internal(self):
        # Zoom object definition at plots
        # (See pyqt_ZoomItem.py)
        self.ZoomMain = ZoomItem(self.MainPlot)
        self.ZoomMDC = ZoomItem(self.MDCPlot)

        # Create image objects
        self.Image = pg.ImageItem()

        # Set the initial colormap
        self.Image.setLookupTable(self.colormap)

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
            callers_local_vars = inspect.currentframe().f_back.f_back.f_back.f_back.f_back.f_locals.items()
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
    def GUI_setdata(self, x, y, Int, xdim, ydim, Idim, xuts, yuts, Iuts, EF, angles, BL_conv):
        # Convert the input data to attributes of the Panel
        self.x = x
        self.y = y
        self.Int = Int

        self.xdim = xdim
        self.ydim = ydim
        self.Idim = Idim

        self.xuts = xuts
        self.yuts = yuts
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
            self.ana_polar = np.Nan

        self.ana_type = angles['ana_type']

        self.BL_conv = BL_conv


    ############################################################################

    # Construct Panel GUI initial content (dependant on loaded data)
    def GUI_initial(self):

        ### Set the axis labels

        # Dispersion
        self.MainPlot.setLabel('left', text=self.ydim, units=self.yuts, unitPrefix=None)
        self.MainPlot.setLabel('bottom', text=self.xdim, units=self.xuts, unitPrefix=None)

        # MDC
        self.MDCPlot.setLabel('left', text='Intensity', units=None, unitPrefix=None)
        self.MDCPlot.setLabel('bottom', text=self.xdim, units=self.xuts, unitPrefix=None)

        ########################################################################

        ### MainPlot image initialization:

        # Add image objects
        self.MainPlot.addItem(self.Image)

        # Create limits rectangle for the data in MainPlot
        self.ImageRectangle = QtCore.QRectF(self.x_min, self.y_max, self.x_max - self.x_min, self.y_min - self.y_max)

        self.Image.setImage(np.flip(self.Int,1))

        # Run an auto-alignment for this slice
        self.auto_align()

        # Set the contrast
        self.contrast()

        # Scale 2D data
        self.Image.setRect(self.ImageRectangle)


        ########################################################################

        # Set corresponding widgets

        # dE
        self.dE_select.setValue(0.01)

        # Centering
        self.X_cen.setRange(self.x_min,self.x_max)
        self.X_cen.setValue(self.x_off)
        self.X_cen.setSingleStep(self.x_delta)
        self.Delta_X_cen.setText(str(np.round(self.x_delta,3)))

        ########################################################################

        ### Limit zoom and panning of MainPlot and CutPlots:

        # Define zoom/panning limits
        self.xmin_lim = self.x_min - (self.x_max - self.x_min) * self.pd
        self.xmax_lim = self.x_max + (self.x_max - self.x_min) * self.pd
        self.ymin_lim = self.y_min - (self.y_max - self.y_min) * self.pd
        self.ymax_lim = self.y_max + (self.y_max - self.y_min) * self.pd

        # Set zoom/panning limits of plots
        self.MainPlot.getViewBox().setLimits(xMin=self.xmin_lim, xMax=self.xmax_lim, yMin=self.ymin_lim, yMax=self.ymax_lim)

        # MDCPlot intensity minimum and maximum are set to Imin and Imax of data
        self.MDCPlot.getViewBox().setLimits(xMin=self.xmin_lim, xMax=self.xmax_lim, yMin=self.I_min, yMax=self.I_max)

        ########################################################################

        ### Set the initial range of plots:

        # Set the initial range of plots
        self.MainPlot.getViewBox().setXRange(self.xmin_lim, self.xmax_lim, padding=0)
        self.MainPlot.getViewBox().setYRange(self.ymin_lim, self.ymax_lim, padding=0)

        # Set the initial range of MDCPlot equal to MainPlot
        self.MDCPlot.getViewBox().setXRange(self.xmin_lim, self.xmax_lim, padding=0)

        # Set the initial range of MDCPlot intensity equal to the I limits of the data in MainPlot
        self.MDCPlot.setYRange(self.I_min, self.I_max, padding=0)

        ########################################################################

        ### Set the initial autorange policy:

        # Disable auto range for plots
        self.MainPlot.disableAutoRange()

        # Enable auto range for MDCPlot
        self.MDCPlot.enableAutoRange()

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
        self.spaceMain = [[self.x_min, self.y_min], [self.x_max, self.y_max]]
        self.spaceMDC = [[self.x_min, self.I_min], [self.x_max, self.I_max]]

        # Set the space for the zoom objects
        self.ZoomMain.setSpace(self.spaceMain)
        self.ZoomMDC.setSpace(self.spaceMDC)

        # Set the space for the cursors
        self.cursors.setSpace(self.spaceMain)

        # Add the leader cursors in MainPlot
        self.MainPlot.addItem(self.cursors)

        # Set the initial cursors configuration
        self.ispany =  self.dE_select.value() / 2
        self.dicts[0]['ispany'] = self.ispany
        self.cursors.setData(pos=np.array([[self.x_off, self.EF]]), **self.dicts[0])

        # Create infinite vertical lines in MDCPlot (vertical side cursors)
        self.scsrvs = pg.InfiniteLine(pos=self.cursors.data['pos'][0][0], angle=90, pen=self.csrcs[0])

        # Add the leader vertical side cursors in MainPlot
        self.MDCPlot.addItem(self.scsrvs)

        # Add the cursor stats labels in CursorStats
        self.CursorStats.addItem(self.labels[0], row=0, col=0)  # Row 1 - offset info
        self.CursorStats.addItem(self.labels[1], row=1, col=0)  # Row 2 - peaks info
        self.CursorStats.addItem(self.labels[2], row=2, col=0)  # Row 3 - current manipulator info
        self.CursorStats.addItem(self.labels[3], row=3, col=0)  # Row 4 - new manipulator info

        ### Define the cursor stats format

        # String particles (dependant on user preferences)
        str1 = "<span style='font-size:" + self.fontsz + "pt'> Theta_par offset: "
        str2 = "%0." + self.digits + "f"
        str3 = "peaks norm emission: "
        str4 = "<span style='font-size:" + self.fontsz + "pt'> Current manip angle: "
        str5 = "Normal emission: "


        # Full strings
        self.format_csr = str1 + str2
        self.format_csr2 = str3 + "%s =" + str2
        self.format_csr3 = str4 + "%s = " + str2
        self.format_csr4 = str5 + "%s = " + str2


        # Extract the normal emissions
        self.get_norm_values()

        # Set the initial cursor stats labels info (cursors)
        # Set label text
        self.update_labels([0,1,2,3])

        # Set the initial MDC plots
        # Place holders
        self.MDCs = []
        self.MDCs.append(self.MDCPlot.plot(self.x,np.zeros(len(self.x)), pen=self.sidecs[0]))
        self.MDCs.append(self.MDCPlot.plot(self.x,np.zeros(len(self.x)), pen=self.sidecs[1]))

        # Update the proper MDC plots
        self.update_MDCs()

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
        self.csrs_mov = [None] * 2
        self.csrs_mov[0] = partial(self.arrowmove, 0)  # Signal object 26
        self.csrs_mov[1] = partial(self.arrowmove, 1)  # Signal object 27



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
        # Signals when the dE_select value is changed
        i = 6
        if not cn or i in cn:
            if self.cns[i] == False:
                self.dE_select.valueChanged.connect(self.update_dE)
                self.cns[i] = True

        # ----------------------------------------------------------------------#
        # Signals when the Plot range is changed
        i = 7
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainPlot.sigRangeChanged.connect(self.updaterange_from_MainPlot)
                self.cns[i] = True

        # Signals when the Plot1 range is changed
        i = 8
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MDCPlot.sigRangeChanged.connect(self.updaterange_from_MDCPlot)
                self.cns[i] = True

        # ----------------------------------------------------------------------#

        # Signals when the theta_par center value is changed
        i = 14
        if not cn or i in cn:
            if self.cns[i] == False:
                self.X_cen.valueChanged.connect(self.update_cursor_pos)
                self.cns[i] = True

        # Signals when enter is pressed over self.Delta_X_cen or when self.Delta_X_cen loses focus
        i = 15
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Delta_X_cen.editingFinished.connect(self.X_cen_step_update)
                self.cns[i] = True

        # ----------------------------------------------------------------------#

        # Update when cursor is dragged
        i = 16
        if not cn or i in cn:
            if self.cns[i] == False:
                self.cursors.scatter.sigPlotChanged.connect(self.update_from_cursor)
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
                self.MainPlot.sigKeyRelease.connect(self.show_csr_sig)
                self.cns[i] = True

        i = 21
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MDCPlot.sigKeyRelease.connect(self.show_csr_sig)
                self.cns[i] = True



        # Signals to hide all cursor objects
        i = 23
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainPlot.sigKeyPress.connect(self.hide_csr_sig)
                self.cns[i] = True

        i = 24
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MDCPlot.sigKeyPress.connect(self.hide_csr_sig)
                self.cns[i] = True


        # ----------------------------------------------------------------------#

        # Individual cursor arrow movements
        i = 26
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainPlot.sigKeyPress.connect(self.csrs_mov[0])
                self.cns[i] = True

        i = 27
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MDCPlot.sigKeyPress.connect(self.csrs_mov[1])
                self.cns[i] = True


        # # ----------------------------------------------------------------------#

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
          # Signals when the dE_select value is changed
        i = 6
        if not cn or i in cn:
            try:
                self.dE_select.valueChanged.disconnect(self.update_dE)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # ----------------------------------------------------------------------#
        # Signals when the MainPlot range is changed
        i = 7
        if not cn or i in cn:
            try:
                self.Plot2.sigRangeChanged.disconnect(self.updaterange_from_MainPlot)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # Signals when the MDCPlot range is changed
        i = 8
        if not cn or i in cn:
            try:
                self.Plot1.sigRangeChanged.disconnect(self.updaterange_from_MDCPlot)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # ----------------------------------------------------------------------

        # Signals when the theta_par center value is changed
        i = 14
        if not cn or i in cn:
            try:
                self.X_cen.valueChanged.disconnect(self.update_cursor_pos)
            except:
                print(i)
            finally:
                self.cns[i] = False

        # Signals when enter is pressed over self.Delta_X_cen or when self.Delta_X_cen loses focus
        i = 15
        if not cn or i in cn:
            try:
                self.Delta_X_cen.editingFinished.disconnect(self.X_cen_step_update)
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
                self.MainPlot.sigKeyRelease.disconnect(self.show_csr_sig)
            except:
                print(i)
            finally:
                self.cns[i] = False

        i = 21
        if not cn or i in cn:
            try:
                self.MDCPlot.sigKeyRelease.disconnect(self.show_csr_sig)
            except:
                print(i)
            finally:
                self.cns[i] = False



        # Signals to hide all cursor objects
        i = 23
        if not cn or i in cn:
            try:
                self.MainPlot.sigKeyPress.disconnect(self.hide_csr_sig)
            except:
                print(i)
            finally:
                self.cns[i] = False

        i = 24
        if not cn or i in cn:
            try:
                self.MDCPlot.sigKeyPress.disconnect(self.hide_csr_sig)
            except:
                print(i)
            finally:
                self.cns[i] = False


        # ----------------------------------------------------------------------#

        # Individual cursor arrow movement
        i = 26
        if not cn or i in cn:
            try:
                self.MainPlot.sigKeyPress.disconnect(self.csrs_mov[0])
            except:
                print(i)
            finally:
                self.cns[i] = False

        i = 27
        if not cn or i in cn:
            try:
                self.MDCPlot.sigKeyPress.disconnect(self.csrs_mov[1])
            except:
                print(i)
            finally:
                self.cns[i] = False

        #
        # # ----------------------------------------------------------------------#
        #
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

    # Update the angular slices
    def update_MDCs(self):

        self.MDCPlot.removeItem(self.MDCs[0])
        self.MDCPlot.removeItem(self.MDCs[1])

        # Get new energy limits
        Emin = self.cursors.data['pos'][0][1] - self.dE_select.value() / 2
        Emin_i = self.find_nearest(self.y, Emin)
        Emax = self.cursors.data['pos'][0][1] + self.dE_select.value() / 2
        Emax_i = self.find_nearest(self.y, Emax)

        # Select corresponding slices

        if Emax_i > Emin_i:  # Need integration
            MDC = np.sum(self.Int[:,Emin_i:Emax_i+1],axis=1)/(Emax_i-Emin_i+1)
        else:
            MDC = self.Int[:, Emin_i]

        # Make a new x-array reversed around the centre point for checking how good the centre line is
        flipped_x = np.flip(self.x) + 2*(self.X_cen.value() - np.median(self.x))

        # Update the plots
        self.MDCs[0] = self.MDCPlot.plot(self.x, MDC, pen=self.sidecs[0])
        self.MDCs[1] = self.MDCPlot.plot(flipped_x, MDC, pen=self.sidecs[1])


    # Update energy integration
    def update_dE(self):
        # Update cursors
        self.ispany = self.dE_select.value() / 2
        self.dicts[0]['ispany'] = self.ispany

        # Set the position of the cursors on MainPlot
        self.cursors.ispany = self.dicts[0]['ispany']

        # Refresh the cursor
        self.cursors.setICrosshair()

        # Refresh the integration-crosshair position
        self.cursors.updateGraph()

        # Update the MDCs
        self.update_MDCs()



    # Update cursor position from numerical boxes
    def update_cursor_pos(self):
        # Disconnect other cursor updates
        self.cursors.scatter.sigPlotChanged.disconnect(self.update_from_cursor)

        # Set the position of the cursors on MainPlot
        self.cursors.setData(pos=[[self.X_cen.value(), self.cursors.data['pos'][0][1]]], **self.dicts[0])

        # Set the positions of the normal emission lines on MDCPlot
        self.scsrvs.setPos(pos=self.cursors.data['pos'][0][0])

        # Update the slices
        self.update_MDCs()

        # Refresh the normal emission values
        self.get_norm_values()

        # Update the labels
        self.update_labels([0,1,3])

        # Reconnect other cursor updates
        self.cursors.scatter.sigPlotChanged.connect(self.update_from_cursor)


    # Update from cursor drag
    def update_from_cursor(self):
        # Disconnect other cursor updates
        self.X_cen.valueChanged.disconnect(self.update_cursor_pos)

        # From drag of cursor on MainPlot, set the positions of the normal emission lines on MDCPlot
        self.scsrvs.setPos(pos=self.cursors.data['pos'][0][0])

        # Update the dialog boxes
        self.X_cen.setValue(self.cursors.data['pos'][0][0])

        # Update the slices
        self.update_MDCs()

        # Refresh the normal emission values
        self.get_norm_values()

        # Update the labels
        self.update_labels([0, 1, 3])

        # Reconnect other cursor updates
        self.X_cen.valueChanged.connect(self.update_cursor_pos)


    # Cursor arrow movements
    def arrowmove(self, plt_num, evt):


        # Current positions
        x_pos = self.cursors.data['pos'][0][0]
        y_pos = self.cursors.data['pos'][0][1]

        # Move cursor data-row up
        if evt.key() == QtCore.Qt.Key.Key_Up and y_pos < self.y_max:
            # Get new position
            y_pos_new = self.y[self.find_nearest(self.y, y_pos) + 1]
            # Set the position of the cursors on slice2
            self.cursors.setData(pos=[[x_pos, y_pos_new]], **self.dicts[0])

        # Move cursor data-row down
        if evt.key() == QtCore.Qt.Key.Key_Down and y_pos > self.y_min:
            # Get new position
            y_pos_new = self.y[self.find_nearest(self.y, y_pos) - 1]
            # Set the position of the cursors on slice2
            self.cursors.setData(pos=[[x_pos, y_pos_new]], **self.dicts[0])

        # Move cursor data-row left
        if evt.key() == QtCore.Qt.Key.Key_Left and x_pos > self.x_min:
            # Get new position
            x_pos_new = self.x[self.find_nearest(self.x, x_pos) - 1]
            # Set the position of the cursors on slice2
            self.cursors.setData(pos=[[x_pos_new, y_pos]], **self.dicts[0])

        # Move cursor data-row right
        if evt.key() == QtCore.Qt.Key.Key_Right and x_pos < self.x_max:
            # Get new position
            x_pos_new = self.x[self.find_nearest(self.x, x_pos) + 1]
            # Set the position of the cursors on slice2
            self.cursors.setData(pos=[[x_pos_new, y_pos]], **self.dicts[0])


    # Update step size for rotation angle selector box
    def X_cen_step_update(self):
        # set the step size for self.X_cen
        self.X_cen.setSingleStep(float(self.Delta_X_cen.text()))



    # Link range: MainPlot -> MDCPlot
    def updaterange_from_MainPlot(self):
        # Disconnect other auto-range updates
        self.MDCPlot.sigRangeChanged.disconnect(self.updaterange_from_MDCPlot)

        # Determine angular range from plot2
        rng = self.MainPlot.viewRange()

        # Set relevant range for MDC plot
        self.MDCPlot.setXRange(rng[0][0],rng[0][1], padding=0)

        # Reconnect other auto-range methods
        self.MDCPlot.sigRangeChanged.connect(self.updaterange_from_MDCPlot)

    # Link range: MDCPlot -> MainPlot
    def updaterange_from_MDCPlot(self):
        # Disconnect other auto-range updates
        self.MainPlot.sigRangeChanged.disconnect(self.updaterange_from_MainPlot)

        # Determine angular range from plot2
        rng = self.MDCPlot.viewRange()

        # Set relevant ranges to plots 1 and 0
        self.MainPlot.setXRange(rng[0][0], rng[0][1], padding=0)

        # Reconnect other auto-range methods
        self.MainPlot.sigRangeChanged.connect(self.updaterange_from_MainPlot)



    # Estimate the normal emission from cross-correlation analysis of the map and a flipped version
    def auto_align(self):

        # Original data (in original array order, not flipped for display)
        orig_data = self.Int

        # Get a version of the dispersion flipped LR
        flipData = np.flipud(orig_data)

        # Work out offset with subpixel precision
        self.shift, error, diffphase = phase_cross_correlation(orig_data, flipData, upsample_factor=100)
        self.x_mid = np.median(self.x)  # Centre of theta_par scale

        # Work out estimated normal emissions (note map was flipped around centre point of array, not angle=0 so need to take care of that offset too
        self.x_off = np.round(self.x_mid + ((self.shift[0] / 2) * self.x_delta), 3)


    def refresh_auto_align(self):

        # Determine current angular range from plot2
        rng = self.MainPlot.viewRange()

        # Get indices of range from this
        x0 = self.find_nearest(self.x, rng[0][0])
        x1 = self.find_nearest(self.x, rng[0][1])
        y0 = self.find_nearest(self.y, rng[1][0])
        y1 = self.find_nearest(self.y, rng[1][1])

        # Arrays over reduced range
        x_red = self.x[x0:x1]
        y_red = self.y[y0:y1]
        slice_red = self.Int[x0:x1,y0:y1]  # Subselect from main dispersion
        # Get a version of the dispersion flipped LR
        flipData = np.flipud(slice_red)

        # Work out offset with subpixel precision
        self.shift, error, diffphase = phase_cross_correlation(slice_red, flipData, upsample_factor=100)
        self.x_mid = np.median(x_red)  # Centre of reduced theta_par scale

        # Work out estimated normal emissions (note map was flipped around centre point of array, not angle=0 so need to take care of that offset too
        self.x_off = np.round(self.x_mid + ((self.shift[0] / 2) * self.x_delta), 3)

        # Set the position of the cursors on MainPlot (other updates follow from this)
        self.cursors.setData(pos=[[self.x_off, self.cursors.data['pos'][0][1]]], **self.dicts[0])

    # Extract normal emission from determined offsets, including current manip values and with beamline specific sign changes
    def get_norm_values(self):
        # Extract normal emission based on analyser configuration
        # self.norm_x is peaks convention
        # self.manip_norm_x is angle to set on the manipulator for normal emission
        if self.ana_type == 'I' or self.ana_type == 'Ip':  # Type I, with or without deflector
            # Norm tilt
            self.norm_tilt = (self.BL_conv['tilt'] * self.tilt) + (self.BL_conv['defl_par'] * self.defl_par) - (self.BL_conv['theta_par'] * self.X_cen.value())
            self.manip_norm_tilt = (self.BL_conv['tilt'] * self.tilt) + (self.BL_conv['defl_perp'] * self.defl_par) - (self.BL_conv['theta_par'] * self.X_cen.value())

        else:  # Type II, with or without deflector
            # Norm polar
            self.norm_polar = (self.BL_conv['polar'] * self.polar) + (self.BL_conv['defl_par'] * self.defl_par) - (self.BL_conv['theta_par'] * self.X_cen.value())
            self.manip_norm_polar = (self.BL_conv['polar'] * self.polar) + (self.BL_conv['defl_par'] * self.defl_par) - (self.BL_conv['theta_par'] * self.X_cen.value())


    def update_labels(self, lcn):
        if 0 in lcn:
            self.labels[0].setText(self.format_csr % (self.X_cen.value()))
        if self.ana_type == 'I' or self.ana_type == 'Ip':  # Type I analyser
            if 1 in lcn:
                self.labels[1].setText(self.format_csr2 % ('norm_tilt', self.norm_tilt))
            if 2 in lcn:
                self.labels[2].setText(self.format_csr3 % (self.BL_conv['tilt_name'], self.tilt))
            if 3 in lcn:
                self.labels[3].setText(self.format_csr4 % (self.BL_conv['tilt_name'], self.manip_norm_tilt))
        else:  # Type II analyser
            if 1 in lcn:
                self.labels[1].setText(self.format_csr2 % ('norm_polar', self.norm_polar))
            if 2 in lcn:
                self.labels[2].setText(self.format_csr3 % (self.BL_conv['polar_name'], self.polar))
            if 3 in lcn:
                self.labels[3].setText(self.format_csr4 % (self.BL_conv['polar_name'], self.manip_norm_polar))

    # Hide all cursor objects
    def hide_csr(self,key,evt):
        # Trigger if key is pressed
        if evt.key() == key and not evt.isAutoRepeat():

            # Hide cursors
            self.MainPlot.removeItem(self.cursors)
            self.MDCPlot.removeItem(self.scsrvs)

            # # Disconnect the signals for cursor arrow movements
            # self.GUI_disconnect(0)

    # Show all cursor objects
    def show_csr(self, key, evt):
        # Trigger if key is pressed
        if evt.key() == key and not evt.isAutoRepeat():

            # Hide cursors
            self.MainPlot.addItem(self.cursors)
            self.MDCPlot.addItem(self.scsrvs)

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
        self.Image.setLevels([(self.LCutOff.value() / 100) * (self.I_max - self.I_min) + self.I_min,
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
        self.Image.setLookupTable(self.colormap)

    # Retrieve colormap from matplotlib
    def get_mpl_colormap(self, cmap):
        # Get the colormap from matplotlib
        colormap = cm.get_cmap(cmap)
        colormap._init()

        # Convert the matplotlib colormap from 0-1 to 0-255 for PyQt6
        # Ignore the last 3 rows of the colormap._lut:
        # colormap._lut has shape (N+3,4) where the first N rows are the RGBA color representations
        # The last 3 rows deal with the treatment of out-of-range and masked values
        # They need to be ignored to avoid the generation of artifacts possibly related to floating-point errors
        lut = (colormap._lut * 255)[1:-3][:]

        return lut

    # Close the app
    def close_app(self):
        self.Aligndisp_panel.close()

    # Set normal emission and close the app
    def set_and_close_app(self):
        # Determine correct string to write
        if self.ana_type == 'I' or self.ana_type == 'Ip':  # Type I analyser
            text = '.attrs[\'norm_tilt\']=' + str(self.norm_tilt)
        else:  # Type II analyser
            text = '.attrs[\'norm_polar\']=' + str(self.norm_polar)

        try:
            # Write the relevant normal emission settings into a new cell in the Jupyter notebook
            cell_above(str(self.wname[0]) + text)
        except:
            # Copy it to the clipboard
            pyperclip.copy(text)

            print('Normal emission copied to the clipboard. Set \"a.<<INSERT COPIED TEXT>>\" to add to attributes of DataArray `a`')

        # Close the panel
        self.Aligndisp_panel.close()


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

