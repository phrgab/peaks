# Display Panel
# Edgar Abarca Morales
# Minor edits EAM 25/10/2021

# Display panel import requirements (non-authorship)
from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import numpy as np
from matplotlib import cm
from functools import partial

# Display panel import requirements (autorship)
from peaks.core.GUI.pyqt_CursorItem import CursorItem
from peaks.core.GUI.pyqt_ZoomItem import ZoomItem
from peaks.core.GUI.pyqt_PlotWidgetKP import PlotWidgetKP
from peaks.core.GUI.pyqt_ResizeItem import ResizeItem
from peaks.core.GUI.pyqt_CircularListWidget import CircularListWidget

# File loading import requirements (autorship)
from .feedDPfromLoaders import feedDPfromLoaders

################################################################################

# Class to create a Display Panel object
class GUI_DisplayPanel(object):

    #--------------------------------------------------------------------------#

    # This class creates a Display Panel

    # Required input:
    # x -> 1D array of shape n (x-axis)
    # y -> 1D array of shape m (y-axis)
    # z -> 2D array of shape nxm (data)

    # xdim -> String with the x-axis name
    # ydim -> String with the y-axis name
    # zdim -> String with the z-axis name
    # (If zdim='counts' it is treated as an integer)

    # xuts -> String with the x-axis units (it can be None)
    # yuts -> String with the y-axis units (it can be None)
    # zuts -> String with the z-axis units (it can be None)

    #--------------------------------------------------------------------------#

    # Run when Display Panel object is created
    def __init__(self, DisplayPanel):

        # Store the framework of the Display Panel
        self.DisplayPanel = DisplayPanel

        # Retrieve user preferences
        self.mysetup()

        # Initialize the Display Panel GUI
        self.GUI_setup()

        # Resize GUI elements
        self.GUI_resize()

        # GUI internal objects initialization
        self.GUI_internal()

    ############################################################################

    # User preferences
    def mysetup(self):

        ### General:

        # Set pyqtgraph background color
        pg.setConfigOptions(background='k')

        # Define default Colormap
        self.colormap=self.get_mpl_colormap('Greys')

        # Initial padding in percentage of total range
        self.pd=0.05

        # Default cursor position (offset from center in percentage of total range)
        self.offcs=[-0.05,+0.05,-0.15,+0.15]

        ### Cursors properties:

        # Size
        self.csr_sizes=[15,15,15,15]

        # Symbol
        self.csr_symbols=[['+'],['+'],['+'],['+']]

        # Brush
        self.csr_brushs=['g','r',(255,131,0),(153,90,233)]

        # Text
        self.csr_texts=["1","2","3","4"]

        # Text color
        self.csr_textcs=['g','r',(255,131,0),(153,90,233)]

        # Crosshair state
        # [] -> none
        # [0] -> vertical
        # [1] -> horizontal
        # [2] -> both
        self.csrs=[2,2,2,2]

        # Crosshair color
        self.csrcs=['g','r',(255,131,0),(153,90,233)]

        # Integration-crosshair spanx
        self.ispanx=0

        # Integration-crosshair spany
        self.ispany=0

        # Integration-crosshair state
        # [] -> none
        # [0] -> vertical
        # [1] -> horizontal
        # [2] -> both
        self.icsrs=[2,2,2,2]

        # Integration-crosshair color
        self.icsrcs=[(0,255,0),(255,0,0),(255,131,0),(153,90,233)]

        # Integration-crosshair brush (RGBA)
        self.ibrushs=[(0,255,0,50),(255,0,0,50),(255,131,0,50),(153,90,233,50)]

        # Allow/deny vertical/horizontal cursor movements
        self.h=True
        self.v=True

        ### Keyboard commands:

        # Activate arrows movement key
        # (Cursor 1 moves using the arrows only)
        self.key2=QtCore.Qt.Key.Key_Shift # (For some reason the shift key does not auto-repeat when pressed)
        self.key3=QtCore.Qt.Key.Key_Y
        self.key4=QtCore.Qt.Key.Key_X
        self.groupkey=QtCore.Qt.Key.Key_C

        # Show/hide cursor key
        self.shkeys=[QtCore.Qt.Key.Key_1,QtCore.Qt.Key.Key_2,QtCore.Qt.Key.Key_3,QtCore.Qt.Key.Key_4]

        # Switch crosshair state key
        self.switchkeys=[QtCore.Qt.Key.Key_Q,QtCore.Qt.Key.Key_W,QtCore.Qt.Key.Key_E,QtCore.Qt.Key.Key_R]

        # Show/hide DCs key
        self.DCkeys=[QtCore.Qt.Key.Key_A,QtCore.Qt.Key.Key_S,QtCore.Qt.Key.Key_D,QtCore.Qt.Key.Key_F]

        # Switch integration-crosshair state key
        self.iswitchkeys=[QtCore.Qt.Key.Key_5,QtCore.Qt.Key.Key_6,QtCore.Qt.Key.Key_7,QtCore.Qt.Key.Key_8]

        # Switch integration-crosshair fade key
        self.fadekey=QtCore.Qt.Key.Key_G

        # Hide/show all active objects key
        self.allkey=QtCore.Qt.Key.Key_Space

        # Reset cursors key
        self.resetkey=QtCore.Qt.Key.Key_Return

        ### Label properties:

        # Label color
        self.labelcs=['g','r',(255,131,0),(153,90,233),'y','y']

        ### Side plots properties

        # Plot color
        self.sidecs=[(0,255,150),(255,0,150),(255,131,150),(153,150,233)]

        ### Whole range integration properties

        # Plot color
        self.peniMDC=(0,255,255)
        self.peniEDC=(255,255,0)

        # Region brush
        self.brushiX=(255,255,0,50)
        self.brushiY=(0,255,255,50)

        # Region pen
        self.peniX=(255,255,0)
        self.peniY=(0,255,255)

        ### CursorStats format

        # Font size
        self.fontsz="12"

        # Digits after decimal point
        self.digits="3"

        # drc color (in label6)
        self.drcc="cyan"

        ### Half-span/step digits after decimal point (dimension units mode)
        self.spandec=4

        ### Create cursors dictionaries (do not edit)
        self.dicts=[]
        for i in range(4):
            self.dicts.append({'dtype': float, 'size': self.csr_sizes[i], 'symbol': self.csr_symbols[i], 'brush': self.csr_brushs[i], 'text': self.csr_texts[i], 'textc': self.csr_textcs[i], 'pxMode': True, 'csr': self.csrs[i], 'csrc': self.csrcs[i], 'ispanx': self.ispanx, 'ispany': self.ispany, 'icsr': self.icsrs[i], 'icsrc': self.icsrcs[i], 'ibrush': self.ibrushs[i], 'h': self.h, 'v': self.v})

    ############################################################################

    # Initialize the Display Panel GUI
    def GUI_setup(self):

        ### Define general GUI elements and geometry:

        # Specify main window properties
        self.DisplayPanel.setObjectName("DisplayPanel")
        self.DisplayPanel.resize(871, 801)

        # Define central widget within main window
        self.centralwidget = QtWidgets.QWidget(self.DisplayPanel)
        self.centralwidget.setMouseTracking(False)
        self.centralwidget.setObjectName("centralwidget")

        # Set central widget background color
        self.centralwidget.setAutoFillBackground(True)
        p = self.centralwidget.palette()
        p.setColor(self.centralwidget.backgroundRole(), QtCore.Qt.GlobalColor.white) # Input color here
        self.centralwidget.setPalette(p)

        # Define MainPlot frame
        self.MainPlot = PlotWidgetKP(self.centralwidget)
        self.MainPlot.setGeometry(QtCore.QRect(30, 10, 541, 381))
        self.MainPlot.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))
        self.MainPlot.setObjectName("MainPlot")

        # Define MDCPlot frame
        self.MDCPlot = pg.PlotWidget(self.centralwidget)
        self.MDCPlot.setGeometry(QtCore.QRect(30, 500, 541, 241))
        self.MDCPlot.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))
        self.MDCPlot.setObjectName("MDCPlot")

        # Define EDCPlot frame
        self.EDCPlot = pg.PlotWidget(self.centralwidget)
        self.EDCPlot.setGeometry(QtCore.QRect(580, 10, 261, 381))
        self.EDCPlot.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))
        self.EDCPlot.setObjectName("EDCPlot")

        # Define CursorStats frame
        # CursorStats must not be in contact with MainPlot and MDCPlot:
        # (to avoid random artefacts in the cursor stats labels when moving the cursors)
        self.CursorStats = pg.GraphicsLayoutWidget(self.centralwidget) # The associated graphics layout is 'self.CursorStats.ci'
        self.CursorStats.setGeometry(QtCore.QRect(30, 395, 541, 101))
        self.CursorStats.setObjectName("CursorStats")
        self.CursorStats.ci.setSpacing(0) # Set to zero the spacing between the cells in the GraphicsLayout
        self.CursorStats.ci.setBorder(color='k') # Color the borders between the cells in the GraphicsLayout (useful to see the cell limits)

        # Define Follow me checkbox
        self.Followme = QtWidgets.QCheckBox(self.centralwidget)
        self.Followme.setGeometry(QtCore.QRect(590, 400, 87, 20))
        self.Followme.setObjectName("Followme")

        # Define Bilateral Link checkbox
        self.Bilateral = QtWidgets.QCheckBox(self.centralwidget)
        self.Bilateral.setGeometry(QtCore.QRect(690, 400, 101, 20))
        self.Bilateral.setObjectName("Bilateral")

        # Define High cutoff slider
        self.HCutOff = QtWidgets.QSlider(self.centralwidget)
        self.HCutOff.setGeometry(QtCore.QRect(580, 470, 251, 22))
        self.HCutOff.setMaximum(100)
        self.HCutOff.setProperty("value", 100)
        self.HCutOff.setSliderPosition(100)
        self.HCutOff.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.HCutOff.setInvertedAppearance(False)
        self.HCutOff.setInvertedControls(False)
        self.HCutOff.setObjectName("HCutOff")

        # Define Low cutoff slider
        self.LCutOff = QtWidgets.QSlider(self.centralwidget)
        self.LCutOff.setGeometry(QtCore.QRect(580, 500, 251, 22))
        self.LCutOff.setMaximum(100)
        self.LCutOff.setProperty("value", 0)
        self.LCutOff.setSliderPosition(0)
        self.LCutOff.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.LCutOff.setInvertedAppearance(False)
        self.LCutOff.setInvertedControls(False)
        self.LCutOff.setObjectName("LCutOff")

        # Define Invert contrast checkbox
        self.Invert = QtWidgets.QCheckBox(self.centralwidget)
        self.Invert.setGeometry(QtCore.QRect(840, 470, 21, 21))
        self.Invert.setObjectName("Invert")

        # Define Colormap dropdown menu
        self.Cmaps = QtWidgets.QComboBox(self.centralwidget)
        self.Cmaps.setGeometry(QtCore.QRect(590, 440, 151, 26))
        self.Cmaps.setEditable(False)
        self.Cmaps.setCurrentText("Greys")
        self.Cmaps.setObjectName("Cmaps")
        self.Cmaps.addItems(["Greys", "Blues", "viridis", "plasma", "inferno", "magma", "cividis", "bone", "spring", "summer", "autumn", "winter", "cool", "jet", "ocean", "RdBu", "BrBG", "PuOr", "coolwarm", "bwr", "seismic"])  # Choose your favourites

        # Define integration-crosshairs span x spinbox
        self.SpanX = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.SpanX.setGeometry(QtCore.QRect(670, 550, 91, 24))
        self.SpanX.setObjectName("SpanX")
        self.SpanX.setDecimals(self.spandec)
        self.SpanX.setSingleStep(0)
        self.SpanX.setLocale(QtCore.QLocale(QtCore.QLocale.Language.English, QtCore.QLocale.Country.UnitedStates)) # Decimal separator '.'

        # Define integration-crosshairs unit change x text entry
        self.DeltaX = QtWidgets.QLineEdit(self.centralwidget)
        self.DeltaX.setGeometry(QtCore.QRect(770, 550, 61, 24))
        self.DeltaX.setObjectName("DeltaX")
        self.DeltaX.setText(('%.'+str(self.spandec)+'f')%(0))

        # Define integration-crosshairs span y spinbox
        self.SpanY = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.SpanY.setGeometry(QtCore.QRect(670, 580, 91, 24))
        self.SpanY.setObjectName("SpanY")
        self.SpanY.setDecimals(self.spandec)
        self.SpanY.setSingleStep(0)
        self.SpanY.setLocale(QtCore.QLocale(QtCore.QLocale.Language.English, QtCore.QLocale.Country.UnitedStates)) # Decimal separator '.'

        # Define integration-crosshairs unit change y text entry
        self.DeltaY = QtWidgets.QLineEdit(self.centralwidget)
        self.DeltaY.setGeometry(QtCore.QRect(770, 580, 61, 24))
        self.DeltaY.setObjectName("DeltaY")
        self.DeltaY.setText(('%.'+str(self.spandec)+'f')%(0))

        # Define integration-crosshair xdim label
        self.LabelX = QtWidgets.QLabel(self.centralwidget)
        self.LabelX.setGeometry(QtCore.QRect(600, 555, 60, 16))
        self.LabelX.setObjectName("LabelX")

        # Define integration-crosshair ydim label
        self.LabelY = QtWidgets.QLabel(self.centralwidget)
        self.LabelY.setGeometry(QtCore.QRect(600, 585, 60, 16))
        self.LabelY.setObjectName("LabelY")

        # Define integration-crosshair half span label
        self.LabelHSpan = QtWidgets.QLabel(self.centralwidget)
        self.LabelHSpan.setGeometry(QtCore.QRect(680, 530, 60, 16))
        self.LabelHSpan.setObjectName("LabelHSpan")

        # Define integration-crosshair step label
        self.LabelStep = QtWidgets.QLabel(self.centralwidget)
        self.LabelStep.setGeometry(QtCore.QRect(780, 530, 60, 16))
        self.LabelStep.setObjectName("LabelStep")

        # Define integrate whole x range checkbox
        self.allX = QtWidgets.QCheckBox(self.centralwidget)
        self.allX.setGeometry(QtCore.QRect(840, 550, 21, 21))
        self.allX.setObjectName("allX")

        # Define integrate whole y range checkbox
        self.allY = QtWidgets.QCheckBox(self.centralwidget)
        self.allY.setGeometry(QtCore.QRect(840, 580, 21, 21))
        self.allY.setObjectName("allY")

        # Define cursor mode label
        self.LabelCsrMode = QtWidgets.QLabel(self.centralwidget)
        self.LabelCsrMode.setGeometry(QtCore.QRect(40, 750, 171, 16))
        self.LabelCsrMode.setObjectName("LabelCsrMode")

        # Define integration cursor units checkbox
        self.DimorPix = QtWidgets.QCheckBox(self.centralwidget)
        self.DimorPix.setGeometry(QtCore.QRect(590, 528, 101, 21))
        self.DimorPix.setObjectName("DimorPix")

        # Define files listbox
        self.Files = CircularListWidget(self.centralwidget)
        self.Files.setGeometry(QtCore.QRect(590, 620, 261, 121))
        self.Files.setObjectName("Files")

        # Define other GUI elements
        self.DisplayPanel.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(self.DisplayPanel)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 860, 24))
        self.menubar.setObjectName("menubar")
        self.DisplayPanel.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self.DisplayPanel)
        self.statusbar.setObjectName("statusbar")
        self.DisplayPanel.setStatusBar(self.statusbar)

        # Initialize widget's labels
        self.GUI_retranslate()
        QtCore.QMetaObject.connectSlotsByName(self.DisplayPanel)

    ############################################################################

    # Set GUI text labels
    def GUI_retranslate(self):
        _translate = QtCore.QCoreApplication.translate
        self.DisplayPanel.setWindowTitle(_translate("DisplayPanel","Display panel"))
        self.Followme.setText(_translate("DisplayPanel","Follow me"))
        self.Bilateral.setText(_translate("DisplayPanel","Bilateral Link"))
        self.LabelHSpan.setText(_translate("DisplayPanel","Span"))
        self.LabelStep.setText(_translate("DisplayPanel","Step"))
        self.LabelCsrMode.setText(_translate("DisplayPanel","Cursor mode: Normal"))
        self.DimorPix.setText(_translate("DisplayPanel","Dim/Pix"))

    ############################################################################

    # Resize GUI elements
    # (See pyqt_ResizeItem.py)
    def GUI_resize(self):

        ResizeItem(self.DisplayPanel,self.MainPlot)
        ResizeItem(self.DisplayPanel,self.MDCPlot)
        ResizeItem(self.DisplayPanel,self.EDCPlot)
        ResizeItem(self.DisplayPanel,self.CursorStats)
        ResizeItem(self.DisplayPanel,self.Followme)
        ResizeItem(self.DisplayPanel,self.Bilateral)
        ResizeItem(self.DisplayPanel,self.HCutOff)
        ResizeItem(self.DisplayPanel,self.LCutOff)
        ResizeItem(self.DisplayPanel,self.Invert)
        ResizeItem(self.DisplayPanel,self.Cmaps)
        ResizeItem(self.DisplayPanel,self.SpanX)
        ResizeItem(self.DisplayPanel,self.DeltaX)
        ResizeItem(self.DisplayPanel,self.SpanY)
        ResizeItem(self.DisplayPanel,self.DeltaY)
        ResizeItem(self.DisplayPanel,self.LabelX)
        ResizeItem(self.DisplayPanel,self.LabelY)
        ResizeItem(self.DisplayPanel,self.LabelHSpan)
        ResizeItem(self.DisplayPanel,self.LabelStep)
        ResizeItem(self.DisplayPanel,self.allX)
        ResizeItem(self.DisplayPanel,self.allY)
        ResizeItem(self.DisplayPanel,self.LabelCsrMode)
        ResizeItem(self.DisplayPanel,self.DimorPix)
        ResizeItem(self.DisplayPanel,self.Files)

    ############################################################################

    # Internal GUI elements initialization
    def GUI_internal(self):

        # Zoom object definition at MainPlot, MDCPlot and EDCPlot
        # (See pyqt_ZoomItem.py)
        self.ZoomMain=ZoomItem(self.MainPlot)
        self.ZoomMDC=ZoomItem(self.MDCPlot)
        self.ZoomEDC=ZoomItem(self.EDCPlot)

        # Create image object
        self.Image = pg.ImageItem()

        # Set the initial colormap
        self.Image.setLookupTable(self.colormap)

        # Create cursor stats labels (force their height using setFixedHeight inherited to LabelItem from GraphicsWidget)
        self.labels=[]
        for i in range(6):
            label=pg.LabelItem(justify='left',color=self.labelcs[i])
            label.setFixedHeight(20)
            self.labels.append(label)

    ############################################################################

    # Retrieve input files
    def GUI_files(self,files):

        # Store the input files in the Display Panel
        self.files=files

        # Add the file names to self.Files
        for item in self.files:
            self.Files.addItem(item.attrs['scan_name'])

        # Select the first input file
        self.Files.setCurrentItem(self.Files.item(0))

        # Define the Display Panel data from the first input file
        self.GUI_setdata(**feedDPfromLoaders(self.files[0]))

        # Construct the Display Panel GUI initial content
        self.GUI_initial()

        # Create auxiliary signal objects
        self.GUI_auxiliary()

        # Initialize Display Panel GUI elements updates
        self.GUI_connect(None)

        # Select the Display Panel data from the list in self.Files
        def select_file():

            # Disconnect all Display Panel GUI elements updates
            self.GUI_disconnect(None)

            # Clear the Display Panel GUI
            self.MainPlot.clear()
            self.MDCPlot.clear()
            self.EDCPlot.clear()
            self.CursorStats.clear()

            # Clear PyQt5.QtWidgets.QGraphicsRectItem objects from the scene() in self.CursorStats.ci:
            # These objects are created everytime a label is added in self.CursorStats
            # These objects are not cleared using self.CursorStats.clear() and if they are not deleted
            # they keep accumulating everytime a new file is selected
            # Their accumulation considerably reduces the performance of the Display Panel
            # Use the method .items() to see the all the items in self.CursorStats
            # If the number of items is n, the first n-1 are PyQt5.QtWidgets.QGraphicsRectItem objects
            # (Methods of an object: print([method for method in dir(obj) if callable(getattr(obj, method))]))
            for i in range(len(self.CursorStats.items())-1):
                self.CursorStats.ci.scene().removeItem(self.CursorStats.items()[0])

            # Define the Display Panel data from the selected input file
            self.GUI_setdata(**feedDPfromLoaders(self.files[self.Files.currentRow()]))

            # Construct the Display Panel GUI initial content
            self.GUI_initial()

            # Create auxiliary signal objects
            self.GUI_auxiliary()

            # Initialize Display Panel GUI elements updates
            self.GUI_connect(None)

        # Signals when an item in self.Files is selected
        self.Files.itemSelectionChanged.connect(select_file)

    ############################################################################

    # Set GUI data
    def GUI_setdata(self,x, y, z, xdim, ydim, zdim, xuts, yuts, zuts):

        # Convert the input data to attributes of the Display Panel
        self.x=x
        self.y=y
        self.z=np.nan_to_num(z)

        self.xdim=xdim
        self.ydim=ydim
        self.zdim=zdim

        self.xuts=xuts
        self.yuts=yuts
        self.zuts=zuts

        # Extract x parameters
        self.x_min=min(self.x)
        self.x_max=max(self.x)
        self.x_size=len(self.x)
        self.x_delta=x[1]-x[0]

        # Extract y parameters
        self.y_min=min(self.y)
        self.y_max=max(self.y)
        self.y_size=len(self.y)
        self.y_delta=y[1]-y[0]

        # Extract z parameters
        self.z_min=np.min(self.z)
        self.z_max=np.max(self.z)

        # The array displayed in the ImageItem
        # Flipping is needed to account for the difference between coordinates and indexes
        self.zz=np.fliplr(self.z)

    ############################################################################

    # Construct Display Panel GUI initial content (dependant on loaded data)
    def GUI_initial(self):

        ### Set the axis labels

        # MainPlot
        self.MainPlot.setLabel('left', text=self.ydim, units=self.yuts, unitPrefix=None)
        self.MainPlot.setLabel('bottom', text=self.xdim, units=self.xuts, unitPrefix=None)

        # MDCPlot
        self.MDCPlot.setLabel('left', text=self.zdim, units=self.zuts, unitPrefix=None)
        self.MDCPlot.setLabel('bottom', text=self.xdim, units=self.xuts, unitPrefix=None)

        # EDCPlot
        self.EDCPlot.setLabel('left', text=self.ydim, units=self.yuts, unitPrefix=None)
        self.EDCPlot.setLabel('bottom', text=self.zdim, units=self.zuts, unitPrefix=None)

        ########################################################################

        ### Set the integration crosshairs labels

        self.LabelX.setText(self.xdim)
        self.LabelY.setText(self.ydim)

        ########################################################################

        ### Set the initial range for the cursor integration spinboxes

        # (Initially the cursor integration is in dimension units)
        self.SpanX.setMaximum(self.x_max-self.x_min)
        self.SpanY.setMaximum(self.y_max-self.y_min)

        ########################################################################

        ### Define the cursor stats format

        # String particles (dependant on user preferences)
        str1="<span style='font-size:"+self.fontsz+"pt'>"
        str2="%0."+self.digits+"f"
        if self.zdim == 'counts': # (Display z as an integer)
            str3="%i"
        else:
            str3=str2
        str4="<span style='color:"+self.drcc+"'>"

        # Full strings
        self.format_csr = str1+"[%i,%i] %s="+str2+" %s="+str2+" %s="+str3 # Cursors
        self.format_dxy = str1+"d(%s)="+str2+" d(%s)="+str2 # Delta dx dy
        self.format_dzrc = str1+"d(%s)="+str3+" &nbsp;&nbsp;"+str4+" d[%i,%i]" # Delta dz dr dc

        ########################################################################

        ### MainPlot image initialization:

        # Add image object in MainPlot
        self.MainPlot.addItem(self.Image)

        # Create limits rectangle for the data in MainPlot
        self.ImageRectangle=QtCore.QRectF(self.x_min,self.y_max,self.x_max-self.x_min,self.y_min-self.y_max)

        # Plot 2D data in MainPlot
        self.Image.setImage(self.zz)

        # Scale 2D data in MainPlot
        self.Image.setRect(self.ImageRectangle)

        ########################################################################

        ### Limit zoom and panning of MainPlot, EDCPlot and MDCPlot:

        # Define zoom/panning limits
        self.le=self.x_min-(self.x_max-self.x_min)*self.pd # Left edge
        self.re=self.x_max+(self.x_max-self.x_min)*self.pd # Right edge
        self.be=self.y_min-(self.y_max-self.y_min)*self.pd # Bottom edge
        self.te=self.y_max+(self.y_max-self.y_min)*self.pd # Top edge

        # Set zoom/panning limits of MainPlot
        self.MainPlot.getViewBox().setLimits(xMin=self.le, xMax=self.re, yMin=self.be, yMax=self.te)

        # MDCPlot intensity minimum and maximum are set to zmin and zmax of data
        self.MDCPlot.getViewBox().setLimits(xMin=self.le, xMax=self.re, yMin=self.z_min, yMax=self.z_max)

        # EDCPlot intensity minimum and maximum are set to zmin and zmax of data
        self.EDCPlot.getViewBox().setLimits(xMin=self.z_min, xMax=self.z_max, yMin=self.be, yMax=self.te)

        ########################################################################

        ### Set the initial range of MainPlot, EDCPlot and MDCPlot:

        # Set the initial range of MainPlot
        self.MainPlot.getViewBox().setXRange(self.le,self.re,padding=0)
        self.MainPlot.getViewBox().setYRange(self.be,self.te,padding=0)

        # Set the initial range of EDCPlot and MDCPlot equal to MainPlot
        self.MDCPlot.getViewBox().setXRange(self.le,self.re,padding=0)
        self.EDCPlot.getViewBox().setYRange(self.be,self.te,padding=0)

        # Set the initial range of EDCPlot and MDCPlot intensity equal to the z limits of the data in MainPlot
        self.MDCPlot.setYRange(self.z_min,self.z_max,padding=0)
        self.EDCPlot.setXRange(self.z_min,self.z_max,padding=0)

        ########################################################################

        ### Set the initial autorange policy:

        # Disable auto range for MainPlot
        self.MainPlot.disableAutoRange()

        # Enable auto range for MDCPlot and EDCPlot
        self.MDCPlot.enableAutoRange()
        self.EDCPlot.enableAutoRange()

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
        self.cursors=[]
        for i in range(4):
            self.cursors.append(CursorItem())

        # Define the zoom/cursor objects space
        self.space=[[self.x_min,self.y_min],[self.x_max,self.y_max]]
        self.spaceMDC=[[self.x_min,self.z_min],[self.x_max,self.z_max]]
        self.spaceEDC=[[self.z_min,self.y_min],[self.z_max,self.y_max]]

        # Set the space for the zoom objects
        self.ZoomMain.setSpace(self.space)
        self.ZoomMDC.setSpace(self.spaceMDC)
        self.ZoomEDC.setSpace(self.spaceEDC)

        # Set the space for the cursors
        for i in range(4):
            self.cursors[i].setSpace(self.space)

        # Add the leader cursors in MainPlot
        for i in range(2):
            self.MainPlot.addItem(self.cursors[i])

        # Get the initial MainPlot range
        self.rng0=self.MainPlot.viewRange()

        # Set the initial cursors configuration
        for i in range(4):
            self.cursors[i].setData(pos=np.array([[self.dcp(self.rng0,i+1,0),self.dcp(self.rng0,i+1,1)]]), **self.dicts[i])

        # Create infinite vertical lines in MDCPlot (vertical side cursors)
        self.scsrvs=[]
        for i in range(4):
            self.scsrvs.append(pg.InfiniteLine(pos=self.cursors[i].data['pos'][0][0],angle=90,pen=self.csrcs[i]))

        # Add the leader vertical side cursors in MainPlot
        for i in range(2):
            self.MDCPlot.addItem(self.scsrvs[i])

        # Create infinite horizontal lines in EDCPlot (horizontal side cursors)
        self.scsrhs=[]
        for i in range(4):
            self.scsrhs.append(pg.InfiniteLine(pos=self.cursors[i].data['pos'][0][1],angle=0,pen=self.csrcs[i]))

        # Add the leader horizontal side cursors in MainPlot
        for i in range(2):
            self.EDCPlot.addItem(self.scsrhs[i])

        # Add the cursor stats labels in CursorStats
        self.CursorStats.addItem(self.labels[0],row=0,col=0) # Cursor 1 (leader)
        self.CursorStats.addItem(self.labels[1],row=1,col=0) # Cursor 2 (leader)
        self.CursorStats.addItem(self.labels[4],row=0,col=1) # Delta dx dy
        self.CursorStats.addItem(self.labels[5],row=1,col=1) # Delta dz dr dc

        # Set the initial cursor stats labels info (cursors)
        for i in range(4):
            self.definfo(i+1)

        # Set the initial cursor stats labels info (leader cursors delta)
        self.defdelta()

        # Set the initial side plots
        self.MDCs=[]
        self.EDCs=[]
        for i in range(4):
            if i<2:
                self.MDCs.append(self.MDCPlot.plot(self.x,self.z[:,self.find_nearest(self.y,self.cursors[i].data['pos'][0][1])], pen=self.sidecs[i]))
                self.EDCs.append(self.EDCPlot.plot(self.z[self.find_nearest(self.x,self.cursors[i].data['pos'][0][0]),:],self.y, pen=self.sidecs[i]))
            else:
                self.MDCs.append(None)
                self.EDCs.append(None)

        # Initialize the signals array
        # self.cns[0] = None, such that the signal count starts at 1
        self.cns=[None]+[False]*self.GUI_connect([0])

    ############################################################################

    # Create auxiliary signal objects
    def GUI_auxiliary(self):

        # Lambda functions can be used to pass arguments to a connected function
        # Partials can be used to pass arguments to a connected function while also handling event objects

        #----------------------------------------------------------------------#

        # Update side cursors
        self.update_scsrs=[None]*4
        self.update_scsrs[0]=lambda: self.update_scsr(1) # Signal object 1
        self.update_scsrs[1]=lambda: self.update_scsr(2) # Signal object 2
        self.update_scsrs[2]=lambda: self.update_scsr(3) # Signal object 3
        self.update_scsrs[3]=lambda: self.update_scsr(4) # Signal object 4

        #----------------------------------------------------------------------#

        # Update cursor stats labels info (cursors)
        self.updateinfos=[None]*4
        self.updateinfos[0]=lambda: self.updateinfo(1) # Signal object 5
        self.updateinfos[1]=lambda: self.updateinfo(2) # Signal object 6
        self.updateinfos[2]=lambda: self.updateinfo(3) # Signal object 7
        self.updateinfos[3]=lambda: self.updateinfo(4) # Signal object 8

        #----------------------------------------------------------------------#

        # Update MDCs
        self.updateMDCs=[None]*4
        self.updateMDCs[0]=lambda: self.updateMDC(1) # Signal object 11
        self.updateMDCs[1]=lambda: self.updateMDC(2) # Signal object 12
        self.updateMDCs[2]=lambda: self.updateMDC(3) # Signal object 13
        self.updateMDCs[3]=lambda: self.updateMDC(4) # Signal object 14

        # Update EDCs
        self.updateEDCs=[None]*4
        self.updateEDCs[0]=lambda: self.updateEDC(1) # Signal object 15
        self.updateEDCs[1]=lambda: self.updateEDC(2) # Signal object 16
        self.updateEDCs[2]=lambda: self.updateEDC(3) # Signal object 17
        self.updateEDCs[3]=lambda: self.updateEDC(4) # Signal object 18

        #----------------------------------------------------------------------#

        # Signals if key is pressed to reset the cursors
        self.resetcsrs_sig=partial(self.resetcsrs,self.resetkey) # Signal object 20

        #----------------------------------------------------------------------#

        # Individual cursor arrow movement
        self.csrs_mov=[None]*4
        self.csrs_mov[0]=partial(self.arrowmove,1) # Signal object 25
        self.csrs_mov[1]=partial(self.arrowmove,2) # Signal object aux
        self.csrs_mov[2]=partial(self.arrowmove,3) # Signal object aux
        self.csrs_mov[3]=partial(self.arrowmove,4) # Signal object aux

        # Individual cursor key pressed + arrow movement
        self.csrs_kmov=[None]*4
        self.csrs_kmov[1]=partial(self.karrowmove,2,self.key2) # Signal object 26
        self.csrs_kmov[2]=partial(self.karrowmove,3,self.key3) # Signal object 27
        self.csrs_kmov[3]=partial(self.karrowmove,4,self.key4) # Signal object 28

        # Signals for groupal cursors movements
        self.karrowall_sig=partial(self.karrowall,self.groupkey) # Signal object 29

        #----------------------------------------------------------------------#

        # Signals to hide all active objects
        self.hidecurrent_sig=partial(self.hidecurrent,[25,26,27,28],self.allkey) # Signal object 30

        # Signals to show all active objects
        self.showcurrent_sig=partial(self.showcurrent,[25,26,27,28],self.allkey) # Signal object 31

        #----------------------------------------------------------------------#

        # Show/hide individual cursors
        self.showhides=[None]*4
        self.showhides[0]=partial(self.showhide,1,self.shkeys[0]) # Signal object 32
        self.showhides[1]=partial(self.showhide,2,self.shkeys[1]) # Signal object 33
        self.showhides[2]=partial(self.showhide,3,self.shkeys[2]) # Signal object 34
        self.showhides[3]=partial(self.showhide,4,self.shkeys[3]) # Signal object 35

        #----------------------------------------------------------------------#

        # Hide/show MDCs
        self.hsMDCs=[None]*4
        self.hsMDCs[0]=partial(self.hideshowMDCs,[11],1,self.DCkeys[0]) # Signal object 36
        self.hsMDCs[1]=partial(self.hideshowMDCs,[12],2,self.DCkeys[1]) # Signal object 37
        self.hsMDCs[2]=partial(self.hideshowMDCs,[13],3,self.DCkeys[2]) # Signal object 38
        self.hsMDCs[3]=partial(self.hideshowMDCs,[14],4,self.DCkeys[3]) # Signal object 39

        # Hide/show EDCs
        self.hsEDCs=[None]*4
        self.hsEDCs[0]=partial(self.hideshowEDCs,[15],1,self.DCkeys[0]) # Signal object 40
        self.hsEDCs[1]=partial(self.hideshowEDCs,[16],2,self.DCkeys[1]) # Signal object 41
        self.hsEDCs[2]=partial(self.hideshowEDCs,[17],3,self.DCkeys[2]) # Signal object 42
        self.hsEDCs[3]=partial(self.hideshowEDCs,[18],4,self.DCkeys[3]) # Signal object 43

        #----------------------------------------------------------------------#

        # Signals to line up the cursors vertically
        self.lineupv_sig=partial(self.lineupv,QtCore.Qt.Key.Key_V) # Signal object 44

        # Signals to line up the cursors horizontally
        self.lineuph_sig=partial(self.lineuph,QtCore.Qt.Key.Key_H) # Signal object 45

        #----------------------------------------------------------------------#

        # Switch crosshair states
        self.switchcsrs=[None]*4
        self.switchcsrs[0]=partial(self.switchcsr,1,'csr',self.switchkeys[0]) # Signal object 46
        self.switchcsrs[1]=partial(self.switchcsr,2,'csr',self.switchkeys[1]) # Signal object 47
        self.switchcsrs[2]=partial(self.switchcsr,3,'csr',self.switchkeys[2]) # Signal object 48
        self.switchcsrs[3]=partial(self.switchcsr,4,'csr',self.switchkeys[3]) # Signal object 49

        # Switch integration-crosshair states
        self.iswitchcsrs=[None]*4
        self.iswitchcsrs[0]=partial(self.switchcsr,1,'icsr',self.iswitchkeys[0]) # Signal object 50
        self.iswitchcsrs[1]=partial(self.switchcsr,2,'icsr',self.iswitchkeys[1]) # Signal object 51
        self.iswitchcsrs[2]=partial(self.switchcsr,3,'icsr',self.iswitchkeys[2]) # Signal object 52
        self.iswitchcsrs[3]=partial(self.switchcsr,4,'icsr',self.iswitchkeys[3]) # Signal object 53

        # Signals to switch integration-crosshair fading
        self.fadeicsrc_sig=partial(self.fadeicsrc,self.fadekey) # Signal object 54

        #----------------------------------------------------------------------#

        # Signals when allX checkbox is checked/unchecked
        self.integrateallx_sig=partial(self.integrateallx,[15,16,17,18,40,41,42,43]) # Signal object 60

        # Signals when allY checkbox is checked/unchecked
        self.integrateally_sig=partial(self.integrateally,[11,12,13,14,36,37,38,39]) # Signal object 61

    ############################################################################

    # Connect Display Panel GUI elements updates
    # If cn = None, connect all the signals
    # If cn = [0], return the total number of signals
    # If cn is a list of positive integers, connect all the signals in the list
    def GUI_connect(self,cn):

        # Keep track of the connections using booleans:
        # There are no methods to determine if an object is connected to a function
        # Keeping track of the connections let us avoid issues such as:
        # Already disconnected errors
        # Descrease in performance due to duplicated connections

        #----------------------------------------------------------------------#

        # Update side cursors
        i = 1
        if not cn or i in cn:
            if self.cns[i] == False:
                self.cursors[0].scatter.sigPlotChanged.connect(self.update_scsrs[0])
                self.cns[i] = True

        i = 2
        if not cn or i in cn:
            if self.cns[i] == False:
                self.cursors[1].scatter.sigPlotChanged.connect(self.update_scsrs[1])
                self.cns[i] = True

        i = 3
        if not cn or i in cn:
            if self.cns[i] == False:
                self.cursors[2].scatter.sigPlotChanged.connect(self.update_scsrs[2])
                self.cns[i] = True

        i = 4
        if not cn or i in cn:
            if self.cns[i] == False:
                self.cursors[3].scatter.sigPlotChanged.connect(self.update_scsrs[3])
                self.cns[i] = True

        #----------------------------------------------------------------------#

        # Update cursor stats labels info (cursors)
        i = 5
        if not cn or i in cn:
            if self.cns[i] == False:
                self.cursors[0].scatter.sigPlotChanged.connect(self.updateinfos[0])
                self.cns[i] = True

        i = 6
        if not cn or i in cn:
            if self.cns[i] == False:
                self.cursors[1].scatter.sigPlotChanged.connect(self.updateinfos[1])
                self.cns[i] = True

        i = 7
        if not cn or i in cn:
            if self.cns[i] == False:
                self.cursors[2].scatter.sigPlotChanged.connect(self.updateinfos[2])
                self.cns[i] = True

        i = 8
        if not cn or i in cn:
            if self.cns[i] == False:
                self.cursors[3].scatter.sigPlotChanged.connect(self.updateinfos[3])
                self.cns[i] = True

        #----------------------------------------------------------------------#

        # Update cursor stats labels info (leader cursors delta)
        i = 9
        if not cn or i in cn:
            if self.cns[i] == False:
                self.cursors[0].scatter.sigPlotChanged.connect(self.updatedelta)
                self.cns[i] = True

        i = 10
        if not cn or i in cn:
            if self.cns[i] == False:
                self.cursors[1].scatter.sigPlotChanged.connect(self.updatedelta)
                self.cns[i] = True

        #----------------------------------------------------------------------#

        # Update MDCs
        i = 11
        if not cn or i in cn:
            if self.cns[i] == False:
                self.cursors[0].scatter.sigPlotChanged.connect(self.updateMDCs[0])
                self.cns[i] = True

        i = 12
        if not cn or i in cn:
            if self.cns[i] == False:
                self.cursors[1].scatter.sigPlotChanged.connect(self.updateMDCs[1])
                self.cns[i] = True

        i = 13
        if not cn or i in cn:
            if self.cns[i] == False:
                self.cursors[2].scatter.sigPlotChanged.connect(self.updateMDCs[2])
                self.cns[i] = True

        i = 14
        if not cn or i in cn:
            if self.cns[i] == False:
                self.cursors[3].scatter.sigPlotChanged.connect(self.updateMDCs[3])
                self.cns[i] = True

        # Update EDCs
        i = 15
        if not cn or i in cn:
            if self.cns[i] == False:
                self.cursors[0].scatter.sigPlotChanged.connect(self.updateEDCs[0])
                self.cns[i] = True

        i = 16
        if not cn or i in cn:
            if self.cns[i] == False:
                self.cursors[1].scatter.sigPlotChanged.connect(self.updateEDCs[1])
                self.cns[i] = True

        i = 17
        if not cn or i in cn:
            if self.cns[i] == False:
                self.cursors[2].scatter.sigPlotChanged.connect(self.updateEDCs[2])
                self.cns[i] = True

        i = 18
        if not cn or i in cn:
            if self.cns[i] == False:
                self.cursors[3].scatter.sigPlotChanged.connect(self.updateEDCs[3])
                self.cns[i] = True

        #----------------------------------------------------------------------#

        # Signals if Followme checkbox state is changed
        i = 19
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Followme.stateChanged.connect(self.follow)
                self.cns[i] = True

        # Signals if key is pressed to reset the cursors
        i = 20
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainPlot.sigKeyPress.connect(self.resetcsrs_sig)
                self.cns[i] = True

        #----------------------------------------------------------------------#

        # Signals when the MainPlot range is changed
        i = 21
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainPlot.sigRangeChanged.connect(self.updatesides)
                self.cns[i] = True

        # Signals when the MDCPlot range is changed
        i = 22
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MDCPlot.sigRangeChanged.connect(self.updatemainfromMDC)
                self.cns[i] = True

        # Signals when the EDCPlot range is changed
        i = 23
        if not cn or i in cn:
            if self.cns[i] == False:
                self.EDCPlot.sigRangeChanged.connect(self.updatemainfromEDC)
                self.cns[i] = True

        # Signals if Bilateral state is changed
        i = 24
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Bilateral.stateChanged.connect(self.bilateral)
                self.cns[i] = True

        #----------------------------------------------------------------------#

        # Individual cursor arrow movement
        i = 25
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainPlot.sigKeyPress.connect(self.csrs_mov[0]) # (Cursor 1 only)
                self.cns[i] = True

        # Individual cursor key pressed + arrow movement
        i = 26
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainPlot.sigKeyPress.connect(self.csrs_kmov[1]) # (Cursor 2)
                self.cns[i] = True

        i = 27
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainPlot.sigKeyPress.connect(self.csrs_kmov[2]) # (Cursor 3)
                self.cns[i] = True

        i = 28
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainPlot.sigKeyPress.connect(self.csrs_kmov[3]) # (Cursor 4)
                self.cns[i] = True

        # Signals for groupal cursors movements
        i = 29
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainPlot.sigKeyPress.connect(self.karrowall_sig)
                self.cns[i] = True

        #----------------------------------------------------------------------#

        # Signals to hide all active objects
        i = 30
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainPlot.sigKeyPress.connect(self.hidecurrent_sig)
                self.cns[i] = True

        # Signals to show all active objects
        i = 31
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainPlot.sigKeyRelease.connect(self.showcurrent_sig)
                self.cns[i] = True

        #----------------------------------------------------------------------#

        # Show/hide individual cursors
        i = 32
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainPlot.sigKeyPress.connect(self.showhides[0])
                self.cns[i] = True

        i = 33
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainPlot.sigKeyPress.connect(self.showhides[1])
                self.cns[i] = True

        i = 34
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainPlot.sigKeyPress.connect(self.showhides[2])
                self.cns[i] = True

        i = 35
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainPlot.sigKeyPress.connect(self.showhides[3])
                self.cns[i] = True

        #----------------------------------------------------------------------#

        # Hide/show MDCs
        i = 36
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainPlot.sigKeyPress.connect(self.hsMDCs[0])
                self.cns[i] = True

        i = 37
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainPlot.sigKeyPress.connect(self.hsMDCs[1])
                self.cns[i] = True

        i = 38
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainPlot.sigKeyPress.connect(self.hsMDCs[2])
                self.cns[i] = True

        i = 39
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainPlot.sigKeyPress.connect(self.hsMDCs[3])
                self.cns[i] = True

        # Hide/show EDCs
        i = 40
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainPlot.sigKeyPress.connect(self.hsEDCs[0])
                self.cns[i] = True

        i = 41
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainPlot.sigKeyPress.connect(self.hsEDCs[1])
                self.cns[i] = True

        i = 42
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainPlot.sigKeyPress.connect(self.hsEDCs[2])
                self.cns[i] = True

        i = 43
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainPlot.sigKeyPress.connect(self.hsEDCs[3])
                self.cns[i] = True

        #----------------------------------------------------------------------#

        # Signals to line up the cursors vertically
        i = 44
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainPlot.sigKeyPress.connect(self.lineupv_sig)
                self.cns[i] = True

        # Signals to line up the cursors horizontally
        i = 45
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainPlot.sigKeyPress.connect(self.lineuph_sig)
                self.cns[i] = True

        #----------------------------------------------------------------------#

        # Switch crosshair states
        i = 46
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainPlot.sigKeyPress.connect(self.switchcsrs[0])
                self.cns[i] = True

        i = 47
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainPlot.sigKeyPress.connect(self.switchcsrs[1])
                self.cns[i] = True

        i = 48
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainPlot.sigKeyPress.connect(self.switchcsrs[2])
                self.cns[i] = True

        i = 49
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainPlot.sigKeyPress.connect(self.switchcsrs[3])
                self.cns[i] = True

        # Switch integration-crosshair states
        i = 50
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainPlot.sigKeyPress.connect(self.iswitchcsrs[0])
                self.cns[i] = True

        i = 51
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainPlot.sigKeyPress.connect(self.iswitchcsrs[1])
                self.cns[i] = True

        i = 52
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainPlot.sigKeyPress.connect(self.iswitchcsrs[2])
                self.cns[i] = True

        i = 53
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainPlot.sigKeyPress.connect(self.iswitchcsrs[3])
                self.cns[i] = True

        # Signals to switch integration-crosshair fading
        i = 54
        if not cn or i in cn:
            if self.cns[i] == False:
                self.MainPlot.sigKeyPress.connect(self.fadeicsrc_sig)
                self.cns[i] = True

        #----------------------------------------------------------------------#

        # Signals if the DimorPix checkbox state is changed
        i = 55
        if not cn or i in cn:
            if self.cns[i] == False:
                self.DimorPix.stateChanged.connect(self.dim2pix)
                self.cns[i] = True

        # Signals when enter is pressed over self.DeltaX or when self.DeltaX loses focus
        i = 56
        if not cn or i in cn:
            if self.cns[i] == False:
                self.DeltaX.editingFinished.connect(self.stepx)
                self.cns[i] = True

        # Signals when enter is pressed over self.DeltaY or when self.DeltaY loses focus
        i = 57
        if not cn or i in cn:
            if self.cns[i] == False:
                self.DeltaY.editingFinished.connect(self.stepy)
                self.cns[i] = True

        # Signals to update the x integration-cursor
        i = 58
        if not cn or i in cn:
            if self.cns[i] == False:
                self.SpanX.valueChanged.connect(self.changeix)
                self.cns[i] = True

        # Signals to update the y integration-cursor
        i = 59
        if not cn or i in cn:
            if self.cns[i] == False:
                self.SpanY.valueChanged.connect(self.changeiy)
                self.cns[i] = True

        #----------------------------------------------------------------------#

        # Signals when allX checkbox is checked/unchecked
        i = 60
        if not cn or i in cn:
            if self.cns[i] == False:
                self.allX.stateChanged.connect(self.integrateallx_sig)
                self.cns[i] = True

        # Signals when allY checkbox is checked/unchecked
        i = 61
        if not cn or i in cn:
            if self.cns[i] == False:
                self.allY.stateChanged.connect(self.integrateally_sig)
                self.cns[i] = True

        #----------------------------------------------------------------------#

        # Signals when the HCutOff slider is moved
        i = 62
        if not cn or i in cn:
            if self.cns[i] == False:
                self.HCutOff.valueChanged.connect(self.contrast)
                self.cns[i] = True

        # Signals when the LCutOff slider is moved
        i = 63
        if not cn or i in cn:
            if self.cns[i] == False:
                self.LCutOff.valueChanged.connect(self.contrast)
                self.cns[i] = True

        # Signals when the Invert checkbox is checked/unchecked
        i = 64
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Invert.clicked.connect(self.invert)
                self.cns[i] = True

        # Signals when an option in Cmaps is selected
        i = 65
        if not cn or i in cn:
            if self.cns[i] == False:
                self.Cmaps.activated.connect(self.cmap_select)
                self.cns[i] = True

        #----------------------------------------------------------------------#

        # Return the total number of signals
        if cn == [0]:
            return i

    ############################################################################

    # Disconnect Display Panel GUI elements updates
    # If cn = None, disconnect all the signals
    # If cn is a list of positive integers, disconnect all the signals in the list
    def GUI_disconnect(self,cn):

        # If an object is already disconnected an exception is raised and ignored
        # (Disconnecting signals using for loops does not work)

        #----------------------------------------------------------------------#

        # Update side cursors
        i = 1
        if not cn or i in cn:
            try:
                self.cursors[0].scatter.sigPlotChanged.disconnect(self.update_scsrs[0])
            except:
                print(i)
            finally: self.cns[i] = False

        i = 2
        if not cn or i in cn:
            try:
                self.cursors[1].scatter.sigPlotChanged.disconnect(self.update_scsrs[1])
            except:
                print(i)
            finally: self.cns[i] = False

        i = 3
        if not cn or i in cn:
            try:
                self.cursors[2].scatter.sigPlotChanged.disconnect(self.update_scsrs[2])
            except:
                print(i)
            finally: self.cns[i] = False

        i = 4
        if not cn or i in cn:
            try:
                self.cursors[3].scatter.sigPlotChanged.disconnect(self.update_scsrs[3])
            except:
                print(i)
            finally: self.cns[i] = False

        #----------------------------------------------------------------------#

        # Update cursor stats labels info (cursors)
        i = 5
        if not cn or i in cn:
            try:
                self.cursors[0].scatter.sigPlotChanged.disconnect(self.updateinfos[0])
            except:
                print(i)
            finally: self.cns[i] = False

        i = 6
        if not cn or i in cn:
            try:
                self.cursors[1].scatter.sigPlotChanged.disconnect(self.updateinfos[1])
            except:
                print(i)
            finally: self.cns[i] = False

        i = 7
        if not cn or i in cn:
            try:
                self.cursors[2].scatter.sigPlotChanged.disconnect(self.updateinfos[2])
            except:
                print(i)
            finally: self.cns[i] = False

        i = 8
        if not cn or i in cn:
            try:
                self.cursors[3].scatter.sigPlotChanged.disconnect(self.updateinfos[3])
            except:
                print(i)
            finally: self.cns[i] = False

        #----------------------------------------------------------------------#

        # Update cursor stats labels info (leader cursors delta)
        i = 9
        if not cn or i in cn:
            try:
                self.cursors[0].scatter.sigPlotChanged.disconnect(self.updatedelta)
            except:
                print(i)
            finally: self.cns[i] = False

        i = 10
        if not cn or i in cn:
            try:
                self.cursors[1].scatter.sigPlotChanged.disconnect(self.updatedelta)
            except:
                print(i)
            finally: self.cns[i] = False

        #----------------------------------------------------------------------#

        # Update MDCs
        i = 11
        if not cn or i in cn:
            try:
                self.cursors[0].scatter.sigPlotChanged.disconnect(self.updateMDCs[0])
            except:
                print(i)
            finally: self.cns[i] = False

        i = 12
        if not cn or i in cn:
            try:
                self.cursors[1].scatter.sigPlotChanged.disconnect(self.updateMDCs[1])
            except:
                print(i)
            finally: self.cns[i] = False

        i = 13
        if not cn or i in cn:
            try:
                self.cursors[2].scatter.sigPlotChanged.disconnect(self.updateMDCs[2])
            except:
                print(i)
            finally: self.cns[i] = False

        i = 14
        if not cn or i in cn:
            try:
                self.cursors[3].scatter.sigPlotChanged.disconnect(self.updateMDCs[3])
            except:
                print(i)
            finally: self.cns[i] = False

        # Update EDCs
        i = 15
        if not cn or i in cn:
            try:
                self.cursors[0].scatter.sigPlotChanged.disconnect(self.updateEDCs[0])
            except:
                print(i)
            finally: self.cns[i] = False

        i = 16
        if not cn or i in cn:
            try:
                self.cursors[1].scatter.sigPlotChanged.disconnect(self.updateEDCs[1])
            except:
                print(i)
            finally: self.cns[i] = False

        i = 17
        if not cn or i in cn:
            try:
                self.cursors[2].scatter.sigPlotChanged.disconnect(self.updateEDCs[2])
            except:
                print(i)
            finally: self.cns[i] = False

        i = 18
        if not cn or i in cn:
            try:
                self.cursors[3].scatter.sigPlotChanged.disconnect(self.updateEDCs[3])
            except:
                print(i)
            finally: self.cns[i] = False

        #----------------------------------------------------------------------#

        # Signals if Followme checkbox state is changed
        i = 19
        if not cn or i in cn:
            try:
                self.Followme.stateChanged.disconnect(self.follow)
            except:
                print(i)
            finally: self.cns[i] = False

        # Signals if key is pressed to reset the cursors
        i = 20
        if not cn or i in cn:
            try:
                self.MainPlot.sigKeyPress.disconnect(self.resetcsrs_sig)
            except:
                print(i)
            finally: self.cns[i] = False

        #----------------------------------------------------------------------#

        # Signals when the MainPlot range is changed
        i = 21
        if not cn or i in cn:
            try:
                self.MainPlot.sigRangeChanged.disconnect(self.updatesides)
            except:
                print(i)
            finally: self.cns[i] = False

        # Signals when the MDCPlot range is changed
        i = 22
        if not cn or i in cn:
            try:
                self.MDCPlot.sigRangeChanged.disconnect(self.updatemainfromMDC)
            except:
                print(i)
            finally: self.cns[i] = False

        # Signals when the EDCPlot range is changed
        i = 23
        if not cn or i in cn:
            try:
                self.EDCPlot.sigRangeChanged.disconnect(self.updatemainfromEDC)
            except:
                print(i)
            finally: self.cns[i] = False

        # Signals if Bilateral state is changed
        i = 24
        if not cn or i in cn:
            try:
                self.Bilateral.stateChanged.disconnect(self.bilateral)
            except:
                print(i)
            finally: self.cns[i] = False

        #----------------------------------------------------------------------#

        # Individual cursor arrow movement
        i = 25
        if not cn or i in cn:
            try:
                self.MainPlot.sigKeyPress.disconnect(self.csrs_mov[0]) # (Cursor 1 only)
            except:
                print(i)
            finally: self.cns[i] = False

        # Individual cursor key pressed + arrow movement
        i = 26
        if not cn or i in cn:
            try:
                self.MainPlot.sigKeyPress.disconnect(self.csrs_kmov[1]) # (Cursor 2)
            except:
                print(i)
            finally: self.cns[i] = False

        i = 27
        if not cn or i in cn:
            try:
                self.MainPlot.sigKeyPress.disconnect(self.csrs_kmov[2]) # (Cursor 3)
            except:
                print(i)
            finally: self.cns[i] = False

        i = 28
        if not cn or i in cn:
            try:
                self.MainPlot.sigKeyPress.disconnect(self.csrs_kmov[3]) # (Cursor 4)
            except:
                print(i)
            finally: self.cns[i] = False

        # Signals for groupal cursors movements
        i = 29
        if not cn or i in cn:
            try:
                self.MainPlot.sigKeyPress.disconnect(self.karrowall_sig)
            except:
                print(i)
            finally: self.cns[i] = False

        #----------------------------------------------------------------------#

        # Signals to hide all active objects
        i = 30
        if not cn or i in cn:
            try:
                self.MainPlot.sigKeyPress.disconnect(self.hidecurrent_sig)
            except:
                print(i)
            finally: self.cns[i] = False

        # Signals to show all active objects
        i = 31
        if not cn or i in cn:
            try:
                self.MainPlot.sigKeyRelease.disconnect(self.showcurrent_sig)
            except:
                print(i)
            finally: self.cns[i] = False

        #----------------------------------------------------------------------#

        # Show/hide individual cursors
        i = 32
        if not cn or i in cn:
            try:
                self.MainPlot.sigKeyPress.disconnect(self.showhides[0])
            except:
                print(i)
            finally: self.cns[i] = False

        i = 33
        if not cn or i in cn:
            try:
                self.MainPlot.sigKeyPress.disconnect(self.showhides[1])
            except:
                print(i)
            finally: self.cns[i] = False

        i = 34
        if not cn or i in cn:
            try:
                self.MainPlot.sigKeyPress.disconnect(self.showhides[2])
            except:
                print(i)
            finally: self.cns[i] = False

        i = 35
        if not cn or i in cn:
            try:
                self.MainPlot.sigKeyPress.disconnect(self.showhides[3])
            except:
                print(i)
            finally: self.cns[i] = False

        #----------------------------------------------------------------------#

        # Hide/show MDCs
        i = 36
        if not cn or i in cn:
            try:
                self.MainPlot.sigKeyPress.disconnect(self.hsMDCs[0])
            except:
                print(i)
            finally: self.cns[i] = False

        i = 37
        if not cn or i in cn:
            try:
                self.MainPlot.sigKeyPress.disconnect(self.hsMDCs[1])
            except:
                print(i)
            finally: self.cns[i] = False

        i = 38
        if not cn or i in cn:
            try:
                self.MainPlot.sigKeyPress.disconnect(self.hsMDCs[2])
            except:
                print(i)
            finally: self.cns[i] = False

        i = 39
        if not cn or i in cn:
            try:
                self.MainPlot.sigKeyPress.disconnect(self.hsMDCs[3])
            except:
                print(i)
            finally: self.cns[i] = False

        # Hide/show EDCs
        i = 40
        if not cn or i in cn:
            try:
                self.MainPlot.sigKeyPress.disconnect(self.hsEDCs[0])
            except:
                print(i)
            finally: self.cns[i] = False

        i = 41
        if not cn or i in cn:
            try:
                self.MainPlot.sigKeyPress.disconnect(self.hsEDCs[1])
            except:
                print(i)
            finally: self.cns[i] = False

        i = 42
        if not cn or i in cn:
            try:
                self.MainPlot.sigKeyPress.disconnect(self.hsEDCs[2])
            except:
                print(i)
            finally: self.cns[i] = False

        i = 43
        if not cn or i in cn:
            try:
                self.MainPlot.sigKeyPress.disconnect(self.hsEDCs[3])
            except:
                print(i)
            finally: self.cns[i] = False

        #----------------------------------------------------------------------#

        # Signals to line up the cursors vertically
        i = 44
        if not cn or i in cn:
            try:
                self.MainPlot.sigKeyPress.disconnect(self.lineupv_sig)
            except:
                print(i)
            finally: self.cns[i] = False

        # Signals to line up the cursors horizontally
        i = 45
        if not cn or i in cn:
            try:
                self.MainPlot.sigKeyPress.disconnect(self.lineuph_sig)
            except:
                print(i)
            finally: self.cns[i] = False

        #----------------------------------------------------------------------#

        # Switch crosshair states
        i = 46
        if not cn or i in cn:
            try:
                self.MainPlot.sigKeyPress.disconnect(self.switchcsrs[0])
            except:
                print(i)
            finally: self.cns[i] = False

        i = 47
        if not cn or i in cn:
            try:
                self.MainPlot.sigKeyPress.disconnect(self.switchcsrs[1])
            except:
                print(i)
            finally: self.cns[i] = False

        i = 48
        if not cn or i in cn:
            try:
                self.MainPlot.sigKeyPress.disconnect(self.switchcsrs[2])
            except:
                print(i)
            finally: self.cns[i] = False

        i = 49
        if not cn or i in cn:
            try:
                self.MainPlot.sigKeyPress.disconnect(self.switchcsrs[3])
            except:
                print(i)
            finally: self.cns[i] = False

        # Switch integration-crosshair states
        i = 50
        if not cn or i in cn:
            try:
                self.MainPlot.sigKeyPress.disconnect(self.iswitchcsrs[0])
            except:
                print(i)
            finally: self.cns[i] = False

        i = 51
        if not cn or i in cn:
            try:
                self.MainPlot.sigKeyPress.disconnect(self.iswitchcsrs[1])
            except:
                print(i)
            finally: self.cns[i] = False

        i = 52
        if not cn or i in cn:
            try:
                self.MainPlot.sigKeyPress.disconnect(self.iswitchcsrs[2])
            except:
                print(i)
            finally: self.cns[i] = False

        i = 53
        if not cn or i in cn:
            try:
                self.MainPlot.sigKeyPress.disconnect(self.iswitchcsrs[3])
            except:
                print(i)
            finally: self.cns[i] = False

        # Signals to switch integration-crosshair fading
        i = 54
        if not cn or i in cn:
            try:
                self.MainPlot.sigKeyPress.disconnect(self.fadeicsrc_sig)
            except:
                print(i)
            finally: self.cns[i] = False

        #----------------------------------------------------------------------#

        # Signals if the DimorPix checkbox state is changed
        i = 55
        if not cn or i in cn:
            try:
                self.DimorPix.stateChanged.disconnect(self.dim2pix)
            except:
                print(i)
            finally: self.cns[i] = False

        # Signals when enter is pressed over self.DeltaX or when self.DeltaX loses focus
        i = 56
        if not cn or i in cn:
            try:
                self.DeltaX.editingFinished.disconnect(self.stepx)
            except:
                print(i)
            finally: self.cns[i] = False

        # Signals when enter is pressed over self.DeltaY or when self.DeltaY loses focus
        i = 57
        if not cn or i in cn:
            try:
                self.DeltaY.editingFinished.disconnect(self.stepy)
            except:
                print(i)
            finally: self.cns[i] = False

        # Signals to update the x integration-cursor
        i = 58
        if not cn or i in cn:
            try:
                self.SpanX.valueChanged.disconnect(self.changeix)
            except:
                print(i)
            finally: self.cns[i] = False

        # Signals to update the y integration-cursor
        i = 59
        if not cn or i in cn:
            try:
                self.SpanY.valueChanged.disconnect(self.changeiy)
            except:
                print(i)
            finally: self.cns[i] = False

        #----------------------------------------------------------------------#

        # Signals when allX checkbox is checked/unchecked
        i = 60
        if not cn or i in cn:
            try:
                self.allX.stateChanged.disconnect(self.integrateallx_sig)
            except:
                print(i)
            finally: self.cns[i] = False

        # Signals when allY checkbox is checked/unchecked
        i = 61
        if not cn or i in cn:
            try:
                self.allY.stateChanged.disconnect(self.integrateally_sig)
            except:
                print(i)
            finally: self.cns[i] = False

        #----------------------------------------------------------------------#

        # Signals when the HCutOff slider is moved
        i = 62
        if not cn or i in cn:
            try:
                self.HCutOff.valueChanged.disconnect(self.contrast)
            except:
                print(i)
            finally: self.cns[i] = False

        # Signals when the LCutOff slider is moved
        i = 63
        if not cn or i in cn:
            try:
                self.LCutOff.valueChanged.disconnect(self.contrast)
            except:
                print(i)
            finally: self.cns[i] = False

        # Signals when the Invert checkbox is checked/unchecked
        i = 64
        if not cn or i in cn:
            try:
                self.Invert.clicked.disconnect(self.invert)
            except:
                print(i)
            finally: self.cns[i] = False

        # Signals when an option in Cmaps is selected
        i = 65
        if not cn or i in cn:
            try:
                self.Cmaps.activated.disconnect(self.cmap_select)
            except:
                print(i)
            finally: self.cns[i] = False

    ############################################################################

    ### GUI internal methods (Display Panel functionalities)

    # Find index of element in array closest to value
    # (An elegant function used to map values into indexes with high efficiency)
    def find_nearest(self,array,value):

        # Find the array length
        n = len(array)

        l = 0      # Initialize lower index
        u = n-1    # Initialize upper index

        # Do while the difference between the upper limit and the lower limit is more than 1 unit
        while u-l > 1:

            m = u+l >> 1 # Compute midpoint with a bitshift

            if value >= array[m]:
                l = m # Update the lower limit
            else:
                u = m # Update the upper limit

        # Return index closest to value
        if value == array[0]:
            return 0
        elif value == array[n-1]:
            return n-1
        else:
            return l

    # Get default cursor position
    def dcp(self,rng,csr_num,i):
        return (rng[i][0]+rng[i][1])/2-(rng[i][1]-rng[i][0])*self.offcs[csr_num-1]

    # Set initial cursor stats labels info (cursors)
    def definfo(self,csr_num):

        x=self.dcp(self.rng0,csr_num,0) # x position
        y=self.dcp(self.rng0,csr_num,1) # y position

        ix=self.find_nearest(self.x,x) # x index
        iy=self.find_nearest(self.y,y) # y index

        # Set label text
        self.labels[csr_num-1].setText(self.format_csr % (ix, iy, self.xdim, x, self.ydim, y, self.zdim, self.z[ix,iy]))

    # Set initial cursor stats labels info (leader cursors delta)
    def defdelta(self):

        x1=self.dcp(self.rng0,1,0) # x position 1
        x2=self.dcp(self.rng0,2,0) # x position 2

        y1=self.dcp(self.rng0,1,1) # y position 1
        y2=self.dcp(self.rng0,2,1) # y position 2

        ix1=self.find_nearest(self.x,x1) # x index 1
        ix2=self.find_nearest(self.x,x2) # x index 2

        iy1=self.find_nearest(self.y,y1) # y index 1
        iy2=self.find_nearest(self.y,y2) # y index 2

        # Set labels text
        self.labels[4].setText(self.format_dxy % (self.xdim, x2-x1, self.ydim, y2-y1))
        self.labels[5].setText(self.format_dzrc % (self.zdim, self.z[ix2,iy2]-self.z[ix1,iy1], ix2-ix1, iy2-iy1))

    # Update the positions of the side cursors
    def update_scsr(self,csr_num):
        self.scsrvs[csr_num-1].setValue(self.cursors[csr_num-1].data['pos'][0][0]) # Vertical
        self.scsrhs[csr_num-1].setValue(self.cursors[csr_num-1].data['pos'][0][1]) # Horizontal

    # Update cursor stats labels info (cursors)
    def updateinfo(self,csr_num):

        x=self.cursors[csr_num-1].data['pos'][0][0] # x position
        y=self.cursors[csr_num-1].data['pos'][0][1] # y position

        ix=self.find_nearest(self.x,x) # x index
        iy=self.find_nearest(self.y,y) # y index

        # Set label text
        self.labels[csr_num-1].setText(self.format_csr % (ix, iy, self.xdim, x, self.ydim, y, self.zdim, self.z[ix,iy]))

    # Update cursor stats labels info (leader cursors delta)
    def updatedelta(self):

        x1=self.cursors[0].data['pos'][0][0] # x position 1
        x2=self.cursors[1].data['pos'][0][0] # x position 2

        y1=self.cursors[0].data['pos'][0][1] # y position 1
        y2=self.cursors[1].data['pos'][0][1] # y position 2

        ix1=self.find_nearest(self.x,x1) # x index 1
        ix2=self.find_nearest(self.x,x2) # x index 2

        iy1=self.find_nearest(self.y,y1) # y index 1
        iy2=self.find_nearest(self.y,y2) # y index 2

        # Set labels text
        self.labels[4].setText(self.format_dxy % (self.xdim, x2-x1, self.ydim, y2-y1))
        self.labels[5].setText(self.format_dzrc % (self.zdim, self.z[ix2,iy2]-self.z[ix1,iy1], ix2-ix1, iy2-iy1))

    # Update the side MDC plots for a cursor
    def updateMDC(self,csr_num):

        # Remove current MDC plot
        self.MDCPlot.removeItem(self.MDCs[csr_num-1])

        # If the y-integration is zero plot the MDC at the crosshair y-position
        if self.ispany == 0:
            self.MDCs[csr_num-1]=self.MDCPlot.plot(self.x,self.z[:,self.find_nearest(self.y,self.cursors[csr_num-1].data['pos'][0][1])], pen=self.sidecs[csr_num-1])

        # If the y-integration is non-zero perform the integration
        else:

            # Calculate y-indexes of the integration-crosshair boundaries
            a=self.find_nearest(self.y,self.cursors[csr_num-1].data['pos'][0][1]-self.ispany)
            b=self.find_nearest(self.y,self.cursors[csr_num-1].data['pos'][0][1]+self.ispany)

            # If the indexes are equal plot the MDC at the crosshair y-position
            # (This happens if the integration value is smaller than the length of one data pixel)
            if a == b:
                self.MDCs[csr_num-1]=self.MDCPlot.plot(self.x,self.z[:,self.find_nearest(self.y,self.cursors[csr_num-1].data['pos'][0][1])], pen=self.sidecs[csr_num-1])

            # If the indexes are not equal average the MDCs within the integration-crosshair
            else:
                self.MDCs[csr_num-1]=self.MDCPlot.plot(self.x,np.sum(self.z[:,a:b+1],axis=1)/(b-a+1), pen=self.sidecs[csr_num-1])

    # Update the side EDC plots for a cursor
    def updateEDC(self,csr_num):

        # Remove current EDC plot
        self.EDCPlot.removeItem(self.EDCs[csr_num-1])

        # If the x-integration is zero plot the EDC at the crosshair x-position
        if self.ispanx == 0:
            self.EDCs[csr_num-1]=self.EDCPlot.plot(self.z[self.find_nearest(self.x,self.cursors[csr_num-1].data['pos'][0][0]),:],self.y, pen=self.sidecs[csr_num-1])

        # If the x-integration is non-zero perform the integration
        else:

            # Calculate x-indexes of the integration-crosshair boundaries
            a=self.find_nearest(self.x,self.cursors[csr_num-1].data['pos'][0][0]-self.ispanx)
            b=self.find_nearest(self.x,self.cursors[csr_num-1].data['pos'][0][0]+self.ispanx)

            # If the indexes are equal plot the EDC at the crosshair x-position
            # (This happens if the integration value is smaller than the length of one data pixel)
            if a == b:
                self.EDCs[csr_num-1]=self.EDCPlot.plot(self.z[self.find_nearest(self.x,self.cursors[csr_num-1].data['pos'][0][0]),:],self.y, pen=self.sidecs[csr_num-1])

            # If the indexes are not equal average the EDCs within the integration-crosshair
            else:
                self.EDCs[csr_num-1]=self.EDCPlot.plot(np.sum(self.z[a:b+1,:],axis=0)/(b-a+1),self.y, pen=self.sidecs[csr_num-1])

    # "Cursors follow me" feature
    def follow_core(self,rng):

        # Bring the cursors to fixed relative positions with respect to the provided range
        for i in range(4):
            if self.cursors[i] in self.MainPlot.getViewBox().allChildren():
                self.cursors[i].setData(pos=np.array([[self.dcp(rng,i+1,0),self.dcp(rng,i+1,1)]]), **self.dicts[i])

    # Bring the cursors to follow position if Followme is checked
    def follow(self):
        if self.Followme.isChecked():
            self.follow_core(self.MainPlot.viewRange())

    # Bring the cursors to follow position if a key is pressed
    # (This action also disables vertical and horizontal modes)
    def resetcsrs(self,key,evt):

        # Use key to trigger movement of cursors
        if evt.key() == key:
            self.follow_core(self.MainPlot.viewRange())

            # Disable vertical mode
            if self.h == False:

                self.h=True

                self.LabelCsrMode.setText("Cursor mode: Normal")

                # Enable horizontal cursor dragging
                for i,item in enumerate(self.cursors):
                    self.dicts[i]['h']=True
                    item.h=True

            # Disable horizontal mode
            if self.v == False:

                self.v=True

                self.LabelCsrMode.setText("Cursor mode: Normal")

                # Enable vertical cursor dragging
                for i,item in enumerate(self.cursors):
                    self.dicts[i]['v']=True
                    item.v=True

    # Link range:  MainPlot -> EDCPlot, MDCPlot (unilateral)
    def updatesides(self):

        self.MDCPlot.sigRangeChanged.disconnect(self.updatemainfromMDC)
        self.EDCPlot.sigRangeChanged.disconnect(self.updatemainfromEDC)

        rng=self.MainPlot.viewRange()
        if self.Followme.isChecked():
            self.follow_core(rng)

        self.MDCPlot.setXRange(rng[0][0],rng[0][1],padding=0)
        self.EDCPlot.setYRange(rng[1][0],rng[1][1],padding=0)

        self.MDCPlot.sigRangeChanged.connect(self.updatemainfromMDC)
        self.EDCPlot.sigRangeChanged.connect(self.updatemainfromEDC)

    # Link range:  MainPlot <-> EDCPlot, MDCPlot (bilateral)
    # Can also be achieved using:
    # MainPlotItem.setXLink(MDCPlotItem)
    # MainPlotItem.setYLink(EDCPlotItem)
    # But it's kinda buggy...
    def updatemainfromMDC(self):

        if self.Bilateral.isChecked():

            self.MainPlot.sigRangeChanged.disconnect(self.updatesides)

            rng=self.MDCPlot.viewRange()
            self.MainPlot.setXRange(rng[0][0],rng[0][1],padding=0)

            self.MainPlot.sigRangeChanged.connect(self.updatesides)

    def updatemainfromEDC(self):

        if self.Bilateral.isChecked():

            self.MainPlot.sigRangeChanged.disconnect(self.updatesides)

            rng=self.EDCPlot.viewRange()
            self.MainPlot.setYRange(rng[1][0],rng[1][1],padding=0)

            self.MainPlot.sigRangeChanged.connect(self.updatesides)

    # Set the range of MDCPlot and EDCPlot to MainPlot when Bilateral is checked
    def bilateral(self):

        if self.Bilateral.isChecked():

            self.MDCPlot.sigRangeChanged.disconnect(self.updatemainfromMDC)
            self.EDCPlot.sigRangeChanged.disconnect(self.updatemainfromEDC)

            rng=self.MainPlot.viewRange()
            self.MDCPlot.setXRange(rng[0][0],rng[0][1],padding=0)
            self.EDCPlot.setYRange(rng[1][0],rng[1][1],padding=0)

            self.MDCPlot.sigRangeChanged.connect(self.updatemainfromMDC)
            self.EDCPlot.sigRangeChanged.connect(self.updatemainfromEDC)

    ### Find all active objects
    def findall(self):

        # Initialize active objects lists
        self.act_csrs=[None,None,None,None]
        self.act_scsrvs=[None,None,None,None]
        self.act_scsrhs=[None,None,None,None]
        self.act_MDCs=[None,None,None,None]
        self.act_EDCs=[None,None,None,None]

        # Find cursors in MainPlot
        for i,item in enumerate(self.cursors):
            if item in self.MainPlot.getViewBox().allChildren():
                self.act_csrs[i]=item

        # Find side cursors in MDCPlot
        for i,item in enumerate(self.scsrvs):
            if item in self.MDCPlot.getViewBox().allChildren():
                self.act_scsrvs[i]=item

        # Find side cursors in EDCPlot
        for i,item in enumerate(self.scsrhs):
            if item in self.EDCPlot.getViewBox().allChildren():
                self.act_scsrhs[i]=item

        # Find side plots in MDCPlot
        for i,item in enumerate(self.MDCs):
            if item in self.MDCPlot.getViewBox().allChildren():
                self.act_MDCs[i]=item

        # Find side plots in EDCPlot
        for i,item in enumerate(self.EDCs):
            if item in self.EDCPlot.getViewBox().allChildren():
                self.act_EDCs[i]=item

    # Cursor arrow movements
    def arrowmove(self,csr_num,evt):

        if self.cursors[csr_num-1] in self.MainPlot.getViewBox().allChildren():

            # Move cursor data-row up
            if evt.key() == QtCore.Qt.Key.Key_Up and self.cursors[csr_num-1].data['pos'][0][1] + self.ispany < self.y_max:
                self.cursors[csr_num-1].setData(pos=np.array([[self.cursors[csr_num-1].data['pos'][0][0],self.y[self.find_nearest(self.y,self.cursors[csr_num-1].data['pos'][0][1])+1]]]), **self.dicts[csr_num-1])

            # Move cursor data-row down
            if evt.key() == QtCore.Qt.Key.Key_Down and self.cursors[csr_num-1].data['pos'][0][1] - self.ispany > self.y_min:
                self.cursors[csr_num-1].setData(pos=np.array([[self.cursors[csr_num-1].data['pos'][0][0],self.y[self.find_nearest(self.y,self.cursors[csr_num-1].data['pos'][0][1])-1]]]), **self.dicts[csr_num-1])

            # Move cursor data-column right
            if evt.key() == QtCore.Qt.Key.Key_Right and self.cursors[csr_num-1].data['pos'][0][0] + self.ispanx < self.x_max:
                self.cursors[csr_num-1].setData(pos=np.array([[self.x[self.find_nearest(self.x,self.cursors[csr_num-1].data['pos'][0][0])+1],self.cursors[csr_num-1].data['pos'][0][1]]]), **self.dicts[csr_num-1])

            # Move cursor data-column left
            if evt.key() == QtCore.Qt.Key.Key_Left and self.cursors[csr_num-1].data['pos'][0][0] - self.ispanx > self.x_min:
                self.cursors[csr_num-1].setData(pos=np.array([[self.x[self.find_nearest(self.x,self.cursors[csr_num-1].data['pos'][0][0])-1],self.cursors[csr_num-1].data['pos'][0][1]]]), **self.dicts[csr_num-1])

    # Cursor key pressed + arrow movements
    def karrowmove(self,csr_num,key,evt):

        # Use key to trigger movement of cursor
        if evt.key() == key and not evt.isAutoRepeat():

            def keyrelease(evt):

                if evt.key() == key:

                    # Disconnect cursor keyrelease
                    self.MainPlot.sigKeyRelease.disconnect(keyrelease)

                    # Disconnect cursor movement
                    self.MainPlot.sigKeyPress.disconnect(self.csrs_mov[csr_num-1])

                    # Reconnect cursor 1 (special, no key pressed)
                    self.MainPlot.sigKeyPress.connect(self.csrs_mov[0])

            # Disconnect cursor 1 (special, no key pressed)
            self.MainPlot.sigKeyPress.disconnect(self.csrs_mov[0])

            # Connect cursor movement
            self.MainPlot.sigKeyPress.connect(self.csrs_mov[csr_num-1])

            # Connect cursor keyrelease
            self.MainPlot.sigKeyRelease.connect(keyrelease)

    # All cursors key pressed + arrow movements
    def karrowall(self,key,evt):

        # Use key to trigger movement of cursors
        if evt.key() == key and not evt.isAutoRepeat():

            def keyrelease(evt):

                if evt.key() == key:

                    # Disconnect cursors keyrelease
                    self.MainPlot.sigKeyRelease.disconnect(keyrelease)

                    # Disconnect cursors movements
                    for i in self.csrs_mov:
                        if i == self.csrs_mov[0]:
                            pass
                        else:
                            self.MainPlot.sigKeyPress.disconnect(i)

            # Connect cursors movements
            for i in self.csrs_mov:
                if i == self.csrs_mov[0]:
                    pass
                else:
                    self.MainPlot.sigKeyPress.connect(i)

            # Connect cursors keyrelease
            self.MainPlot.sigKeyRelease.connect(keyrelease)

    # Hide all active objects
    # (Except objects created using whole range integration modes)
    # (See 'Integrate whole range' section)
    def hidecurrent(self,cn,key,evt):                     #----> Non-trivial cns

        # Trigger if key is pressed
        if evt.key() == key and not evt.isAutoRepeat():

            # Find all active objects
            self.findall()

            # Hide cursors in MainPlot
            for item in self.act_csrs:
                if item:
                    self.MainPlot.removeItem(item)

            # Hide side cursors in MDCPlot
            for item in self.act_scsrvs:
                if item:
                    self.MDCPlot.removeItem(item)

            # Hide side cursors in EDCPlot
            for item in self.act_scsrhs:
                if item:
                    self.EDCPlot.removeItem(item)

            # Hide side plots in MDCPlot
            for item in self.act_MDCs:
                if item:
                    self.MDCPlot.removeItem(item)

            # Hide side plots in EDCPlot
            for item in self.act_EDCs:
                if item:
                    self.EDCPlot.removeItem(item)

            # Disconnect the signals for cursor arrow movements
            self.GUI_disconnect(cn)

    # Show all active objects
    # (Except objects created using whole range integration modes)
    # (See 'Integrate whole range' section)
    def showcurrent(self,cn,key,evt):                     #----> Non-trivial cns

        # Trigger if key is pressed
        if evt.key() == key:

            # Show cursors in MainPlot
            for item in self.act_csrs:
                if item:
                    self.MainPlot.addItem(item)

            # Show side cursors in MDCPlot
            for item in self.act_scsrvs:
                if item:
                    self.MDCPlot.addItem(item)

            # Show side cursors in EDCPlot
            for item in self.act_scsrhs:
                if item:
                    self.EDCPlot.addItem(item)

            # Show side plots in MDCPlot
            for item in self.act_MDCs:
                if item:
                    self.MDCPlot.addItem(item)

            # Show side plots in EDCPlot
            for item in self.act_EDCs:
                if item:
                    self.EDCPlot.addItem(item)

            # Connect the signals for cursor arrow movements
            self.GUI_connect(cn)

            # Run immediate cursor follow function
            self.follow()

    # Hide/show a single cursor
    def showhide(self,csr_num,key,evt):

        # Trigger if key is pressed
        if evt.key() == key:

            # Hide cursor
            if self.cursors[csr_num-1] in self.MainPlot.getViewBox().allChildren():

                # Remove cursor in MainPlot
                self.MainPlot.removeItem(self.cursors[csr_num-1])

                # Remove cursor in MDCPlot and EDCPlot
                self.MDCPlot.removeItem(self.scsrvs[csr_num-1])
                self.EDCPlot.removeItem(self.scsrhs[csr_num-1])

                # Remove side sideplots
                self.MDCPlot.removeItem(self.MDCs[csr_num-1])
                self.EDCPlot.removeItem(self.EDCs[csr_num-1])

                # Remove cursor stats labels (hidden cursors only)
                if csr_num > 2:
                    self.CursorStats.removeItem(self.labels[csr_num-1])

            # Show cursor
            else:

                # Show cursor in MainPlot
                self.MainPlot.addItem(self.cursors[csr_num-1])

                # Set the integration-crosshair:
                # To apply the changes in the integration-crosshair when the cursor was hidden)
                # (Changes in csr.icsrc for example)
                self.cursors[csr_num-1].setICrosshair()

                # Refresh the integration-crosshair position
                self.cursors[csr_num-1].updateGraph()

                # Show cursor in MDCPlot and EDCPlot
                self.MDCPlot.addItem(self.scsrvs[csr_num-1])
                self.EDCPlot.addItem(self.scsrhs[csr_num-1])

                # The sideplots are automatically updated:
                # When the cursor is brought back self.csr.scatter.sigPlotChanged is triggered
                # (See updateMDC and updateEDC)

                # Add cursor stats labels (hidden cursors only)
                if csr_num > 2:
                    self.CursorStats.addItem(self.labels[csr_num-1],row=csr_num-1,col=0)

                    # Clear the new PyQt5.QtWidgets.QGraphicsRectItem object from the scene() in self.CursorStats.ci
                    # The new object is the first in self.CursorStats.items()
                    # (See discussion in GUI_files)
                    self.CursorStats.ci.scene().removeItem(self.CursorStats.items()[0])

                # Run immediate cursor follow function
                self.follow()

    # Hide/show MDCs for a cursor
    def hideshowMDCs(self,cn,csr_num,key,evt):            #----> Non-trivial cns

        # Trigger if key is pressed
        if evt.key() == key:

            # Hide show MDCs
            if self.MDCs[csr_num-1] in self.MDCPlot.getViewBox().allChildren():

                self.MDCPlot.removeItem(self.MDCs[csr_num-1])

                # Disconnect MDC updates with the cursors
                self.GUI_disconnect(cn)

            else:

                self.MDCPlot.addItem(self.MDCs[csr_num-1])

                # Connect MDC updates with the cursors
                self.GUI_connect(cn)

    # Hide/show EDCs for a cursor
    def hideshowEDCs(self,cn,csr_num,key,evt):            #----> Non-trivial cns

        # Trigger if key is pressed
        if evt.key() == key:

            # Hide show EDCs
            if self.EDCs[csr_num-1] in self.EDCPlot.getViewBox().allChildren():

                self.EDCPlot.removeItem(self.EDCs[csr_num-1])

                # Disconnect EDC updates with the cursors
                self.GUI_disconnect(cn)

            else:

                self.EDCPlot.addItem(self.EDCs[csr_num-1])

                # Connect EDC updates with the cursors
                self.GUI_connect(cn)

    # Vertical cursors line up
    def lineupv(self,key,evt):

        # Trigger if key is pressed
        if evt.key() == key:

            # Do not let lineupv act if lineuph is active
            if self.v == False:
                return

            if self.h == True:

                self.h=False

                self.LabelCsrMode.setText("Cursor mode: Vertical")

                # Disable horizontal cursor dragging
                for i,item in enumerate(self.cursors):
                    self.dicts[i]['h']=False
                    item.h=False

                # Find all objects
                self.findall()

                # If the self.C1 cursor is in MainPlot the remaining cursors line up vertical to it
                if self.cursors[0] in self.act_csrs:
                    for i,item in enumerate(self.act_csrs):
                        if item:
                            item.setData(pos=np.array([[self.cursors[0].pos[0][0],item.pos[0][1]]]),**self.dicts[i])

            else:

                self.h=True

                self.LabelCsrMode.setText("Cursor mode: Normal")

                # Enable horizontal cursor dragging
                for i,item in enumerate(self.cursors):
                    self.dicts[i]['h']=True
                    item.h=True

    # Horizontal cursors line up
    def lineuph(self,key,evt):

        # Trigger if key is pressed
        if evt.key() == key:

            # Do not let lineuph act if lineupv is active
            if self.h == False:
                return

            if self.v == True:

                self.v=False

                self.LabelCsrMode.setText("Cursor mode: Horizontal")

                # Disable vertical cursor dragging
                for i,item in enumerate(self.cursors):
                    self.dicts[i]['v']=False
                    item.v=False

                # Find all objects
                self.findall()

                # If the self.C1 cursor is in MainPlot the remaining cursors line up horizontal to it
                if self.cursors[0] in self.act_csrs:
                    for i,item in enumerate(self.act_csrs):
                        if item:
                            item.setData(pos=np.array([[item.pos[0][0],self.cursors[0].pos[0][1]]]),**self.dicts[i])

            else:

                self.v=True

                self.LabelCsrMode.setText("Cursor mode: Normal")

                # Enable vertical cursor dragging
                for i,item in enumerate(self.cursors):
                    self.dicts[i]['v']=True
                    item.v=True

    # Switch between none, vertical, horizontal and both crosshair states
    # Mode = 'csr' (crosshair)
    # Mode = 'icsr' (integration-crosshair)
    def switchcsr(self,csr_num,mode,key,evt):

        # Trigger if key is pressed
        if evt.key() == key:

            if self.cursors[csr_num-1] in self.MainPlot.getViewBox().allChildren():

                # Switch between the possible cursor states encoded in 'csrs'
                # (See pyqt_CursorItem.py)
                if self.dicts[csr_num-1][mode] == 2:
                    self.dicts[csr_num-1][mode]=1
                elif self.dicts[csr_num-1][mode] == 1:
                    self.dicts[csr_num-1][mode]=0
                elif self.dicts[csr_num-1][mode] == 0:
                    self.dicts[csr_num-1][mode]=[]
                elif self.dicts[csr_num-1][mode] == []:
                    self.dicts[csr_num-1][mode]=2

                # Update the crosshair state and set the crosshair
                if mode == 'csr':
                    self.cursors[csr_num-1].csr=self.dicts[csr_num-1][mode]
                    self.cursors[csr_num-1].setCrosshair()
                else:
                    self.cursors[csr_num-1].icsr=self.dicts[csr_num-1][mode]
                    self.cursors[csr_num-1].setICrosshair()

                # Refresh the crosshair position
                # (This is required because setCrosshair and setICrosshair reset the crosshair position to default)
                self.cursors[csr_num-1].updateGraph()

    # Make icsrc to have the Alpha in ibrush and back
    def fadeicsrc(self,key,evt):

        # Trigger if key is pressed
        if evt.key() == key:

            for i, item in enumerate(self.cursors):

                # Switch between having icsrc equal to ibrush or not
                # (See pyqt_CursorItem.py)
                if self.dicts[i]['icsrc'] == item.ibrush:
                    self.dicts[i]['icsrc']=item.ibrush[0:3]
                else:
                    self.dicts[i]['icsrc']=item.ibrush

                # Update the integration-crosshair state
                item.icsrc=self.dicts[i]['icsrc']

                # Refresh the cursor if in MainPlot
                if item in self.MainPlot.getViewBox().allChildren():

                    # Set the integration-crosshair
                    item.setICrosshair()

                    # Refresh the integration-crosshair position
                    item.updateGraph()

    # Switch between dimension units and pixels in cursor integration
    def dim2pix(self):

        # Dim -> Pix
        if self.DimorPix.isChecked():

            # Convert current steps to pixels
            csx=round(float(self.DeltaX.text())/self.x_delta)
            csy=round(float(self.DeltaY.text())/self.y_delta)

            # Convert current spans to pixels
            chx=round(self.SpanX.value()/self.x_delta)+1
            if chx > self.x_size:
                chx = self.x_size
            chy=round(self.SpanY.value()/self.y_delta)+1
            if chy > self.y_size:
                chy = self.y_size

            # Set the range for the spinboxes in pixels
            self.SpanX.setMinimum(1)
            self.SpanY.setMinimum(1)
            self.SpanX.setMaximum(self.x_size)
            self.SpanY.setMaximum(self.y_size)

            # Set the spinboxes digits after decimal point to zero
            self.SpanX.setDecimals(0)
            self.SpanY.setDecimals(0)

            # Set converted steps
            self.DeltaX.setText(('%i')%(csx))
            self.DeltaY.setText(('%i')%(csy))

            # Set spinboxes singlestep
            self.SpanX.setSingleStep(csx)
            self.SpanY.setSingleStep(csy)

            # Set converted spans
            self.SpanX.setValue(chx)
            self.SpanY.setValue(chy)

        # Pix -> Dim
        else:

            # Convert current steps to dimension units
            csx=int(self.DeltaX.text())*self.x_delta
            csy=int(self.DeltaY.text())*self.y_delta

            # Convert current spans to dimension units
            chx=(self.SpanX.value()-1)*self.x_delta
            chy=(self.SpanY.value()-1)*self.y_delta

            # Set the range for the spinboxes in dimension units
            self.SpanX.setMinimum(0)
            self.SpanY.setMinimum(0)
            self.SpanX.setMaximum(self.x_max-self.x_min)
            self.SpanY.setMaximum(self.y_max-self.y_min)

            # Set the user defined spinboxes digits after decimal point
            self.SpanX.setDecimals(self.spandec)
            self.SpanY.setDecimals(self.spandec)

            # Set converted steps
            self.DeltaX.setText(('%.'+str(self.spandec)+'f')%(csx))
            self.DeltaY.setText(('%.'+str(self.spandec)+'f')%(csy))

            # Set spinboxes singlestep
            self.SpanX.setSingleStep(csx)
            self.SpanY.setSingleStep(csy)

            # Set converted spans
            self.SpanX.setValue(chx)
            self.SpanY.setValue(chy)

    # Set single step change in self.SpanX
    def stepx(self):
        if self.DimorPix.isChecked():
            self.SpanX.setSingleStep(int(self.DeltaX.text()))
        else:
            self.SpanX.setSingleStep(float(self.DeltaX.text()))

    # Set single step change in self.SpanY
    def stepy(self):
        if self.DimorPix.isChecked():
            self.SpanY.setSingleStep(int(self.DeltaY.text()))
        else:
            self.SpanY.setSingleStep(float(self.DeltaY.text()))

    # Change ispanx of the integration-crosshair using the SpanX spinbox
    def changeix(self):

        for i, item in enumerate(self.cursors):

            # Update the integration-crosshair ispanx in DP.py
            if self.DimorPix.isChecked():
                self.ispanx=(self.SpanX.value()-1)*self.x_delta/2
            else:
                self.ispanx=self.SpanX.value()/2

            # Update the cursor dictionary
            self.dicts[i]['ispanx']=self.ispanx

            # Update the integration-crosshair ispanx
            item.ispanx=self.dicts[i]['ispanx']

            # Refresh the cursor if in MainPlot
            if item in self.MainPlot.getViewBox().allChildren():

                # Set the integration-crosshair
                item.setICrosshair()

                # Refresh the integration-crosshair position
                item.updateGraph()

    # Change ispany of the integration-crosshair using the SpanY spinbox
    def changeiy(self):

        for i, item in enumerate(self.cursors):

            # Update the integration-crosshair ispany in DP.py
            if self.DimorPix.isChecked():
                self.ispany=(self.SpanY.value()-1)*self.y_delta/2
            else:
                self.ispany=self.SpanY.value()/2

            # Update the cursor dictionary
            self.dicts[i]['ispany']=self.ispany

            # Update the integration-crosshair ispany
            item.ispany=self.dicts[i]['ispany']

            # Refresh the cursor if in MainPlot
            if item in self.MainPlot.getViewBox().allChildren():

                # Set the integration-crosshair
                item.setICrosshair()

                # Refresh the integration-crosshair position
                item.updateGraph()

    # Integrate whole x range
    def integrateallx(self,cn):                           #----> Non-trivial cns

        if self.allX.isChecked():

            self.tempEDC=[]
            self.tempispanx=[]

            # Remove and store current EDCs
            for i,item in enumerate(self.EDCs):

                if item in self.EDCPlot.getViewBox().allChildren():

                    self.EDCPlot.removeItem(item)
                    self.tempEDC.append(i)

            # Disconnect EDC updates with the cursors
            # Disconnect hide/show EDCs objects
            self.GUI_disconnect(cn)

            # Disable x integration-crosshairs
            self.tempispanx=self.SpanX.value()
            self.SpanX.setValue(0)
            self.SpanX.setEnabled(False)
            self.DeltaX.setEnabled(False)

            # Plot the integrated EDC covering the whole x range
            self.iX=self.EDCPlot.plot(np.sum(self.z[0:len(self.x),:],axis=0)/(len(self.x)),self.y, pen=self.peniEDC)

            # Create a linear region covering the whole x range over MainPlot
            self.iregionX=pg.LinearRegionItem(values=(self.x_min,self.x_max),orientation='vertical',brush=self.brushiX,pen=self.peniX,movable=False)
            self.MainPlot.addItem(self.iregionX)

        else:

            # Restore the EDCs
            for i in self.tempEDC:
                self.EDCPlot.addItem(self.EDCs[i])

            # Connect EDC updates with the cursors
            # Connect hide/show EDCs objects
            self.GUI_connect(cn)

            # Enable x integration-crosshairs
            self.SpanX.setValue(self.tempispanx)
            self.SpanX.setEnabled(True)
            self.DeltaX.setEnabled(True)

            # Remove the integrated EDC and the linear region
            self.EDCPlot.removeItem(self.iX)
            self.MainPlot.removeItem(self.iregionX)

    # Integrate whole y range
    def integrateally(self,cn):                           #----> Non-trivial cns

        if self.allY.isChecked():

            self.tempMDC=[]
            self.tempispany=[]

            # Remove and store current MDCs
            for i,item in enumerate(self.MDCs):

                if item in self.MDCPlot.getViewBox().allChildren():

                    self.MDCPlot.removeItem(item)
                    self.tempMDC.append(i)

            # Disconnect MDC updates with the cursors
            # Disconnect hide/show MDCs objects
            self.GUI_disconnect(cn)

            # Disable y integration-crosshairs
            self.tempispany=self.SpanY.value()
            self.SpanY.setValue(0)
            self.SpanY.setEnabled(False)
            self.DeltaY.setEnabled(False)

            # Plot the integrated MDC covering the whole y range
            self.iY=self.MDCPlot.plot(self.x,np.sum(self.z[:,0:len(self.y)],axis=1)/(len(self.y)), pen=self.peniMDC)

            # Create a linear region covering the whole y range over MainPlot
            self.iregionY=pg.LinearRegionItem(values=(self.y_min,self.y_max),orientation='horizontal',brush=self.brushiY,pen=self.peniY,movable=False)
            self.MainPlot.addItem(self.iregionY)

        else:

            # Restore the MDCs
            for i in self.tempMDC:
                self.MDCPlot.addItem(self.MDCs[i])

            # Connect MDC updates with the cursors
            # Connect hide/show MDCs objects
            self.GUI_connect(cn)

            # Enable y integration-crosshairs
            self.SpanY.setValue(self.tempispany)
            self.SpanY.setEnabled(True)
            self.DeltaY.setEnabled(True)

            # Remove the integrated MDC and the linear region
            self.MDCPlot.removeItem(self.iY)
            self.MainPlot.removeItem(self.iregionY)

    # Set the levels in the image according to the positions of HCutOff and LCutOff
    def contrast(self):
        self.Image.setLevels([(self.LCutOff.value()/100)*(self.z_max-self.z_min)+self.z_min,(self.HCutOff.value()/100)*(self.z_max-self.z_min)+self.z_min])

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
        self.colormap=self.get_mpl_colormap(self.Cmaps.currentText())

        # Apply selected colormap
        self.Image.setLookupTable(self.colormap)

    # Retrieve colormap from matplotlib
    def get_mpl_colormap(self,cmap):

        # Get the colormap from matplotlib
        colormap = cm.get_cmap(cmap)
        colormap._init()

        # Convert the matplotlib colormap from 0-1 to 0-255 for PyQt5
        # Ignore the last 3 rows of the colormap._lut:
        # colormap._lut has shape (N+3,4) where the first N rows are the RGBA color representations
        # The last 3 rows deal with the treatment of out-of-range and masked values
        # They need to be ignored to avoid the generation of artifacts possibly related to floating-point errors
        lut = (colormap._lut*255)[1:-3][:]

        return lut

################################################################################
