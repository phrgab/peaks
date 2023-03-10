# Fit Panel
# Tommaso Antonelli 29/10/2021


import sys

from PyQt6 import QtCore, QtGui, QtWidgets
from functools import partial

# Fit Panel panel import requirements (autorship)
from peaks.core.GUI.pyqt_ZoomItem import ZoomItem
from peaks.core.GUI.pyqt_ResizeItem import ResizeItem
# File loading import requirements (autorship)
import pyqtgraph as pg

# for fitting
from peaks.core.fit import *

class UiDialogCons(object):
    """
    dialog window for constrain
    """
    def setupUi(self, Dialog, model_params,param_key):

        Dialog.resize(260, 123)
        #old model parameters
        self.model_params = model_params
        #main grid layout
        self.gridDialog = QtWidgets.QGridLayout(Dialog)

        #label stating the name of the parameter
        self.labelparamName = QtWidgets.QLabel(Dialog)
        self.gridDialog.addWidget(self.labelparamName, 0, 0, 1, 2)

        #botton for save and close dialog window
        self.pushClose = QtWidgets.QPushButton(Dialog)
        self.pushClose.setText("Save and Close")
        self.gridDialog.addWidget(self.pushClose, 0, 2, 1, 2)

        #widget to set max and min values
        self.labelMin = QtWidgets.QLabel(Dialog)
        self.labelMin.setText('min =')
        self.gridDialog.addWidget(self.labelMin, 1, 0, 1, 1)
        self.lineEditmin = QtWidgets.QLineEdit(Dialog)
        self.gridDialog.addWidget(self.lineEditmin, 1, 1, 1, 1)
        self.labelMax = QtWidgets.QLabel(Dialog)
        self.labelMax.setText('max =')
        self.gridDialog.addWidget(self.labelMax, 1, 2, 1, 1)
        self.lineEditmax = QtWidgets.QLineEdit(Dialog)
        self.gridDialog.addWidget(self.lineEditmax, 1, 3, 1, 1)

        #check box to enable contrain
        self.checkConsDialog = QtWidgets.QCheckBox(Dialog)
        self.checkConsDialog.setText('constrain')
        self.checkConsDialog.setChecked(True)
        self.gridDialog.addWidget(self.checkConsDialog, 2, 3, 1, 1)
        #widget to edit the contrain expression
        self.lineEditcons = QtWidgets.QLineEdit(Dialog)
        self.gridDialog.addWidget(self.lineEditcons, 2, 0, 1, 3)
        # label to print error message
        self.labelmessage = QtWidgets.QLabel(Dialog)
        self.gridDialog.addWidget(self.labelmessage, 3, 0, 1, 4)


        ################ init = pass values to widgets
        self.labelmessage.setText('Valid')
        self.labelparamName.setText(param_key)
        self.lineEditmin.setText(str(self.model_params[param_key].min))
        self.lineEditmax.setText(str( self.model_params[param_key].max))
        self.lineEditcons.setText(str(self.model_params[param_key].expr).split("None")[0])
        #########  signals
        self.lineEditmin_sig = partial(self.change_cons, param_key, 'min', self.lineEditmin.textChanged)
        self.lineEditmax_sig = partial(self.change_cons, param_key, 'max', self.lineEditmax.textChanged)
        self.lineEditcons_sig = partial(self.change_cons, param_key, 'cons', self.lineEditcons.textChanged)
        self.pushClose_sig = partial(self.close, Dialog, self.pushClose.clicked)
        ######### connect
        self.lineEditmin.textChanged.connect(self.lineEditmin_sig)
        self.lineEditmax.textChanged.connect(self.lineEditmax_sig)
        self.lineEditcons.textChanged.connect(self.lineEditcons_sig)
        self.pushClose.clicked.connect(self.pushClose_sig)

    def close(self,Window,evt):
        #close the Window only if the expression is valid
        if self.labelmessage.text() == 'Valid':
            Window.accept()

    def change_cons(self, param_key, wid_key, evt):
        """
        change min, max and constrain expression of a specific parameter param_key
        """
        # change min and max
        if wid_key == 'min':
            self.model_params[param_key].min = float(self.lineEditmin.text())
        if wid_key == 'max':
            self.model_params[param_key].max = float(self.lineEditmax.text())

        #change constrian extrepssion
        if wid_key == 'cons':
            try:
                # space might generate bugs--> strip space from expression
                self.model_params[param_key].expr = self.lineEditcons.text().strip(' ')
                print(self.model_params[param_key])
                self.labelmessage.setText('Valid')
                print('valid expression for constrain enterd')
            except: # catch *all* exceptions
                #print error message in the dialog window
                error_messsage = sys.exc_info()[0]
                self.labelmessage.setText(str(error_messsage))
                #no constrain is None in lmfit module
                self.model_params[param_key].expr = None
                print('IGNORE ERROR MESSAGE, EVERYTHING IS FINE!!!')
                pass

    def get_newmodel(self,param_key):
        #if the checkbox is not enable set the expression as  None
        if not self.checkConsDialog.isChecked():
            self.model_params[param_key].expr = None
        return self.model_params[param_key].expr


class scrollArea(QtWidgets.QScrollArea):
    """
    scroll Area where to insert the parameter widgets
    include lables from a list passed as argument (label_list)
    """
    def __init__(self, label_list = ['label1']):
        super().__init__()

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(sizePolicy.hasHeightForWidth())

        #SCROLL AREA
        #self. is already a QtWidgets.QScrollArea
        self.setSizePolicy(sizePolicy)
        self.setWidgetResizable(True)
        self.scroll = QtWidgets.QWidget()
        self.scroll.setGeometry(QtCore.QRect(0, 0, 634, 163))
        self.setWidget(self.scroll)

        # grid layout inside scroll area
        self.grid = QtWidgets.QGridLayout(self.scroll)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(0)

        #Spacers
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.grid.addItem(spacerItem4, 10, 0, 1, 8)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.grid.addItem(spacerItem5, 0, 7, 9, 1)

        # Row Labels
        col = 0
        for text in label_list:
            label = labelRow(self.scroll)
            label.setText(text)
            self.grid.addWidget(label, col, 0, 1, 1)
            col+=1


class labelRow(QtWidgets.QLabel):
    """
    class for labels definig size properties
    """
    def __init__(self, parent):
        super().__init__(parent)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        self.setMinimumSize(QtCore.QSize(80, 15))
        self.setSizePolicy(sizePolicy)


class ParamWidgets(QtWidgets.QHBoxLayout):
    """
    class defining the group of  3 widgets for a single parameter: value , hold and constrain
    they are nicely packed in a HBoxLayout
    """

    def __init__(self, Enabled = True):
        super().__init__()

        #HBox size proepries
        self.setSpacing(6)
        self.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetDefaultConstraint)

        #Line edit for value
        self.lineValue = QtWidgets.QLineEdit()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineValue.sizePolicy().hasHeightForWidth())
        self.lineValue.setSizePolicy(sizePolicy)
        self.lineValue.setMinimumSize(QtCore.QSize(70, 0))
        self.lineValue.setEnabled(Enabled)
        self.lineValue.setMaxLength(12)
        self.lineValue.setFont(QtGui.QFont('Arial', pointSize = 10))
        self.lineValue.setValidator(QtGui.QDoubleValidator(notation=QtGui.QDoubleValidator.Notation.ScientificNotation))
        self.addWidget(self.lineValue)

        #CheckBox for Hold
        self.checkHold = QtWidgets.QCheckBox()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.checkHold.sizePolicy().hasHeightForWidth())
        self.checkHold.setSizePolicy(sizePolicy)
        self.checkHold.setText("")
        self.checkHold.setEnabled(Enabled)
        self.addWidget(self.checkHold)

        #CheckBox for Constrain
        self.checkCons = QtWidgets.QCheckBox()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.checkCons.sizePolicy().hasHeightForWidth())
        self.checkCons.setSizePolicy(sizePolicy)
        self.checkCons.setMinimumSize(QtCore.QSize(0, 0))
        self.checkCons.setText("")
        self.checkCons.setEnabled(Enabled)
        self.addWidget(self.checkCons)

class Fit_GUI(object):

    """
    main window of the Fit Panel
    """

    def __init__(self, MainWindow):

        # Store the framework of the app
        self.MainWindow = MainWindow
        self.mySetup()
        self.setupUi()

    ############################################################################

    def mySetup(self):
        # Set pyqtgraph background color
        pg.setConfigOptions(background='w')


    ############################################################################

    def setupUi(self):

        # main window of the app
        self.MainWindow.setObjectName("MainWindow")
        self.MainWindow.resize(656, 661)

        self.centralwidget = QtWidgets.QWidget(self.MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.maingrid = QtWidgets.QGridLayout(self.centralwidget)
        self.maingrid.setContentsMargins(10, 10, 10, 10)
        self.maingrid.setSpacing(0)
        self.maingrid.setObjectName("gridLayout")

        self.MainWindow.setCentralWidget(self.centralwidget)
        QtCore.QMetaObject.connectSlotsByName(self.MainWindow)


        #############################################################################

        # scroll area peak components
        self.scrollAreaPeak = scrollArea(label_list = ["prefix","type","color","center","sigma","amplitude","gamma",'fwhm', 'height'])
        self.maingrid.addWidget(self.scrollAreaPeak, 1, 0, 2, 4)


        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.maingrid.addItem(spacerItem2, 0, 2, 1, 2)

        ###############################################################
        #side scroll Area for special functions (background, fermi, convolution)
        self.sideLayout = QtWidgets.QGridLayout()
        self.maingrid.addLayout(self.sideLayout, 3, 3, 4, 1)

        self.sideLayout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetFixedSize)
        self.sideLayout.setSpacing(0)
        self.sideLayout.setObjectName("sideLayout")
        self.sideLayout.setColumnMinimumWidth(0, 10)
        self.sideLayout.setColumnMinimumWidth(1, 10)

        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.sideLayout.addItem(spacerItem3, 2, 0, 1, 1)

        #####################TABWidget For SPECIALS

        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.sideLayout.addWidget(self.tabWidget, 0, 0, 1, 2)

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy)
        self.tabWidget.setObjectName("tabWidget")

        ##########bkg TAB
        self.tab_bkg = QtWidgets.QWidget()
        self.tab_bkg.setEnabled(True)
        self.tab_bkg.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        tabgrid = QtWidgets.QGridLayout(self.tab_bkg)
        self.tabWidget.addTab(self.tab_bkg,'bkg')
        tabgrid.setContentsMargins(0, 0, 0, 0)

        self.scrollAreaBkg = scrollArea(label_list = ['label', 'type', 'color','slope','intercept'])
        tabgrid.addWidget(self.scrollAreaBkg, 0, 0, 1, 1)


        ##########Fermi TAB
        self.tab_fermi = QtWidgets.QWidget()
        self.tab_fermi.setEnabled(True)
        self.tab_fermi.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        tabgrid = QtWidgets.QGridLayout(self.tab_fermi)
        self.tabWidget.addTab(self.tab_fermi,'fermi')
        tabgrid.setContentsMargins(0, 0, 0, 0)

        self.scrollAreaFermi = scrollArea(label_list = ['label', 'type', 'color'])
        tabgrid.addWidget(self.scrollAreaFermi, 0, 0, 1, 1)

        ##########convolution TAB
        self.tab_conv = QtWidgets.QWidget()
        self.tab_conv.setEnabled(True)
        self.tab_conv.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        tabgrid = QtWidgets.QGridLayout(self.tab_conv)
        self.tabWidget.addTab(self.tab_conv,'convolution')
        tabgrid.setContentsMargins(0, 0, 0, 0)

        self.scrollAreaConv = scrollArea(label_list = ['label', 'type', 'color', 'sigma_gauss'])
        tabgrid.addWidget(self.scrollAreaConv, 0, 0, 1, 1)

        ##########################################################################################
        # add peak button
        self.pushAddPeak = QtWidgets.QPushButton(self.centralwidget)
        sizePolicyButton = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicyButton.setHorizontalStretch(0)
        sizePolicyButton.setVerticalStretch(0)
        sizePolicyButton.setHeightForWidth(self.pushAddPeak.sizePolicy().hasHeightForWidth())
        self.pushAddPeak.setSizePolicy(sizePolicyButton)
        self.pushAddPeak.setText('Add Peak')
        self.maingrid.addWidget(self.pushAddPeak, 0, 0, 1, 1)

        # AutoGuess button
        self.pushAutoGuess = QtWidgets.QPushButton(self.centralwidget)
        sizePolicyButton.setHeightForWidth(self.pushAutoGuess.sizePolicy().hasHeightForWidth())
        self.pushAutoGuess.setSizePolicy(sizePolicyButton)
        self.pushAutoGuess.setText('Auto Guess')
        self.maingrid.addWidget(self.pushAutoGuess, 0, 1, 1, 1)

        # Run fit button
        self.pushRunFit = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushRunFit.sizePolicy().hasHeightForWidth())
        self.pushRunFit.setSizePolicy(sizePolicy)
        self.pushRunFit.setText('Run Fit')
        self.sideLayout.addWidget(self.pushRunFit, 2, 0, 1, 1)

        # Save fit button
        self.pushSaveFit = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushSaveFit.sizePolicy().hasHeightForWidth())
        self.pushSaveFit.setSizePolicy(sizePolicy)
        self.pushSaveFit.setText('Save')
        self.sideLayout.addWidget(self.pushSaveFit, 2, 1, 1, 1)

        # reset params button
        self.pushResetParams = QtWidgets.QPushButton(self.centralwidget)
        sizePolicyButton = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicyButton.setHorizontalStretch(0)
        sizePolicyButton.setVerticalStretch(0)
        sizePolicyButton.setHeightForWidth(self.pushResetParams.sizePolicy().hasHeightForWidth())
        self.pushResetParams.setSizePolicy(sizePolicyButton)
        self.pushResetParams.setText('ResetParams')
        self.sideLayout.addWidget(self.pushResetParams, 3, 0, 1, 1)

        # plot widget for DC
        self.graphicsDC = pg.PlotWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(4)
        sizePolicy.setHeightForWidth(self.graphicsDC.sizePolicy().hasHeightForWidth())
        self.graphicsDC.setSizePolicy(sizePolicy)
        self.graphicsDC.setObjectName("graphicsDC")
        self.graphicsDC.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))
        self.maingrid.addWidget(self.graphicsDC, 6, 0, 1, 3)
        # plot widget for Residue
        self.graphicsRes = pg.PlotWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.graphicsRes.sizePolicy().hasHeightForWidth())
        self.graphicsRes.setSizePolicy(sizePolicy)
        self.graphicsRes.setObjectName("graphicsRes")
        self.maingrid.addWidget(self.graphicsRes, 3, 0, 1, 3)



        ###############################################################

        QtCore.QMetaObject.connectSlotsByName(self.MainWindow)

    ############################################################################

    def GUI_resize(self):

        ResizeItem(self.MainWindow,self.graphicsDC)
        ResizeItem(self.MainWindow,self.pushRunFit)
        ResizeItem(self.MainWindow,self.pushSaveFit)

    ############################################################################

    def GUI_internal(self):
        self.ZoomMain=ZoomItem(self.graphicsDC)

    ############################################################################
    # Border between static elements and dynamic elements
    ############################################################################


    # Retrieve input files
    def GUI_files(self,files, model_tot = FitFunc()):

        # Define the Display Panel data from the first input file
        self.GUI_setdata(files)

        self.GUI_internal()
        # Construct the Display Panel GUI initial content
        self.GUI_initial()
        # Construct the Display Panel GUI initial content
        self.GUI_LoadModel(model_tot)

    ############################################################################

    def GUI_setdata(self,files):
        #check if files is a 1D xarray with only 1 dimension
        try:
            files.dims
        except:
            print("data are not a xarray, please define your data as xarray with 1 dimension")

        if len(files.dims) == 1:
            self.x=files[files.dims[0]]
            self.y=files

            self.x_min=min(self.x)
            self.x_max=max(self.x)
            self.y_min=min(self.y)
            self.y_max=max(self.y)
        else:
            print('data passed to fit panel has more than 1 dimension, please pass only 1D xarray')

    ##########################################################################################

    def GUI_initial(self):

        self.space=[[self.x_min,self.y_min],[self.x_max,self.y_max]]
        self.ZoomMain.setSpace(self.space)


        #initialise dictionaries
        #dictionary of dictionaries storing the widgets for the parameters (both peak components and specials)
        self.wid={}
        #dictionary storing the FitFunc instances
        self.mod = {}
        #signals
        self.sig = {}
        #connections
        self.con = {}
        # plots of the components
        self.plots = {}

        #plot raw data
        self.plots['raw'] = self.graphicsDC.plot(self.x,self.y,pen=None ,symbol='o', symbolSize=3, symbolBrush='k')
        self.graphicsDC.disableAutoRange()

        #init first component p1
        self.init_model('p1', func_type='Lorentzian')
        self.create_wid('p1', self.scrollAreaPeak.grid)
        self.update_wid('p1')

        #init background
        self.init_model('Bkg', func_type='Linear')
        self.create_wid('Bkg', self.scrollAreaBkg.grid)
        self.update_wid('Bkg')

        #init Fermi function
        self.init_model('Fermi', func_type = 'None')
        self.create_wid('Fermi', self.scrollAreaFermi.grid)
        self.update_wid('Fermi')

        #init Gaussian Convolution
        self.init_model('Conv', func_type = 'Gauss_convolution')
        self.create_wid('Conv', self.scrollAreaConv.grid)
        self.update_wid('Conv')

        #compose the Total Model
        self.compose_modelTot()

        # crate signals and connection for components
        self.create_sig('p1')
        self.con['p1']={}
        self.connect_sig('p1')
        self.create_sig('Bkg')
        self.con['Bkg']={}
        self.connect_sig('Bkg')
        self.create_sig('Fermi')
        self.con['Fermi']={}
        self.connect_sig('Fermi')

        self.create_sig('Conv')
        self.con['Conv']={}
        self.connect_sig('Conv')

        # first the Bkg because needs to be evaluate
        self.refreshPlotDC('Bkg')
        self.refreshPlotDC('p1')
        self.refreshPlotRes()

        #crate connections for static widgets like
        #Addpeak, RunFit, SaveFit... buttons
        self.connect_static_wid()

    def GUI_LoadModel(self, model_tot = FitFunc()):
        for prefix in model_tot.components.keys():
            #if it is already a component in the fit panel
            print('comp', model_tot.components[prefix].model)
            if self.mod.get(prefix):
                if self.mod[prefix].func_type != 'None':
                    #over ride the model and change the func type ( it will update params and plots automatically)
                    self.mod[prefix] = model_tot.components[prefix]
                    print(self.mod[prefix].func_type)
                    self.change_type(prefix)

                else:
                    #self.wid[prefix]['type'].setCurrentText(model_tot.components[prefix].func_type)
                    self.mod[prefix] = model_tot.components[prefix]
                    self.update_wid(prefix)

            #if the components doesn't exist yet
            else:
                #create a new peak and load the new params
                self.mod[prefix] = model_tot.components[prefix]
                self.AddPeak()





    ############################################################################
    # SIGNALS AND CONNECTIONS
    ############################################################################


    def connect_static_wid(self):
        """
        connection for static widgets = bottons only (not for widgets related to components)
        """
        # pushRunFit
        self.pushRunFit.clicked.connect(self.RunFit)

        # pushSaveFit
        self.pushSaveFit.clicked.connect(self.SaveFit)

        # pushAddPeak
        self.pushAddPeak.clicked.connect(self.AddPeak)

        # pushAutoGuess
        #self.pushAutoGuess.clicked.connect(AutoGuess)

        # pushLoadModel
        self.pushResetParams.clicked.connect(self.ResetParams)


    def connect_sig(self, prefix):
        for wid_key in self.wid[prefix].keys():
            #these are the comboboxes
            if wid_key in ['type']:
                if not self.con[prefix].get(wid_key): #True when self.con[prefix].get(wid_key) = None
                    self.wid[prefix][wid_key].currentTextChanged.connect(self.sig[prefix][wid_key])
                    self.con[prefix][wid_key]=True

            #these are the pushColor
            if wid_key in ['color']:
                if not self.con[prefix].get(wid_key): #True when self.con[prefix].get(wid_key) = None
                    self.wid[prefix][wid_key].clicked.connect(self.sig[prefix][wid_key])
                    self.con[prefix][wid_key]=True

            #these are the lineedits:
            if '_val' in wid_key:
                if not self.con[prefix].get(wid_key): #True when self.con[prefix].get(wid_key) = None
                    self.wid[prefix][wid_key].textChanged.connect(self.sig[prefix][wid_key])
                    self.con[prefix][wid_key]=True

            #these are the checkboxes:
            if '_hold' in wid_key:
                if not self.con[prefix].get(wid_key): #True when self.con[prefix].get(wid_key) = None
                    self.wid[prefix][wid_key].stateChanged.connect(self.sig[prefix][wid_key])
                    self.con[prefix][wid_key]=True

             #these are the checkboxes:
            if '_cons' in wid_key:
                if not self.con[prefix].get(wid_key): #True when self.con[prefix].get(wid_key) = None
                    self.wid[prefix][wid_key].clicked.connect(self.sig[prefix][wid_key])
                    self.con[prefix][wid_key]=True



    def diconnect_sig(self, prefix):
        for wid_key in self.wid[prefix].keys():
            #these are the comboboxes
            if wid_key in ['type']:
                if self.con[prefix].get(wid_key): #True when self.con[prefix].get(wid_key) = None
                    self.wid[prefix][wid_key].currentTextChanged.disconnect(self.sig[prefix][wid_key])
                    self.con[prefix].pop(wid_key)

            #these are the pushColor
            if wid_key in ['color']:
                if self.con[prefix].get(wid_key): #True when self.con[prefix].get(wid_key) = None
                    self.wid[prefix][wid_key].clicked.disconnect(self.sig[prefix][wid_key])
                    self.con[prefix].pop(wid_key)

            #these are the lineedits:
            if '_val' in wid_key:
                if self.con[prefix].get(wid_key): #True when self.con[prefix].get(wid_key) = None
                    self.wid[prefix][wid_key].textChanged.disconnect(self.sig[prefix][wid_key])
                    self.con[prefix].pop(wid_key)

            #these are the checkboxes:
            if '_hold' in wid_key:
                if self.con[prefix].get(wid_key): #True when self.con[prefix].get(wid_key) = None
                    self.wid[prefix][wid_key].stateChanged.disconnect(self.sig[prefix][wid_key])
                    self.con[prefix].pop(wid_key)

            #these are the checkboxes:
            if '_cons' in wid_key:
                if self.con[prefix].get(wid_key): #True when self.con[prefix].get(wid_key) = None
                    self.wid[prefix][wid_key].clicked.disconnect(self.sig[prefix][wid_key])
                    self.con[prefix].pop(wid_key)



    def create_sig(self,prefix):
        # if self.sig[prefix] doesn't exist yet create an empty one
        if self.sig.get(prefix) is None:
            self.sig[prefix]={}
        for wid_key in self.wid[prefix].keys():
            #check if the signal already exist
            if not self.sig[prefix].get(wid_key): #true when get returns None
                #these are the comboboxes
                if wid_key in ['type']:
                    self.sig[prefix][wid_key] = lambda: self.change_type(prefix)

                #these are the pushColor
                if wid_key in ['color']:
                    self.sig[prefix][wid_key] = partial(self.change_color,prefix,self.wid[prefix][wid_key].clicked)

                #these are the lineedits:
                if '_val' in wid_key:
                    self.sig[prefix][wid_key] = partial(self.change_param, prefix, wid_key,self.wid[prefix][wid_key].textChanged)

                #these are the hold checkboxes:
                if '_hold' in wid_key:
                    self.sig[prefix][wid_key] = partial(self.change_hold,prefix,wid_key,self.wid[prefix][wid_key].stateChanged)

                #these are the cons checkboxes:
                if '_cons' in wid_key:
                    self.sig[prefix][wid_key] = partial(self.open_dialog_cons,prefix,wid_key,self.wid[prefix][wid_key].clicked)

    def open_dialog_cons(self,prefix,wid_key,evt):
        """
        open a dialog to insert constrain expression and min/max values of a parameter
        """
        #extract the exact parameter key (name) from wid_key and prefix
        param_key = prefix+wid_key.split('_cons')[0]

        #init and show the dialog window
        Dialog = QtWidgets.QDialog()
        ui = UiDialogCons()
        ui.setupUi(Dialog,self.mod['Tot'].params,param_key)
        Dialog.show()

        # when the dialog window has been closed:
        if Dialog.exec():
            #return the new expression to the model tot only if it is  a valid expression (see UiDialogCons methonds)
            if ui.labelmessage.text() == 'Valid':
                self.mod["Tot"].params[param_key].expr = ui.get_newmodel(param_key)

            #return min and max
            self.mod["Tot"].params[param_key].min = ui.model_params[param_key].min
            self.mod[prefix].params[param_key].min = ui.model_params[param_key].min
            self.mod["Tot"].params[param_key].max = ui.model_params[param_key].max
            self.mod[prefix].params[param_key].max = ui.model_params[param_key].max

            #refresh the plot...some parameters might be changed
            self.refreshPlotDC('Tot')
            self.refreshPlotRes()

            #set the checkbox in th Main widow as the checkbox in the dialog one
            self.wid[prefix][wid_key].setChecked(ui.checkConsDialog.isChecked())
            #but if the expression is None e.g. is not valid or empty set as false
            # be careful that an empty expression is a valid expression but it do not generate any constrain
            if self.mod["Tot"].params[param_key].expr is None:
                self.wid[prefix][wid_key].setChecked(False)
                self.mod[prefix].params[param_key].expr = None

        #if closed without clicking the close and save botton
        # return nothing and set the constain checkbox in the main window as False
        else:
            self.wid[prefix][wid_key].setChecked(False)


    def change_hold(self,prefix,wid_key,evt):
        # update hold/unhold
        self.update_param(prefix,wid_key)

    def change_param(self,prefix,wid_key,evt):

        # change a parameter value or constrain and refresh the plots accordingly

        self.update_param(prefix,wid_key)
        # if Shirely you need to recompose all the model because the bkg is changed
        if self.mod[prefix].func_type == 'Shirley':
            self.compose_modelTot()
        self.refreshPlotDC(prefix)
        self.refreshPlotRes()



    def change_type(self,prefix):
        print('change type', prefix)
        #change type of a fit function component
        #recreate the correct parameters,widgets,signals and connections
        #recompose the model tot and finally refresh the plots

        #save parameters and color from old function
        old_color = self.mod[prefix].color
        old_params = self.mod[prefix].params

        #disconnect all the signals of the component
        self.diconnect_sig(prefix)

        # based on the prefix identify the layout where to destroy and create the parameters
        if prefix =='Bkg':
            Layout = self.scrollAreaBkg
        elif prefix =='Fermi':
            Layout = self.scrollAreaFermi
        elif prefix =='Conv':
            Layout = self.scrollAreaConv
        elif prefix.startswith('p'):
            Layout = self.scrollAreaPeak
            #extrac the correct colum from prefix
            col = int(prefix.strip('p'))

        #remove all widgets for all the parameters and if it a special component also the parameter labels
        for irow,param_key in enumerate(self.mod[prefix].params.keys()):
            self.remove_wid(prefix,param_key, Layout.grid,irow)


        #create new model with the new func_type
        self.init_model(prefix, func_type=str(self.wid[prefix]['type'].currentText()),
                        color = old_color)

        #if the component is a Peak
        if prefix.startswith('p'):
            #new parameters have to be ordered as ["center","sigma","amplitude","gamma",'fwhm', 'height']
            for irow,wid_key in enumerate(["center","sigma","amplitude","gamma",'fwhm', 'height']):
                #if it is a parameter of the model
                if prefix+wid_key in self.mod[prefix].params.keys():
                    # and if the widget doesn't exist yet
                    wid_key_val = wid_key+'_val'
                    if wid_key_val not in self.wid[prefix].keys():
                        #create the widgets
                        param_key = prefix+wid_key
                        newpwid = ParamWidgets(Enabled = True)
                        self.wid[prefix][wid_key+'_val'] = newpwid.lineValue
                        self.wid[prefix][wid_key+'_hold'] = newpwid.checkHold
                        self.wid[prefix][wid_key+'_cons'] = newpwid.checkCons
                        Layout.grid.addLayout(newpwid, 3+irow, col, 1, 1)

        #if it is a 'special' component create a widget for all the params
        else:
            irow = 0
            for param_key in self.mod[prefix].params.keys():

                wid_key = param_key.split(prefix)[1]
                # and if the widget doesn't exist yet
                wid_key_val = wid_key+'_val'
                if wid_key_val not in self.wid[prefix].keys():
                    #create the widget
                    pwid = ParamWidgets(Enabled = True)
                    self.wid[prefix][wid_key+'_val'] = pwid.lineValue
                    self.wid[prefix][wid_key+'_hold'] = pwid.checkHold
                    self.wid[prefix][wid_key+'_cons'] = pwid.checkCons
                    Layout.grid.addLayout(pwid, 3+irow, 1, 1, 1)

                    #create label
                    label = labelRow(Layout)
                    label.setText(param_key)
                    Layout.grid.addWidget(label, 3+irow, 0, 1, 1)
                irow += 1

        #if possible re-use the same param value of the old component
        for param_key in self.mod[prefix].params.keys():
            if  param_key in old_params.keys():
                self.mod[prefix].params[param_key].value = old_params[param_key].value


        self.update_wid(prefix)
        #re-connect all the widgets of component prefix
        self.create_sig(prefix)
        self.connect_sig(prefix)
        #re-compose model Tot
        self.compose_modelTot()
        self.update_all_params(prefix)
        self.refreshPlotDC(prefix)
        self.refreshPlotRes()




    def change_color(self,prefix,evt):

        #chage color of the component

        #open color dialog window
        color = QtWidgets.QColorDialog.getColor()
        self.mod[prefix].color = color.name()
        self.wid[prefix]['color'].setStyleSheet('QPushButton {margin: 8px ; background-color:' + str(color.name())+'}')
        self.refreshPlotDC(prefix)

    def refreshPlotRes(self):
        #refresh residue plot
        if self.plots.get('Res') is not None:
            self.plots['Res'].clear()

        self.plots['Res'] = self.graphicsRes.plot(self.x,self.y-self.mod['Tot'].eval,
                                                       pen='r')

    def refreshPlotDC(self, prefix):
        # refresh main plot

        #check if the component is already plotted. if yes clear the plot
        if self.plots.get(prefix) is not None:
            self.plots[prefix].clear()
        #check if the model tot is already plotted. if yes clear the plot
        if self.plots.get('Tot') is not None:
            self.plots['Tot'].clear()

        #if you need to refresh the bkg you need to refresh all the other components too!!!
        if prefix in ['Bkg']:
            #check if it is an actual function
            if self.mod['Bkg'].model is not None:

                #evaluate the bkg (needed specially in case of Shirley)
                self.mod['Bkg'].evaluate(x=self.x, y=self.y)
                self.plots['Bkg'] = self.graphicsDC.plot(self.x,self.mod['Bkg'].eval, pen = pg.mkPen(self.mod['Bkg'].color, width=3))
                #update all the components too
                for prefix2 in self.mod.keys():
                    if prefix2.startswith('p'):
                        if self.mod[prefix2].model is not None:
                            self.mod[prefix2].evaluate(x=self.x,y=self.y)
                            if self.plots.get(prefix2) is not None:
                                    self.plots[prefix2].clear()
                            self.plots[prefix2] = self.graphicsDC.plot(self.x,self.mod[prefix2].eval+self.mod['Bkg'].eval,
                                                               pen=pg.mkPen(self.mod[prefix2].color, width=3))

        # if it is a component
        if prefix.startswith('p'):
            #check if it is an actual function
            if self.mod[prefix].model is not None:
                self.mod[prefix].evaluate(x=self.x,y=self.y)
                self.plots[prefix] = self.graphicsDC.plot(self.x,self.mod[prefix].eval+self.mod['Bkg'].eval,
                                                           pen=pg.mkPen(self.mod[prefix].color, width=3))

        # re-plot model tot (in case of Fermi you need to refresh only the model tot)
        if self.mod['Tot'].model is not None:
            self.mod['Tot'].evaluate(x=self.x,y=self.y)
            self.plots['Tot'] = self.graphicsDC.plot(self.x,self.mod['Tot'].eval,
                                                       pen=pg.mkPen('r', width=3))

    def compose_modelTot(self):
        """
        compose the total model for the fitting as
        ((sum(peak_components)+background)*Fermi)convolution with gaussian
        """
        #put peak componets in a list
        peak_comp = []
        for prefix in self.mod.keys():
            if prefix.startswith('p'):
                peak_comp.append(self.mod[prefix])

        #init model TOT (see frontend_fit_GUI code)
        self.mod['Tot'] = compose_model(y = self.y,
                                        components = peak_comp,
                                        background = self.mod['Bkg'],
                                        Fermi = self.mod['Fermi'],
                                        Convolution = self.mod['Conv'])


    def init_model(self, prefix, func_type=None, color = '#70a9e7'):
        """
        create a model from the function type chosen in the qt widget
        Args:
            prefix: prefix of the component (model)
            func_type : if it is provided create a fit function this that specific function
        """
        #if funct_type not specified override funct_type with the one in the widget
        if func_type is None:
            func_type = str(self.wid[prefix]['type'].currentText())

        #(see frontend_fit_GUI code)
        self.mod[prefix] = FitFunc(func_type=func_type,
                                    prefix=prefix,
                                   color = color)

        #create model+params (see frontend_fit_GUI code)
        self.mod[prefix].create_model()

    def update_param_from_modelTot(self):
        #update all the paramters of the components from the model Tot
        #only the values are updated (!!!!!do not update the constaints!!!!!)
        for param_key in self.mod['Tot'].params.keys():
            for prefix in self.mod.keys():
                if prefix in param_key:
                    self.mod[prefix].params[param_key].value = self.mod['Tot'].params[param_key].value



    def update_param(self,prefix,wid_key):
        #update a single lmfit.parameter from the setting choosen in the Qtwidget (wid_key)

        #extract the param name from prefix and wid_key
        param_key = prefix+wid_key.split("_")[0]

        # update_constraints() is a lmfit internal method that update all the values of the model
        # considering also the dependece form the constaint expressions
        self.mod['Tot'].params.update_constraints()
        self.update_param_from_modelTot()

        #is the param constrained?
        constrain = None
        try:
            constrain = self.mod['Tot'].params[param_key].expr
        except:
            pass

        # if the param is constrained
        if constrain is not None:
            # override the change with the value returned by the expression in the model tot
            self.mod[prefix].params[param_key].value = self.mod['Tot'].params[param_key].value
            #update the widget with the value returned by the expression
            wid_key_val=wid_key.split("_")[0]+"_val"
            try:
                self.wid[prefix][wid_key_val].setText(str(self.mod[prefix].params[param_key].value))
            except:
                pass

        # if not constrained update also the model "Tot"
        elif "_val" in wid_key:
            try:
                self.mod[prefix].params[param_key].value = float(self.wid[prefix][wid_key].text())
                self.mod['Tot'].params[param_key].value = float(self.wid[prefix][wid_key].text())
            except:
                pass
        elif "_hold" in wid_key:
            self.mod[prefix].params[param_key].vary = not(self.wid[prefix][wid_key].isChecked())
            try:
                self.mod['Tot'].params[param_key].vary = not(self.wid[prefix][wid_key].isChecked())
            except:
                pass

        # re-define the color fo the component
        if "color" in wid_key:
            self.mod[prefix].color = self.wid[prefix][wid_key].currentText()

        # if Shirely you need to recompose the model TOT because the bkg is changed
        if self.mod[prefix].func_type == 'Shirley':
            self.compose_modelTot()

    def update_all_params(self, prefix):
        """
        update all the lmfit.parameters of a specific component from the widgets
        """
        #sanity check the function type is the same of the widget
        if self.mod[prefix].func_type != self.wid[prefix]['type'].currentText():
            print('model and widget have different funct_type')
        else:
            #update everything but not type, color and prefix
            for wid_key in self.wid[prefix].keys():
                if wid_key not in ['color','prefix','type']:
                    self.update_param(prefix,wid_key)


    def update_wid(self, prefix):
        """
        update all the widgets related to a specific component (model) taking the values from the lmfit.models
        Args:
            prefix: prefix of the component (model)
        """
        model = self.mod[prefix]
        #check model is not a None model

        if model.func_type !='None':
            #check passed: start update wid
            self.wid[prefix]['prefix'].setText(model.prefix)
            self.wid[prefix]['type'].setCurrentText(model.func_type)
            self.wid[prefix]['color'].setStyleSheet('QPushButton {margin: 8px ; background-color:' + str(model.color)+'}')

            #update the widget for value Hold and constaints
            for param_key in model.params.keys():
                wid_key = param_key.split(prefix)[1]+'_val'
                self.wid[prefix][wid_key].setText(str(model.params[param_key].value))
                wid_key = param_key.split(prefix)[1]+'_hold'
                self.wid[prefix][wid_key].setChecked(not(model.params[param_key].vary))
                wid_key = param_key.split(prefix)[1]+'_cons'
                if model.cons.get(param_key) is None:
                    self.wid[prefix][wid_key].setChecked(False)
                else:
                    self.wid[prefix][wid_key].setChecked(True)


    def remove_wid(self, prefix, param_key, Layout,irow):
        #remove the 3 wigets for value, hold and constrain of a single parameter

        key = param_key.split(prefix)[1]
        #if it is a peak component
        if prefix.startswith('p'):
            #extract component number from prefix
            col = int(''.join(filter(str.isdigit, prefix)))
            #remove widget for parameters following a certain order
            wid_row=key.split('_')[0]
            for irow, row in enumerate(["center","sigma","amplitude","gamma",'fwhm', 'height']):
                if wid_row == row:

                    Layout.itemAtPosition(3+irow, col).layout().itemAt(0).widget().deleteLater()
                    self.wid[prefix].pop(wid_row+'_val')
                    Layout.itemAtPosition(3+irow, col).layout().itemAt(1).widget().deleteLater()
                    self.wid[prefix].pop(wid_row+'_hold')
                    Layout.itemAtPosition(3+irow, col).layout().itemAt(2).widget().deleteLater()
                    self.wid[prefix].pop(wid_row+'_cons')
                    #delete the complete Hbox layout
                    Layout.itemAtPosition(3+irow, col).layout().deleteLater()

        #if instead it is a 'special' component
        else:
            #remove all the widgets related to that parameter
            wid_row=key.split('_')[0]
            #sanity check if there is at least the widget for the value to remove
            if self.wid[prefix].get(wid_row+'_val'):
                Layout.itemAtPosition(3+irow, 1).layout().itemAt(0).widget().deleteLater()
                self.wid[prefix].pop(wid_row+'_val')
                Layout.itemAtPosition(3+irow, 1).layout().itemAt(1).widget().deleteLater()
                self.wid[prefix].pop(wid_row+'_hold')
                Layout.itemAtPosition(3+irow, 1).layout().itemAt(2).widget().deleteLater()
                self.wid[prefix].pop(wid_row+'_cons')
                #delete the complete layout
                Layout.itemAtPosition(3+irow, 1).layout().deleteLater()

                #delete label too
                Layout.itemAtPosition(3+irow, 0).widget().deleteLater()




    def create_wid(self, prefix, Layout):
        """
        create Qt widgets to control lmfit fit function (both peak and specials)
        all the widgets are stored in self.wid
        Args:
            prefix: prefix of the lmfit fit function,
                    from it the number of the component is extracted and the widgets are inserted in the proper column
            Layout: Qt layout

        """
        #create dictionary where to store the widgets
        self.wid[prefix] = {}
        # insert special function always in column 1
        col = 1
        # widgets for components are inserted in the column determinate by the prefix
        # identify components that are peak from prefix 'p#'
        if prefix.startswith('p'):
            #extract colum number from prefix
            col = int(''.join(filter(str.isdigit, prefix)))
            combo_list = fit_function_list()[0]
        else:
            combo_list = fit_function_list()[1]

        # label stating the prefix
        label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(label.sizePolicy().hasHeightForWidth())
        label.setSizePolicy(sizePolicy)
        label.setScaledContents(False)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        label.setObjectName("label")
        label.setText(prefix)
        self.wid[prefix]['prefix'] = label
        Layout.addWidget(self.wid[prefix]['prefix'], 0, col, 1, 1)

        # combobox form type
        comboBox = QtWidgets.QComboBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(comboBox.sizePolicy().hasHeightForWidth())
        comboBox.setSizePolicy(sizePolicy)
        comboBox.setIconSize(QtCore.QSize(16, 16))
        comboBox.setObjectName("comboBox")

        # populate the combo Box with the list of models available in Frontend_fit_GUI
        comboBox.insertItems(1,combo_list)
        comboBox.setCurrentIndex(0)
        self.wid[prefix]['type'] = comboBox
        Layout.addWidget(self.wid[prefix]['type'], 1, col, 1, 1)

        # push button to select color
        PushColor = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(PushColor.sizePolicy().hasHeightForWidth())
        PushColor.setSizePolicy(sizePolicy)
        PushColor.setIconSize(QtCore.QSize(16, 16))
        self.wid[prefix]['color'] = PushColor
        Layout.addWidget(self.wid[prefix]['color'], 2, col, 1, 1)

        # for peak components only
        if prefix.startswith('p'):
            # create a widget for every parameter in the lmfit.model following the order "center","sigma","amplitude","gamma",'fwhm', 'height'
            for irow,wid_key in enumerate(["center","sigma","amplitude","gamma",'fwhm', 'height']):
                param_key = prefix+wid_key
                if param_key in self.mod[prefix].params.keys():
                    pwid = ParamWidgets(Enabled = True)
                    self.wid[prefix][wid_key+'_val'] = pwid.lineValue
                    self.wid[prefix][wid_key+'_hold'] = pwid.checkHold
                    self.wid[prefix][wid_key+'_cons'] = pwid.checkCons
                    Layout.addLayout(pwid, 3+irow, col, 1, 1)

        # for special components
        else:
            irow = 0
            # create a widget for every parameter in the lmfit.model
            for param_key in self.mod[prefix].params.keys():
                wid_key = param_key.split(prefix)[1]
                pwid = ParamWidgets(Enabled = True)
                self.wid[prefix][wid_key+'_val'] = pwid.lineValue
                self.wid[prefix][wid_key+'_hold'] = pwid.checkHold
                self.wid[prefix][wid_key+'_cons'] = pwid.checkCons
                Layout.addLayout(pwid, 3+irow, 1, 1, 1)
                irow +=1


    def load_params(self,fit_result_params):
        # pass the parameters from fit_result to the specific component
        for param_key in fit_result_params.keys():
            if  param_key.startswith('p'):
                #extract component number from prefix
                index = int(''.join(filter(str.isdigit,  param_key)))
                prefix = 'p'+str(index)
                self.mod[prefix].params[param_key].value = fit_result_params[param_key].value
                self.mod[prefix].params[param_key].vary = fit_result_params[param_key].vary
                self.mod[prefix].params[param_key].min = fit_result_params[param_key].min
                self.mod[prefix].params[param_key].max = fit_result_params[param_key].max
                self.mod[prefix].cons[param_key] = fit_result_params[param_key].expr

            elif  param_key.startswith('Bkg'):
                self.mod['Bkg'].params[param_key].value = fit_result_params[param_key].value
                self.mod['Bkg'].params[param_key].vary = fit_result_params[param_key].vary
                self.mod['Bkg'].params[param_key].min = fit_result_params[param_key].min
                self.mod['Bkg'].params[param_key].max = fit_result_params[param_key].max
                self.mod['Bkg'].cons[param_key] = fit_result_params[param_key].expr

            elif  param_key.startswith('Fermi'):
                self.mod['Fermi'].params[param_key].value = fit_result_params[param_key].value
                self.mod['Fermi'].params[param_key].vary = fit_result_params[param_key].vary
                self.mod['Fermi'].params[param_key].min = fit_result_params[param_key].min
                self.mod['Fermi'].params[param_key].max = fit_result_params[param_key].max
                self.mod['Fermi'].cons[param_key] = fit_result_params[param_key].expr

            elif  param_key.startswith('Conv'):
                self.mod['Conv'].params[param_key].value = fit_result_params[param_key].value
                self.mod['Conv'].params[param_key].vary = fit_result_params[param_key].vary
                self.mod['Conv'].params[param_key].min = fit_result_params[param_key].min
                self.mod['Conv'].params[param_key].max = fit_result_params[param_key].max
                self.mod['Conv'].cons[param_key] = fit_result_params[param_key].expr


    def RunFit(self):
        #run the fitting and update the paramters with the optimised values
        self.fit_result = self.mod['Tot'].fit(x=self.x,y=self.y)
        self.mod["Tot"].params = self.fit_result.params
        self.load_params(self.fit_result.params)
        for prefix in self.mod.keys():
            if prefix not in ['Tot']:
                self.update_wid(prefix)

    def ResetParams(self):
        # reset parameters as before the last fit
        try:
            self.mod["Tot"].params = self.fit_result.init_params
            self.load_params(self.fit_result.init_params)
        except:
            pass
        for prefix in self.mod.keys():
            if prefix not in ['Tot']:
                self.update_wid(prefix)


    def SaveFit(self):
        filename = 'lastfit.nc'
        save_fit(self.mod['Tot'], self.fit_result, self.x, self.y, filename=filename)
        print('load the saved fit in jupyter notebook by running: model_tot,ds = load_fit("lastfit.nc")')


    def AddPeak(self):
        #add a new colum in the grid of the peak scroll area and:
        #recompose model tot and refresh plots
        #create widgets, signals and connection for the new component

        #extract number of Peak components
        last_peakprefix = max(list(self.wid.keys()))

        #prefix of the new Peak component
        newindex = int(last_peakprefix.strip('p'))+1

        prefix = 'p'+str(newindex)
        #if a model for the new component doesn't exist yet init a new one
        if not self.mod.get(prefix):
            self.init_model(prefix, func_type='Lorentzian')
            #if there is already a peak take the parameters value form the previous peak where possible
            if newindex > 1:
                #take a paremter of the new component
                for param_key in self.mod[prefix].params.keys():
                    #change the prefix with the one of the previous peak
                    param_type = param_key.split(prefix)[1]
                    param_key1 = 'p'+str(newindex-1)+param_type
                    if param_key1 in self.mod['p'+str(newindex-1)].params.keys():
                        self.mod[prefix].params[param_key].value = self.mod['p'+str(newindex-1)].params[param_key1].value


        self.compose_modelTot()
        self.create_wid(prefix, self.scrollAreaPeak.grid)
        self.update_wid(prefix)
        self.refreshPlotDC(prefix)
        self.refreshPlotRes()
        self.sig[prefix]={}
        self.create_sig(prefix)
        self.con[prefix]={}
        self.connect_sig(prefix)




    ############################################################################
