# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Calibration_template.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from peaks.core.fileIO.loaders.LEED_load import *

import xarray as xr

from .plotwidgetsignal import PlotWidgetSig
#from .rot_disorder import rotation_disorder

import numpy as np

from PyQt5 import QtCore, QtGui

from PyQt5 import QtWidgets
#from pyqtgraph import PlotWidget, plot
#import pyqtgraph as pg
import sys  # We need sys so that we can pass argv to QApplication
from PyQt5.QtWidgets import QGraphicsEllipseItem, QGraphicsLineItem
from PyQt5.QtCore import Qt, QPointF

import cv2
from skimage.feature import blob_log

from scipy.spatial import KDTree

import pyqtgraph as pg

from IPython.display import display, Javascript


def cell_below(text):
    text = text.replace("\n", "\\n").replace("'", "\\'")
    display(Javascript("""
    var cell = IPython.notebook.insert_cell_below('code')
    cell.set_text('"""+text+"""')
    cell.execute()
    """))

def setAttrs(data, **kwargs): 
    
    if isinstance(data, xr.core.dataset.Dataset):
        if 'camera_factor' in kwargs:
                for ii in data.data_vars:
                    data[ii].attrs['camera_factor'] = kwargs['camera_factor']
                data.attrs['camera_factor'] = kwargs['camera_factor']
                
        if 'calibration_energy' in kwargs:
            for ii in data.data_vars:
                    data[ii].attrs['calibration_energy'] = kwargs['calibration_energy']
            data.attrs['calibration_energy'] = kwargs['calibration_energy']
            
        if 'name' in kwargs:
            name = kwargs['name']
            if 'energy' in kwargs:
                data[name].attrs['energy'] = kwargs['energy']
            if 'center' in kwargs:  
                data[name].attrs['center'] = kwargs['center']
            if 'azi_offset' in kwargs:
                data[name].attrs['azi_offset'] = kwargs['azi_offset']

def find_center(points):
    """
    A function to locate the center of an arbitary shape

    Parameters
    ----------
    points : 
        Bragg peaks.

    Returns
    -------
    center : nparray
        Center of shape.
    angle : Float
        azi offset.

    """
    x, y = [], []
    for i in range(len(points)):
        x.append(points[i].x())
        y.append(points[i].y())
    array = np.vstack((x,y)).T

    center = array.mean(axis=0)
    
    #taps_nparray = np.array((self.point_stream.data['x'],self.point_stream.data['y'])).T
    #Find the Bragg peak with smallest y
    A=center[1]-array[:,1].min()
    
    center = center.astype(int)
    
    #Identify the index for this Bragg peak
    index_value=array[:,1].argmin()
    
    #Find the x position of this Bragg peak
    O=array[index_value][0]-center[0]
    
    #Calculate the angle from the vertical (anticlockwise)
    angle = np.arctan(float(O/A))*180/np.pi# + self.data.spectrum.attrs['azi']
    
    return center, angle

def offsets(image, Bragg_peak, center, energy, lattice):
    #Calculate theta and phi offsets
    
    gray = image.astype(np.uint8)
    
    gray = cv2.Canny(gray,0,20)
    
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=900,minRadius=200)
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
    
    #Wavelength of eletron
    lamda = (6.63*10**-34)/np.sqrt(2*(9.11*10**-31)*float(energy.text())*(1.6*10**-19))
    
    #Distance in pixels between 1st Order Bragg peak and zeroeth peak
    mod_moveObject = np.sqrt((Bragg_peak[0].x()-center[0])**2 + (Bragg_peak[0].y()-center[1])**2)
    
    #Theta value at the first order Bragg peak
    theta= np.arcsin(lamda/(float(lattice.text())*10**-10))*180/np.pi
    
    #Ratio of theta to pixel for length from center to 1st Bragg peak
    theta_ratio = theta / mod_moveObject
    
    #Distance between real center and center of LEED spots
    diff_x = circles[0][0] - center[0]
    diff_y = circles[0][1] - center[1]
    
    print(circles[0][0])
    print(circles[0][1])
    
    #Convert from pixel to theta
    theta_offset = diff_x * theta_ratio
    phi_offset = diff_y * theta_ratio

    print(theta_offset)
    print(phi_offset)

class Ui_MainWindow(object):
    def __init__(self, file):
        self.file = file
        
    def setupUi(self, MainWindow):
        
        #Setup the MainWindow
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(884, 805)
        MainWindow.setMinimumSize(QtCore.QSize(884, 805))
        MainWindow.setMaximumSize(QtCore.QSize(884, 805))
        
        #Font size = 10
        font = QtGui.QFont()
        font.setPointSize(10)
        
        #A lot of code just to change the execute button to be green
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        #---------------------------------------------------------------------
        #Calibrate tab
        
        #Widget to contain calibrate graph
        self.calibrate = QtWidgets.QWidget()
        self.calibrate.setObjectName("calibrate")
        
        #Scene for calibrate graph
        self.cal_view = QtWidgets.QGraphicsScene()  
        
        #Graphicsview for the above scene
        self.view = QtWidgets.QGraphicsView(self.cal_view, self.calibrate)
        self.view.setGeometry(QtCore.QRect(0, 0, 884, 643))
        self.view.setFrameStyle(QtWidgets.QFrame.NoFrame)

        #Create widget that will contain the layout grid
        self.layoutWidget = QtWidgets.QWidget(self.calibrate)
        self.layoutWidget.setGeometry(QtCore.QRect(0, 655, 884, 97))
        self.layoutWidget.setObjectName("layoutWidget")
        
        #Create the layout grd that the buttons will be placed in
        self.gridLayout_2 = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setVerticalSpacing(7)
        self.gridLayout_2.setObjectName("gridLayout_2")
        
        #Helper instructions to guide the user
        self.instruct_cal = QtWidgets.QLabel(self.layoutWidget)
        self.instruct_cal.setAlignment(QtCore.Qt.AlignCenter)
        self.instruct_cal.setFont(font)
        self.instruct_cal.setObjectName("instruct_cal")
        self.gridLayout_2.addWidget(self.instruct_cal, 0, 0, 1, 9)
        
        #Label before shape comboBox        
        self.Shape = QtWidgets.QLabel(self.layoutWidget)
        self.Shape.setFont(font)
        self.Shape.setObjectName("Shape")
        self.gridLayout_2.addWidget(self.Shape, 1, 0, 1, 1)
        
        #ComboBox to select shape of Brillouin zone
        self.Shape_combo = QtWidgets.QComboBox(self.layoutWidget)
        self.Shape_combo.setFont(font)
        self.Shape_combo.setModelColumn(0)
        self.Shape_combo.setObjectName("Shape_combo")
        self.Shape_combo.addItem("")
        self.Shape_combo.addItem("")
        self.Shape_combo.addItem("")
        self.gridLayout_2.addWidget(self.Shape_combo, 1, 1, 1, 1)
        
        #First spacer at col=2
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem1, 1, 2, 1, 1)
        
        #Lattice parameter a label
        self.Lat_par_a = QtWidgets.QLabel(self.layoutWidget)
        self.Lat_par_a.setFont(font)
        self.Lat_par_a.setObjectName("Lat_par_a")
        self.gridLayout_2.addWidget(self.Lat_par_a, 1, 3, 1, 1)
        
        #Input box for first lattice parameter
        self.lineEdit_3 = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.gridLayout_2.addWidget(self.lineEdit_3, 1, 4, 1, 1)
        
        #Second spacer at col=5
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem, 1, 5, 1, 1)
        
        self.Clear1 = QtWidgets.QPushButton(self.layoutWidget)
        self.Clear1.setFont(font)
        self.Clear1.setObjectName("Clear1")
        self.gridLayout_2.addWidget(self.Clear1, 1, 6, 1, 1)
        
        #Third spacer at col=7
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem2, 1, 7, 1, 1)
        
        #Calibrate button
        self.Calibrate1 = QtWidgets.QPushButton(self.layoutWidget)
        self.Calibrate1.setEnabled(True)
        self.Calibrate1.setSizeIncrement(QtCore.QSize(1, 2))
        self.Calibrate1.setPalette(palette)
        self.Calibrate1.setFont(font)
        self.Calibrate1.setAutoFillBackground(True)
        self.Calibrate1.setAutoDefault(False)
        self.Calibrate1.setDefault(False)
        self.Calibrate1.setFlat(True)
        self.Calibrate1.setObjectName("Calibrate1")
        self.gridLayout_2.addWidget(self.Calibrate1, 1, 8, 1, 1)
        
        #Spacer item at col=9
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem3, 1, 9, 1, 1)
        
        #Filename label
        self.Filename2 = QtWidgets.QLabel(self.layoutWidget)
        self.Filename2.setText("Filename")
        self.Filename2.setFont(font)
        self.Filename2.setObjectName("Filename2")
        self.gridLayout_2.addWidget(self.Filename2, 1, 10, 1, 1)
        
        #ComboBox to select file
        self.combo2 = QtWidgets.QComboBox(self.layoutWidget)
        self.combo2.setFont(font)
        self.combo2.setModelColumn(0)
        self.combo2.setObjectName("combo2")
        self.gridLayout_2.addWidget(self.combo2, 1, 11, 1, 1)
        
        #Label for electron energy
        self.Energy = QtWidgets.QLabel(self.layoutWidget)
        self.Energy.setFont(font)
        self.Energy.setObjectName("Energy")
        self.gridLayout_2.addWidget(self.Energy, 2, 0, 1, 1)
        
        #Input box for electron energy
        self.lineEditEnergy = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEditEnergy.setObjectName("lineEditEnergy")
        self.gridLayout_2.addWidget(self.lineEditEnergy, 2, 1, 1, 1)
        
        #Second lattice parameter label (for anisotropic Brillouin zones)
        self.lat_par_b = QtWidgets.QLabel(self.layoutWidget)
        self.lat_par_b.setFont(font)
        self.lat_par_b.setObjectName("lat_par_b")
        self.gridLayout_2.addWidget(self.lat_par_b, 2, 3, 1, 1)
        
        #Input for 2nd lattice parameter
        self.lineEdit_4 = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.gridLayout_2.addWidget(self.lineEdit_4, 2, 4, 1, 1)
        
        #Automatically select Bragg peaks
        self.Automate1 = QtWidgets.QPushButton(self.layoutWidget)
        self.Automate1.setFont(font)
        self.Automate1.setObjectName("Automate1")
        self.gridLayout_2.addWidget(self.Automate1, 2, 6, 1, 1)
        
        #Button to calibrate and center - use when calibration and final
        #LEED images are the same
        self.cal_center = QtWidgets.QPushButton(self.layoutWidget)
        self.cal_center.setFont(font)
        self.cal_center.setObjectName("cal_center")
        self.gridLayout_2.addWidget(self.cal_center, 2, 8, 1, 1)
        
        #Button to calibrate and then use the calibration center as the same
        #center for all images
        self.calibrate_center_all = QtWidgets.QPushButton(self.layoutWidget)
        self.calibrate_center_all.setFont(font)
        self.calibrate_center_all.setObjectName("calibrate_center_all")
        self.gridLayout_2.addWidget(self.calibrate_center_all, 2, 10, 1, 2)
        
        #Calculate rotation disorder label
        self.offset_label = QtWidgets.QLabel(self.layoutWidget)
        self.offset_label.setFont(font)
        self.offset_label.setObjectName("offset_label")
        self.gridLayout_2.addWidget(self.offset_label, 0, 10, 1, 1)
        
        #Disorder Boolean
        self.Check2 = QtWidgets.QCheckBox(self.layoutWidget)
        self.Check2.setFont(font)
        self.Check2.setObjectName("off")
        self.gridLayout_2.addWidget(self.Check2, 0, 11, 1, 1)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Clear1.setText(_translate("MainWindow", "Clear"))
        self.instruct_cal.setText(_translate("MainWindow", "Identify the first order Bragg Peaks, fill in the parameters below and click calibrate"))
        self.Calibrate1.setText(_translate("MainWindow", "Calibrate"))
        self.Lat_par_a.setText(_translate("MainWindow", "Lattice Parameter (a):"))
        self.cal_center.setText(_translate("MainWindow", "Calibrate and Center"))
        self.Automate1.setText(_translate("MainWindow", "Automate"))
        self.Energy.setText(_translate("MainWindow", "Energy:"))
        self.Shape.setText(_translate("MainWindow", "Shape:"))
        self.lat_par_b.setText(_translate("MainWindow", "Lattice Parameter (b):"))
        self.Shape_combo.setItemText(0, _translate("MainWindow", "Hexagon"))
        self.Shape_combo.setItemText(1, _translate("MainWindow", "Square"))
        self.Shape_combo.setItemText(2, _translate("MainWindow", "Rectangle"))
        self.Filename2.setText(_translate("MainWindow", "Filename:"))
        self.calibrate_center_all.setText(_translate("MainWindow","Calibrate and Center All"))
        self.offset_label.setText(_translate("MainWindow","Offset"))
        
class MovingObject(QGraphicsEllipseItem):
    def __init__(self, x, y, r, app, cal_view):
        super(MovingObject, self).__init__(0, 0, r, r)
        self.setPos(x, y)
        self.r = r
        self.setBrush(Qt.yellow)
        self.setAcceptHoverEvents(True)
        self.cal_view = cal_view
        
    # mouse click event
    def mousePressEvent(self, event):
        pass

    def hoverEnterEvent(self, event):
        self.cal_view.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.OpenHandCursor))

    def hoverLeaveEvent(self, event):
        self.cal_view.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.CrossCursor))

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Delete:
            print("another delete")          

    def mouseMoveEvent(self, event):
        #mousePoint = self.cal_view.plotItem.vb.mapSceneToView(event.scenePos())
        orig_cursor_position = self.cal_view.plotItem.vb.mapSceneToView(event.lastScenePos())
        updated_cursor_position = self.cal_view.plotItem.vb.mapSceneToView(event.scenePos())

        orig_position = self.cal_view.plotItem.vb.mapSceneToView(event.scenePos())

        self.updated_cursor_x = updated_cursor_position.x() - orig_cursor_position.x() + orig_position.x()
        self.updated_cursor_y = updated_cursor_position.y() - orig_cursor_position.y() + orig_position.y()
        self.setPos(QPointF(self.updated_cursor_x, self.updated_cursor_y))
        
class LineObject(QGraphicsLineItem):
    def __init__(self, x, y, view):
        super(LineObject, self).__init__(x, y, x, y)
        self.setPen(Qt.yellow)
        self.view = view

class CalibrateGUI(QtWidgets.QMainWindow):
     def __init__(self, file, xarray, app, **kwargs):
         super(CalibrateGUI, self).__init__()
         self.file = file
         self.xarray = xarray
         self.ui = Ui_MainWindow(file)
         self.ui.setupUi(self)
         self.circle2 = []
         self.circle_item = []
         self.app = app
         self.moveObject = []
         self.moveObject2 = []
         self.rightclick = False
         self.circ_cal = 10
         self.circ_cen = 10         
         self.added = False
         self.text = []
         self.markdown = []
         
         for ii in self.xarray.data_vars:
             self.ui.combo2.addItem(ii)

         self.ui.combo2.setCurrentIndex(0)
         self.combo2value = self.ui.combo2.currentText()
         self.img = xarray[self.combo2value].data
         self.combovalue = self.ui.combo2.currentText()
         
         self.ui.cal_view = PlotWidgetSig(self.ui.view)
         self.ui.cal_view.setMenuEnabled(False)
         self.ui.cal_view.setGeometry(QtCore.QRect(0, 0, 884, 643))
         self.ui.cal_view.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.CrossCursor))
         
         self.myobject = self.moveObject
         self.myscene = self.ui.cal_view
         self.circ_size = self.circ_cal
         
         self.create_graph(xarray[self.combo2value].data, self.ui.cal_view)
         
         self.ui.Clear1.clicked.connect(self.Clear1Clicked)
         self.ui.Calibrate1.clicked.connect(self.ExecuteClicked)
         self.ui.Automate1.clicked.connect(self.AutomateButtonClicked)
         self.ui.cal_center.clicked.connect(self.calAndCenter)
         self.ui.combo2.currentIndexChanged.connect(self.combochange)
         self.ui.calibrate_center_all.clicked.connect(self.calibrateCenterAll)
         
         self.ui.cal_view.scene().sigMouseClicked.connect(self.mouseClickEvent)
         self.ui.cal_view.scene().sigMouseMoved.connect(self.mouseMoveEvent)
         
         self.ui.cal_view.sigKeyPress.connect(self.circleSizeChange)
         self.ui.cal_view.sigKeyPress.connect(self.keyPressEvent)

         if 'energy' in xarray[self.combovalue].attrs and xarray[self.combovalue].attrs['energy'] != None:
             self.ui.lineEditEnergy.setText(str(xarray[self.combovalue].attrs['energy']))
             
         self.show()     
     
     def changed(self):
        if self.ui.tabWidget.currentIndex() == 0:
            self.myscene = self.ui.cal_view
            self.myobject = self.moveObject
            self.circ_size = self.circ_cal
        else:
            self.myscene = self.ui.cen_view
            self.myobject = self.moveObject2
            self.circ_size = self.circ_cen
     
     def circleSizeChange(self, event):
         if event.key() == QtCore.Qt.Key_Up:
             self.circ_size += 1
             for i in range(len(self.myobject)):
                 self.myscene.removeItem(self.myobject[i])
                 self.myobject[i] = MovingObject(self.myobject[i].x()-1/2, self.myobject[i].y()-1/2, self.circ_size, self.app, self.myscene)
                 self.myscene.addItem(self.myobject[i])
                 
         if event.key() == QtCore.Qt.Key_Down:     
             self.circ_size -= 1
             for i in range(len(self.myobject)):
                 self.myscene.removeItem(self.myobject[i])
                 self.myobject[i] = MovingObject(self.myobject[i].x()+1/2, self.myobject[i].y()+1/2, self.circ_size, self.app, self.myscene)
                 self.myscene.addItem(self.myobject[i])
            
      
     def mouseClickEvent(self, event):
        #if event.button() == Qt.LeftButton:
            
        mousePoint = self.myscene.plotItem.vb.mapSceneToView(event.scenePos())
        
        if event.button() == Qt.LeftButton:
            self.myobject.append(MovingObject(mousePoint.x()- self.circ_size/2, mousePoint.y()-self.circ_size/2, self.circ_size, self.app, self.myscene))
            self.myscene.addItem(self.myobject[-1])
            
        elif event.button() == Qt.RightButton :
            if self.rightclick == False:
                if hasattr(self, 'line'):
                    self.myscene.removeItem(self.line)
                    del self.line
                self.rightclick = True
                self.x = mousePoint.x()
                self.y = mousePoint.y()
                self.linePos = [self.x, self.y]
                self.line = LineObject(mousePoint.x(), mousePoint.y(), self.ui.cal_view)
                self.myscene.addItem(self.line)
            elif self.rightclick == True:
                self.rightclick = False
                self.linePos.append(self.mousePointLine.x())
                self.linePos.append(self.mousePointLine.y())

     def mouseMoveEvent(self, event):
        if self.rightclick == True:
            self.mousePointLine = self.ui.cal_view.plotItem.vb.mapSceneToView(event)

            self.line.setLine(self.x, self.y, self.mousePointLine.x(), self.mousePointLine.y())
                          
     def Clear1Clicked(self):
            
         for i in self.myobject:
             self.myscene.removeItem(i)
         self.Image.setImage(self.img.T)
         self.myobject.clear()
         
         if hasattr(self, 'line'):
             self.myscene.removeItem(self.line)
             del self.line
         
     def AutomateButtonClicked(self,event):
             
        self.img_t = self.img
        gray = self.img_t.astype(np.uint8)
            
        #Blob detection method not yet fully implemented
        blobs_log = blob_log(gray, max_sigma=30, num_sigma=10,min_sigma=5, threshold=.1, overlap =0)
        x_vals = []
        y_vals = []
        for blob in blobs_log:
            y5, x5, r5 = blob
            #Condition to avoid picking up stray marks on screen
            if r5 > 3:
                x_vals.append(x5)
                y_vals.append(y5)
            
        x_vals= np.array(x_vals)
        y_vals= np.array(y_vals)
        vals = np.vstack((x_vals,y_vals))
        vals = vals.T
                
        kdtree = KDTree(vals)
        
        auto_bragg = []
        
        a = len(self.myobject)
        for m in range(0,a):
            d, i = kdtree.query((self.myobject[m].x(),self.myobject[m].y()))
            auto_bragg.append(vals[i]) 
        
        for i in self.myobject:
            self.myscene.removeItem(i)
        self.myobject.clear()

        for m in range(0,a):
            x = self.myscene.mapToScene(int(auto_bragg[m][0]),int(auto_bragg[m][1]))
            self.myobject.append(MovingObject(x.x()-self.circ_size/2, x.y()-self.circ_size/2, self.circ_size, self.app, self.myscene))
            self.myscene.addItem(self.myobject[-1])
            
     def calibrateCenterAll(self):
         self.ExecuteClicked()
         
         for ii in self.xarray.data_vars:
             self.text.append("setAttrs(data = {0}, name = '{1}', center = [{2},{3}])".format(self.file, ii, self.center[0], self.center[1]))
     
         self.close()   
            
     def centerAll(self):
         self.add()
         if self.added == True:
             del self.text[-1]
             
             for ii in self.xarray.data_vars:
                 self.text.append("setAttrs(data = {0}, name = '{1}', center = [{2},{3}])".format(self.file, ii, self.center[0], self.center[1]))
         
             self.close()
        
     def add(self):
        self.added = True
         
        if self.ui.lineEditEnergy2.text() == '':
            self.ui.instruct_cen.setText("Please input energy value")
            self.ui.instruct_cen.setStyleSheet("background-color: red; border: 1px solid black;")
            self.added = False
        
        elif len(self.moveObject2)==0:
            self.ui.instruct_cen.setText("Select Bragg peaks")
            self.ui.instruct_cen.setStyleSheet("background-color: red; border: 1px solid black;")
            self.added = False
            
        elif len(self.moveObject2)==1:
            self.center = [int(self.moveObject2[-1].x()),int(self.moveObject2[-1].y())]
            self.text.append("setAttrs(data = {0}, name = '{1}', center = [{2},{3}], energy = {4})".format(self.file, self.combovalue2, self.center[0], self.center[1], self.ui.lineEditEnergy2.text()))
            
            self.ui.instruct_cen.setText("{} has been centered".format(self.combovalue2))
            self.ui.instruct_cen.setStyleSheet("background-color: green; border: 1px solid black;")
            
        elif len(self.moveObject2) == 6 and str(self.ui.shapeComboBox.currentText()) == 'Hexagon':
            
            #Find center and azimuthal offset from Bragg peaks
            self.center, self.angle = find_center(self.moveObject2)

            self.text.append("setAttrs(data = {0}, name = '{1}', center = [{2},{3}], energy = {4}, azi_offset = {5:.2f})".format(self.file, self.combovalue2, self.center[0], self.center[1], self.ui.lineEditEnergy2.text(), self.angle))
            
            self.ui.instruct_cen.setText("{} has been centered".format(self.combovalue2))
            self.ui.instruct_cen.setStyleSheet("background-color: green; border: 1px solid black;")
            
        elif len(self.moveObject2) == 4 and str(self.ui.shapeComboBox.currentText()) == 'Square':

            #Find center and azimuthal offset from Bragg peaks
            self.center, self.angle = find_center(self.moveObject2)

            #Calulate camera factor - relationship between pixel and angle
            q_coord = 2*np.pi/(float(self.ui.lineEdit_3.text()))
            camera_factor = q_coord/np.sqrt((self.circle[0].x()-self.center[0])**2 +(self.circle[0].y()-self.center[1])**2)
            
            self.text.append("setAttrs(data = {0}, name = '{1}', center = [{2},{3}], energy = {4}, azi_offset = {5:.2f})".format(self.file, self.combovalue2, self.center[0], self.center[1], self.ui.lineEditEnergy2.text(), self.angle))
            
            self.ui.instruct_cen.setText("{} has been centered".format(self.combovalue2))
            self.ui.instruct_cen.setStyleSheet("background-color: green; border: 1px solid black;")

        elif len(self.moveObject2) > 4 and str(self.ui.shapeComboBox.currentText()) in {'Square', 'Rectangle'}:
            self.ui.instruct_cen.setText("Too many Bragg peaks for a quadrilateral Brillouin zone")
            self.ui.instruct_cen.setStyleSheet("background-color: red; border: 1px solid black;")
            self.added = False
            
        elif len(self.moveObject2) > 4:
            self.ui.instruct_cen.setText("Too many Bragg peaks")
            self.ui.instruct_cen.setStyleSheet("background-color: red; border: 1px solid black;")

        else:
            self.ui.instruct_cen.setText("Not enough Bragg peaks")
            self.ui.instruct_cen.setStyleSheet("background-color: red; border: 1px solid black;")
            self.added = False
            
        if len(self.moveObject2) > 1 and self.ui.Check.isChecked():
            self.text_rot = "#Calculating rotation disorder \n"
            self.text_rot += "azi, rad = rotation_disorder(data = {0}['{1}'], bragg = [{2},{3}], center =[{4},{5}])".format(self.file, self.combovalue2, self.moveObject2[0].x(), self.moveObject2[0].y(), self.center[0], self.center[1])            
            
     def calAndCenter(self):
         self.ExecuteClicked()
         self.close()
         
     def ExecuteClicked(self):        
        #Check that enough Bragg peaks have been selected
        
        if self.ui.lineEdit_3.text() == '' and self.ui.lineEditEnergy.text() == '':
            self.ui.instruct_cal.setText("Please input lattice parameter and energy values")
            self.ui.instruct_cal.setStyleSheet("background-color: red; border: 1px solid black;")
            
        elif self.ui.lineEditEnergy.text() == '':
            self.ui.instruct_cal.setText("Please input energy value")
            self.ui.instruct_cal.setStyleSheet("background-color: red; border: 1px solid black;")
            
        elif self.ui.lineEdit_3.text() == '':
            self.ui.instruct_cal.setText("Please input lattice parameter value")
            self.ui.instruct_cal.setStyleSheet("background-color: red; border: 1px solid black;")
            
        else:           
            if hasattr(self, 'line'):
                self.center = [int((self.linePos[0]+self.linePos[2])/2),int((self.linePos[1]+self.linePos[3])/2)]
                q_coord = 4*np.pi/(np.sqrt(3)*float(self.ui.lineEdit_3.text()))
                self.camera_factor = q_coord/np.sqrt((self.linePos[0]-self.center[0])**2 +(self.linePos[1]-self.center[1])**2)
                
                self.text.append("setAttrs(data = {0}, name = '{1}', center = [{2},{3}], camera_factor = {4:.4f}, calibration_energy = {5}, energy = {5})".format(self.file, self.combovalue, self.center[0], self.center[1], self.camera_factor, self.ui.lineEditEnergy.text()))
                
                #Change tab and show progress in progress bar
                self.ui.tabWidget.setCurrentIndex(1)
                self.ui.progressBar.setValue(50)
                
            elif len(self.moveObject)==0:
                self.ui.instruct_cal.setText("Select Bragg peaks")
                self.ui.instruct_cal.setStyleSheet("background-color: red; border: 1px solid black;")
                
            elif len(self.moveObject) == 6 and str(self.ui.Shape_combo.currentText()) == 'Hexagon':
                
                #Find center and azimuthal offset from Bragg peaks
                self.center, self.angle = find_center(self.moveObject)
    
                #Calulate camera factor - relationship between pixel and angle
                q_coord = 4*np.pi/(np.sqrt(3)*float(self.ui.lineEdit_3.text()))
                self.camera_factor = q_coord/np.sqrt((self.moveObject[0].x()-self.center[0])**2 +(self.moveObject[0].y()-self.center[1])**2)
                
                #Text that will be used to code a function for setting metadata
                self.text.append("setAttrs(data = {0}, name = '{1}', center = [{2},{3}], camera_factor = {4:.4f}, calibration_energy = {5}, energy = {5})".format(self.file, self.combovalue, self.center[0], self.center[1], self.camera_factor, self.ui.lineEditEnergy.text()))
                
                if self.ui.Check2.isChecked():
                    offsets(self.img, self.moveObject, self.center, self.ui.lineEditEnergy2, self.ui.lineEdit_3)
                
                #Change tab and show progress in progress bar
                self.ui.tabWidget.setCurrentIndex(1)
                self.ui.progressBar.setValue(50)
    
            elif len(self.moveObject) == 4 and str(self.ui.Shape_combo.currentText()) == 'Square':
    
                #Find center and azimuthal offset from Bragg peaks
                self.center, self.angle = find_center(self.moveObject)
    
                #Calulate camera factor - relationship between pixel and angle
                q_coord = 2*np.pi/(float(self.ui.lineEdit_3.text()))
                self.camera_factor = q_coord/np.sqrt((self.circle[0][0]-self.center[0])**2 +(self.circle[0][1]-self.center[1])**2)
    
                #Text that will be used to code a function for setting metadata
                self.text.append("setAttrs(data = {0}, name = '{1}', center = [{2},{3}], camera_factor = {4:.4f}, calibration_energy = {5}, energy = {5})".format(self.file, self.combovalue, self.center[0], self.center[1], self.camera_factor, self.ui.lineEditEnergy.text()))
                
                if self.ui.Check2.isChecked():
                    offsets(self.img, self.moveObject, self.center, self.ui.lineEditEnergy2, self.ui.lineEdit_3)
                
                #Change tab and show progress in progress bar
                self.ui.tabWidget.setCurrentIndex(1)
                self.ui.progressBar.setValue(50)
                
            elif len(self.moveObject) == 4 and str(self.ui.Shape_combo.currentText()) == 'Rectangle':
    
                #Find center and azimuthal offset from Bragg peaks
                self.center, self.angle = find_center(self.moveObject)
    
                #Calulate camera factor - relationship between pixel and angle
                q_coord = 2*np.pi/(float(self.ui.lineEdit_3.text()))
                camera_factor = q_coord/np.sqrt((self.circle[0][0]-self.center[0])**2 +(self.circle[0][1]-self.center[1])**2)
                
                q_coord2 = 2*np.pi/(float(self.ui.lineEdit_4.text()))
                self.camera_factor = q_coord2/np.sqrt((self.circle[1][0]-self.center[0])**2 +(self.circle[1][1]-self.center[1])**2)
    
                #Text that will be used to code a function for setting metadata
                self.text.append("setAttrs(data = {0}, name = '{1}', center = [{2},{3}], camera_factor = {4:.4f}, calibration_energy = {5}, energy = {5})".format(self.file, self.combovalue, self.center[0], self.center[1], self.camera_factor, self.ui.lineEditEnergy.text()))
                
                if self.ui.Check2.isChecked():
                    offsets(self.img, self.moveObject, self.center, self.ui.lineEditEnergy2, self.ui.lineEdit_3)
                    
                #Change tab and show progress in progress bar
                self.ui.tabWidget.setCurrentIndex(1)
                self.ui.progressBar.setValue(50)
                
            elif len(self.moveObject) > 4 and str(self.ui.Shape_combo.currentText()) in {'Square', 'rectangle'}:
                self.ui.instruct_cal.setText("Too many Bragg peaks for a quadrilateral Brillouin zone")
                self.ui.instruct_cal.setStyleSheet("background-color: red; border: 1px solid black;")
                
            elif len(self.moveObject) > 6 and str(self.ui.Shape_combo.currentText()) == 'Hexagon':
                self.ui.instruct_cal.setText("Too many Bragg peaks")
                self.ui.instruct_cal.setStyleSheet("background-color: red; border: 1px solid black;")
    
            else:
                self.ui.instruct_cal.setText("Not enough Bragg peaks")
                self.ui.instruct_cal.setStyleSheet("background-color: red; border: 1px solid black;")
                
                
         
     def ExecuteClicked2(self):        
         if self.added == True:
                 self.close()
         else:
             self.add()
             if self.added == True:
                 self.close()
                 
     def create_graph(self, img, view):
        self.Image = pg.ImageItem()
        self.Image.setImage(img.T)
        
        view.plotItem.addItem(self.Image)
        cm = pg.colormap.get('CET-L1')
        self.bar = pg.ColorBarItem( values= (0, 255), cmap = cm, limits=[0,255]) # prepare interactive color bar
        # Have ColorBarItem control colors of img and appear in 'plot':
        self.bar.setImageItem( self.Image, insert_in= view.plotItem ) 
        #self.level = self.Image.getLevels()
        
     def create_graph2(self, img, view):
        self.Image2 = pg.ImageItem()
        self.Image2.setImage(img.T)
        
        view.plotItem.addItem(self.Image2)
        cm = pg.colormap.get('CET-L1')
        self.bar2 = pg.ColorBarItem( values= (0, 255), cmap = cm, limits=[0,255]) # prepare interactive color bar
        # Have ColorBarItem control colors of img and appear in 'plot':
        self.bar2.setImageItem( self.Image2, insert_in= view.plotItem ) 
        #self.level = self.Image.getLevels()
        
     def combochange(self):
         self.combovalue = self.ui.combo2.currentText()
         self.img = self.xarray[self.combovalue].data
         
         self.myscene.clear()
         self.myscene.addItem(self.Image)
         
         # Plot 2D data in MainPlot
         self.Image.setImage(self.img.T)
         self.myobject.clear()
         
         if 'energy' in self.xarray[self.combovalue].attrs and self.xarray[self.combovalue].attrs['energy'] != None:
             self.ui.lineEditEnergy.setText(str(self.xarray[self.combovalue].attrs['energy'])) 
              
     def combochange2(self):
         self.combovalue2 = self.ui.comboBox.currentText()
         self.img = self.xarray[self.combovalue2].data
         
         self.myscene.clear()
         self.myscene.addItem(self.Image2)
         
         # Plot 2D data in MainPlot
         self.Image2.setImage(self.img.T)
         self.myobject.clear()
         
         if 'energy' in self.xarray[self.combovalue2].attrs and self.xarray[self.combovalue2].attrs['energy'] != None:
             self.ui.lineEditEnergy2.setText(str(self.xarray[self.combovalue2].attrs['energy'])) 
              
def cal_tool2(file, xarray, *kargs, **kwargs):
    """
    

    Parameters
    ----------
    file : String
        Xarray name.
    xarray : xr.Datset
        The input dataset.
        
    Optional parameters
    -----------
    q   : String
      Adding thi parameter will convert the xarray
      to q space. Syntax: q = string - with the string being 
      the name of the new dataset created after q conversion
    calibration  : String
      If you would like to calibrate to a LEED image not in the
      dataset, add calibration = String - with the Strig being 
      the path of the LEED image
    *kargs : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    if 'calibration' in kwargs:
        test = LEED_load(filename = kwargs['calibration'])
        xarray = xr.merge([test, xarray])
        merge_text = "_merge = LEED_load(filename = '{0}')".format(kwargs['calibration']) + "\n"
        merge_text += "{0} = xr.merge([_merge, {0}])".format(file)
        
    elif xarray.attrs['calibration'] != None:
        test = LEED_load(filename = xarray.attrs['calibration'])
        xarray = xr.merge([test, xarray])
        merge_text = "_merge = LEED_load(filename = '{0}')".format(kwargs['calibration']) + "\n"
        merge_text += "{0} = xr.merge([_merge, {0}])".format(file)
    
    global app
    
    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = QtWidgets.QApplication.instance() 
    GUI = CalibrateGUI(file, xarray, app)
    (app.exec_())
    
    #If all the details have been introduced to the GUI
    #a function will be written to the notebook
    #to assign all updated metadata to the dataset
    if len(GUI.text) > 0 and 'calibration' in kwargs:
        text = merge_text + '\n' + GUI.text[0]
    elif len(GUI.text) > 0:
        text = GUI.text[0]
            
    if len(GUI.text) > 0:
        for i in range(1, len(GUI.text)):
            text = text + '\n' + GUI.text[i]          
    
        if 'q' in kwargs:
            q_conv_command = "{0} = LEED_q_convert({1})".format(kwargs['q'], file)
            text = text + '\n' + q_conv_command
        
        if hasattr(GUI, 'text_rot'):
            #if 'q' in kwargs:
                #GUI.text_rot = GUI.text_rot.replace(file, kwargs['q'])
            text = text + '\n' + GUI.text_rot
        
        cell_below(text)
            
    display(Javascript("""
    var output_area = this
    var cell_element = output_area.element.parents('.cell')
    var cell_idx = Jupyter.notebook.get_cell_elements().index(cell_element)
    let code = Jupyter.notebook.get_cell(cell_idx).get_text()
    const a = code.indexOf("cal_tool")
    let code2 = code.slice(0, a) + '#' + code.slice(a) + '"""+ text +"""'
    code = '#' + code
    var cell = IPython.notebook.insert_cell_below('code')
    cell.set_text(code2)
    IPython.notebook.cut_cell(cell_idx)
    """))
    