# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Calibration_template.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QGraphicsEllipseItem
from PyQt5.QtCore import Qt, QPointF

from PyQt5 import QtWidgets
#from pyqtgraph import PlotWidget, plot
#import pyqtgraph as pg
import sys  # We need sys so that we can pass argv to QApplication
import os
from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGridLayout
from PyQt5.QtCore import Qt, QPointF
import cv2
from PyQt5.QtGui import (QIcon, QBrush, QColor, QPainter, QPixmap)

import cv2
from skimage.feature import blob_log

from scipy.spatial import KDTree

class Ui_MainWindow(object):
    def __init__(self, file):
        self.file = file
        
    def setupUi(self, MainWindow):
        
        #Setup the MainWindow
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(966, 850)
        MainWindow.setMinimumSize(QtCore.QSize(966, 900))
        MainWindow.setMaximumSize(QtCore.QSize(966, 900))
        
        #Create two tabs - Calibrate and Center
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 966, 850))
        self.tabWidget.setMinimumSize(QtCore.QSize(0, 850))
        self.tabWidget.setMaximumSize(QtCore.QSize(16777215, 850))
        self.tabWidget.setTabShape(QtWidgets.QTabWidget.Triangular)
        self.tabWidget.setIconSize(QtCore.QSize(100, 20))
        self.tabWidget.setObjectName("tabWidget")
        self.calibrate = QtWidgets.QWidget()
        self.calibrate.setObjectName("calibrate")
        
        #This is a holder for the LEED image
        #self.cal_image = QtWidgets.QLabel(self.calibrate)
        #self.cal_image.setGeometry(QtCore.QRect(0, 0, 966, 723))
        #self.cal_image.setPixmap(QtGui.QPixmap(self.file))
        #self.cal_image.setScaledContents(True)
        #self.cal_image.setObjectName("cal_image")
        
        #######################################################
        
        self.cal_view = QtWidgets.QGraphicsScene()
        self.test = QPixmap(self.file)
        self.test = self.test.scaledToWidth(966)
        self.item = self.cal_view.addPixmap(self.test)
        #cal_view.render(QtCore.QRect(0,0,960,723))
        #self.cal_view.setGeometry(QtCore.QRect(0, 0, 966, 723))
        #self.cal_view.setLineWidth(1)
        #cal_view.setObjectName("cal_view")
    
     
    
    
        #self.moveObject2 = MovingObject(100, 100, 20)
        #cal_view.addItem(self.moveObject2)  
        
        self.view = QtWidgets.QGraphicsView(self.cal_view, self.calibrate)
        self.view.setGeometry(QtCore.QRect(0, 0, 966, 723))
        self.view.setFrameStyle(QtWidgets.QFrame.NoFrame)

        
        #Create layout for Calibrate tab
        self.layoutWidget = QtWidgets.QWidget(self.calibrate)
        self.layoutWidget.setGeometry(QtCore.QRect(0, 730, 961, 97))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setVerticalSpacing(7)
        self.gridLayout_2.setObjectName("gridLayout_2")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem, 1, 7, 1, 1)
        self.Clear1 = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.Clear1.setFont(font)
        self.Clear1.setObjectName("Clear1")
        self.gridLayout_2.addWidget(self.Clear1, 1, 8, 2, 1)
        self.lineEdit_4 = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.gridLayout_2.addWidget(self.lineEdit_4, 3, 6, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.gridLayout_2.addWidget(self.label_4, 0, 3, 1, 10)
        self.Execute1 = QtWidgets.QPushButton(self.layoutWidget)
        self.Execute1.setEnabled(True)
        self.Execute1.setSizeIncrement(QtCore.QSize(1, 2))
        
        #A lot of code just to change the execute button to be green
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(127, 255, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(63, 255, 63))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 127, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 170, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(127, 255, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(127, 255, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(63, 255, 63))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 127, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 170, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(127, 255, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 127, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(127, 255, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(63, 255, 63))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 127, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 170, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 127, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 127, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipText, brush)
        self.Execute1.setPalette(palette)
        
        
        font = QtGui.QFont()
        font.setPointSize(10)
        self.Execute1.setFont(font)
        self.Execute1.setAutoFillBackground(True)
        self.Execute1.setAutoDefault(False)
        self.Execute1.setDefault(False)
        self.Execute1.setFlat(True)
        self.Execute1.setObjectName("Execute1")
        self.gridLayout_2.addWidget(self.Execute1, 1, 10, 3, 1)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.gridLayout_2.addWidget(self.lineEdit_2, 3, 3, 1, 1)
        self.Lat_par_a = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.Lat_par_a.setFont(font)
        self.Lat_par_a.setObjectName("Lat_par_a")
        self.gridLayout_2.addWidget(self.Lat_par_a, 1, 5, 2, 1)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.gridLayout_2.addWidget(self.lineEdit_3, 1, 6, 2, 1)
        self.save = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.save.setFont(font)
        self.save.setObjectName("save")
        self.gridLayout_2.addWidget(self.save, 1, 12, 3, 1)
        self.Automate1 = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.Automate1.setFont(font)
        self.Automate1.setObjectName("Automate1")
        self.gridLayout_2.addWidget(self.Automate1, 3, 8, 1, 1)
        self.Energy = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.Energy.setFont(font)
        self.Energy.setObjectName("Energy")
        self.gridLayout_2.addWidget(self.Energy, 3, 0, 1, 3)
        self.Shape = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.Shape.setFont(font)
        self.Shape.setObjectName("Shape")
        self.gridLayout_2.addWidget(self.Shape, 1, 0, 2, 3)
        self.lat_par_b = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lat_par_b.setFont(font)
        self.lat_par_b.setObjectName("lat_par_b")
        self.gridLayout_2.addWidget(self.lat_par_b, 3, 5, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem1, 1, 4, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem2, 1, 9, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem3, 1, 11, 1, 1)
        self.Shape_combo = QtWidgets.QComboBox(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.Shape_combo.setFont(font)
        self.Shape_combo.setModelColumn(0)
        self.Shape_combo.setObjectName("Shape_combo")
        self.Shape_combo.addItem("")
        self.Shape_combo.addItem("")
        self.Shape_combo.addItem("")
        self.gridLayout_2.addWidget(self.Shape_combo, 1, 3, 2, 1)
        self.tabWidget.addTab(self.calibrate, "")
        self.center = QtWidgets.QWidget()
        self.center.setObjectName("center")
        self.label_17 = QtWidgets.QLabel(self.center)
        self.label_17.setGeometry(QtCore.QRect(0, 0, 966, 723))
        self.label_17.setText("")
        self.label_17.setPixmap(QtGui.QPixmap(self.file))
        self.label_17.setScaledContents(True)
        self.label_17.setObjectName("label_17")
        self.center_view = QtWidgets.QGraphicsView(self.center)
        self.center_view.setGeometry(QtCore.QRect(0, 0, 966, 723))
        self.center_view.setObjectName("center_view")
        self.gridLayoutWidget = QtWidgets.QWidget(self.center)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(0, 730, 961, 97))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout_8.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.label_18 = QtWidgets.QLabel(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_18.setFont(font)
        self.label_18.setObjectName("label_18")
        self.gridLayout_8.addWidget(self.label_18, 1, 0, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_8.addItem(spacerItem4, 1, 2, 1, 1)
        self.pushButton_17 = QtWidgets.QPushButton(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pushButton_17.setFont(font)
        self.pushButton_17.setObjectName("pushButton_17")
        self.gridLayout_8.addWidget(self.pushButton_17, 1, 3, 1, 1)
        self.Execute2 = QtWidgets.QPushButton(self.gridLayoutWidget)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(127, 255, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(63, 255, 63))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 127, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 170, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(127, 255, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(127, 255, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(63, 255, 63))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 127, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 170, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(127, 255, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 127, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(127, 255, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(63, 255, 63))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 127, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 170, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 127, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 127, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipText, brush)
        self.Execute2.setPalette(palette)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.Execute2.setFont(font)
        self.Execute2.setAutoFillBackground(True)
        self.Execute2.setAutoDefault(False)
        self.Execute2.setDefault(False)
        self.Execute2.setFlat(True)
        self.Execute2.setObjectName("Execute2")
        self.gridLayout_8.addWidget(self.Execute2, 2, 5, 1, 1)
        self.Add = QtWidgets.QPushButton(self.gridLayoutWidget)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(127, 255, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(63, 255, 63))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 127, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 170, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(127, 255, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(127, 255, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(63, 255, 63))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 127, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 170, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(127, 255, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 127, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(127, 255, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(63, 255, 63))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 127, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 170, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 127, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 127, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipText, brush)
        self.Add.setPalette(palette)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.Add.setFont(font)
        self.Add.setAutoFillBackground(True)
        self.Add.setFlat(True)
        self.Add.setObjectName("Add")
        self.gridLayout_8.addWidget(self.Add, 1, 5, 1, 1)
        self.comboBox = QtWidgets.QComboBox(self.gridLayoutWidget)
        self.comboBox.setObjectName("comboBox")
        self.gridLayout_8.addWidget(self.comboBox, 1, 1, 1, 1)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_8.addItem(spacerItem5, 1, 4, 1, 1)
        self.pushButton_19 = QtWidgets.QPushButton(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pushButton_19.setFont(font)
        self.pushButton_19.setObjectName("pushButton_19")
        self.gridLayout_8.addWidget(self.pushButton_19, 2, 3, 1, 1)
        self.label_19 = QtWidgets.QLabel(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_19.setFont(font)
        self.label_19.setObjectName("label_19")
        self.gridLayout_8.addWidget(self.label_19, 0, 2, 1, 3)
        self.center_view.raise_()
        self.label_17.raise_()
        self.gridLayoutWidget.raise_()
        self.tabWidget.addTab(self.center, "")
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(0, 850, 971, 23))
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setAutoFillBackground(False)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Clear1.setText(_translate("MainWindow", "Clear"))
        self.label_4.setText(_translate("MainWindow", "Identify the first order Bragg Peaks, fill in the parameters below and click execute"))
        self.Execute1.setText(_translate("MainWindow", "Execute"))
        self.Lat_par_a.setText(_translate("MainWindow", "Lattice Parameter (a):"))
        self.save.setText(_translate("MainWindow", "Save Calibration"))
        self.Automate1.setText(_translate("MainWindow", "Automate"))
        self.Energy.setText(_translate("MainWindow", "Energy:"))
        self.Shape.setText(_translate("MainWindow", "Shape:"))
        self.lat_par_b.setText(_translate("MainWindow", "Lattice Parameter (b):"))
        self.Shape_combo.setItemText(0, _translate("MainWindow", "Hexagon"))
        self.Shape_combo.setItemText(1, _translate("MainWindow", "Sqaure"))
        self.Shape_combo.setItemText(2, _translate("MainWindow", "Rectangle"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.calibrate), _translate("MainWindow", "Calibrate"))
        self.label_18.setText(_translate("MainWindow", "Filename"))
        self.pushButton_17.setText(_translate("MainWindow", "Clear"))
        self.Execute2.setText(_translate("MainWindow", "Execute"))
        self.Add.setText(_translate("MainWindow", "Add"))
        self.pushButton_19.setText(_translate("MainWindow", "Automate"))
        self.label_19.setText(_translate("MainWindow", "Click on center of LEED, or click on the 1st Order Bragg Peaks and press execute"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.center), _translate("MainWindow", "Center"))
        

        
class MovingObject(QGraphicsEllipseItem):
    def __init__(self, x, y, r):
        super(MovingObject, self).__init__(0, 0, r, r)
        self.setPos(x, y)
        self.setBrush(Qt.yellow)
        self.setAcceptHoverEvents(True)

    # mouse hover event
    def hoverEnterEvent(self, event):
        app.instance().setOverrideCursor(Qt.OpenHandCursor)

    def hoverLeaveEvent(self, event):
        app.instance().setOverrideCursor(Qt.CrossCursor)

    # mouse click event
    def mousePressEvent(self, event):
        pass

    def mouseMoveEvent(self, event):
        orig_cursor_position = event.lastScenePos()
        updated_cursor_position = event.scenePos()

        orig_position = self.scenePos()

        self.updated_cursor_x = updated_cursor_position.x() - orig_cursor_position.x() + orig_position.x()
        self.updated_cursor_y = updated_cursor_position.y() - orig_cursor_position.y() + orig_position.y()
        self.setPos(QPointF(self.updated_cursor_x, self.updated_cursor_y))

    def mouseReleaseEvent(self, event):
        print('x: {0}, y: {1}'.format(self.pos().x(), self.pos().y()))
        #return self.pos().x(), self.pos().y()
        
    #def deleteEvent(self, event):
        #if qt.OpenHandCursor:
            
        

#import test_rc

file = r"C:\Users\lsh8\Documents\LEED_16-07-19\LEED_16-07-19\GM2-365_IV_100.tiff"

class CalibrateGUI(QtWidgets.QMainWindow):
     def __init__(self, file):
         super(CalibrateGUI, self).__init__()
         self.file = file
         self.ui = Ui_MainWindow(file)
         self.ui.setupUi(self)
         #self.ui.cal_view.instance().setOverrideCursor(Qt.CrossCursor)
         self.circle = []
         self.circle_item = []
         #self.t = GraphicView(self.file)
         #self.initializeUI()
         
         self.ui.Clear1.clicked.connect(self.Clear1Clicked)
         self.ui.Execute1.clicked.connect(self.ExecuteClicked)
         self.ui.Automate1.clicked.connect(self.AutomateButtonClicked)
         self.show()
         
     def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.moveObject = MovingObject(event.x()-7.5-2, event.y()-7.5-20, 15)
            print(event.x())
            print(event.y())
            self.ui.cal_view.addItem(self.moveObject)
            #self.circle_item.append[obj]
            cursor = QtGui.QCursor()
            self.circle.append(self.moveObject)
            print(self.circle[0].pos().y())
            print(len(self.circle))
            #print(self.moveObject.pos().x())
            
     def wheelEvent(self, event):
 # if event.delta() > 0: # Roller up, PyQt4
        # This function has been deprecated, use pixelDelta() or angleDelta() instead.
        angle=event.angleDelta() / 8 # Returns the QPoint object, the value of the wheel, in 1/8 degrees
        angleX=angle.x()/10 # The distance rolled horizontally (not used here)
        angleY=angle.y()/10 # The distance that is rolled vertically
        if angleY > 0:
                         print(angleY)
                         print(self.moveObject) # response test statement
        else: #roll down
                         print(angleY)
                         print("mouse wheel down") # response test statement
                         
     def Clear1Clicked(self):
         self.ui.cal_view.clear()
         self.test = QPixmap(self.file)
         self.test = self.test.scaledToWidth(960)
         self.ui.cal_view.addPixmap(self.test)
         del self.circle[:]
         
     def AutomateButtonClicked(self,event):
        
        #image = region.data.astype(np.uint8)
        gray = cv2.imread(self.file, 0)
        print(gray)
        #gray.astype(np.uint8)
        #output = image.copy()
        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #ret,grays = cv2.threshold(gray,10,255,cv2.THRESH_BINARY)
        #grays = cv2.Canny(grays,0,20)
            
        #Blob detection method not yet fully implemented
        blobs_log = blob_log(gray, max_sigma=30, num_sigma=10,min_sigma=5, threshold=.1, overlap =0)
        x_vals = []
        y_vals = []
        for blob in blobs_log:
            y5, x5, r5 = blob
            #Condition to avoid picking up stray marks on screen
            if r5 > 3:
                x_vals.append(x5*3/4)
                y_vals.append(y5*3/4)
            
        x_vals= np.array(x_vals)
        y_vals= np.array(y_vals)
        vals = np.vstack((x_vals,y_vals))
        vals = vals.T
                
        kdtree = KDTree(vals)
    
        auto_bragg = []
        
        a = len(self.circle)
        for m in range(0,a):
            d, i = kdtree.query((self.circle[m].pos().x(),self.circle[m].pos().y()))
            print("closest point:", vals[i])
            auto_bragg.append(vals[i])
            print(self.circle[m].pos().x()*3/2)   
        
        self.ui.cal_view.clear()
        self.test = QPixmap(self.file)
        self.test = self.test.scaledToWidth(960)
        self.ui.cal_view.addPixmap(self.test)
        del self.circle[:]
        
        for m in range(0,a):
            self.moveObject = MovingObject(auto_bragg[m][0], auto_bragg[m][1], 15)
            self.ui.cal_view.addItem(self.moveObject)
                
     def ExecuteClicked(self):        
        #Check that enough Bragg peaks have been selected
        
        if len(self.circle)==0:
            print("Select Bragg peaks")
            
        elif len(self.circle) == 6 and str(self.ui.Shape_combo.currentText()) == 'Hexagon':
            
            f = np.array([self.circle[0].pos().x(),self.circle[0].pos().y()])
            g = np.array([self.circle[1].pos().x(),self.circle[1].pos().y()])
            h = np.array([self.circle[2].pos().x(),self.circle[2].pos().y()])
            taps_nparray = np.vstack((f,g,h))

            #Locate center of the 6 Bragg peaks
            center = np.ones((3, 2))
            for i in range(0,3):
                    center[0+i][0] = (self.circle[0+i].pos().x()+self.circle[3+i].pos().x())/2
                    center[0+i][1] = (self.circle[0+i].pos().y()+self.circle[3+i].pos().y())/2
            self.center = center.mean(axis=0)

            #taps_nparray = np.array((self.point_stream.data['x'],self.point_stream.data['y'])).T
            #Find the Bragg peak with smallest y
            print(taps_nparray)
            A=-taps_nparray[:,1].min()+center[0][1]
            #Identify the index for this Bragg peak
            index_value=taps_nparray[:,1].argmin()
            #Find the x position of this Bragg peak
            O=taps_nparray[index_value][0]-center[0][0]
            #Calculate the angle from the vertical (anticlockwise)
            self.angle = np.arctan(float(O/A))*180/np.pi# + self.data.spectrum.attrs['azi']

            print(self.ui.lineEdit_3.text())
            #Calulate camera factor - relationship between pixel and angle
            q_coord = 4*np.pi/(np.sqrt(3)*float(self.ui.lineEdit_3.text()))
            camera_factor = q_coord/np.sqrt((self.circle[0].pos().x()-center[0][0])**2 +(self.circle[0].pos().y()-center[0][1])**2)

            #Assign all useful calibration variables
            self.camera_factor = camera_factor
            #self.cal_energy = self.energy
            self.center = self.center.astype(int)

            #Display all values
            print("Center of LEED image is {0}".format(self.center))
            print("Camera factor is {0}".format(self.camera_factor))
            print("Azimuth offset is {0}".format(self.angle))
                
            #self.data.attrs['center'] = self.center
            #self.data.attrs['camera_factor'] = self.camera_factor
            #self.data.attrs['calibration_energy'] = self.energy.value 
            
            #for ii in self.data.data_vars:
                #self.data[ii].attrs['center'] = self.data.attrs['center']
                #self.data[ii].attrs['camera_factor'] = self.data.attrs['camera_factor']
                #self.data[ii].attrs['calibration_energy'] = self.data.attrs['calibration_energy']

        elif len(self.point_stream.data['x']) == 4 and str(self.ui.Shape_combo.currentText()) == 'Square':

            #Locate center of the 4 Bragg peaks
            center = np.ones((2, 2))
            for i in range(0,2):
                    center[0+i][0] = (self.point_stream.data['x'][0+i]+self.point_stream.data['x'][2+i])/2
                    center[0+i][1] = (self.point_stream.data['y'][0+i]+self.point_stream.data['y'][2+i])/2
            self.center = center.mean(axis=0)

            taps_nparray = np.array((self.point_stream.data['x'],self.point_stream.data['y'])).T
            #Find the Bragg peak with smallest y
            A=-taps_nparray[:,1].min()+center[0][1]
            #Identify the index for this Bragg peak
            index_value=taps_nparray[:,1].argmin()
            #Find the x position of this Bragg peak
            O=taps_nparray[index_value][0]-center[0][0]
            #Calculate the angle from the vertical (anticlockwise)
            self.angle = np.arctan(float(O/A))*180/np.pi + self.data.spectrum.attrs['azi']

            #Calulate camera factor - relationship between pixel and angle
            q_coord = 2*np.pi/(self.lattice.value)
            camera_factor = q_coord/np.sqrt((self.point_stream.data['x'][0]-center[0][0])**2 +(self.point_stream.data['y'][0]-center[0][1])**2)

            #Assign all useful calibration variables
            self.camera_factor = camera_factor
            self.cal_energy = self.energy
            self.center = self.center.astype(int)
            
            self.data.attrs['center'] = self.center

            #Display all values
            with self.output:
                print("Center of LEED image is {0}".format(self.center))
                print("Camera factor is {0}".format(self.camera_factor))
                print("Azimuth offset is {0}".format(self.angle))
                
            self.data.attrs['center'] = self.center
            self.data.attrs['camera_factor'] = self.camera_factor
            self.data.attrs['calibration_energy'] = self.energy.value 
            
            for ii in self.data.data_vars:
                self.data[ii].attrs['center'] = self.data.attrs['center']
                self.data[ii].attrs['camera_factor'] = self.data.attrs['camera_factor']
                self.data[ii].attrs['calibration_energy'] = self.data.attrs['calibration_energy']


        elif len(self.point_stream.data) > 4 and self.shape.value == 'Square':
            with self.output:
                print("Too many Bragg peaks for a square Brillouin zone")

        else:
            with self.output:
                print("Not enough Bragg peaks")

        #def cell_below():
            #display(Javascript("""
            #var cell = IPython.notebook.insert_cell_below('code')
            #cell.set_text("print('center is {0}'.format(test.center))
            #cell.execute()
            #"""))

        #cell_below()
        

    

app = QtWidgets.QApplication(sys.argv)
GUI = CalibrateGUI(file)
sys.exit(app.exec_())