# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'align_disp_panel.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Aligndisp_panel(object):
    def setupUi(self, Aligndisp_panel):
        Aligndisp_panel.setObjectName("Aligndisp_panel")
        Aligndisp_panel.setEnabled(True)
        Aligndisp_panel.resize(746, 523)
        self.centralwidget = QtWidgets.QWidget(Aligndisp_panel)
        self.centralwidget.setMouseTracking(False)
        self.centralwidget.setAutoFillBackground(False)
        self.centralwidget.setObjectName("centralwidget")
        self.MainPlot = PlotWidget(self.centralwidget)
        self.MainPlot.setGeometry(QtCore.QRect(30, 10, 401, 291))
        self.MainPlot.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.MainPlot.setObjectName("MainPlot")
        self.MDC_Plot = PlotWidget(self.centralwidget)
        self.MDC_Plot.setGeometry(QtCore.QRect(30, 300, 401, 171))
        self.MDC_Plot.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.MDC_Plot.setObjectName("MDC_Plot")
        self.graphicsView = GraphicsLayoutWidget(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(440, 140, 301, 111))
        self.graphicsView.setObjectName("graphicsView")
        self.HCutOff = QtWidgets.QSlider(self.centralwidget)
        self.HCutOff.setGeometry(QtCore.QRect(440, 360, 251, 22))
        self.HCutOff.setMaximum(100)
        self.HCutOff.setProperty("value", 100)
        self.HCutOff.setOrientation(QtCore.Qt.Horizontal)
        self.HCutOff.setInvertedAppearance(False)
        self.HCutOff.setInvertedControls(False)
        self.HCutOff.setObjectName("HCutOff")
        self.LCutOff = QtWidgets.QSlider(self.centralwidget)
        self.LCutOff.setGeometry(QtCore.QRect(440, 390, 251, 22))
        self.LCutOff.setMaximum(100)
        self.LCutOff.setProperty("value", 0)
        self.LCutOff.setSliderPosition(0)
        self.LCutOff.setOrientation(QtCore.Qt.Horizontal)
        self.LCutOff.setInvertedAppearance(False)
        self.LCutOff.setInvertedControls(False)
        self.LCutOff.setObjectName("LCutOff")
        self.checkBox = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox.setGeometry(QtCore.QRect(700, 380, 21, 21))
        self.checkBox.setText("")
        self.checkBox.setObjectName("checkBox")
        self.Cmaps = QtWidgets.QComboBox(self.centralwidget)
        self.Cmaps.setGeometry(QtCore.QRect(440, 320, 151, 26))
        self.Cmaps.setEditable(False)
        self.Cmaps.setCurrentText("")
        self.Cmaps.setObjectName("Cmaps")
        self.Cen_label = QtWidgets.QLabel(self.centralwidget)
        self.Cen_label.setGeometry(QtCore.QRect(440, 70, 171, 16))
        self.Cen_label.setObjectName("Cen_label")
        self.Label_X_cen = QtWidgets.QLabel(self.centralwidget)
        self.Label_X_cen.setGeometry(QtCore.QRect(439, 100, 71, 20))
        self.Label_X_cen.setObjectName("Label_X_cen")
        self.X_cen = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.X_cen.setGeometry(QtCore.QRect(520, 100, 91, 24))
        self.X_cen.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.X_cen.setObjectName("X_cen")
        self.Delta_X_cen = QtWidgets.QLineEdit(self.centralwidget)
        self.Delta_X_cen.setGeometry(QtCore.QRect(620, 100, 61, 24))
        self.Delta_X_cen.setObjectName("Delta_X_cen")
        self.Auto_cen = QtWidgets.QPushButton(self.centralwidget)
        self.Auto_cen.setGeometry(QtCore.QRect(550, 60, 113, 32))
        self.Auto_cen.setObjectName("Auto_cen")
        self.Label_dE = QtWidgets.QLabel(self.centralwidget)
        self.Label_dE.setGeometry(QtCore.QRect(460, 20, 60, 16))
        self.Label_dE.setObjectName("Label_dE")
        self.dE_select = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.dE_select.setGeometry(QtCore.QRect(560, 20, 91, 24))
        self.dE_select.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.dE_select.setObjectName("dE_select")
        self.set_norm = QtWidgets.QPushButton(self.centralwidget)
        self.set_norm.setGeometry(QtCore.QRect(630, 440, 113, 32))
        self.set_norm.setObjectName("set_norm")
        self.exit = QtWidgets.QPushButton(self.centralwidget)
        self.exit.setGeometry(QtCore.QRect(520, 440, 113, 32))
        self.exit.setObjectName("exit")
        Aligndisp_panel.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Aligndisp_panel)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 746, 22))
        self.menubar.setObjectName("menubar")
        Aligndisp_panel.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(Aligndisp_panel)
        self.statusbar.setObjectName("statusbar")
        Aligndisp_panel.setStatusBar(self.statusbar)

        self.retranslateUi(Aligndisp_panel)
        QtCore.QMetaObject.connectSlotsByName(Aligndisp_panel)

    def retranslateUi(self, Aligndisp_panel):
        _translate = QtCore.QCoreApplication.translate
        Aligndisp_panel.setWindowTitle(_translate("Aligndisp_panel", "Align dispersion"))
        self.Cen_label.setText(_translate("Aligndisp_panel", "Centering:"))
        self.Label_X_cen.setText(_translate("Aligndisp_panel", "theta_par"))
        self.Delta_X_cen.setText(_translate("Aligndisp_panel", "1"))
        self.Auto_cen.setText(_translate("Aligndisp_panel", "Auto"))
        self.Label_dE.setText(_translate("Aligndisp_panel", "dE"))
        self.set_norm.setText(_translate("Aligndisp_panel", "Set and Exit"))
        self.exit.setText(_translate("Aligndisp_panel", "Close"))


from pyqtgraph import GraphicsLayoutWidget, PlotWidget


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Aligndisp_panel = QtWidgets.QMainWindow()
    ui = Ui_Aligndisp_panel()
    ui.setupUi(Aligndisp_panel)
    Aligndisp_panel.show()
    sys.exit(app.exec_())
