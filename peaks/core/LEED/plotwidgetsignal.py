# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 19:30:44 2021

@author: lsh8
"""

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

# Upgrade PlotWidget to support key press events
class PlotWidgetSig(pg.PlotWidget):

    # Initialize PlotWidget class
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Establish a signal for key press events
    sigKeyPress = QtCore.pyqtSignal(object)

    # Add keyPressEvent method to PlotWidget class
    def keyPressEvent(self, ev):
        self.scene().keyPressEvent(ev)
        self.sigKeyPress.emit(ev)