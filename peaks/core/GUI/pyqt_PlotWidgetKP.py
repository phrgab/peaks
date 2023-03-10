# PlotWidget with KeyPress & KeyRelease for pyqtgraph
# Edgar Abarca Morales

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

# Upgrade PlotWidget to support key press events
class PlotWidgetKP(pg.PlotWidget):

    # Initialize PlotWidget class
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Establish a signal for key press events
    sigKeyPress = QtCore.pyqtSignal(object)

    # Add keyPressEvent method to PlotWidget class
    def keyPressEvent(self, ev):
        self.scene().keyPressEvent(ev)
        self.sigKeyPress.emit(ev)

    # Establish a signal for key release events
    sigKeyRelease = QtCore.pyqtSignal(object)

    # Add keyReleaseEvent method to PlotWidget class
    def keyReleaseEvent(self, ev):
        self.scene().keyReleaseEvent(ev)
        self.sigKeyRelease.emit(ev)

################################################################################

# Uncomment to test the class

#def keyPressed(evt):
    #print("Key pressed")

#def keyReleased(evt):
    #print("Key released")

#if __name__ == "__main__":
    #import sys
    #app = QtWidgets.QApplication(sys.argv)
    #win = PlotWidgetKP()
    #win.sigKeyPress.connect(keyPressed)
    #win.sigKeyRelease.connect(keyReleased)
    #win.show()
    #sys.exit(app.exec_())
