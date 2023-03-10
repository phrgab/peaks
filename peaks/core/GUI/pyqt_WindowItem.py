# QtWidgets.QMainWindow with resize for pyqt5
# Edgar Abarca Morales

from PyQt6 import QtCore, QtWidgets

# Upgrade QtWidgets.QMainWindow to support resize events
class WindowItem(QtWidgets.QMainWindow):

    resized = QtCore.pyqtSignal()

    def  __init__(self, parent=None):
        super(WindowItem, self).__init__(parent=parent)

        # Connect the resize event to a function
        # self.resized.connect(self.someFunction)

    def resizeEvent(self, event):
        self.resized.emit()
        return super(WindowItem, self).resizeEvent(event)

    #def someFunction(self):
        #print("Resized")

# Uncomment to test the class
#if __name__ == "__main__":
    #import sys
    #app = QtWidgets.QApplication(sys.argv)
    #w = WindowItem()
    #w.show()
    #sys.exit(app.exec_())
