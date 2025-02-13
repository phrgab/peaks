import pyqtgraph as pg
from PyQt6 import QtCore


class _KeyPressGraphicsLayoutWidget(pg.GraphicsLayoutWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    keyPressed = QtCore.pyqtSignal(object)
    keyReleased = QtCore.pyqtSignal(object)

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        self.keyPressed.emit(event)

    def keyReleaseEvent(self, event):
        super().keyReleaseEvent(event)
        self.keyReleased.emit(event)
