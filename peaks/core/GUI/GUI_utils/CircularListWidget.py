# CircularListWidget for PyQt5
# Edgar Abarca Morales
from PyQt6 import QtCore, QtWidgets


# Upgrade QtWidgets.QListWidget to support infinite arrow scrolling
class _CircularListWidget(QtWidgets.QListWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def keyPressEvent(self, event):
        # Circular scrolling behaviour
        if event.key() == QtCore.Qt.Key.Key_Down:
            if self.currentRow() == self.count() - 1:
                self.setCurrentRow(0)
                return
        elif event.key() == QtCore.Qt.Key.Key_Up:
            if self.currentRow() == 0:
                self.setCurrentRow(self.count() - 1)
                return

        # Parent behavior
        super().keyPressEvent(event)
