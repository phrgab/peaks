# Run Align FS Panel
# Phil King
# 19/6/2021
#LH modified 2/10/21

from PyQt6 import QtWidgets
from peaks.core.GUI.pyqt_WindowItem import WindowItem
from peaks.core.GUI.spatial_map.disp4d_panel import Ui_disp4d_Panel
import sys

def disp4d(data):

    # App should be a global attribute so that
    # it is not removed when the function has been executed
    global app

    # If an instance of an app exists - do not make
    # a new instance
    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = QtWidgets.QApplication.instance() 

    # Create a new app
    #app = QtWidgets.QApplication(sys.argv)

    # Create an alignment Panel in a WindowItem
    # (See pyqt_WindowItem.py)
    disp4d=Ui_disp4d_Panel(WindowItem())

    # Show the Panel
    disp4d.disp4d_panel.show()

    # Retrieve input file
    disp4d.GUI_files(data)

    # Run the app main loop
    app.exec()