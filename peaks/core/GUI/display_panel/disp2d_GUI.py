# Run Display Panel
# Edgar Abarca Morales
# Edit LH 11/08/2021
# Minor edits EAM 25/10/2021
from PyQt6 import QtWidgets
from peaks.core.GUI.pyqt_WindowItem import WindowItem
from .DP import GUI_DisplayPanel
import sys

def disp2d(files):
    
    # Declare app as a global attribute so that it is not removed after execution
    global app
    
    # Do not make a new instance if an instance of app already exists
    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = QtWidgets.QApplication.instance() 

    # Create a Display Panel in a WindowItem
    # (See pyqt_WindowItem.py)
    DP=GUI_DisplayPanel(WindowItem())

    # Show the Display Panel
    DP.DisplayPanel.show()

    # Retrieve input files
    DP.GUI_files(files)

    # Run the app main loop
    app.exec()
