# Run Fit Panel
# Tommaso Antonelli 29/10/2021

from peaks.core.GUI.fit1d.fit_GUI8 import Fit_GUI
from peaks.core.GUI.fit1d.SelectDC_GUI import SelectDC_GUI
from peaks.core.GUI.pyqt_WindowItem import WindowItem
from peaks.core.fit import *
from PyQt6 import QtWidgets
import sys


#debug option at the end of the code

def fit1d(files, model_tot = FitFunc()):
    """
    open fit panel for 1D xarray in a PyQt5 window
    :param files: x1D or 2D array
    :param model_tot: fitting model as FitFunc instance to be loaded in the fit panel
    """


    # Declare app as a global attribute so that it is not removed after execution
    global app

    # Do not make a new instance if an instance of app already exists
    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = QtWidgets.QApplication.instance()

    if len(files.dims)==2:
        # Create a Display Panel in a WindowItem to select DC to fit
        # (See pyqt_WindowItem.py)
        SelectDC = SelectDC_GUI(WindowItem())

        # Show the SelectDC panel
        SelectDC.SelectDC_GUI.show()

        # Retrieve input files
        SelectDC.GUI_files([files])

        # Run the app main loop
        app.exec_()
        files = SelectDC.send2FP

    # Create a Display Panel in a WindowItem
    # (See pyqt_WindowItem.py)
    FP=Fit_GUI(WindowItem())

    # Show the Display Panel
    FP.MainWindow.show()

    # Retrieve input files
    FP.GUI_files(files)
    # Retrieve model tot
    FP.GUI_LoadModel(model_tot)

    # Run the app main loop
    app.exec_()

##### for debugging####
debug = False

if debug:
    a = load_data("peaks/examples/example_data/ARPES/i05-59819.nxs") #Diamond i05-HR Disp.
    files=EDC(a,-9)
    fit1d(files)
