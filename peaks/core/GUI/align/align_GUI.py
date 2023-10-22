# Run Align FS Panel
# Phil King
# 19/6/2021
# LH modified 02/10/21

import xarray as xr
from PyQt6 import QtWidgets
from peaks.core.GUI.pyqt_WindowItem import WindowItem
from peaks.core.GUI.align.align_FS_panel import Ui_AlignFS_panel
from peaks.core.GUI.align.align_disp_panel import Ui_Aligndisp_panel
from peaks.core.utils.OOP_method import add_methods
import sys

@add_methods(xr.DataArray)
def align(data):
    ''' Master load function for GUI data alignment panels. Currently works with single 2D dataarray,
        and a 3D dataarray of Fermi map data.

            Input:
                data - the data to align

            Returns:
                null - opens the relevant data alignment panel '''
    ndim = len(data.dims)
    if ndim == 2:
        align_disp(data)
    elif ndim == 3:
        align_FS(data.transpose(...,'theta_par','eV'))  # NB Transposition required to get array ordering right for quick scrolling in the GUI
    else:
        print('Data type not currently supported for interactive display')


def align_FS(data):
    
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
    align=Ui_AlignFS_panel(WindowItem())

    # Show the Panel
    align.AlignFS_panel.show()

    # Retrieve input file
    align.GUI_files(data)

    # Run the app main loop
    #sys.exit(app.exec_())
    app.exec()

def align_disp(data):
    
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
    align = Ui_Aligndisp_panel(WindowItem())

    # Show the Panel
    align.Aligndisp_panel.show()

    # Retrieve input file
    align.GUI_files(data)

    # Run the app main loop
    app.exec()