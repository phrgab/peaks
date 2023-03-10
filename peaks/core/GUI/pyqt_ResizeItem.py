# ResizeItem for PyQt5
# Edgar Abarca Morales

from PyQt6 import QtCore

class ResizeItem:

    def __init__(self, Parent, Child):

        # Initial Parent's geometry
        self.W0 = Parent.geometry().width()
        self.H0 = Parent.geometry().height()

        # Initial Child's geometry
        self.x0 = Child.geometry().x() # Absolute position relative to Parent (0,0)
        self.y0 = Child.geometry().y() # Absolute position relative to Parent (0,0)
        self.w0 = Child.geometry().width()
        self.h0 = Child.geometry().height()

        # Child's relative x-position (fixed in Ui design)
        self.rx = self.x0/self.W0

        # Child's relative y-position (fixed in Ui design)
        self.ry = self.y0/self.H0

        # Relative resize function
        def relative_resize(rx,ry,W0,H0,Wf,Hf,w0,h0):

            # Final Child's x-position
            xf = rx*Wf

            # Final Child's y-position
            yf = ry*Hf

            # Final Child's width
            wf = (Wf/W0)*w0

            # Final Child's height
            hf = (Hf/H0)*h0

            return xf, yf, wf, hf

        # Resize widgets function
        def widget_resize():

            # Get current Parent's geometry
            Wf = Parent.geometry().width()
            Hf = Parent.geometry().height()

            # Calculate new Child's geometry
            xf, yf, wf, hf = relative_resize(self.rx,self.ry,self.W0,self.H0,Wf,Hf,self.w0,self.h0)

            # Set new Child's geometry
            Child.setGeometry(QtCore.QRect(int(xf), int(yf), int(wf), int(hf)))

            # Update initial Parent's geometry
            self.W0 = Wf
            self.H0 = Hf

            # Update initial Child's geometry
            self.w0 = wf
            self.h0 = hf

        # Signal when Parent's window is resized
        Parent.resized.connect(widget_resize)
