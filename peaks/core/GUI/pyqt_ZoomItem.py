# ZoomItem for pyqtgraph
# Edgar Abarca Morales
from PyQt6 import QtCore
import numpy as np
import pyqtgraph as pg

class ZoomItem:

    def __init__(self, Parent):

        # Set required parameters
        self.Parent=Parent

        # Margin defining the ZoomItem boundaries as a percentage of the SceneRange
        self.mg=0.01

        # Signal when the mouse is clicked over self.Parent
        self.Parent.getViewBox().scene().sigMouseClicked.connect(self.mouseDoubleClickEvent)

    # Define the space for the zoom item
    # (Data boundaries within the scene in self.Parent)
    def setSpace(self,space):
        self.space=space

    # Main zoom function
    def mouseDoubleClickEvent(self,ev):

        # Return the mouse button that generated the click event
        # If it was not the mouse left button ignore the event
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            ev.ignore()
            return

        # Ignore single-click events
        if ev.double() == False:
            ev.ignore()
            return

        # Run if the event is a double-click
        if ev.double() == True:

            # Get the mouse position at the time of the double-click
            pos1 = self.Parent.getViewBox().mapSceneToView(ev.scenePos())

            # Ignore double-click events outside the data in self.Parent
            if pos1.x() < self.space[0][0] or pos1.x() > self.space[1][0] or pos1.y() < self.space[0][1] or pos1.y() > self.space[1][1]:
                ev.ignore()
                return

            # Run if the double-click event ocurred within self.Parent
            else:

                # Create a point in self.Parent at the double-click location (fixed)
                iPoint=pg.GraphItem(pos=np.array([[pos1.x(),pos1.y()]]), dtype=float, size=15, symbol=['+'], brush='y')

                # Create a point in self.Parent at the double-click location (movable)
                fPoint=pg.GraphItem(pos=np.array([[pos1.x(),pos1.y()]]), dtype=float, size=15, symbol=['+'], brush='y')

                # Create a ROI item
                Region=pg.ROI(pos=(pos1.x(),pos1.y()),size=(0,0),pen='b',hoverPen='b',handlePen='b',handleHoverPen='b')

                # Add the points to self.Parent
                self.Parent.addItem(iPoint)
                self.Parent.addItem(fPoint)

                # Add the ROI to self.Parent
                self.Parent.addItem(Region)

                # Create a mouseMoved event
                def mouseMovedEvent(mov):

                    # Get the current range of the scene
                    SceneRange=self.Parent.getViewBox().viewRange()

                    # Get the mouse position along the mouse movement
                    pos2 = self.Parent.getViewBox().mapSceneToView(mov)

                    ############################################################

                    # Keep the cursors within the data in MainPlot
                    if self.space[0][0] <= pos2.x() <= self.space[1][0] and self.space[0][1] <= pos2.y() <= self.space[1][1]:

                        # Update the position of the movable point
                        fPoint.setData(pos=np.array([[pos2.x(),pos2.y()]]), dtype=float, size=15, symbol=['+'], brush='y')

                        # Update the ROI
                        Region.setSize((pos2.x()-pos1.x(),pos2.y()-pos1.y()))

                    # Do not let the cursor stick in the top and bottom edges
                    elif self.space[0][0] <= pos2.x() <= self.space[1][0] and (self.space[0][1] > pos2.y() or pos2.y() > self.space[1][1]):

                        # Let the cursor arrive to the top and bottom edges even for fast mouse movements
                        if self.space[0][1] > pos2.y():

                            # Update the position of the movable point
                            fPoint.setData(pos=np.array([[pos2.x(),self.space[0][1]]]), dtype=float, size=15, symbol=['+'], brush='y')

                            # Update the ROI
                            Region.setSize((pos2.x()-pos1.x(),self.space[0][1]-pos1.y()))

                        else:

                            # Update the position of the movable point
                            fPoint.setData(pos=np.array([[pos2.x(),self.space[1][1]]]), dtype=float, size=15, symbol=['+'], brush='y')

                            # Update the ROI
                            Region.setSize((pos2.x()-pos1.x(),self.space[1][1]-pos1.y()))

                    # Do not let the cursor stick in the left and right edges
                    elif (self.space[0][0] > pos2.x() or pos2.x() > self.space[1][0]) and self.space[0][1] <= pos2.y() <= self.space[1][1]:

                        # Let the cursor arrive to the left and right edges even for fast mouse movements
                        if self.space[0][0] > pos2.x():

                            # Update the position of the movable point
                            fPoint.setData(pos=np.array([[self.space[0][0],pos2.y()]]), dtype=float, size=15, symbol=['+'], brush='y')

                            # Update the ROI
                            Region.setSize((self.space[0][0]-pos1.x(),pos2.y()-pos1.y()))

                        else:

                            # Update the position of the movable point
                            fPoint.setData(pos=np.array([[self.space[1][0],pos2.y()]]), dtype=float, size=15, symbol=['+'], brush='y')

                            # Update the ROI
                            Region.setSize((self.space[1][0]-pos1.x(),pos2.y()-pos1.y()))

                    # Let the cursor arrive to the corners horizontally even for fast mouse movements
                    elif (self.space[0][0] > pos2.x() or pos2.x() > self.space[1][0]) and (self.space[0][1] > pos2.y() or pos2.y() > self.space[1][1]):

                        # Left-Down corner
                        if self.space[0][0] > pos2.x() and self.space[0][1] > pos2.y():

                            # Update the position of the movable point
                            fPoint.setData(pos=np.array([[self.space[0][0],self.space[0][1]]]), dtype=float, size=15, symbol=['+'], brush='y')

                            # Update the ROI
                            Region.setSize((self.space[0][0]-pos1.x(),self.space[0][1]-pos1.y()))

                        # Right-Down corner
                        elif pos2.x() > self.space[1][0] and self.space[0][1] > pos2.y():

                            # Update the position of the movable point
                            fPoint.setData(pos=np.array([[self.space[1][0],self.space[0][1]]]), dtype=float, size=15, symbol=['+'], brush='y')

                            # Update the ROI
                            Region.setSize((self.space[1][0]-pos1.x(),self.space[0][1]-pos1.y()))

                        # Left-Up corner
                        elif self.space[0][0] > pos2.x() and pos2.y() > self.space[1][1]:

                            # Update the position of the movable point
                            fPoint.setData(pos=np.array([[self.space[0][0],self.space[1][1]]]), dtype=float, size=15, symbol=['+'], brush='y')

                            # Update the ROI
                            Region.setSize((self.space[0][0]-pos1.x(),self.space[1][1]-pos1.y()))

                        # Right-Up corner
                        elif pos2.x() > self.space[1][0] and pos2.y() > self.space[1][1]:

                            # Update the position of the movable point
                            fPoint.setData(pos=np.array([[self.space[1][0],self.space[1][1]]]), dtype=float, size=15, symbol=['+'], brush='y')

                            # Update the ROI
                            Region.setSize((self.space[1][0]-pos1.x(),self.space[1][1]-pos1.y()))

                    ############################################################

                    # Disconnect signals if the mouse goes outside self.Parent
                    if pos2.x() < SceneRange[0][0] + (SceneRange[0][1]-SceneRange[0][0])*self.mg or pos2.x() > SceneRange[0][1] - (SceneRange[0][1]-SceneRange[0][0])*self.mg or pos2.y() < SceneRange[1][0] + (SceneRange[1][1]-SceneRange[1][0])*self.mg or pos2.y() > SceneRange[1][1] - (SceneRange[1][1]-SceneRange[1][0])*self.mg:

                        # Disconnect mouse movement events
                        self.Parent.getViewBox().scene().sigMouseMoved.disconnect(mouseMovedEvent)

                        # Disconnect single-click events
                        self.Parent.getViewBox().scene().sigMouseClicked.disconnect(mouseClickEvent)

                        # Delete the points and ROI
                        self.Parent.removeItem(iPoint)
                        self.Parent.removeItem(fPoint)
                        self.Parent.removeItem(Region)

                # Create a mouseClicked event
                def mouseClickEvent(one):

                    # Return the mouse button that generated the click event
                    # If it was not the mouse left button ignore the event
                    if one.button() != QtCore.Qt.MouseButton.LeftButton:
                        one.ignore()
                        return

                    # Disconnect mouse movement events
                    self.Parent.getViewBox().scene().sigMouseMoved.disconnect(mouseMovedEvent)

                    # Disconnect single-click events
                    self.Parent.getViewBox().scene().sigMouseClicked.disconnect(mouseClickEvent)

                    # Update self.Parent range
                    self.Parent.getViewBox().setXRange(min(Region.pos()[0],Region.pos()[0]+Region.size()[0]),max(Region.pos()[0],Region.pos()[0]+Region.size()[0]),padding=0)
                    self.Parent.getViewBox().setYRange(min(Region.pos()[1],Region.pos()[1]+Region.size()[1]),max(Region.pos()[1],Region.pos()[1]+Region.size()[1]),padding=0)

                    # Delete the points and ROI
                    self.Parent.removeItem(iPoint)
                    self.Parent.removeItem(fPoint)
                    self.Parent.removeItem(Region)

                # Signal when the mouse is clicked over self.Parent
                self.Parent.getViewBox().scene().sigMouseClicked.connect(mouseClickEvent)

                # Signal when the mouse is moved over self.Parent
                self.Parent.getViewBox().scene().sigMouseMoved.connect(mouseMovedEvent)
