# CursorItem for pyqtgraph
# Edgar Abarca Morales
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

# Subclass from pyqtgraph.GraphItem
class CursorItem(pg.GraphItem):

    # Execute when CursorItem object is created
    def __init__(self):

        # Define the point being dragged
        self.dragPoint = None

        # Distance between the point position and the cursor when the mouse button is clicked over a point
        self.dragOffset = None

        # Label item for the point
        self.textItem = []

        # Crosshair item for the point
        self.vcrosshairItem = [] # Vertical
        self.hcrosshairItem = [] # Horizontal

        # Integration-crosshair item for the point
        self.ivcrosshairItem = [] # Vertical
        self.ihcrosshairItem = [] # Horizontal

        # Initialize pyqtgraph.GraphItem class (CursorItem inherits the properties of GraphItem)
        # In particular (see GraphItem source code):
        # GraphItem.scatter = ScatterPlotItem
        # Thus, by inheritance: CursorItem.scatter = ScatterPlotItem
        # Then, the signal tracking defined for a ScatterPlotItem is available for CursorItem.scatter
        pg.GraphItem.__init__(self)

    # Define the space for the cursor
    def setSpace(self,space):
        self.space=space

    # Set the data for CursorItem object
    def setData(self, **kwds):

        # Remove 'text' item from kwds and save it as self.text
        self.text = kwds.pop('text', [])

        # Remove 'textc' item from kwds and save it as self.textc
        self.textc = kwds.pop('textc', [])

        # Remove 'csr' item from kwds and save it as self.csr
        self.csr = kwds.pop('csr', [])

        # Remove 'csrc' item from kwds and save it as self.csrc
        self.csrc = kwds.pop('csrc', [])

        # Remove 'ispanx' item from kwds and save it as self.ispanx
        self.ispanx = kwds.pop('ispanx', [])

        # Remove 'ispany' item from kwds and save it as self.ispany
        self.ispany = kwds.pop('ispany', [])

        # Remove 'icsr' item from kwds and save it as self.icsr
        self.icsr = kwds.pop('icsr', [])

        # Remove 'icsrc' item from kwds and save it as self.icsrc
        self.icsrc = kwds.pop('icsrc', [])

        # Remove 'ibrush' item from kwds and save it as self.ibrush
        self.ibrush = kwds.pop('ibrush', [])

        # Remove 'h' item from kwds and save it as self.h
        self.h = kwds.pop('h', [])

        # Remove 'v' item from kwds and save it as self.v
        self.v = kwds.pop('v', [])

        # Save the remaining items in kwds as self.data
        self.data = kwds

        # Update the label in the plot
        self.setText()

        # Update the crosshair in the plot
        self.setCrosshair()

        # Update the integration-crosshair in the plot
        self.setICrosshair()

        # Update the point and label position in the plot
        self.updateGraph()

    # Set the point label
    def setText(self):

        # Check that self.textItem is not empty (in a very pythonic way)
        # self.textItem is empty when pg.GraphItem.__init__(self) runs setData(self, **kwds) internally in CursorItem.__init__(self)
        # This kind of check is applied for different objects in the lines below
        if self.textItem:

            # If the label exist remove it from the scene and clear it
            self.textItem.scene().removeItem(self.textItem)
            self.textItem = []

        if self.text:

            # Update the label in the plot
            self.textItem = pg.TextItem(self.text,color=self.textc)

            # Define the graph object as the parent for the label
            self.textItem.setParentItem(self)

    # Set the crosshair
    def setCrosshair(self):

        # If the crosshair exist remove it from the scene and clear it
        if self.vcrosshairItem:
            self.vcrosshairItem.scene().removeItem(self.vcrosshairItem)
            self.vcrosshairItem = []

        if self.hcrosshairItem:
            self.hcrosshairItem.scene().removeItem(self.hcrosshairItem)
            self.hcrosshairItem = []

        # Build the crosshair with the selected configuration
        if self.csr == []: # None
            pass

        elif self.csr == 0: # Vertical

            self.vcrosshairItem = pg.InfiniteLine(pos=None,angle=90,pen=self.csrc)
            self.vcrosshairItem.setParentItem(self)

        elif self.csr == 1: # Horizontal

            self.hcrosshairItem = pg.InfiniteLine(pos=None,angle=0,pen=self.csrc)
            self.hcrosshairItem.setParentItem(self)

        elif self.csr == 2: # Both

            self.vcrosshairItem = pg.InfiniteLine(pos=[0,0],angle=90,pen=self.csrc)
            self.vcrosshairItem.setParentItem(self)

            self.hcrosshairItem = pg.InfiniteLine(pos=None,angle=0,pen=self.csrc)
            self.hcrosshairItem.setParentItem(self)

    # Set the integration-crosshair
    # The integration-crosshair will only be created if self.ispanx or self.ispany are not zero
    def setICrosshair(self):

        # If the crosshair exist remove it from the scene and clear it
        if self.ivcrosshairItem:
            self.ivcrosshairItem.scene().removeItem(self.ivcrosshairItem)
            self.ivcrosshairItem = []

        if self.ihcrosshairItem:
            self.ihcrosshairItem.scene().removeItem(self.ihcrosshairItem)
            self.ihcrosshairItem = []

        # Build the integration-crosshair with the selected configuration
        if self.icsr == []: # None
            pass

        elif self.icsr == 0: # Vertical

            if self.ispanx != 0:
                self.ivcrosshairItem = pg.LinearRegionItem(values=(0,0),orientation='vertical',brush=self.ibrush,pen=self.icsrc,movable=False)
                self.ivcrosshairItem.setParentItem(self)

        elif self.icsr == 1: # Horizontal

            if self.ispany != 0:
                self.ihcrosshairItem = pg.LinearRegionItem(values=(0,0),orientation='horizontal',brush=self.ibrush,pen=self.icsrc,movable=False)
                self.ihcrosshairItem.setParentItem(self)

        elif self.icsr == 2: # Both

            if self.ispanx != 0:
                self.ivcrosshairItem = pg.LinearRegionItem(values=(0,0),orientation='vertical',brush=self.ibrush,pen=self.icsrc,movable=False)
                self.ivcrosshairItem.setParentItem(self)

            if self.ispany != 0:
                self.ihcrosshairItem = pg.LinearRegionItem(values=(0,0),orientation='horizontal',brush=self.ibrush,pen=self.icsrc,movable=False)
                self.ihcrosshairItem.setParentItem(self)

    # Update the point, label and crosshair position
    def updateGraph(self):

        # Define the fixed GraphItem object with the data in self.data (position,etc.)
        pg.GraphItem.setData(self, **self.data)

        # Set the label position
        if self.textItem:
            self.textItem.setPos(*self.data['pos'][0])

        # Set the crosshair position
        if self.vcrosshairItem:
            if 'angle' in self.data:
                self.vcrosshairItem.setValue([self.data['pos'][0][0], self.data['pos'][0][1]])
                self.vcrosshairItem.setAngle(90 + self.data['angle'])
            else:
                self.vcrosshairItem.setValue(self.data['pos'][0][0])

        if self.hcrosshairItem:
            if 'angle' in self.data:
                self.hcrosshairItem.setValue([self.data['pos'][0][0],self.data['pos'][0][1]])
                self.hcrosshairItem.setAngle(self.data['angle'])
            else:
                self.hcrosshairItem.setValue(self.data['pos'][0][1])

        # Set the integration-crosshair values
        if self.ivcrosshairItem:
            if self.ispanx != 0:
                self.ivcrosshairItem.setRegion((self.data['pos'][0][0]-self.ispanx,self.data['pos'][0][0]+self.ispanx))

        if self.ihcrosshairItem:
            if self.ispany != 0:
                self.ihcrosshairItem.setRegion((self.data['pos'][0][1]-self.ispany,self.data['pos'][0][1]+self.ispany))

    # Give the point the functionality to be dragged using the mouse
    def mouseDragEvent(self, ev):

        # Return the mouse button that generated the click event
        # If it was not the mouse left button ignore the event
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            ev.ignore()
            return

        # Run if a mouse drag is initiated (if the mouse moves before the click above is released)
        if ev.isStart():

            # Get the position of the cursor when the button was first pressed
            pos = ev.buttonDownPos()

            # Find the point at the cursor when the button was first pressed
            # If the point was not under the cursor when the button was first pressed ignore the event
            if len(self.scatter.pointsAt(pos)) == 0:
                ev.ignore()
                return

            # Get the drag point
            self.dragPoint = self.scatter.pointsAt(pos)

            # Get the distance between the original point position and the cursor when the button was first pressed
            self.dragOffset = self.data['pos'][0] - pos

        # Run when the click is released after the drag
        elif ev.isFinish():

            # Clear the drag point
            self.dragPoint = None
            return

        # Run if there is no drag, ignore the event
        else:
            if self.dragPoint is None:
                ev.ignore()
                return

        # Keep the cursors within the data in MainPlot
        if self.space[0][0] + self.ispanx <= ev.pos()[0] + self.dragOffset[0] <= self.space[1][0] - self.ispanx and self.space[0][1] + self.ispany <= ev.pos()[1] + self.dragOffset[1] <= self.space[1][1] - self.ispany:

            # Update the position data of the dragged point =
            # The final position of the cursor after the drag +
            # the offset between the original point position and the cursor when the button was first pressed
            if self.h == True: # Horizontal movement allowed
                self.data['pos'][0][0] = ev.pos()[0] + self.dragOffset[0]
            if self.v == True: # Vertical movement allowed
                self.data['pos'][0][1] = ev.pos()[1] + self.dragOffset[1]

            # Update the point and label position in the plot
            self.updateGraph()

        # Do not let the cursor stick in the top and bottom edges
        elif self.space[0][0] + self.ispanx <= ev.pos()[0] + self.dragOffset[0] <= self.space[1][0] - self.ispanx and (self.space[0][1] + self.ispany > ev.pos()[1] + self.dragOffset[1] or ev.pos()[1] + self.dragOffset[1] > self.space[1][1] - self.ispany):

            # Update the position data of the dragged point =
            # The final position of the cursor after the drag +
            # the offset between the original point position and the cursor when the button was first pressed
            if self.h == True: # Horizontal movement allowed
                self.data['pos'][0][0] = ev.pos()[0] + self.dragOffset[0]

            # Let the cursor arrive to the top and bottom edges even for fast mouse movements
            if self.v == True:
                if self.space[0][1] + self.ispany > ev.pos()[1] + self.dragOffset[1]:
                    self.data['pos'][0][1] = self.space[0][1] + self.ispany
                else:
                    self.data['pos'][0][1] = self.space[1][1] - self.ispany

            # Update the points and labels positions in the plot
            self.updateGraph()

        # Do not let the cursor stick in the left and right edges
        elif (self.space[0][0] + self.ispanx > ev.pos()[0] + self.dragOffset[0] or ev.pos()[0] + self.dragOffset[0] > self.space[1][0] - self.ispanx) and self.space[0][1] + self.ispany <= ev.pos()[1] + self.dragOffset[1] <= self.space[1][1] - self.ispany:

            # Update the position data of the dragged point =
            # The final position of the cursor after the drag +
            # the offset between the original point position and the cursor when the button was first pressed
            if self.v == True: # Vertical movement allowed
                self.data['pos'][0][1] = ev.pos()[1] + self.dragOffset[1]

            # Let the cursor arrive to the left and right edges even for fast mouse movements
            if self.h == True:
                if self.space[0][0] + self.ispanx > ev.pos()[0] + self.dragOffset[0]:
                    self.data['pos'][0][0] = self.space[0][0] + self.ispanx
                else:
                    self.data['pos'][0][0] = self.space[1][0] - self.ispanx

            # Update the points and labels positions in the plot
            self.updateGraph()

        # Do not let the cursor jam near the corners
        elif (self.space[0][0] + self.ispanx > ev.pos()[0] + self.dragOffset[0] or ev.pos()[0] + self.dragOffset[0] > self.space[1][0] - self.ispanx) and (self.space[0][1] + self.ispany > ev.pos()[1] + self.dragOffset[1] or ev.pos()[1] + self.dragOffset[1] > self.space[1][1] - self.ispany):

            # Let the cursor arrive to the corners horizontally even for fast mouse movements
            if self.h == True:
                if self.space[0][0] + self.ispanx > ev.pos()[0] + self.dragOffset[0]:
                    self.data['pos'][0][0] = self.space[0][0] + self.ispanx
                elif ev.pos()[0] + self.dragOffset[0] > self.space[1][0] - self.ispanx:
                    self.data['pos'][0][0] = self.space[1][0] - self.ispanx

            # Let the cursor arrive to the corners vertically even for fast mouse movements
            if self.v == True:
                if self.space[0][1] + self.ispany > ev.pos()[1] + self.dragOffset[1]:
                    self.data['pos'][0][1] = self.space[0][1] + self.ispany
                elif ev.pos()[1] + self.dragOffset[1] > self.space[1][1] - self.ispany:
                    self.data['pos'][0][1] = self.space[1][1] - self.ispany

            # Update the points and labels positions in the plot
            self.updateGraph()

        # Accept the event (the drag event will not be delivered to other items)
        ev.accept()
