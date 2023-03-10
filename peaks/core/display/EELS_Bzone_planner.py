#!/usr/bin/env python
# coding: utf-8

# In[17]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI in Jupyter notebook to display the region of Brillouin zone
that is measured in an EELS measuremnt

v0.1
Created on Tue Aug 04 2020

@author: Lewis Hart, lsh8@st_andrews.ac.uk
"""
#Numpy
from numpy import radians

#Math
from math import sin, cos, sqrt, pi

#Widgets
import ipywidgets as widgets
from ipycanvas import Canvas
from IPython.display import clear_output, display

style = {'description_width': '150px'}

#GUI gile loader
class EELS_Bzone():

    #GUI for loading data using PyARPES. data_path is path to data files to load
    def __init__(self):
        self.canvas = Canvas(width=200, height=200)
        self.display_widgets()

    def _create_widgets(self):
        #Action buttons
        self.executeButton = widgets.Button(description="Execute",
                                         tooltip='Click to create Brillouin zone',
                                         button_style='success'
                                         )
        self.executeButton.on_click(self._on_executeButton_clicked)
        
        #Output
        self.out = widgets.Output(layout={'border': '1px solid black'})
        self.out2 = widgets.Output()

        #Shape selection
        self.BL = widgets.Dropdown(options=['Hexagon', 'Square'],
                                   value='Hexagon',
                                   description='Source:',
                                  )
        
        #Input parameters
        self.angleRange = widgets.FloatText(value=30,
                                           description='Detector Angle Range:',
                                           layout=widgets.Layout(width='300px'),style=style
                                          )
        self.latticeParameter = widgets.FloatText(value=3.54,
                                         description='Lattice parameter:',
                                         layout=widgets.Layout(width='300px'),style=style
                                        )
        self.theta = widgets.FloatText(value=67.5,
                                         description='Theta:',
                                         layout=widgets.Layout(width='300px'),style=style
                                        )
        self.omega = widgets.FloatText(value=0,
                                         description='Omega:',
                                         layout=widgets.Layout(width='300px'),style=style
                                        )
        self.electronEnergy = widgets.FloatText(value=30,
                                         description='Electron Energy:',
                                         layout=widgets.Layout(width='300px'),style=style
                                        )
        
       
    #Calculate the size of the Brillouin zone and cut
    def _on_executeButton_clicked(self, change):
        #Obtain all input parameters from Gui
        a=self.latticeParameter.value
        angle_range = self.angleRange.value
        elE=self.electronEnergy.value
        omega = self.omega.value
        
        #Theta is the angle of incidence between sample and incoming electrons
        #Increasing the angle on the SMC controller decreases this angle of incidence
        theta = 135 - self.theta.value
        theta *= pi/180
        
        #Q and T are the maximum and minimum angles as measured by the analyser
        Q = angle_range/2
        Q *= pi/180
        
        T = -angle_range/2
        T *= pi/180
        
        #Multiplier to make the Brillouin zone fill the Canvas
        rb=75
        
        #Case 1: Brillouin zone is hexagonol
        if self.BL.value == "Hexagon":
            
            #Caluclate real space lattice parameters
            a1x,a1y = a, 0
            a2x,a2y = a/2,(a/2)*sqrt(3)
            
            #Calculate reciprocal space lattice parameters
            b1x,b1y=2*(pi)/a,-2*(pi)/(sqrt(3)*a)
            b2x,b2y=0, 4*(pi)/(sqrt(3)*a)
            
            #Magnitude of reciprocal space lattice parameters
            r=(2/3)*b1x + (1/3)*b2x
            
            #Calculate cut of Brillouin zone measured by EELS
            #k1 is start of cut, k2 is end
            #100 is added as an offset to center cut and Brillouin zone in Canvas 
            a1 = (sin(theta) - sin((theta+Q-(theta-67.5*pi/180)*2)))
            k1 = (0.51231681*sqrt(elE)*a1)*rb/r + 100

            a2 = (sin(theta) - sin((theta+T-(theta-67.5*pi/180)*2)))
            k2 = (0.51231681*sqrt(elE)*a2)*rb/r + 100
            
            #Calculate vertices of Brillouin zone
            x0=rb*cos(omega*pi/180) +100
            y0=rb*sin(omega*pi/180) +100

            x1=rb*cos((omega+60)*pi/180) + 100
            y1=rb*sin((omega+60)*pi/180) + 100

            x2=rb*cos((omega+120)*pi/180) + 100
            y2=rb*sin((omega+120)*pi/180) + 100

            x3=rb*cos((omega+180)*pi/180) + 100
            y3=rb*sin((omega+180)*pi/180) + 100

            x4=rb*cos((omega+240)*pi/180) + 100
            y4=rb*sin((omega+240)*pi/180) + 100

            x5=rb*cos((omega+300)*pi/180) + 100
            y5=rb*sin((omega+300)*pi/180) + 100
            
            #Draw Brillouin zone and cut
            self.canvas.clear()
            self.canvas.begin_path()
            self.canvas.move_to(x0, y0)
            self.canvas.line_to(x1, y1)
            self.canvas.line_to(x2, y2)
            self.canvas.line_to(x3, y3)
            self.canvas.line_to(x4, y4)
            self.canvas.line_to(x5, y5)
            self.canvas.line_to(x0, y0)
            self.canvas.move_to(k1, 100)
            self.canvas.line_to(k2, 100)
            self.canvas.stroke()
            self.canvas.fill_style = 'green'
            self.canvas.fill_arc(100, 100, 2, 0, 2*pi)
            
            with self.out2:
                clear_output(wait=True)
                display(self.canvas)
    
        #############################################################################
        else:
            r=2*pi/a
            
            #Calculate cut
            a1 = (sin(theta) - sin((theta+Q-(theta-67.5*pi/180)*2)))
            k1 = (0.51231681*sqrt(elE)*a1)*2*rb/r + 100

            a2 = (sin(theta) - sin((theta+T-(theta-67.5*pi/180)*2)))
            k2 = (0.51231681*sqrt(elE)*a2)*2*rb/r + 100
        
            #Calculate vertices of square
            x0=rb*cos(omega*pi/180) +100
            y0=rb*sin(omega*pi/180) +100

            x1=rb*cos((omega+90)*pi/180) +100 
            y1=rb*sin((omega+90)*pi/180) +100

            x2=rb*cos((omega+180)*pi/180) +100
            y2=rb*sin((omega+180)*pi/180) +100

            x3=rb*cos((omega+270)*pi/180) +100
            y3=rb*sin((omega+270)*pi/180) +100
            
            #Draw Brillouin zone and cut
            self.canvas.clear()
            self.canvas.begin_path()
            self.canvas.move_to(x1, y1)
            self.canvas.line_to(x2, y2)
            self.canvas.line_to(x3, y3)
            self.canvas.line_to(x0, y0)
            self.canvas.line_to(x1, y1)
            self.canvas.move_to(k1, 100)
            self.canvas.line_to(k2, 100)
            self.canvas.stroke()
            self.canvas.fill_arc(100, 100, 2, 0, 2*pi)
            self.canvas.fill_style = 'red'
            
            with self.out2:
                clear_output(wait=True)
                display(self.canvas)
    
    #Display the GUI
    def display_widgets(self):
        self._create_widgets()
        with self.out:
            print('Select Shape of the Brillouin zone, input all parameters and press execute (Theta is angle on SMC controller)')
        #Display panel
        display(widgets.VBox([
            widgets.HBox([
                widgets.VBox([
                    self.executeButton,
                ]),
                widgets.VBox([
                    self.BL,
                    widgets.HBox([self.angleRange]),
                    widgets.HBox([self.latticeParameter]),
                    widgets.HBox([self.theta]),
                    widgets.HBox([self.omega]),
                    widgets.HBox([self.electronEnergy])
                ]),
            self.out2]),
            self.out])) 

