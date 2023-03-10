#Classes describing jupyter widget view panel
#BE 19/10/2021

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from ipywidgets import widgets, interactive_output, VBox, HBox
from ipywidgets.widgets import Layout
import xarray as xr
from peaks.utils.OOP_method import add_methods


@add_methods(xr.DataArray)
def view(xarray):
    '''This functions calls the correct jupyter widget view panel, depending on the dimensions of the xarray input'''
    
    if len(xarray.dims) == 1:
        view_1D(xarray)
    elif len(xarray.dims) == 2:
        view_disp(xarray)
    elif len(xarray.dims) == 3:
        view_FM(xarray)
    elif len(xarray.dims) == 4:
        view_SM(xarray)
    else:
        print("Data format is not supported")



class view_1D(object):
    
    def __init__(self, data_1D):
        '''This function initiates the 1D data panel'''
        
        #start UI
        self.data_1D = data_1D
        self.xlabel = list(data_1D.dims)[0]
        self.min_x = float(min(self.data_1D.coords[self.xlabel].data))
        self.max_x = float(max(self.data_1D.coords[self.xlabel].data))
        self._create_widgets()
        self._displayUI()

    def _plot(self,x_value):
        
        if x_value[0] != x_value[1]:
            self.data_1D.sel({self.xlabel:slice(x_value[0],x_value[1])}).plot(figsize=(6.2,6.2))
        else:
            self.data_1D.sel({self.xlabel:x_value[0]},method='nearest').plot(figsize=(6.2,6.2))
        plt.title(" ")

    def _create_widgets(self):
        self.w = dict(
                 x_value = widgets.FloatRangeSlider(
                    min=self.min_x,
                    max=self.max_x,
                    value=[self.min_x,self.max_x],
                    step=0.001,
                    description=self.xlabel + ':',
                    layout = Layout(width = '404px'),
                    style = {'description_width': 'initial'},
                    disabled=False,
                    continuous_update=False,
                    readout=True,
                    indent=True,
                    orientation='horizontal',
                )
                )

    def _displayUI(self):   
        output = interactive_output(self._plot, self.w)
        box = VBox([ VBox([*self.w.values()][0:]), output])
        display(box)



class view_disp(object):
    
    def __init__(self, dispersion):
        '''This function initiates the dispersion panel'''
        
        #start UI
        self.dispersion = dispersion
        dispersion_dimensions = list(dispersion.dims)
        dispersion_dimensions.remove('eV')
        self.xlabel = dispersion_dimensions[0]
        self.cmap = 'Blues'
        self.min_x = float(min(self.dispersion.coords[self.xlabel].data))
        self.max_x = float(max(self.dispersion.coords[self.xlabel].data))
        self.min_eV = float(min(self.dispersion.coords['eV'].data))
        self.max_eV = float(max(self.dispersion.coords['eV'].data))
        self.max_counts = np.max(self.dispersion.sel({self.xlabel:slice(self.min_x,self.max_x),'eV':slice(self.min_eV,self.max_eV)}).fillna(0).data)
        self.vmin = 0
        self.vmax = self.max_counts
        self._create_widgets()
        self._displayUI()

    def _plot(self,cmap,spacer1,invert_check,EDC_check,MDC_check,colour_scale,x_value,eV_value):
        
        if invert_check == True:
            cmap = cmap + '_r'
            
    
        if EDC_check == False and MDC_check == False:
            try:
                self.max_counts = np.max(self.dispersion.sel({self.xlabel:slice(x_value[0],x_value[1]),'eV':slice(eV_value[0],eV_value[1])}).fillna(0).data)
                if x_value[0] != x_value[1] and eV_value[0] != eV_value[1]:
                    self.dispersion.sel({self.xlabel:slice(x_value[0],x_value[1]),'eV':slice(eV_value[0],eV_value[1])}).T.plot(cmap=cmap,vmin=colour_scale[0]*self.max_counts,vmax=colour_scale[1]*self.max_counts,add_colorbar=False,figsize=(6.2,6.2))
                    plt.title(" ")
                else:
                    self.dispersion.sel({self.xlabel:slice(x_value[0],x_value[1]),'eV':slice(eV_value[0],eV_value[1])}).T.plot(figsize=(6.2,6.2))
            #if the x or eV values are not a range (want to take nearest EDC/MDC)
            except:
                if x_value[0] == x_value[1] and eV_value[0] == eV_value[1]:
                    #not sure why someone would want this... just exists to prevent an error (outputs a histogram)
                    self.dispersion.sel({self.xlabel:x_value[0]},method='nearest').sel({'eV':eV_value[0]},method='nearest').T.plot(figsize=(6.2,6.2))
                elif x_value[0] == x_value[1]:
                    #get EDC at x_value
                    self.dispersion.sel({self.xlabel:x_value[0]},method='nearest').sel({'eV':slice(eV_value[0],eV_value[1])}).T.plot(figsize=(6.2,6.2))
                elif eV_value[0] == eV_value[1]:
                    #get MDC at eV_value
                    self.dispersion.sel({self.xlabel:slice(x_value[0],x_value[1])}).sel({'eV':eV_value[0]},method='nearest').T.plot(figsize=(6.2,6.2))
                    
        elif EDC_check == True and MDC_check == False:
            try:
                self.max_counts = np.max(self.dispersion.sel({self.xlabel:slice(x_value[0],x_value[1]),'eV':slice(eV_value[0],eV_value[1])}).fillna(0).data)
                self.dispersion.sel({self.xlabel:slice(x_value[0],x_value[1]),'eV':slice(eV_value[0],eV_value[1])}).mean(self.xlabel).T.plot(figsize=(6.2,6.2))
                avg_x = np.round((x_value[0]+x_value[1])/2,7)
                window = np.round((x_value[1]-x_value[0])/2,7)
                plt.title(self.xlabel + ' = ' + str(avg_x) + " \u00B1 " + str(window))
            #if the x or eV values are not a range (want to take nearest EDC/MDC)
            except:
                if x_value[0] == x_value[1] and eV_value[0] == eV_value[1]:
                    #not sure why someone would want this... just exists to prevent an error (outputs a histogram)
                    self.dispersion.sel({self.xlabel:x_value[0]},method='nearest').sel({'eV':eV_value[0]},method='nearest').T.plot(figsize=(6.2,6.2))
                elif x_value[0] == x_value[1]:
                    #get EDC at x_value
                    self.dispersion.sel({self.xlabel:x_value[0]},method='nearest').sel({'eV':slice(eV_value[0],eV_value[1])}).T.plot(figsize=(6.2,6.2))
                elif eV_value[0] == eV_value[1]:
                    #get MDC at eV_value
                    self.dispersion.sel({self.xlabel:slice(x_value[0],x_value[1])}).sel({'eV':eV_value[0]},method='nearest').mean(self.xlabel).T.plot(figsize=(6.2,6.2))
                    
        elif EDC_check == False and MDC_check == True:
            try:
                self.max_counts = np.max(self.dispersion.sel({self.xlabel:slice(x_value[0],x_value[1]),'eV':slice(eV_value[0],eV_value[1])}).fillna(0).data)
                self.dispersion.sel({self.xlabel:slice(x_value[0],x_value[1]),'eV':slice(eV_value[0],eV_value[1])}).mean('eV').T.plot(figsize=(6.2,6.2))
                avg_eV = np.round((eV_value[0]+eV_value[1])/2,7)
                window = np.round((eV_value[1]-eV_value[0])/2,7)
                plt.title('eV = ' + str(avg_eV) + " \u00B1 " + str(window))
            #if the x or eV values are not a range (want to take nearest EDC/MDC)
            except:
                if x_value[0] == x_value[1] and eV_value[0] == eV_value[1]:
                    #not sure why someone would want this... just exists to prevent an error (outputs a histogram)
                    self.dispersion.sel({self.xlabel:x_value[0]},method='nearest').sel({'eV':eV_value[0]},method='nearest').T.plot(figsize=(6.2,6.2))
                elif x_value[0] == x_value[1]:
                    #get EDC at x_value
                    self.dispersion.sel({self.xlabel:x_value[0]},method='nearest').sel({'eV':slice(eV_value[0],eV_value[1])}).mean('eV').T.plot(figsize=(6.2,6.2))
                elif eV_value[0] == eV_value[1]:
                    #get MDC at eV_value
                    self.dispersion.sel({self.xlabel:slice(x_value[0],x_value[1])}).sel({'eV':eV_value[0]},method='nearest').T.plot(figsize=(6.2,6.2))
                    
        elif EDC_check == True and MDC_check == True:
            try:
                self.max_counts = np.max(self.dispersion.sel({self.xlabel:slice(x_value[0],x_value[1]),'eV':slice(eV_value[0],eV_value[1])}).fillna(0).data)
                self.dispersion.sel({self.xlabel:slice(x_value[0],x_value[1]),'eV':slice(eV_value[0],eV_value[1])}).mean(self.xlabel).mean('eV').T.plot(figsize=(6.2,6.2))
            #if the x or eV values are not a range (want to take nearest EDC/MDC)
            except:
                if x_value[0] == x_value[1] and eV_value[0] == eV_value[1]:
                    #not sure why someone would want this... just exists to prevent an error (outputs a histogram)
                    self.dispersion.sel({self.xlabel:x_value[0]},method='nearest').sel({'eV':eV_value[0]},method='nearest').T.plot(figsize=(6.2,6.2))
                elif x_value[0] == x_value[1]:
                    #get EDC at x_value
                    self.dispersion.sel({self.xlabel:x_value[0]},method='nearest').sel({'eV':slice(eV_value[0],eV_value[1])}).T.plot(figsize=(6.2,6.2))
                elif eV_value[0] == eV_value[1]:
                    #get MDC at eV_value
                    self.dispersion.sel({self.xlabel:slice(x_value[0],x_value[1])}).sel({'eV':eV_value[0]},method='nearest').T.plot(figsize=(6.2,6.2))

    def _create_widgets(self):
        self.w = dict(cmap = widgets.Dropdown(
                    options = ['Blues','Purples','Oranges','Greys','viridis','terrain'],
                    description = 'Colour map:',
                    style = {'description_width': 'initial'},
                    layout = Layout(width = '165px'),
                    disabled=False,
                ),
                spacer1 = widgets.Label(
                    value="",
                    layout = Layout(width = '20px')
                ),
                 invert_check = widgets.Checkbox(
                    value=False,
                    description='Invert',
                    disabled=False,
                    indent=False,
                    layout = Layout(width = '77px')
                ),
                 EDC_check = widgets.Checkbox(
                    value=False,
                    description='EDC',
                    disabled=False,
                    indent=False,
                    layout = Layout(width = '72px')
                ),
                 MDC_check = widgets.Checkbox(
                    value=False,
                    description='MDC',
                    disabled=False,
                    indent=False,
                    layout = Layout(width = '72px')
                ),
                 colour_scale = widgets.FloatRangeSlider(
                    min=0,
                    max=1,
                    value=[0,1],
                    step=0.01,
                    description='Colour scale:',
                    layout = Layout(width = '404px'),
                    style = {'description_width': 'initial'},
                    disabled=False,
                    continuous_update=False,
                    readout=True,
                    indent=True,
                    orientation='horizontal',
                ),
                 x_value = widgets.FloatRangeSlider(
                    min=self.min_x,
                    max=self.max_x,
                    value=[self.min_x,self.max_x],
                    step=0.001,
                    description=self.xlabel + ':',
                    layout = Layout(width = '404px'),
                    style = {'description_width': 'initial'},
                    disabled=False,
                    continuous_update=False,
                    readout=True,
                    indent=True,
                    orientation='horizontal',
                ),
                 eV_value = widgets.FloatRangeSlider(
                    min=self.min_eV,
                    max=self.max_eV,
                    value=[self.min_eV,self.max_eV],
                    step=0.001,
                    description='eV:',
                    layout = Layout(width = '404px'),
                    style = {'description_width': 'initial'},
                    disabled=False,
                    continuous_update=False,
                    readout=True,
                    indent=True,
                    orientation='horizontal',
                )
                )

    def _displayUI(self):   
        output = interactive_output(self._plot, self.w)
        box = VBox([HBox([*self.w.values()][0:5]), VBox([*self.w.values()][5:]), output])
        display(box)



class view_FM(object):
    
    def __init__(self, FM):
        '''This function initiates the FM panel'''
        
        #start UI
        self.FM = FM
        FM_dimensions = list(FM.dims)
        FM_dimensions.remove('eV')
        self.xlabel = FM_dimensions[1]
        self.ylabel = FM_dimensions[0]
        self.cmap = 'Blues'
        self.window = 0
        self.x = 0
        self.y = 0
        try:
            self.eV = np.round(estimate_EF(self.FM),2)
        except:
            clear_output()
            self.eV = np.round((min(self.FM.coords['eV'])+max(self.FM.coords['eV']))/2,2)
        self.max_counts = np.max(self.FM.fillna(0).data)
        self.vmin = 0
        self.vmax = 1
        self._create_widgets()
        self._displayUI()

    def _plot(self,cmap,spacer1,line_colour,spacer2,eV_window,spacer3,invert_check,colour_scale,eV_value,x_value,y_value,azi_value):
        
        if invert_check == True:
            cmap = cmap + '_r'
        fig, axes = plt.subplots(ncols=3, figsize=(15,5))
        
        #FS plot
        if eV_window == 0:
            self.FM.sel({'eV':eV_value},method='nearest').plot(cmap=cmap,vmin=colour_scale[0]*self.max_counts,vmax=colour_scale[1]*self.max_counts,add_colorbar=False,ax=axes[0])
            nearest_eV = np.round(float(self.FM.sel({'eV':eV_value},method='nearest').coords['eV']),5)
            axes[0].set_title("eV = " + str(nearest_eV))
        else:
            self.FM.sel({'eV':slice(eV_value-(eV_window/2),eV_value+(eV_window/2))}).mean('eV').plot(cmap=cmap,vmin=colour_scale[0]*self.max_counts,vmax=colour_scale[1]*self.max_counts,add_colorbar=False,ax=axes[0])
            axes[0].set_title("eV = " + str(eV_value) + " \u00B1 " + str(eV_window/2))
        
        #plot along mapping angle
        self.FM.sel({self.xlabel:x_value},method='nearest').T.plot(cmap=cmap,vmin=colour_scale[0]*self.max_counts,vmax=colour_scale[1]*self.max_counts,add_colorbar=False,ax=axes[1])
        nearest_x = np.round(float(self.FM.sel({self.xlabel:x_value},method='nearest').coords[self.xlabel]),5)
        axes[1].set_title(self.xlabel + " = " + str(nearest_x))
        
        #plot along angle parallel to slit
        self.FM.sel({self.ylabel:y_value},method='nearest').T.plot(cmap=cmap,vmin=colour_scale[0]*self.max_counts,vmax=colour_scale[1]*self.max_counts,add_colorbar=False,ax=axes[2])
        nearest_y = np.round(float(self.FM.sel({self.ylabel:y_value},method='nearest').coords[self.ylabel]),5)
        axes[2].set_title(self.ylabel + " = " + str(nearest_y))
        
        #reference lines
        axes[0].axvline(x_value,color=line_colour)
        axes[0].axhline(y_value,color=line_colour)
        axes[1].axvline(y_value,color=line_colour)
        axes[1].axhline(eV_value,color=line_colour)
        axes[2].axvline(x_value,color=line_colour)
        axes[2].axhline(eV_value,color=line_colour)
        
        #azi alignment tool
        m = np.tan(np.radians(-1*azi_value))
        c = y_value - (m*x_value)
        x = np.linspace(-100,100,2)
        y = m*x+c
        axes[0].plot(x, y, '--',color=line_colour)
        
        plt.tight_layout()

    def _create_widgets(self):
        self.w = dict(
                 cmap = widgets.Dropdown(
                    options = ['Blues','Purples','Oranges','Greys','viridis','terrain'],
                    description = 'Colour map:',
                    style = {'description_width': 'initial'},
                    layout = Layout(width = '165px'),
                    disabled=False,
                ),
                spacer1 = widgets.Label(
                    value="",
                    layout = Layout(width = '20px')
                ),
                 line_colour = widgets.Dropdown(
                    options = ['Red','Orange','Yellow','Green','Purple','Black','White'],
                    description = 'Line colour:',
                    style = {'description_width': 'initial'},
                    layout = Layout(width = '160px'),
                    disabled=False,
                ),
                spacer2 = widgets.Label(
                    value="",
                    layout = Layout(width = '20px')
                ),
                eV_window = widgets.BoundedFloatText(
                    value=self.window,
                    min=0,
                    step=0.01,
                    description='Window (eV):',
                    style = {'description_width': 'initial'},
                    layout = Layout(width = '150px'),
                    disabled=False
                ),
                spacer3 = widgets.Label(
                    value="",
                    layout = Layout(width = '20px')
                ),
                 invert_check = widgets.Checkbox(
                    value=False,
                    description='Invert',
                    disabled=False,
                    indent=False,
                    layout = Layout(width = '150px')
                ),
                 colour_scale = widgets.FloatRangeSlider(
                    min=0,
                    max=1,
                    value=[0,1],
                    step=0.01,
                    description='Colour scale:',
                    layout = Layout(width = '617px'),
                    style = {'description_width': 'initial'},
                    disabled=False,
                    continuous_update=False,
                    readout=True,
                    indent=True,
                    orientation='horizontal',
                ),
                 eV_value = widgets.FloatSlider(
                    min=float(min(self.FM.coords['eV'].data)),
                    max=float(max(self.FM.coords['eV'].data)),
                    value=self.eV,
                    step=0.001,
                    description='eV:',
                    layout = Layout(width = '617px'),
                    style = {'description_width': 'initial'},
                    disabled=False,
                    continuous_update=False,
                    readout=True,
                    indent=True,
                    orientation='horizontal',
                ),
                 x_value = widgets.FloatSlider(
                    min=float(min(self.FM.coords[self.xlabel].data)),
                    max=float(max(self.FM.coords[self.xlabel].data)),
                    value=self.x,
                    step=0.001,
                    description=self.xlabel + ':',
                    layout = Layout(width = '617px'),
                    style = {'description_width': 'initial'},
                    disabled=False,
                    continuous_update=False,
                    readout=True,
                    indent=True,
                    orientation='horizontal',
                ),
                 y_value = widgets.FloatSlider(
                    min=float(min(self.FM.coords[self.ylabel].data)),
                    max=float(max(self.FM.coords[self.ylabel].data)),
                    value=self.y,
                    step=0.001,
                    description=self.ylabel + ':',
                    layout = Layout(width = '617px'),
                    style = {'description_width': 'initial'},
                    disabled=False,
                    continuous_update=False,
                    readout=True,
                    indent=True,
                    orientation='horizontal',
                ),
                 azi_value = widgets.FloatSlider(
                    min=-90,
                    max=90,
                    value=0,
                    step=0.01,
                    description='azi:',
                    layout = Layout(width = '617px'),
                    style = {'description_width': 'initial'},
                    disabled=False,
                    continuous_update=False,
                    readout=True,
                    indent=True,
                    orientation='horizontal',
                )
                )

    def _displayUI(self):   
        output = interactive_output(self._plot, self.w)
        box = VBox([HBox([*self.w.values()][0:7]), VBox([*self.w.values()][7:]), output])
        display(box)



class view_SM(object):
    
    def __init__(self, SM):
        '''This function initiates the SM panel'''
        
        #start UI
        self.SM = SM
        self.SM_mean = SM.mean(['theta_par','eV'])
        SM_dimensions = list(SM.dims)
        SM_dimensions.remove('eV')
        SM_dimensions.remove('theta_par')
        self.xlabel = SM_dimensions[0]
        self.ylabel = SM_dimensions[1]
        self.cmap = 'Blues'
        self.x = np.round((min(self.SM.coords[self.xlabel])+max(self.SM.coords[self.xlabel]))/2,2)
        self.xstep = (SM.coords[self.xlabel].data[1]-SM.coords[self.xlabel].data[0])
        self.y = np.round((min(self.SM.coords[self.ylabel])+max(self.SM.coords[self.ylabel]))/2,2)
        self.ystep = (SM.coords[self.ylabel].data[1]-SM.coords[self.ylabel].data[0])
        self.max_counts = np.max(self.SM.fillna(0).data)
        self.vmin = 0
        self.vmax = 1
        self._create_widgets()
        self._displayUI()

    def _plot(self,cmap,spacer1,line_colour,spacer2,invert_check,colour_scale,x_value,y_value):
        
        if invert_check == True:
            cmap = cmap + '_r'
        fig, axes = plt.subplots(ncols=2, figsize=(12,6))
        
        self.SM_mean.T.plot(cmap=cmap,vmin=colour_scale[0]*self.max_counts,vmax=colour_scale[1]*self.max_counts,add_colorbar=False,ax=axes[0])
        
        self.SM.sel({self.xlabel:x_value},method='nearest').sel({self.ylabel:y_value},method='nearest').T.plot(cmap=cmap,vmin=colour_scale[0]*self.max_counts,vmax=colour_scale[1]*self.max_counts,add_colorbar=False,ax=axes[1])
        axes[1].set_title(self.xlabel + ' = ' + str(np.round(x_value,7)) + ", " + self.ylabel + ' = ' + str(np.round(y_value,7)))
        
        #reference lines
        axes[0].axvline(x_value,color=line_colour)
        axes[0].axhline(y_value,color=line_colour)
        
        plt.tight_layout()

    def _create_widgets(self):
        self.w = dict(
                 cmap = widgets.Dropdown(
                    options = ['Blues','Purples','Oranges','Greys','viridis','terrain'],
                    description = 'Colour map:',
                    style = {'description_width': 'initial'},
                    layout = Layout(width = '165px'),
                    disabled=False,
                ),
                spacer1 = widgets.Label(
                    value="",
                    layout = Layout(width = '20px')
                ),
                 line_colour = widgets.Dropdown(
                    options = ['Red','Orange','Yellow','Green','Purple','Black','White'],
                    description = 'Line colour:',
                    style = {'description_width': 'initial'},
                    layout = Layout(width = '160px'),
                    disabled=False,
                ),
                spacer2 = widgets.Label(
                    value="",
                    layout = Layout(width = '20px')
                ),
                 invert_check = widgets.Checkbox(
                    value=False,
                    description='Invert',
                    disabled=False,
                    indent=False,
                    layout = Layout(width = '150px')
                ),
                 colour_scale = widgets.FloatRangeSlider(
                    min=0,
                    max=1,
                    value=[0,1],
                    step=0.01,
                    description='Colour scale:',
                    layout = Layout(width = '439px'),
                    style = {'description_width': 'initial'},
                    disabled=False,
                    continuous_update=False,
                    readout=True,
                    indent=True,
                    orientation='horizontal',
                ),
                 x_value = widgets.FloatSlider(
                    min=float(min(self.SM.coords[self.xlabel].data)),
                    max=float(max(self.SM.coords[self.xlabel].data)),
                    value=self.x,
                    step=self.xstep,
                    description=self.xlabel + ':',
                    layout = Layout(width = '439px'),
                    style = {'description_width': 'initial'},
                    disabled=False,
                    continuous_update=False,
                    readout=True,
                    indent=True,
                    orientation='horizontal',
                ),
                 y_value = widgets.FloatSlider(
                    min=float(min(self.SM.coords[self.ylabel].data)),
                    max=float(max(self.SM.coords[self.ylabel].data)),
                    value=self.y,
                    step=self.ystep,
                    description=self.ylabel + ':',
                    layout = Layout(width = '439px'),
                    style = {'description_width': 'initial'},
                    disabled=False,
                    continuous_update=False,
                    readout=True,
                    indent=True,
                    orientation='horizontal',
                ),
                )

    def _displayUI(self):   
        output = interactive_output(self._plot, self.w)
        box = VBox([HBox([*self.w.values()][0:5]), VBox([*self.w.values()][5:]), output])
        display(box)
