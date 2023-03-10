#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v1.0.3
Saturday June 20 2020

@author: Phil King pdk6@st-andrews.ac.uk
"""

import pandas as pd
import math
import os
import pickle
import ipywidgets as widgets
from IPython.display import clear_output, display
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.sampledata.periodic_table import elements
from bokeh.transform import dodge, factor_cmap
from bokeh.models import ColumnDataSource, CustomJS, DataRange1d, Div, Legend, LegendItem, OpenURL, Select, TapTool, Title
from bokeh.layouts import column, row
from pymatgen.core import periodic_table


class ptab(periodic_table.ElementBase):
    # Subclasses pymatgen.core.periodic_table.ElementBase
    '''Periodic table information, for display and general element information.'''
    # This name = value convention is redundant and dumb, but unfortunately is
    # necessary to preserve backwards compatibility with a time when Element is
    # a regular object that is constructed with Element(symbol).
    H = "H"
    He = "He"
    Li = "Li"
    Be = "Be"
    B = "B"
    C = "C"
    N = "N"
    O = "O"
    F = "F"
    Ne = "Ne"
    Na = "Na"
    Mg = "Mg"
    Al = "Al"
    Si = "Si"
    P = "P"
    S = "S"
    Cl = "Cl"
    Ar = "Ar"
    K = "K"
    Ca = "Ca"
    Sc = "Sc"
    Ti = "Ti"
    V = "V"
    Cr = "Cr"
    Mn = "Mn"
    Fe = "Fe"
    Co = "Co"
    Ni = "Ni"
    Cu = "Cu"
    Zn = "Zn"
    Ga = "Ga"
    Ge = "Ge"
    As = "As"
    Se = "Se"
    Br = "Br"
    Kr = "Kr"
    Rb = "Rb"
    Sr = "Sr"
    Y = "Y"
    Zr = "Zr"
    Nb = "Nb"
    Mo = "Mo"
    Tc = "Tc"
    Ru = "Ru"
    Rh = "Rh"
    Pd = "Pd"
    Ag = "Ag"
    Cd = "Cd"
    In = "In"
    Sn = "Sn"
    Sb = "Sb"
    Te = "Te"
    I = "I"
    Xe = "Xe"
    Cs = "Cs"
    Ba = "Ba"
    La = "La"
    Ce = "Ce"
    Pr = "Pr"
    Nd = "Nd"
    Pm = "Pm"
    Sm = "Sm"
    Eu = "Eu"
    Gd = "Gd"
    Tb = "Tb"
    Dy = "Dy"
    Ho = "Ho"
    Er = "Er"
    Tm = "Tm"
    Yb = "Yb"
    Lu = "Lu"
    Hf = "Hf"
    Ta = "Ta"
    W = "W"
    Re = "Re"
    Os = "Os"
    Ir = "Ir"
    Pt = "Pt"
    Au = "Au"
    Hg = "Hg"
    Tl = "Tl"
    Pb = "Pb"
    Bi = "Bi"
    Po = "Po"
    At = "At"
    Rn = "Rn"
    Fr = "Fr"
    Ra = "Ra"
    Ac = "Ac"
    Th = "Th"
    Pa = "Pa"
    U = "U"
    Np = "Np"
    Pu = "Pu"
    Am = "Am"
    Cm = "Cm"
    Bk = "Bk"
    Cf = "Cf"
    Es = "Es"
    Fm = "Fm"
    Md = "Md"
    No = "No"
    Lr = "Lr"
    Rf = "Rf"
    Db = "Db"
    Sg = "Sg"
    Bh = "Bh"
    Hs = "Hs"
    Mt = "Mt"
    Ds = "Ds"
    Rg = "Rg"
    Cn = "Cn"
    Nh = "Nh"
    Fl = "Fl"
    Mc = "Mc"
    Lv = "Lv"
    Ts = "Ts"
    Og = "Og"

    # Show complete periodic table in interactive format
    def show():
        show = ptab_display()

    # List the core properties of a given element
    def properties(self):
        # Extract most properites
        properties = {}
        try:
            properties['Name'] = self.long_name
        except:
            pass
        try:
            properties['Z'] = self.Z
        except:
            pass
        try:
            properties['Atomic mass (amu)'] = self.atomic_mass
        except:
            pass
        try:
            properties['Ionic radii (Ang)'] = self.ionic_radii
        except:
            properties['Ionic radii (Ang)'] = self.average_ionic_radius
        finally:
            pass
        try:
            properties['Atomic radius (Ang)'] = self.atomic_radius_calculated
        except:
            pass
        try:
            properties['Metallic radius (Ang)'] = self.metallic_radius
        except:
            pass
        try:
            properties['Van der Waals radius (Ang)'] = self.van_der_waals_radius
        except:
            pass
        try:
            properties['Density (Kg m^-3)'] = self.density_of_solid
        except:
            pass
        try:
            properties['Thermal expansion coefficient (K^-1)'] = self.coefficient_of_linear_thermal_expansion
        except:
            pass
        try:
            properties['Poisson\'s ratio'] = self.poissons_ratio
        except:
            pass
        try:
            properties['Bulk Modulus (GPa)'] = self.bulk_modulus
        except:
            pass
        try:
            properties['Young`s modulus (GPa)'] = self.youngs_modulus
        except:
            pass
        try:
            properties['Mendeleev number'] = self.medeleev_no
        except:
            pass
        try:
            properties['Electronic configuration'] = self.electronic_structure
        except:
            pass
        try:
            properties['Common oxidation states'] = str(self.common_oxidation_states)
        except:
            pass
        try:
            properties['Known oxidation states'] = str(self.icsd_oxidation_states)
        except:
            pass
        try:
            properties['Electron afinity'] = str(self.electron_affinity)
        except:
            pass
        try:
            properties['Electrical resistivty'] = self.electrical_resistivity
        except:
            pass
        try:
            properties['Superconducting Tc (K)'] = self.superconduction_temperature
        except:
            pass
        try:
            properties['Thermal conductivity (W K^-1 m^-1)'] = self.thermal_conductivity
        except:
            pass
        try:
            properties['Sound velocity (m s^-1)'] = self.velocity_of_sound
        except:
            pass
        try:
            properties['Melting point (K)'] = self.melting_point
        except:
            pass
        try:
            properties['Boiling point (K)'] = self.boiling_point
        except:
            pass
        try:
            properties['Reflectivity'] = self.reflectivity
        except:
            pass
        try:
            properties['Refractive index'] = self.refractive_index
        except:
            pass

        # Make into pandas dataframe and display
        df = pd.DataFrame(properties, index=[self.symbol]).T
        display(df)

        # Extract orbital energy levels
        try:
            df2 = pd.DataFrame(self.atomic_orbitals, index=['Calculated orbital energy levels (LDA, eV)']).T
            display(df2)
        except:
            pass

        # Extract ionization energies
        try:
            df3 = pd.DataFrame(self.ionization_energies, index=['Ionisation energies']).T
            display(df3)
        except:
            pass



def ptab_display():
    '''
    Funtion to initiate display of Periodic Table in Jupyter notebook environment including widgets

    Returns
    -------
    None.

    '''
    #Define relevant interactors
    style_long = {'description_width': 'initial'}
    to_disp = widgets.Dropdown(
        options=['','General', 'XPS', 'MBE'],
        value='',
        description='Select Periodic Table type to display:',
        disabled=False,
        style = style_long
    )
    
    KE_on = widgets.Checkbox(
        value=False,
        description='Show core level kinetic energy?',
        disabled=False,
        indent=True
    )
    
    hv = widgets.BoundedFloatText(
        value = 1486.6,
        min = 0,
        max = 100000,
        step = 0.1,
        description = 'hv (eV):',
        disabled = False
    )
    
    #Define update actions when interactors are modified:
    #Selecting new option in dropdown box
    def show_ptable(to_disp):
        if to_disp == 'General':
            show_ptab(to_disp,0)
        elif to_disp == 'MBE':
            show_ptab(to_disp,0)
        elif to_disp == 'XPS':
            display(widgets.HBox([KE_on,hv]))
            if KE_on.value is False:
                show_ptab(to_disp,0)
            else:
                show_ptab(to_disp,hv.value)
    
    #When showing KE option changed    
    def update_KE(a):
        if a.name == 'value':
            clear_output()
            display(widgets.HBox([KE_on,hv]))
            if KE_on.value is False:
                show_ptab(to_disp.value,0)
            else:
                show_ptab(to_disp.value,hv.value)
    
    #When hv value changed
    def update_hv(a):
        if a.name == 'value':
            if KE_on.value is True:
                clear_output()
                display(widgets.HBox([KE_on,hv]))
                show_ptab(to_disp.value,hv.value)

    #Actions that happen on changing output    
    out = widgets.interactive_output(show_ptable, {'to_disp': to_disp})            
    display(widgets.VBox([to_disp, out]))
    KE_on.observe(update_KE)
    hv.observe(update_hv)
    
    
def show_ptab(to_disp,hv):    
    
    '''Function using Bokeh to plot a periodic table with custom hover-over information displays. 
    see https://bokeh.org
    Currently working for General chemical display and XPS core level.
    
    Parameters
    ----------
    to_disp : STRING
         Key to what additional information is displayed in periodic table: "General", "XPS", or "MBE".
    hv : FLOAT
        Photon energy used to calculate core level kinetic energies.

    Returns
    -------
    disp : TYPE
        handle to Bokeh plot in case you wish to push update (currently not used)    '''
    
    output_notebook()

    ##############################################
    ########## Main periodic table display #######
    ##############################################
    
    #Define period and group information
    periods = ["I", "II", "III", "IV", "V", "VI", "VII", "", "*", "**"]
    groups = [str(x) for x in range(1, 19)]

    #Create Data frame for main group elements
    global df
    df = elements.copy()
    df["atomic mass"] = df["atomic mass"].astype(str)
    df["period"] = [periods[x-1] for x in df.period]

    #Modify group and period information of LA and AC series for plotting purposes
    for ii in range (56,71):
        df.loc[ii,'period'] = "*"
        df.loc[ii,'group'] = 4+ii-56
    for ii in range (88,103):
        df.loc[ii,'period'] = "**"
        df.loc[ii,'group'] = 4+ii-88    
    df["group"] = df["group"].astype(str)    
    df.loc[70,'metal'] = 'lanthanoid'
    df.loc[102,'metal'] = 'actinoid'
    
    #Correct name for Hn
    df.loc[112,'name'] = 'Nihonium'
            
    #Add core level information into data frame if required
    if to_disp == 'XPS':
        df["core levels"] = ''
        for ii in range (0,118):
            df.loc[ii,'core levels'] = add_by_element(elementName=df.loc[ii,"symbol"],hv=hv)
            
    #Add MBE source data if required
    if to_disp == 'MBE':
        path = os.path.dirname(__file__)
        path1 = os.path.join(path, 'evap_guide/Evaporation_checklist.xlsx')
        df_mbe = pd.read_excel(path1, sheet_name='Database')
        
        df['Tv @ e-8 Torr'] = ''
        df['Tv @ e-4 Torr'] = ''
        df['T_idle'] = ''
        df['T_growth'] = ''
        df['Cell type'] = ''
        df['Crucible'] = ''
        df['Cell'] = ''
        df['Evap history'] = ''
        df['cell_sheet'] = ''
        df['cell_sheet2'] = ''
        
        j=0
        try:
            for i in df_mbe['Atomic #']:
                df.loc[i-1,'Tv @ e-8 Torr'] = df_mbe.loc[j,'Tv @e-8 Torr (oC)']
                df.loc[i-1,'Tv @ e-4 Torr'] = df_mbe.loc[j,'Tv @e-4 Torr (oC)']
                if math.isnan(df_mbe.loc[j,'Preferred Ti (exp. confirmed)']) is False:
                    df.loc[i-1,'T_idle'] = 'Conf.: ' + str(df_mbe.loc[j,'Preferred Ti (exp. confirmed)'])
                else:
                    df.loc[i-1,'T_idle'] = 'Est: ' + str(df_mbe.loc[j,'Estimated idle temp. Ti (oC)'])
                
                if type(df_mbe.loc[j,'Preferred Tg (exp. confirmed) GM1']) is not str and type(df_mbe.loc[j,'Preferred Tg (exp. confirmed) GM2']) is not str: 
                    if math.isnan(df_mbe.loc[j,'Preferred Tg (exp. confirmed) GM1']) is False and math.isnan(df_mbe.loc[j,'Preferred Tg (exp. confirmed) GM2']) is False:
                        df.loc[i-1,'T_growth'] = 'Conf. GM1: ' + str(df_mbe.loc[j,'Preferred Tg (exp. confirmed) GM1'])+'; Conf. GM2: ' + str(df_mbe.loc[j,'Preferred Tg (exp. confirmed) GM2'])
                    elif math.isnan(df_mbe.loc[j,'Preferred Tg (exp. confirmed) GM1']) is False:
                        df.loc[i-1,'T_growth'] = 'Conf. GM1: ' + str(df_mbe.loc[j,'Preferred Tg (exp. confirmed) GM1'])
                    elif math.isnan(df_mbe.loc[j,'Preferred Tg (exp. confirmed) GM2']) is False:
                        df.loc[i-1,'T_growth'] = 'Conf. GM2: ' + str(df_mbe.loc[j,'Preferred Tg (exp. confirmed) GM2'])
                    else:
                        df.loc[i-1,'T_growth'] = 'Est.: ' + str(df_mbe.loc[j,'Estimated growth temp. Tg (oC)'])
                elif type(df_mbe.loc[j,'Preferred Tg (exp. confirmed) GM1']) is str:
                    if math.isnan(df_mbe.loc[j,'Preferred Tg (exp. confirmed) GM2']) is False:
                        df.loc[i-1,'T_growth'] = 'Conf. GM1: '+str(df_mbe.loc[j,'Preferred Tg (exp. confirmed) GM1'])+'; Conf. GM2: ' + str(df_mbe.loc[j,'Preferred Tg (exp. confirmed) GM2'])
                    else:
                        df.loc[i-1,'T_growth'] = 'Conf. GM1: '+str(df_mbe.loc[j,'Preferred Tg (exp. confirmed) GM1'])
                else:
                    df.loc[i-1,'T_growth'] = 'Conf. GM2: ' + str(df_mbe.loc[j,'Preferred Tg (exp. confirmed) GM2'])
                df.loc[i-1,'Cell type'] = df_mbe.loc[j,'Thermal evap.']
                df.loc[i-1,'Crucible'] = df_mbe.loc[j,'Crucible materials']
                try: 
                    if math.isnan(df_mbe.loc[j,'Preferred cell (exp. confirmed)']) is True:
                        df.loc[i-1,'Cell'] = 'Suitable: ' + str(df_mbe.loc[j,'suitable cell '])
                except:
                    df.loc[i-1,'Cell'] = '<table style=\"float:left\"> <tr><td> Preferred: ' + str(df_mbe.loc[j,'Preferred cell (exp. confirmed)']) + '</td></tr><tr><td> Suitable: ' + str(df_mbe.loc[j,'suitable cell ']) + "</td></tr></table>"
                df.loc[i-1,'Evap history'] = df_mbe.loc[j,'Evaporation history\n(Previous condition (Period, cell type/crucible material, Ti(oC)&Tg(oC) or power(%)&Ie(mA), flux (A/s))']
                try:
                    if math.isnan(df_mbe.loc[j,'Current spec sheet GM1']) is True:
                        df.loc[i-1,'cell_sheet'] = '-'
                except:
                    df.loc[i-1,'cell_sheet'] = df_mbe.loc[j,'Current spec sheet GM1']
                try:
                    if math.isnan(df_mbe.loc[j,'Current spec sheet GM2']) is True:
                        df.loc[i-1,'cell_sheet2'] = '-'
                except:
                    df.loc[i-1,'cell_sheet2'] = df_mbe.loc[j,'Current spec sheet GM2']
                j += 1
        except:
            pass
            
        #Convert melting point from K to deg C
        df['melting point'] -= 273
    
    pt_data = ColumnDataSource(df)
    
    #Define color choices for display
    cmap = {
        "alkali metal"         : "#a6cee3",
        "alkaline earth metal" : "#1f78b4",
        "metal"                : "#d93b43",
        "halogen"              : "#999d9a",
        "metalloid"            : "#e08d49",
        "noble gas"            : "#eaeaea",
        "nonmetal"             : "#f1d4Af",
        "transition metal"     : "#65C99C",
        "lanthanoid"          : "#FFFFE0",
        "actinoid"            : "#FFD9D9",
    }

    #Define tooltips to set roll-over text content
    if to_disp == 'General':
        TOOLTIPS = [
            ("Name", "@name"),
            ("Atomic number", "@{atomic number}"),
            ("Atomic mass", "@{atomic mass}"),
            ("Density", "@{density} g/cm^3"),
            ("Ionic radius", "@{ion radius}"),
            ("Type", "@metal"),
            ("Electronegativity", "@{electronegativity}"),
            ("Melting temperature", "@{melting point} K"),
        #    ("CPK color", "$color[hex, swatch]:CPK"),
            ("Electronic configuration", "@{electronic configuration}"),
        ]
    elif to_disp == 'XPS':
        TOOLTIPS = [
            ("Name", "@name"),
            ("Atomic number", "@{atomic number}"),
            ("Atomic mass", "@{atomic mass}"),
            ("Type", "@metal"),
        #    ("CPK color", "$color[hex, swatch]:CPK"),
            ("Electronic configuration", "@{electronic configuration}"),
            ("Core levels", "@{core levels}{safe}"),
        ]
    elif to_disp == 'MBE':
        TOOLTIPS = [
            ("Name", "@name"),
            ("Atomic number", "@{atomic number}"),
            ("Atomic mass", "@{atomic mass}"),
            ("Density", "@{density} g/cm^3"),
            ("Melting temperature", "@{melting point} degC"),
        #    ("CPK color", "$color[hex, swatch]:CPK"),
            ("Electronic configuration", "@{electronic configuration}"),
            ("Vapour pressure (10^-8 Torr)", "@{Tv @ e-8 Torr} degC"),
            ("Vapour pressure (10^-4 Torr)", "@{Tv @ e-4 Torr} degC"),
            ("Idle cell temperature", "@{T_idle} degC"),
            ("Growth cell temperature", "@{T_growth} degC"),
            ("Classification", "@{Cell type}"),
            ("Cell", "@{Cell}{safe}"),
            ("Crucible", "@{Crucible}"),
        ]
        
    #Define the plot
    p = figure(plot_width=950, plot_height=600,
               x_range=groups, y_range=list(reversed(periods)),
               tools=["hover","tap"], toolbar_location=None, tooltips=TOOLTIPS)
    t = Title()
    
    #Make selection-specific title
    if to_disp == 'General':
        t.text = "Periodic Table - click on element to show more information"
    elif to_disp == 'XPS':
        t.text = "Periodic Table - click on element (#1-#83 and #92) to show core level and cross-section information (hold shift for multiple selections)"
    elif to_disp == 'MBE':
        t.text = "Periodic Table - click on element to show approved cell sheet if available"
    p.title = t
    # "hover","tap"

    
    r = p.rect("group", "period", 0.95, 0.95, source=pt_data, fill_alpha=0.6, legend_field="metal",
               color=factor_cmap('metal', palette=list(cmap.values()), factors=list(cmap.keys())))

    text_props = {"source": pt_data, "text_align": "left", "text_baseline": "middle"}
    x = dodge("group", -0.4, range=p.x_range)
    p.text(x=x, y="period", text="symbol", text_font_style="bold", **text_props)
    p.text(x=x, y=dodge("period", 0.3, range=p.y_range), text="atomic number",
           text_font_size="8pt", **text_props)
    p.text(x=x, y=dodge("period", -0.35, range=p.y_range), text="name",
           text_font_size="5pt", **text_props)
    p.text(x=x, y=dodge("period", -0.2, range=p.y_range), text="atomic mass",
           text_font_size="5pt", **text_props)
    p.text(x=["3", "3"], y=["VI", "VII"], text=["*", "**"], text_align="center", text_baseline="middle")

    #Axis setup etc.
    p.outline_line_color = None
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_standoff = 0
    p.legend.orientation = "horizontal"
    p.legend.location = (0,117)

    #Setup hover rendering
    p.hover.renderers = [r] # only hover element boxes
    
    
    ##############################################
    ######### Extra plots for XPS mode ###########
    ##############################################
    if to_disp == 'XPS':
    
        ##############################################
        ############## XPS core level plot  ##########
        ##############################################
        
        #Make a dictionary containing all the binding/kinetic energies of core levels and intensity values grouped by element up to Bi, plus U
        core_level_data = {}
        for ii in range (0,83):
            core_level_data = add_by_element(df.loc[ii,"symbol"],hv,1,core_level_data)
        core_level_data = add_by_element(df.loc[91,"symbol"],hv,1,core_level_data)
        
        
        #Line colour definition
        line_colours = ['cadetblue','brown','khaki','darkmagenta','darkolivegreen','navy','darkcyan','red','green']
    
        #Define tooltips for roll-over highlighting
        TOOLTIPS2 = [
                ("", "@CL"),
                ("", "@En eV"),
            ]
    
        #Define figure
        p2 = figure(plot_width=400, plot_height=320, toolbar_location="below", 
                    tools=['xbox_zoom','hover','reset','save'], tooltips=TOOLTIPS2,
                    title='Core levels')

        
        #Set up plots for core levels (define data arrays and line segments)
        n_max = 6 #maximum number of lines that can be shown
       
        CLi = []
        li = []
        ll = []
        for i in range (0,n_max):
            CLi.append(ColumnDataSource (data=dict(En=[],Zero=[],Int=[],CL=[])))
            li.append(p2.segment(x0='En', y0='Zero', x1='En', y1='Int', source=CLi[i], line_color=line_colours[i], alpha = 0.8, line_width=3))
            ll.append(LegendItem(label='', renderers=[li[i]]))

        #Add legend based on entries in ll
        p2.add_layout(Legend(items=ll), 'above')
        p2.legend.orientation = 'horizontal'
        p2.legend.border_line_alpha = 0
        p2.legend.background_fill_alpha = 0
        p2.legend.margin = 0
        p2.legend.click_policy="hide"
        
        #Axis definitions
        if hv == 0:
            p2.xaxis.axis_label = 'Binding Energy (eV)'
            p2.x_range.flipped = True
        else:
            p2.xaxis.axis_label = 'Kinetic Energy (eV)'
            p2.x_range.flipped = False
        
        #p2.yaxis.axis_label = 'Intensity (arb. units)'
        p2.yaxis.major_label_text_font_size = '0pt'  # turn off x-axis tick labels
        p2.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
        p2.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
        
        
        
        ##############################################
        ############## Cross section plot  ###########
        ##############################################
        
        #Define tooltips for roll-over highlighting
        TOOLTIPS3 = [
                ("Photon energy", "@En eV"),
                ("Cross section", "@xc Mbarn"),
            ]
    
        #Define figure
        p3 = figure(plot_width=400, plot_height=500, y_axis_type="log", x_range = DataRange1d(range_padding=0.05, only_visible=True), y_range = DataRange1d(range_padding=0.05, only_visible=True), toolbar_location="below", 
                    tools=['xwheel_zoom','box_zoom','hover','reset','save'], active_drag='box_zoom', tooltips=TOOLTIPS3,
                    title='Photoionization cross sections')
        
        #Load cross-section data - read all from xsec folder
        '''The data are taken from: J.J. Yeh, Atomic Calculation of Photoionization Cross-Sections 
            and Asymmetry Parameters, Gordon and Breach Science Publishers, Langhorne, PE (USA), 1993 
            and from J.J. Yeh and I.Lindau, Atomic Data and Nuclear Data Tables, 32, 1-155 (1985). 
            The data are those calculated in the dipole length approximation.
            They have been stored in a dictionary saved using pickle, saved in the folder of the script, and loaded from that'''
        path = os.path.dirname(__file__)
        
        with open(os.path.join(path, 'xc.p'), 'rb') as fp:
            xc = pickle.load(fp)
        
        with open(os.path.join(path, 'xc_name.p'), 'rb') as fp:
            xc_name = pickle.load(fp)
        
        
        #Set up plots
        xci = [] #Dictionary for column data sources   
        xc_li = [] #plot lines
        xc_ll = [] #legend labels
        for i in range (0,n_max):
            xci.append(ColumnDataSource (data=dict(En=[],xc=[])))
            xc_li.append(p3.line(x='En', y='xc', source=xci[i], line_color=line_colours[i], alpha = 0.8, line_width=3))    
            xc_ll.append(LegendItem(label='', renderers=[xc_li[i]]))

        #Add legend based on entries in ll
        p3.add_layout(Legend(items=xc_ll), 'above')
        p3.legend.orientation = 'horizontal'
        p3.legend.border_line_alpha = 0
        p3.legend.background_fill_alpha = 0
        p3.legend.margin = 0
        p3.legend.click_policy="hide"
        
        #Dropdown menus
        selections = []
        xc_entries = {}
        for i in range (0,n_max):
            xc_entries['i'] = ['-----']
            selections.append(Select(title="_", width=100, value="", options=[]))
        
        #Axis labels
        p3.xaxis.axis_label = 'Photon Energy (eV)'
        p3.yaxis.axis_label = 'Cross section (Mbarn)'
    
        
        
    ##############################################
    ################# Callbacks ##################
    ##############################################
    taptool = p.select(type=TapTool)
    taptool.renderers = [r]   
    if to_disp == 'General':
        #Open webelements page for corresponding element when clicked in general ptab mode
        taptool.callback = OpenURL(url="https://www.webelements.com/@symbol/")
             
    elif to_disp == 'MBE':
        #Open source data sheet file when element clicked in MBE mode
        callback_mbe = CustomJS(args=dict(waddress = df['cell_sheet'],waddress2 = df['cell_sheet2']), code="""
            var inds = cb_obj.indices;
            if (inds > 0) {
                var a = waddress[inds]
                var b = waddress2[inds]
                if (a == '-') {
                        if (b == '-') {
                        return
                        } else {
                        window.open(b)
                        }
                } else {
                if (b == '-') {
                        window.open(a)
                } else {
                        window.open(a)
                        window.open(b)
                        }
                }
            }
        """)
            
        pt_data.selected.js_on_change("indices", callback_mbe)

    elif to_disp == 'XPS':
        #Setup custom selection action to show core levels and select relevant cross section data when selecting elements on periodic table
        callback_cl = CustomJS(args=dict(cld = core_level_data, CLi=CLi, ll = ll, xc_select = selections, xc_name = xc_name, xc=xc, xci=xci, xc_ll=xc_ll, el_list = df['symbol']), code="""
            var inds
            inds = cb_obj.indices;        
            var el
            
            //set all plots back to null if selection removes
            if (inds.length == 0) {
                    for (i=0; i < CLi.length; i++) {
                        var dd = CLi[i].data
                        dd['En'] = []; dd['Zero'] = []; dd['Int'] = []; dd['CL'] = [];
                        ll[i].label.value = '';
                        xc_select[i].options = []
                        xc_select[i].title = '_'
                        var dd1 = xci[i].data
                        dd1['En'] = []; dd1['xc'] = [];
                        xc_ll[i].label.value = [];
                        
                    }
            } else {
                //for each of the selected values in p-table, update the display
                for (i=0; i < inds.length; i++) {
                        var i
                        var index = inds.length - i - 1
                        var el = el_list[inds[index]] //selected elemnent in p-table 
                        
                        //Update core level data for CL plot
                        var dd = CLi[i].data
                        dd['En'] = cld[el+'_En']; dd['Zero'] = cld[el+'_Zero']; dd['Int'] = cld[el+'_Int']; dd['CL'] = cld[el+'_CL'];
                        ll[i].label.value = el;
                        
                        //Update fields in xc selection widgets
                        function el_search(entry) {
                            return entry.includes(el+'_')
                        }
                        
                        let choices = xc_name.filter(el_search).sort()
                        xc_select[i].options = choices
                        xc_select[i].title = el+':'
                        
                        //Update corresponding plot
                        var dd1 = xci[i].data
                        dd1['En'] = xc['En_'+choices[0]]
                        dd1['xc'] = xc['xc_'+choices[0]]
                        xc_ll[i].label.value = choices[0];
                    }
            }

            for (i=0; i < CLi.length; i++) {
                    CLi[i].change.emit();
                    xc_select[i].change.emit()
                    xci[i].change.emit();
            }
            """)
            
        pt_data.selected.js_on_change("indices", callback_cl)
        
        #Change displayed cross section plot based on dropdown box selection
        for i in range (0,n_max):        
            selections[i].js_on_change("value", CustomJS(args=dict(xc=xc, xci=xci, xc_ll=xc_ll, i = i), code="""
                var selected
                selected = cb_obj.value;   
                
                var dd = xci[i].data
                dd['En'] = xc['En_'+selected]
                dd['xc'] = xc['xc_'+selected]
                xc_ll[i].label.value = selected;
                            
                xci[i].change.emit();
                """))
        
           
    ##############################################
    ############### Display plots ################
    ##############################################
    if to_disp == 'XPS':
        dumdiv1 = Div(text='',height=20)
        dumdiv2 = Div(text='',width=20)
        layout = column(p, dumdiv1, row(p2,dumdiv2,p3,column(selections)))
        show(layout)
    elif to_disp == 'MBE':
        div1 = Div(text="""<b>Safety Information:</b></br>
                   Check SDS and evaporation checklist for detailed safety information and for details on avoiding risks.</br>
                   </br>
                    Types of risks</br>
                     - X: explosive </br>
                     - O: oxidising agent</br>
                     - F: flammable solid/emit flammable gas</br>
                     - T: toxic to human body</br>
                     - A: corrosive or irritant (skin, eyes etc.)</br>
                     - C: carcinogenic</br>
                     - R: radioactive</br>
                     - N: dangerous to the environment </br>
                    </br>
                     Risk level rating</br>
                     - Smaller numbers/alphabetically earlier letters mean higher risks in general.</br>
                     - Especially for the environmental risk, "ac"/"ch" indicate "acute" (short-term)/"chronic" (long-term) effects on the environment. </br>
                     """,
        width=400, height=350)
        div2 = Div(text="""<b>Cell catagory:</b></br>
                   * Category I: Materials which sublime/can be evaporated in solid phase, in general set max cell temperature at 100 oC below melting point. </br>
                   * Category II: CAUTION: Materials which are in liquid phase during evaporation, set max SP at either 100 oC above Tv @1e-4 Torr or operatinal max tem. of the cell. Cool/heat the cell at 1oC/min rate while the cell temperature is within 100 oC from Tm. Take out the cell regularly to investigate the crucible/cell condition. </br>
                   * Category III: AVOID. In exceptional circumstances, if approved, cool/heat the cell at 1oC/min rate while the cell temperature is within 100 oC from Tm. Take out the cell regularly to investigate the crucible/cell condition.""",
        width=550, height=350)
        layout = column(p,row(div1,div2))
        show(layout)
    else:
        show(p)
    
    
    
    #


def add_by_element(elementName,hv=0,flag=0,core_level_data=None):
    ''' Function to add XPS core-level info to periodic table plot
    Modified from XPS package of PESTO by Craig Polley

    Parameters
    ----------
    elementName : STRING
        Name of element for core level information to extract
    hv : FLOAT, optional
        Photon energy for calculating kinetic energies. If 0, then only binding energy information used.
    flag : INT, optional
        Set whether dictionary of core-level data is output instead of text string (set to 1 for this)
    core_level_data : DICT, optional
        Pass dictionary for updating core level KE vs Int for plot    

    Returns
    -------
    output: STROMG
        Info for the datatable of the ptable
    core_level_data: DICT
        Core level data in different format.

    '''

    ANALYZER_WORKFUNCTION = 4.4
    output = ''
    matches=[]
    for index,element in enumerate(core_levels):
        if element[1]==elementName:
            matches.append(element)
    
    if len(matches)==0:
        return output
    else:
        if hv>0:
            output += "<table style=\"float:right\"><thead><tr><td>Core level</td><td>&emsp;Binding energy</td><td>&emsp;Ek at hv={0}</td></tr></thead>".format(hv)
            for element in matches:
                name = element[1]
                level = element[2]
                orbital = element[3]
                spin = element[4]
                literatureBindingEnergy = float(element[5])
                Ek_1st_order = hv-ANALYZER_WORKFUNCTION-literatureBindingEnergy
                if Ek_1st_order>0: #Only show if accessible at selected photon energy
                    if flag == 1:
                        try:
                            core_level_data[elementName+'_En'].append(Ek_1st_order)
                            core_level_data[elementName+'_Int'].append(1)
                            core_level_data[elementName+'_Zero'].append(0)
                            if spin=="0":
                                core_level_data[elementName+'_CL'].append("{0} {1}{2}".format(name,level,orbital))
                            else:
                                core_level_data[elementName+'_CL'].append("{0} {1}{2}{3}".format(name,level,orbital,spin))
                        except:
                            core_level_data[elementName+'_En'] = [Ek_1st_order]
                            core_level_data[elementName+'_Int'] = [1]
                            core_level_data[elementName+'_Zero'] = [0]
                            if spin=="0":
                                core_level_data[elementName+'_CL'] = ["{0} {1}{2}".format(name,level,orbital)]
                            else:
                                core_level_data[elementName+'_CL'] = ["{0} {1}{2}{3}".format(name,level,orbital,spin)]
                            
                    else:
                        if spin=="0":
                            output += "<tr><td>{} {}{}</td><td>{:.6}</td><td>{:.6}</td></tr>".format(name,level,orbital,literatureBindingEnergy,Ek_1st_order)
                        else:
                            output += "<tr><td>{} {}{}{}</td><td>{:.6}</td><td>{:.6}</td></tr>".format(name,level,orbital,spin,literatureBindingEnergy,Ek_1st_order)
        else:
            output += "<table style=\"float:right\"><thead><tr><td>Core level</td><td>&emsp;Binding energy</td></tr></thead>"
            for element in matches:
                name = element[1]
                level = element[2]
                orbital = element[3]
                spin = element[4]
                literatureBindingEnergy = float(element[5])
                if literatureBindingEnergy < 2000: #list only core levels <2keV binding energy
                    if flag == 1:
                        try:
                            core_level_data[elementName+'_En'].append(literatureBindingEnergy)
                            core_level_data[elementName+'_Int'].append(1)
                            core_level_data[elementName+'_Zero'].append(0)
                            if spin=="0":
                                core_level_data[elementName+'_CL'].append("{0} {1}{2}".format(name,level,orbital))
                            else:
                                core_level_data[elementName+'_CL'].append("{0} {1}{2}{3}".format(name,level,orbital,spin))
                        except:
                            core_level_data[elementName+'_En'] = [literatureBindingEnergy]
                            core_level_data[elementName+'_Int'] = [1]
                            core_level_data[elementName+'_Zero'] = [0]
                            if spin=="0":
                                core_level_data[elementName+'_CL'] = ["{0} {1}{2}".format(name,level,orbital)]
                            else:
                                core_level_data[elementName+'_CL'] = ["{0} {1}{2}{3}".format(name,level,orbital,spin)]
                    else:
                        if spin=="0":
                            output += "<tr><td>{0} {1}{2}</td><td>{3:.6}</td></tr>".format(name,level,orbital,literatureBindingEnergy)
                        else:
                            output += "<tr><td>{0} {1}{2}{3}</td><td>{4:.6}</td></tr>".format(name,level,orbital,spin,literatureBindingEnergy)
    output += "</table>"
    if flag == 1:
        return core_level_data
    return output

"""
Reference information.
Source: webcore_levels.com, which references mostly Cardona and Ley, 
Photoemission in Solids I (1978)

Contains all non-radioactive core_levels (up to Bi)
"""


core_levels=[]
#Row 1
core_levels.append([1,"H","1","s","0",13.6])
core_levels.append([2,"He","1","s","0",23.4])

#Row 2
core_levels.append([3,"Li","1","s","0",54.7]) 
core_levels.append([4,"Be","1","s","0",111.5])
core_levels.append([5,"B","1","s","0",188])
core_levels.append([6,"C","1","s","0",284.2])
core_levels.append([7,"N","1","s","0",409.9])
core_levels.append([7,"N","2","s","0",37.3])
core_levels.append([8,"O","1","s","0",543.1])
core_levels.append([8,"O","2","s","0",41.6])
core_levels.append([9,"F","1","s","0",696.7])
core_levels.append([10,"Ne","1","s","0",870.2])
core_levels.append([10,"Ne","2","s","0",48.5])
core_levels.append([10,"Ne","2","p","1/2",21.7])
core_levels.append([10,"Ne","2","p","3/2",21.6])


#Row 3
core_levels.append([11,"Na","1","s","0",1070.8])
core_levels.append([11,"Na","2","s","0",63.5])
core_levels.append([11,"Na","2","p","1/2",30.4])
core_levels.append([11,"Na","2","p","3/2",30.5])
core_levels.append([12,"Mg","1","s","0",1303.0])
core_levels.append([12,"Mg","2","s","0",88.6])
core_levels.append([12,"Mg","2","p","1/2",49.6])
core_levels.append([12,"Mg","2","p","3/2",49.2])
core_levels.append([13,"Al","1","s","0",1559.0])
core_levels.append([13,"Al","2","s","0",117.8])
core_levels.append([13,"Al","2","p","1/2",72.9])
core_levels.append([13,"Al","2","p","3/2",72.5])
core_levels.append([14,"Si","1","s","0",1839.0])
core_levels.append([14,"Si","2","s","0",149.7])
core_levels.append([14,"Si","2","p","1/2",99.8])
core_levels.append([14,"Si","2","p","3/2",99.2])
core_levels.append([15,"P","1","s","0",2145.5])
core_levels.append([15,"P","2","s","0",189.0])
core_levels.append([15,"P","2","p","1/2",136.0])
core_levels.append([15,"P","2","p","3/2",135.0])
core_levels.append([16,"S","1","s","0",2472.0])
core_levels.append([16,"S","2","s","0",230.9])
core_levels.append([16,"S","2","p","1/2",163.6])
core_levels.append([16,"S","2","p","3/2",162.5])
core_levels.append([17,"Cl","1","s","0",2822])
core_levels.append([17,"Cl","2","s","0",270])
core_levels.append([17,"Cl","2","p","1/2",202])
core_levels.append([17,"Cl","2","p","3/2",200])
core_levels.append([18,"Ar","1","s","0",3205.9])
core_levels.append([18,"Ar","2","s","0",326.3])
core_levels.append([18,"Ar","2","p","1/2",250.6])
core_levels.append([18,"Ar","2","p","3/2",248.4])
core_levels.append([18,"Ar","3","s","0",29.3])
core_levels.append([18,"Ar","3","p","1/2",15.9])
core_levels.append([18,"Ar","3","p","3/2",15.7])

#Row 4
core_levels.append([19,"K","1","s","0",3608.4])
core_levels.append([19,"K","2","s","0",378.6])
core_levels.append([19,"K","2","p","1/2",297.3])
core_levels.append([19,"K","2","p","3/2",294.6])
core_levels.append([19,"K","3","s","0",34.8])
core_levels.append([19,"K","3","p","1/2",18.3])
core_levels.append([19,"K","3","p","3/2",18.3])
core_levels.append([20,"Ca","1","s","0",4038.5])
core_levels.append([20,"Ca","2","s","0",438.4])
core_levels.append([20,"Ca","2","p","1/2",349.7])
core_levels.append([20,"Ca","2","p","3/2",346.2])
core_levels.append([20,"Ca","3","s","0",44.3])
core_levels.append([20,"Ca","3","p","1/2",25.4])
core_levels.append([20,"Ca","3","p","3/2",25.4])
core_levels.append([21,"Sc","1","s","0",4492])
core_levels.append([21,"Sc","2","s","0",498])
core_levels.append([21,"Sc","2","p","1/2",403.6])
core_levels.append([21,"Sc","2","p","3/2",398.7])
core_levels.append([21,"Sc","3","s","0",51.1])
core_levels.append([21,"Sc","3","p","1/2",28.3])
core_levels.append([21,"Sc","3","p","3/2",28.3])
core_levels.append([22,"Ti","1","s","0",4966])
core_levels.append([22,"Ti","2","s","0",560.9])
core_levels.append([22,"Ti","2","p","1/2",460.2])
core_levels.append([22,"Ti","2","p","3/2",453.8])
core_levels.append([22,"Ti","3","s","0",58.7])
core_levels.append([22,"Ti","3","p","1/2",32.6])
core_levels.append([22,"Ti","3","p","3/2",32.6])
core_levels.append([23,"V","1","s","0",5465])
core_levels.append([23,"V","2","s","0",626.7])
core_levels.append([23,"V","2","p","1/2",519.8])
core_levels.append([23,"V","2","p","3/2",512.1])
core_levels.append([23,"V","3","s","0",66.3])
core_levels.append([23,"V","3","p","1/2",37.2])
core_levels.append([23,"V","3","p","3/2",37.2])
core_levels.append([24,"Cr","1","s","0",5989])
core_levels.append([24,"Cr","2","s","0",696])
core_levels.append([24,"Cr","2","p","1/2",583.8])
core_levels.append([24,"Cr","2","p","3/2",574.1])
core_levels.append([24,"Cr","3","s","0",74.1])
core_levels.append([24,"Cr","3","p","1/2",42.2])
core_levels.append([24,"Cr","3","p","3/2",42.2])
core_levels.append([25,"Mn","1","s","0",6539])
core_levels.append([25,"Mn","2","s","0",769.1])
core_levels.append([25,"Mn","2","p","1/2",649.9])
core_levels.append([25,"Mn","2","p","3/2",638.7])
core_levels.append([25,"Mn","3","s","0",82.3])
core_levels.append([25,"Mn","3","p","1/2",47.2])
core_levels.append([25,"Mn","3","p","3/2",47.2])
core_levels.append([26,"Fe","1","s","0",7112])
core_levels.append([26,"Fe","2","s","0",844.6])
core_levels.append([26,"Fe","2","p","1/2",719.9])
core_levels.append([26,"Fe","2","p","3/2",706.8])
core_levels.append([26,"Fe","3","s","0",91.3])
core_levels.append([26,"Fe","3","p","1/2",52.7])
core_levels.append([26,"Fe","3","p","3/2",52.7])
core_levels.append([27,"Co","1","s","0",7709])
core_levels.append([27,"Co","2","s","0",925.1])
core_levels.append([27,"Co","2","p","1/2",793.2])
core_levels.append([27,"Co","2","p","3/2",778.1])
core_levels.append([27,"Co","3","s","0",101])
core_levels.append([27,"Co","3","p","1/2",58.9])
core_levels.append([27,"Co","3","p","3/2",59.9])
core_levels.append([28,"Ni","1","s","0",8333])
core_levels.append([28,"Ni","2","s","0",1008.6])
core_levels.append([28,"Ni","2","p","1/2",870])
core_levels.append([28,"Ni","2","p","3/2",852.7])
core_levels.append([28,"Ni","3","s","0",110.8])
core_levels.append([28,"Ni","3","p","1/2",68])
core_levels.append([28,"Ni","3","p","3/2",66.2])
core_levels.append([29,"Cu","1","s","0",8979])
core_levels.append([29,"Cu","2","s","0",1096.7])
core_levels.append([29,"Cu","2","p","1/2",952.3])
core_levels.append([29,"Cu","2","p","3/2",932.7])
core_levels.append([29,"Cu","3","s","0",122.5])
core_levels.append([29,"Cu","3","p","1/2",77.3])
core_levels.append([29,"Cu","3","p","3/2",75.1])
core_levels.append([30,"Zn","1","s","0",9659])
core_levels.append([30,"Zn","2","s","0",1196.2])
core_levels.append([30,"Zn","2","p","1/2",1044.9])
core_levels.append([30,"Zn","2","p","3/2",1021.8])
core_levels.append([30,"Zn","3","s","0",139.8])
core_levels.append([30,"Zn","3","p","1/2",91.4])
core_levels.append([30,"Zn","3","p","3/2",88.6])
core_levels.append([30,"Zn","3","d","3/2",10.2])
core_levels.append([30,"Zn","3","d","5/2",10.1])
core_levels.append([31,"Ga","1","s","0",10367])
core_levels.append([31,"Ga","2","s","0",1299])
core_levels.append([31,"Ga","2","p","1/2",1143.2])
core_levels.append([31,"Ga","2","p","3/2",1116.4])
core_levels.append([31,"Ga","3","s","0",159.5])
core_levels.append([31,"Ga","3","p","1/2",103.5])
core_levels.append([31,"Ga","3","p","3/2",100])
core_levels.append([31,"Ga","3","d","3/2",18.7])
core_levels.append([31,"Ga","3","d","5/2",18.7])
core_levels.append([32,"Ge","1","s","0",11103])
core_levels.append([32,"Ge","2","s","0",1414.6])
core_levels.append([32,"Ge","2","p","1/2",1248.1])
core_levels.append([32,"Ge","2","p","3/2",1217])
core_levels.append([32,"Ge","3","s","0",180.1])
core_levels.append([32,"Ge","3","p","1/2",124.9])
core_levels.append([32,"Ge","3","p","3/2",120.8])
core_levels.append([32,"Ge","3","d","3/2",29.8])
core_levels.append([32,"Ge","3","d","5/2",29.2])
core_levels.append([33,"As","1","s","0",11867])
core_levels.append([33,"As","2","s","0",1527])
core_levels.append([33,"As","2","p","1/2",1359.1])
core_levels.append([33,"As","2","p","3/2",1323.6])
core_levels.append([33,"As","3","s","0",204.7])
core_levels.append([33,"As","3","p","1/2",146.2])
core_levels.append([33,"As","3","p","3/2",141.2])
core_levels.append([33,"As","3","d","3/2",41.7])
core_levels.append([33,"As","3","d","5/2",41.7])
core_levels.append([34,"Se","1","s","0",12658])
core_levels.append([34,"Se","2","s","0",1652])
core_levels.append([34,"Se","2","p","1/2",1474.3])
core_levels.append([34,"Se","2","p","3/2",1433.9])
core_levels.append([34,"Se","3","s","0",229.6])
core_levels.append([34,"Se","3","p","1/2",166.5])
core_levels.append([34,"Se","3","p","3/2",160.7])
core_levels.append([34,"Se","3","d","3/2",55.5])
core_levels.append([34,"Se","3","d","5/2",54.6])
core_levels.append([35,"Br","1","s","0",13474])
core_levels.append([35,"Br","2","s","0",1782])
core_levels.append([35,"Br","2","p","1/2",1596])
core_levels.append([35,"Br","2","p","3/2",1550])
core_levels.append([35,"Br","3","s","0",257])
core_levels.append([35,"Br","3","p","1/2",189])
core_levels.append([35,"Br","3","p","3/2",182])
core_levels.append([35,"Br","3","d","3/2",70])
core_levels.append([35,"Br","3","d","5/2",69])
core_levels.append([36,"Kr","1","s","0",14326])
core_levels.append([36,"Kr","2","s","0",1921])
core_levels.append([36,"Kr","2","p","1/2",1730.9])
core_levels.append([36,"Kr","2","p","3/2",1678.4])
core_levels.append([36,"Kr","3","s","0",292.8])
core_levels.append([36,"Kr","3","p","1/2",222.2])
core_levels.append([36,"Kr","3","p","3/2",214.4])
core_levels.append([36,"Kr","3","d","3/2",95])
core_levels.append([36,"Kr","3","d","5/2",93.8])
core_levels.append([36,"Kr","4","s","0",27.5])
core_levels.append([36,"Kr","4","p","1/2",14.1])
core_levels.append([36,"Kr","4","p","3/2",14.1])

#Row 5
core_levels.append([37,"Rb","1","s","0",15200])
core_levels.append([37,"Rb","2","s","0",2065])
core_levels.append([37,"Rb","2","p","1/2",1864])
core_levels.append([37,"Rb","2","p","3/2",1804])
core_levels.append([37,"Rb","3","s","0",326.7])
core_levels.append([37,"Rb","3","p","1/2",248.7])
core_levels.append([37,"Rb","3","p","3/2",239.1])
core_levels.append([37,"Rb","3","d","3/2",113])
core_levels.append([37,"Rb","3","d","5/2",112])
core_levels.append([37,"Rb","4","s","0",30.5])
core_levels.append([37,"Rb","4","p","1/2",16.3])
core_levels.append([37,"Rb","4","p","3/2",15.3])

core_levels.append([38,"Sr","1","s","0",16105])
core_levels.append([38,"Sr","2","s","0",2216])
core_levels.append([38,"Sr","2","p","1/2",2007])
core_levels.append([38,"Sr","2","p","3/2",1940])
core_levels.append([38,"Sr","3","s","0",358.7])
core_levels.append([38,"Sr","3","p","1/2",280.3])
core_levels.append([38,"Sr","3","p","3/2",270])
core_levels.append([38,"Sr","3","d","3/2",136])
core_levels.append([38,"Sr","3","d","5/2",134.2])
core_levels.append([38,"Sr","4","s","0",38.9])
core_levels.append([38,"Sr","4","p","1/2",21.6])
core_levels.append([38,"Sr","4","p","3/2",20.1])

core_levels.append([39,"Y","1","s","0",17038])
core_levels.append([39,"Y","2","s","0",2373])
core_levels.append([39,"Y","2","p","1/2",2156])
core_levels.append([39,"Y","2","p","3/2",2080])
core_levels.append([39,"Y","3","s","0",392])
core_levels.append([39,"Y","3","p","1/2",310.6])
core_levels.append([39,"Y","3","p","3/2",298.8])
core_levels.append([39,"Y","3","d","3/2",157.7])
core_levels.append([39,"Y","3","d","5/2",155.8])
core_levels.append([39,"Y","4","s","0",43.8])
core_levels.append([39,"Y","4","p","1/2",24.4])
core_levels.append([39,"Y","4","p","3/2",23.1])

core_levels.append([40,"Zr","1","s","0",17998])
core_levels.append([40,"Zr","2","s","0",2532])
core_levels.append([40,"Zr","2","p","1/2",2307])
core_levels.append([40,"Zr","2","p","3/2",2223])
core_levels.append([40,"Zr","3","s","0",430.3])
core_levels.append([40,"Zr","3","p","1/2",343.5])
core_levels.append([40,"Zr","3","p","3/2",329.8])
core_levels.append([40,"Zr","3","d","3/2",181.1])
core_levels.append([40,"Zr","3","d","5/2",178.8])
core_levels.append([40,"Zr","4","s","0",50.6])
core_levels.append([40,"Zr","4","p","1/2",28.5])
core_levels.append([40,"Zr","4","p","3/2",27.1])

core_levels.append([41,"Nb","1","s","0",18986])
core_levels.append([41,"Nb","2","s","0",2698])
core_levels.append([41,"Nb","2","p","1/2",2465])
core_levels.append([41,"Nb","2","p","3/2",2371])
core_levels.append([41,"Nb","3","s","0",466.6])
core_levels.append([41,"Nb","3","p","1/2",376.1])
core_levels.append([41,"Nb","3","p","3/2",360.6])
core_levels.append([41,"Nb","3","d","3/2",205])
core_levels.append([41,"Nb","3","d","5/2",202.3])
core_levels.append([41,"Nb","4","s","0",56.4])
core_levels.append([41,"Nb","4","p","1/2",32.6])
core_levels.append([41,"Nb","4","p","3/2",30.8])

core_levels.append([42,"Mo","1","s","0",20000])
core_levels.append([42,"Mo","2","s","0",2866])
core_levels.append([42,"Mo","2","p","1/2",2625])
core_levels.append([42,"Mo","2","p","3/2",2520])
core_levels.append([42,"Mo","3","s","0",506.3])
core_levels.append([42,"Mo","3","p","1/2",411.6])
core_levels.append([42,"Mo","3","p","3/2",394])
core_levels.append([42,"Mo","3","d","3/2",231.1])
core_levels.append([42,"Mo","3","d","5/2",227.9])
core_levels.append([42,"Mo","4","s","0",63.2])
core_levels.append([42,"Mo","4","p","1/2",37.6])
core_levels.append([42,"Mo","4","p","3/2",35.5])

core_levels.append([43,"Tc","1","s","0",21044])
core_levels.append([43,"Tc","2","s","0",3043])
core_levels.append([43,"Tc","2","p","1/2",2793])
core_levels.append([43,"Tc","2","p","3/2",2677])
core_levels.append([43,"Tc","3","s","0",544])
core_levels.append([43,"Tc","3","p","1/2",447.6])
core_levels.append([43,"Tc","3","p","3/2",417.7])
core_levels.append([43,"Tc","3","d","3/2",257.6])
core_levels.append([43,"Tc","3","d","5/2",253.9])
core_levels.append([43,"Tc","4","s","0",69.5])
core_levels.append([43,"Tc","4","p","1/2",42.3])
core_levels.append([43,"Tc","4","p","3/2",39.9])

core_levels.append([44,"Ru","1","s","0",22117])
core_levels.append([44,"Ru","2","s","0",3224])
core_levels.append([44,"Ru","2","p","1/2",2967])
core_levels.append([44,"Ru","2","p","3/2",2838])
core_levels.append([44,"Ru","3","s","0",586.1])
core_levels.append([44,"Ru","3","p","1/2",483.3])
core_levels.append([44,"Ru","3","p","3/2",461.5])
core_levels.append([44,"Ru","3","d","3/2",284.2])
core_levels.append([44,"Ru","3","d","5/2",280])
core_levels.append([44,"Ru","4","s","0",75])
core_levels.append([44,"Ru","4","p","1/2",46.3])
core_levels.append([44,"Ru","4","p","3/2",43.2])

core_levels.append([45,"Rh","1","s","0",23220])
core_levels.append([45,"Rh","2","s","0",3412])
core_levels.append([45,"Rh","2","p","1/2",3146])
core_levels.append([45,"Rh","2","p","3/2",3004])
core_levels.append([45,"Rh","3","s","0",628.1])
core_levels.append([45,"Rh","3","p","1/2",521.3])
core_levels.append([45,"Rh","3","p","3/2",496.5])
core_levels.append([45,"Rh","3","d","3/2",311.9])
core_levels.append([45,"Rh","3","d","5/2",307.2])
core_levels.append([45,"Rh","4","s","0",81.4])
core_levels.append([45,"Rh","4","p","1/2",50.5])
core_levels.append([45,"Rh","4","p","3/2",47.3])

core_levels.append([46,"Pd","1","s","0",24350])
core_levels.append([46,"Pd","2","s","0",3604])
core_levels.append([46,"Pd","2","p","1/2",3330])
core_levels.append([46,"Pd","2","p","3/2",3173])
core_levels.append([46,"Pd","3","s","0",671.6])
core_levels.append([46,"Pd","3","p","1/2",559.9])
core_levels.append([46,"Pd","3","p","3/2",532.3])
core_levels.append([46,"Pd","3","d","3/2",340.5])
core_levels.append([46,"Pd","3","d","5/2",335.2])
core_levels.append([46,"Pd","4","s","0",87.1])
core_levels.append([46,"Pd","4","p","1/2",55.7])
core_levels.append([46,"Pd","4","p","3/2",50.9])

core_levels.append([47,"Ag","1","s","0",25514])
core_levels.append([47,"Ag","2","s","0",3806])
core_levels.append([47,"Ag","2","p","1/2",3524])
core_levels.append([47,"Ag","2","p","3/2",3351])
core_levels.append([47,"Ag","3","s","0",719])
core_levels.append([47,"Ag","3","p","1/2",603.8])
core_levels.append([47,"Ag","3","p","3/2",573])
core_levels.append([47,"Ag","3","d","3/2",374])
core_levels.append([47,"Ag","3","d","5/2",368.3])
core_levels.append([47,"Ag","4","s","0",97])
core_levels.append([47,"Ag","4","p","1/2",63.7])
core_levels.append([47,"Ag","4","p","3/2",58.3])

core_levels.append([48,"Cd","1","s","0",26711])
core_levels.append([48,"Cd","2","s","0",4018])
core_levels.append([48,"Cd","2","p","1/2",3727])
core_levels.append([48,"Cd","2","p","3/2",3538])
core_levels.append([48,"Cd","3","s","0",772])
core_levels.append([48,"Cd","3","p","1/2",652.6])
core_levels.append([48,"Cd","3","p","3/2",618.4])
core_levels.append([48,"Cd","3","d","3/2",411.9])
core_levels.append([48,"Cd","3","d","5/2",405.2])
core_levels.append([48,"Cd","4","s","0",109.8])
core_levels.append([48,"Cd","4","p","1/2",63.9])
core_levels.append([48,"Cd","4","p","3/2",63.9])
core_levels.append([48,"Cd","4","d","3/2",11.7])
core_levels.append([48,"Cd","4","d","5/2",10.7])

core_levels.append([49,"In","1","s","0",27940])
core_levels.append([49,"In","2","s","0",4238])
core_levels.append([49,"In","2","p","1/2",3938])
core_levels.append([49,"In","2","p","3/2",3730])
core_levels.append([49,"In","3","s","0",827.2])
core_levels.append([49,"In","3","p","1/2",703.2])
core_levels.append([49,"In","3","p","3/2",665.3])
core_levels.append([49,"In","3","d","3/2",451.4])
core_levels.append([49,"In","3","d","5/2",443.9])
core_levels.append([49,"In","4","s","0",122.9])
core_levels.append([49,"In","4","p","1/2",73.5])
core_levels.append([49,"In","4","p","3/2",73.5])
core_levels.append([49,"In","4","d","3/2",17.7])
core_levels.append([49,"In","4","d","5/2",16.9])

core_levels.append([50,"Sn","1","s","0",29200])
core_levels.append([50,"Sn","2","s","0",4465])
core_levels.append([50,"Sn","2","p","1/2",4156])
core_levels.append([50,"Sn","2","p","3/2",3929])
core_levels.append([50,"Sn","3","s","0",884.7])
core_levels.append([50,"Sn","3","p","1/2",756.5])
core_levels.append([50,"Sn","3","p","3/2",714.6])
core_levels.append([50,"Sn","3","d","3/2",493.2])
core_levels.append([50,"Sn","3","d","5/2",484.9])
core_levels.append([50,"Sn","4","s","0",137.1])
core_levels.append([50,"Sn","4","p","1/2",83.6])
core_levels.append([50,"Sn","4","p","3/2",83.6])
core_levels.append([50,"Sn","4","d","3/2",24.9])
core_levels.append([50,"Sn","4","d","5/2",23.9])

core_levels.append([51,"Sb","1","s","0",30491])
core_levels.append([51,"Sb","2","s","0",4698])
core_levels.append([51,"Sb","2","p","1/2",4380])
core_levels.append([51,"Sb","2","p","3/2",4132])
core_levels.append([51,"Sb","3","s","0",940])
core_levels.append([51,"Sb","3","p","1/2",812.7])
core_levels.append([51,"Sb","3","p","3/2",766.4])
core_levels.append([51,"Sb","3","d","3/2",537.5])
core_levels.append([51,"Sb","3","d","5/2",528.2])
core_levels.append([51,"Sb","4","s","0",153.2])
core_levels.append([51,"Sb","4","p","1/2",95.6])
core_levels.append([51,"Sb","4","p","3/2",95.6])
core_levels.append([51,"Sb","4","d","3/2",33.3])
core_levels.append([51,"Sb","4","d","5/2",32.1])

core_levels.append([52,"Te","1","s","0",31814])
core_levels.append([52,"Te","2","s","0",4939])
core_levels.append([52,"Te","2","p","1/2",4612])
core_levels.append([52,"Te","2","p","3/2",4341])
core_levels.append([52,"Te","3","s","0",1006])
core_levels.append([52,"Te","3","p","1/2",870.8])
core_levels.append([52,"Te","3","p","3/2",820.8])
core_levels.append([52,"Te","3","d","3/2",583.4])
core_levels.append([52,"Te","3","d","5/2",573])
core_levels.append([52,"Te","4","s","0",169.4])
core_levels.append([52,"Te","4","p","1/2",103.3])
core_levels.append([52,"Te","4","p","3/2",103.3])
core_levels.append([52,"Te","4","d","3/2",41.9])
core_levels.append([52,"Te","4","d","5/2",40.4])

core_levels.append([53,"I","1","s","0",33169])
core_levels.append([53,"I","2","s","0",5188])
core_levels.append([53,"I","2","p","1/2",4852])
core_levels.append([53,"I","2","p","3/2",4557])
core_levels.append([53,"I","3","s","0",1072])
core_levels.append([53,"I","3","p","1/2",931])
core_levels.append([53,"I","3","p","3/2",875])
core_levels.append([53,"I","3","d","3/2",630.8])
core_levels.append([53,"I","3","d","5/2",619.3])
core_levels.append([53,"I","4","s","0",186])
core_levels.append([53,"I","4","p","1/2",123])
core_levels.append([53,"I","4","p","3/2",123])
core_levels.append([53,"I","4","d","3/2",50.6])
core_levels.append([53,"I","4","d","5/2",48.9])

core_levels.append([54,"Xe","1","s","0",34561])
core_levels.append([54,"Xe","2","s","0",5453])
core_levels.append([54,"Xe","2","p","1/2",5107])
core_levels.append([54,"Xe","2","p","3/2",4786])
core_levels.append([54,"Xe","3","s","0",1148.7])
core_levels.append([54,"Xe","3","p","1/2",1002.1])
core_levels.append([54,"Xe","3","p","3/2",940.6])
core_levels.append([54,"Xe","3","d","3/2",689])
core_levels.append([54,"Xe","3","d","5/2",676.4])
core_levels.append([54,"Xe","4","s","0",213.2])
core_levels.append([54,"Xe","4","p","1/2",146.7])
core_levels.append([54,"Xe","4","p","3/2",145.5])
core_levels.append([54,"Xe","4","d","3/2",69.5])
core_levels.append([54,"Xe","4","d","5/2",67.5])
core_levels.append([54,"Xe","5","s","0",23.3])
core_levels.append([54,"Xe","5","p","1/2",13.4])
core_levels.append([54,"Xe","5","p","3/2",12.1])

####Row 6

core_levels.append([55,"Cs","1","s","0",35985])
core_levels.append([55,"Cs","2","s","0",5714])
core_levels.append([55,"Cs","2","p","1/2",5359])
core_levels.append([55,"Cs","2","p","3/2",5012])
core_levels.append([55,"Cs","3","s","0",1211])
core_levels.append([55,"Cs","3","p","1/2",1071])
core_levels.append([55,"Cs","3","p","3/2",1003])
core_levels.append([55,"Cs","3","d","3/2",740.5])
core_levels.append([55,"Cs","3","d","5/2",726.6])
core_levels.append([55,"Cs","4","s","0",232.3])
core_levels.append([55,"Cs","4","p","1/2",172.4])
core_levels.append([55,"Cs","4","p","3/2",161.3])
core_levels.append([55,"Cs","4","d","3/2",79.8])
core_levels.append([55,"Cs","4","d","5/2",77.5])
core_levels.append([55,"Cs","5","s","0",22.7])
core_levels.append([55,"Cs","5","p","1/2",14.2])
core_levels.append([55,"Cs","5","p","3/2",12.1])

core_levels.append([56,"Ba","1","s","0",37441])
core_levels.append([56,"Ba","2","s","0",5989])
core_levels.append([56,"Ba","2","p","1/2",5624])
core_levels.append([56,"Ba","2","p","3/2",5247])
core_levels.append([56,"Ba","3","s","0",1293])
core_levels.append([56,"Ba","3","p","1/2",1137])
core_levels.append([56,"Ba","3","p","3/2",1063])
core_levels.append([56,"Ba","3","d","3/2",795.7])
core_levels.append([56,"Ba","3","d","5/2",780.5])
core_levels.append([56,"Ba","4","s","0",253.5])
core_levels.append([56,"Ba","4","p","1/2",192])
core_levels.append([56,"Ba","4","p","3/2",178.6])
core_levels.append([56,"Ba","4","d","3/2",92.6])
core_levels.append([56,"Ba","4","d","5/2",89.9])
core_levels.append([56,"Ba","5","s","0",30.3])
core_levels.append([56,"Ba","5","p","1/2",17])
core_levels.append([56,"Ba","5","p","3/2",14.8])

core_levels.append([57,"La","1","s","0",38925])
core_levels.append([57,"La","2","s","0",6266])
core_levels.append([57,"La","2","p","1/2",5891])
core_levels.append([57,"La","2","p","3/2",5483])
core_levels.append([57,"La","3","s","0",1362])
core_levels.append([57,"La","3","p","1/2",1209])
core_levels.append([57,"La","3","p","3/2",1128])
core_levels.append([57,"La","3","d","3/2",853])
core_levels.append([57,"La","3","d","5/2",836])
core_levels.append([57,"La","4","s","0",274.7])
core_levels.append([57,"La","4","p","1/2",205.8])
core_levels.append([57,"La","4","p","3/2",196])
core_levels.append([57,"La","4","d","3/2",105.3])
core_levels.append([57,"La","4","d","5/2",102.5])
core_levels.append([57,"La","5","s","0",34.3])
core_levels.append([57,"La","5","p","1/2",19.3])
core_levels.append([57,"La","5","p","3/2",16.8])

core_levels.append([58,"Ce","1","s","0",40443])
core_levels.append([58,"Ce","2","s","0",6548])
core_levels.append([58,"Ce","2","p","1/2",6164])
core_levels.append([58,"Ce","2","p","3/2",5723])
core_levels.append([58,"Ce","3","s","0",1436])
core_levels.append([58,"Ce","3","p","1/2",1274])
core_levels.append([58,"Ce","3","p","3/2",1187])
core_levels.append([58,"Ce","3","d","3/2",902.4])
core_levels.append([58,"Ce","3","d","5/2",883.8])
core_levels.append([58,"Ce","4","s","0",291])
core_levels.append([58,"Ce","4","p","1/2",223.2])
core_levels.append([58,"Ce","4","p","3/2",206.5])
core_levels.append([58,"Ce","4","d","3/2",109])
core_levels.append([58,"Ce","4","f","5/2",0.1])
core_levels.append([58,"Ce","4","f","7/2",0.1])
core_levels.append([58,"Ce","5","s","0",37.8])
core_levels.append([58,"Ce","5","p","1/2",19.8])
core_levels.append([58,"Ce","5","p","3/2",17])

core_levels.append([59,"Pr","1","s","0",41991])
core_levels.append([59,"Pr","2","s","0",6835])
core_levels.append([59,"Pr","2","p","1/2",6440])
core_levels.append([59,"Pr","2","p","3/2",5964])
core_levels.append([59,"Pr","3","s","0",1511])
core_levels.append([59,"Pr","3","p","1/2",1337])
core_levels.append([59,"Pr","3","p","3/2",1242])
core_levels.append([59,"Pr","3","d","3/2",948.3])
core_levels.append([59,"Pr","3","d","5/2",928.8])
core_levels.append([59,"Pr","4","s","0",304.5])
core_levels.append([59,"Pr","4","p","1/2",236.3])
core_levels.append([59,"Pr","4","p","3/2",217.6])
core_levels.append([59,"Pr","4","d","3/2",115.1])
core_levels.append([59,"Pr","4","d","5/2",115.1])
core_levels.append([59,"Pr","4","f","5/2",2])
core_levels.append([59,"Pr","4","f","7/2",2])
core_levels.append([59,"Pr","5","s","0",37.4])
core_levels.append([59,"Pr","5","p","1/2",22.3])
core_levels.append([59,"Pr","5","p","3/2",22.3])

core_levels.append([60,"Nd","1","s","0",43569])
core_levels.append([60,"Nd","2","s","0",7126])
core_levels.append([60,"Nd","2","p","1/2",6722])
core_levels.append([60,"Nd","2","p","3/2",6208])
core_levels.append([60,"Nd","3","s","0",1575])
core_levels.append([60,"Nd","3","p","1/2",1403])
core_levels.append([60,"Nd","3","p","3/2",1297])
core_levels.append([60,"Nd","3","d","3/2",1003.3])
core_levels.append([60,"Nd","3","d","5/2",980.4])
core_levels.append([60,"Nd","4","s","0",319.2])
core_levels.append([60,"Nd","4","p","1/2",243.3])
core_levels.append([60,"Nd","4","p","3/2",224.6])
core_levels.append([60,"Nd","4","d","3/2",120.5])
core_levels.append([60,"Nd","4","d","5/2",120.5])
core_levels.append([60,"Nd","4","f","5/2",1.5])
core_levels.append([60,"Nd","4","f","7/2",1.5])
core_levels.append([60,"Nd","5","s","0",37.5])
core_levels.append([60,"Nd","5","p","1/2",21.1])
core_levels.append([60,"Nd","5","p","3/2",21.1])

core_levels.append([61,"Pm","1","s","0",45184])
core_levels.append([61,"Pm","2","s","0",7428])
core_levels.append([61,"Pm","2","p","1/2",7013])
core_levels.append([61,"Pm","2","p","3/2",6459])
core_levels.append([61,"Pm","3","p","1/2",1471.4])
core_levels.append([61,"Pm","3","p","3/2",1357])
core_levels.append([61,"Pm","3","d","3/2",1052])
core_levels.append([61,"Pm","3","d","5/2",1027])
core_levels.append([61,"Pm","4","p","1/2",242])
core_levels.append([61,"Pm","4","p","3/2",242])
core_levels.append([61,"Pm","4","d","3/2",120])
core_levels.append([61,"Pm","4","d","5/2",120])


core_levels.append([62,"Sm","1","s","0",46834])
core_levels.append([62,"Sm","2","s","0",7737])
core_levels.append([62,"Sm","2","p","1/2",7312])
core_levels.append([62,"Sm","2","p","3/2",6716])
core_levels.append([62,"Sm","3","s","0",1723])
core_levels.append([62,"Sm","3","p","1/2",1541])
core_levels.append([62,"Sm","3","p","3/2",1419.8])
core_levels.append([62,"Sm","3","d","3/2",1110.9])
core_levels.append([62,"Sm","3","d","5/2",1083.4])
core_levels.append([62,"Sm","4","s","0",347.2])
core_levels.append([62,"Sm","4","p","1/2",265.6])
core_levels.append([62,"Sm","4","p","3/2",247.4])
core_levels.append([62,"Sm","4","d","3/2",129])
core_levels.append([62,"Sm","4","d","5/2",129])
core_levels.append([62,"Sm","4","f","5/2",5.2])
core_levels.append([62,"Sm","4","f","7/2",5.2])
core_levels.append([62,"Sm","5","s","0",37.4])
core_levels.append([62,"Sm","5","p","1/2",21.3])
core_levels.append([62,"Sm","5","p","3/2",21.3])

core_levels.append([63,"Eu","1","s","0",48519])
core_levels.append([63,"Eu","2","s","0",8052])
core_levels.append([63,"Eu","2","p","1/2",7617])
core_levels.append([63,"Eu","2","p","3/2",6977])
core_levels.append([63,"Eu","3","s","0",1800])
core_levels.append([63,"Eu","3","p","1/2",1614])
core_levels.append([63,"Eu","3","p","3/2",1481])
core_levels.append([63,"Eu","3","d","3/2",1158.6])
core_levels.append([63,"Eu","3","d","5/2",1127.5])
core_levels.append([63,"Eu","4","s","0",360])
core_levels.append([63,"Eu","4","p","1/2",284])
core_levels.append([63,"Eu","4","p","3/2",257])
core_levels.append([63,"Eu","4","d","3/2",133])
core_levels.append([63,"Eu","4","d","5/2",127.7])
core_levels.append([63,"Eu","4","f","5/2",0])
core_levels.append([63,"Eu","4","f","7/2",0])
core_levels.append([63,"Eu","5","s","0",32])
core_levels.append([63,"Eu","5","p","1/2",22])
core_levels.append([63,"Eu","5","p","3/2",22])

core_levels.append([64,"Gd","1","s","0",50239])
core_levels.append([64,"Gd","2","s","0",8376])
core_levels.append([64,"Gd","2","p","1/2",7930])
core_levels.append([64,"Gd","2","p","3/2",7243])
core_levels.append([64,"Gd","3","s","0",1881])
core_levels.append([64,"Gd","3","p","1/2",1688])
core_levels.append([64,"Gd","3","p","3/2",1544])
core_levels.append([64,"Gd","3","d","3/2",1221.9])
core_levels.append([64,"Gd","3","d","5/2",1189.6])
core_levels.append([64,"Gd","4","s","0",378.6])
core_levels.append([64,"Gd","4","p","1/2",286])
core_levels.append([64,"Gd","4","p","3/2",271])
core_levels.append([64,"Gd","4","d","5/2",142.6])
core_levels.append([64,"Gd","4","f","5/2",8.6])
core_levels.append([64,"Gd","4","f","7/2",8.6])
core_levels.append([64,"Gd","5","s","0",36])
core_levels.append([64,"Gd","5","p","1/2",20])
core_levels.append([64,"Gd","5","p","3/2",20])

core_levels.append([65,"Tb","1","s","0",51996])
core_levels.append([65,"Tb","2","s","0",8708])
core_levels.append([65,"Tb","2","p","1/2",8252])
core_levels.append([65,"Tb","2","p","3/2",7514])
core_levels.append([65,"Tb","3","s","0",1968])
core_levels.append([65,"Tb","3","p","1/2",1768])
core_levels.append([65,"Tb","3","p","3/2",1611])
core_levels.append([65,"Tb","3","d","3/2",1276.9])
core_levels.append([65,"Tb","3","d","5/2",1241.1])
core_levels.append([65,"Tb","4","s","0",396])
core_levels.append([65,"Tb","4","p","1/2",322.4])
core_levels.append([65,"Tb","4","p","3/2",284.1])
core_levels.append([65,"Tb","4","d","3/2",150.5])
core_levels.append([65,"Tb","4","d","5/2",150.5])
core_levels.append([65,"Tb","4","f","5/2",7.7])
core_levels.append([65,"Tb","4","f","7/2",2.4])
core_levels.append([65,"Tb","5","s","0",45.6])
core_levels.append([65,"Tb","5","p","1/2",28.7])
core_levels.append([65,"Tb","5","p","3/2",22.6])

core_levels.append([66,"Dy","1","s","0",53789])
core_levels.append([66,"Dy","2","s","0",9046])
core_levels.append([66,"Dy","2","p","1/2",8581])
core_levels.append([66,"Dy","2","p","3/2",7790])
core_levels.append([66,"Dy","3","s","0",2047])
core_levels.append([66,"Dy","3","p","1/2",1842])
core_levels.append([66,"Dy","3","p","3/2",1676])
core_levels.append([66,"Dy","3","d","3/2",1333])
core_levels.append([66,"Dy","3","d","5/2",1292])
core_levels.append([66,"Dy","4","s","0",414.2])
core_levels.append([66,"Dy","4","p","1/2",333.5])
core_levels.append([66,"Dy","4","p","3/2",293.2])
core_levels.append([66,"Dy","4","d","3/2",153.6])
core_levels.append([66,"Dy","4","d","5/2",153.6])
core_levels.append([66,"Dy","4","f","5/2",8])
core_levels.append([66,"Dy","4","f","7/2",4.3])
core_levels.append([66,"Dy","5","s","0",49.9])
core_levels.append([66,"Dy","5","p","1/2",26.3])
core_levels.append([66,"Dy","5","p","3/2",26.3])

core_levels.append([67,"Ho","1","s","0",55618])
core_levels.append([67,"Ho","2","s","0",9394])
core_levels.append([67,"Ho","2","p","1/2",8918])
core_levels.append([67,"Ho","2","p","3/2",8071])
core_levels.append([67,"Ho","3","s","0",2128])
core_levels.append([67,"Ho","3","p","1/2",1923])
core_levels.append([67,"Ho","3","p","3/2",1741])
core_levels.append([67,"Ho","3","d","3/2",1392])
core_levels.append([67,"Ho","3","d","5/2",1351])
core_levels.append([67,"Ho","4","s","0",432.4])
core_levels.append([67,"Ho","4","p","1/2",343.5])
core_levels.append([67,"Ho","4","p","3/2",308.2])
core_levels.append([67,"Ho","4","d","3/2",160])
core_levels.append([67,"Ho","4","d","5/2",160])
core_levels.append([67,"Ho","4","f","5/2",8.6])
core_levels.append([67,"Ho","4","f","7/2",5.2])
core_levels.append([67,"Ho","5","s","0",49.3])
core_levels.append([67,"Ho","5","p","1/2",30.8])
core_levels.append([67,"Ho","5","p","3/2",24.1])

core_levels.append([68,"Er","1","s","0",57486])
core_levels.append([68,"Er","2","s","0",9751])
core_levels.append([68,"Er","2","p","1/2",9264])
core_levels.append([68,"Er","2","p","3/2",8358])
core_levels.append([68,"Er","3","s","0",2206])
core_levels.append([68,"Er","3","p","1/2",2006])
core_levels.append([68,"Er","3","p","3/2",1812])
core_levels.append([68,"Er","3","d","3/2",1453])
core_levels.append([68,"Er","3","d","5/2",1409])
core_levels.append([68,"Er","4","s","0",449.8])
core_levels.append([68,"Er","4","p","1/2",366.2])
core_levels.append([68,"Er","4","p","3/2",320.2])
core_levels.append([68,"Er","4","d","3/2",167.6])
core_levels.append([68,"Er","4","d","5/2",167.6])
core_levels.append([68,"Er","4","f","7/2",4.7])
core_levels.append([68,"Er","5","s","0",50.6])
core_levels.append([68,"Er","5","p","1/2",31.4])
core_levels.append([68,"Er","5","p","3/2",24.7])

core_levels.append([69,"Tm","1","s","0",59390])
core_levels.append([69,"Tm","2","s","0",10116])
core_levels.append([69,"Tm","2","p","1/2",9617])
core_levels.append([69,"Tm","2","p","3/2",8648])
core_levels.append([69,"Tm","3","s","0",2307])
core_levels.append([69,"Tm","3","p","1/2",2090])
core_levels.append([69,"Tm","3","p","3/2",1885])
core_levels.append([69,"Tm","3","d","3/2",1515])
core_levels.append([69,"Tm","3","d","5/2",1468])
core_levels.append([69,"Tm","4","s","0",470.9])
core_levels.append([69,"Tm","4","p","1/2",385.9])
core_levels.append([69,"Tm","4","p","3/2",332.6])
core_levels.append([69,"Tm","4","d","3/2",175.5])
core_levels.append([69,"Tm","4","d","5/2",175.5])
core_levels.append([69,"Tm","4","f","7/2",4.6])
core_levels.append([69,"Tm","5","s","0",54.7])
core_levels.append([69,"Tm","5","p","1/2",31.8])
core_levels.append([69,"Tm","5","p","3/2",25])

core_levels.append([70,"Yb","1","s","0",61332])
core_levels.append([70,"Yb","2","s","0",10486])
core_levels.append([70,"Yb","2","p","1/2",9978])
core_levels.append([70,"Yb","2","p","3/2",8944])
core_levels.append([70,"Yb","3","s","0",2398])
core_levels.append([70,"Yb","3","p","1/2",2173])
core_levels.append([70,"Yb","3","p","3/2",1950])
core_levels.append([70,"Yb","3","d","3/2",1576])
core_levels.append([70,"Yb","3","d","5/2",1528])
core_levels.append([70,"Yb","4","s","0",480.5])
core_levels.append([70,"Yb","4","p","1/2",388.7])
core_levels.append([70,"Yb","4","p","3/2",339.7])
core_levels.append([70,"Yb","4","d","3/2",191.2])
core_levels.append([70,"Yb","4","d","5/2",182.4])
core_levels.append([70,"Yb","4","f","5/2",2.5])
core_levels.append([70,"Yb","4","f","7/2",1.3])
core_levels.append([70,"Yb","5","s","0",52])
core_levels.append([70,"Yb","5","p","1/2",30.3])
core_levels.append([70,"Yb","5","p","3/2",24.1])

core_levels.append([71,"Lu","1","s","0",63314])
core_levels.append([71,"Lu","2","s","0",10870])
core_levels.append([71,"Lu","2","p","1/2",10349])
core_levels.append([71,"Lu","2","p","3/2",9244])
core_levels.append([71,"Lu","3","s","0",2491])
core_levels.append([71,"Lu","3","p","1/2",2264])
core_levels.append([71,"Lu","3","p","3/2",2024])
core_levels.append([71,"Lu","3","d","3/2",1639])
core_levels.append([71,"Lu","3","d","5/2",1589])
core_levels.append([71,"Lu","4","s","0",506.8])
core_levels.append([71,"Lu","4","p","1/2",412.4])
core_levels.append([71,"Lu","4","p","3/2",359.2])
core_levels.append([71,"Lu","4","d","3/2",206.1])
core_levels.append([71,"Lu","4","d","5/2",196.3])
core_levels.append([71,"Lu","4","f","5/2",8.9])
core_levels.append([71,"Lu","4","f","7/2",7.5])
core_levels.append([71,"Lu","5","s","0",57.3])
core_levels.append([71,"Lu","5","p","1/2",33.6])
core_levels.append([71,"Lu","5","p","3/2",26.7])

core_levels.append([72,"Hf","1","s","0",65351])
core_levels.append([72,"Hf","2","s","0",11271])
core_levels.append([72,"Hf","2","p","1/2",10739])
core_levels.append([72,"Hf","2","p","3/2",9561])
core_levels.append([72,"Hf","3","s","0",2601])
core_levels.append([72,"Hf","3","p","1/2",2365])
core_levels.append([72,"Hf","3","p","3/2",2107])
core_levels.append([72,"Hf","3","d","3/2",1716])
core_levels.append([72,"Hf","3","d","5/2",1662])
core_levels.append([72,"Hf","4","s","0",538])
core_levels.append([72,"Hf","4","p","1/2",438.2])
core_levels.append([72,"Hf","4","p","3/2",380.7])
core_levels.append([72,"Hf","4","d","3/2",220])
core_levels.append([72,"Hf","4","d","5/2",211.5])
core_levels.append([72,"Hf","4","f","5/2",15.9])
core_levels.append([72,"Hf","4","f","7/2",14.2])
core_levels.append([72,"Hf","5","s","0",64.2])
core_levels.append([72,"Hf","5","p","1/2",38])
core_levels.append([72,"Hf","5","p","3/2",29.9])

core_levels.append([73,"Ta","1","s","0",67416])
core_levels.append([73,"Ta","2","s","0",11682])
core_levels.append([73,"Ta","2","p","1/2",11136])
core_levels.append([73,"Ta","2","p","3/2",9881])
core_levels.append([73,"Ta","3","s","0",2708])
core_levels.append([73,"Ta","3","p","1/2",2469])
core_levels.append([73,"Ta","3","p","3/2",2194])
core_levels.append([73,"Ta","3","d","3/2",1793])
core_levels.append([73,"Ta","3","d","5/2",1735])
core_levels.append([73,"Ta","4","s","0",563.4])
core_levels.append([73,"Ta","4","p","1/2",463.4])
core_levels.append([73,"Ta","4","p","3/2",400.9])
core_levels.append([73,"Ta","4","d","3/2",237.9])
core_levels.append([73,"Ta","4","d","5/2",226.4])
core_levels.append([73,"Ta","4","f","5/2",23.5])
core_levels.append([73,"Ta","4","f","7/2",21.6])
core_levels.append([73,"Ta","5","s","0",69.7])
core_levels.append([73,"Ta","5","p","1/2",42.2])
core_levels.append([73,"Ta","5","p","3/2",32.7])

core_levels.append([74,"W","1","s","0",69525])
core_levels.append([74,"W","2","s","0",12100])
core_levels.append([74,"W","2","p","1/2",11544])
core_levels.append([74,"W","2","p","3/2",10207])
core_levels.append([74,"W","3","s","0",2820])
core_levels.append([74,"W","3","p","1/2",2575])
core_levels.append([74,"W","3","p","3/2",2281])
core_levels.append([74,"W","3","d","3/2",1949])
core_levels.append([74,"W","3","d","5/2",1809])
core_levels.append([74,"W","4","s","0",594.1])
core_levels.append([74,"W","4","p","1/2",490.4])
core_levels.append([74,"W","4","p","3/2",423.6])
core_levels.append([74,"W","4","d","3/2",255.9])
core_levels.append([74,"W","4","d","5/2",243.5])
core_levels.append([74,"W","4","f","5/2",33.6])
core_levels.append([74,"W","4","f","7/2",31.4])
core_levels.append([74,"W","5","s","0",75.6])
core_levels.append([74,"W","5","p","1/2",45.3])
core_levels.append([74,"W","5","p","3/2",36.8])

core_levels.append([75,"Re","1","s","0",71676])
core_levels.append([75,"Re","2","s","0",12527])
core_levels.append([75,"Re","2","p","1/2",11959])
core_levels.append([75,"Re","2","p","3/2",10535])
core_levels.append([75,"Re","3","s","0",2932])
core_levels.append([75,"Re","3","p","1/2",2682])
core_levels.append([75,"Re","3","p","3/2",2367])
core_levels.append([75,"Re","3","d","3/2",1949])
core_levels.append([75,"Re","3","d","5/2",1883])
core_levels.append([75,"Re","4","s","0",625.4])
core_levels.append([75,"Re","4","p","1/2",518.7])
core_levels.append([75,"Re","4","p","3/2",446.8])
core_levels.append([75,"Re","4","d","3/2",273.9])
core_levels.append([75,"Re","4","d","5/2",260.5])
core_levels.append([75,"Re","4","f","5/2",42.9])
core_levels.append([75,"Re","4","f","7/2",40.5])
core_levels.append([75,"Re","5","s","0",83])
core_levels.append([75,"Re","5","p","1/2",45.6])
core_levels.append([75,"Re","5","p","3/2",34.6])

core_levels.append([76,"Os","1","s","0",73871])
core_levels.append([76,"Os","2","s","0",12968])
core_levels.append([76,"Os","2","p","1/2",12385])
core_levels.append([76,"Os","2","p","3/2",10871])
core_levels.append([76,"Os","3","s","0",3049])
core_levels.append([76,"Os","3","p","1/2",2792])
core_levels.append([76,"Os","3","p","3/2",2457])
core_levels.append([76,"Os","3","d","3/2",2031])
core_levels.append([76,"Os","3","d","5/2",1960])
core_levels.append([76,"Os","4","s","0",658.2])
core_levels.append([76,"Os","4","p","1/2",549.1])
core_levels.append([76,"Os","4","p","3/2",470.7])
core_levels.append([76,"Os","4","d","3/2",293.1])
core_levels.append([76,"Os","4","d","5/2",278.5])
core_levels.append([76,"Os","4","f","5/2",53.4])
core_levels.append([76,"Os","4","f","7/2",50.7])
core_levels.append([76,"Os","5","s","0",84])
core_levels.append([76,"Os","5","p","1/2",58])
core_levels.append([76,"Os","5","p","3/2",44.5])

core_levels.append([77,"Ir","1","s","0",76111])
core_levels.append([77,"Ir","2","s","0",13419])
core_levels.append([77,"Ir","2","p","1/2",12824])
core_levels.append([77,"Ir","2","p","3/2",11215])
core_levels.append([77,"Ir","3","s","0",3174])
core_levels.append([77,"Ir","3","p","1/2",2909])
core_levels.append([77,"Ir","3","p","3/2",2551])
core_levels.append([77,"Ir","3","d","3/2",2116])
core_levels.append([77,"Ir","3","d","5/2",2040])
core_levels.append([77,"Ir","4","s","0",691.1])
core_levels.append([77,"Ir","4","p","1/2",577.8])
core_levels.append([77,"Ir","4","p","3/2",495.8])
core_levels.append([77,"Ir","4","d","3/2",311.9])
core_levels.append([77,"Ir","4","d","5/2",296.3])
core_levels.append([77,"Ir","4","f","5/2",63.8])
core_levels.append([77,"Ir","4","f","7/2",60.8])
core_levels.append([77,"Ir","5","s","0",95.2])
core_levels.append([77,"Ir","5","p","1/2",63])
core_levels.append([77,"Ir","5","p","3/2",48])

core_levels.append([78,"Pt","1","s","0",78395])
core_levels.append([78,"Pt","2","s","0",13880])
core_levels.append([78,"Pt","2","p","1/2",13273])
core_levels.append([78,"Pt","2","p","3/2",11564])
core_levels.append([78,"Pt","3","s","0",3296])
core_levels.append([78,"Pt","3","p","1/2",3027])
core_levels.append([78,"Pt","3","p","3/2",2645])
core_levels.append([78,"Pt","3","d","3/2",2202])
core_levels.append([78,"Pt","3","d","5/2",2122])
core_levels.append([78,"Pt","4","s","0",725.4])
core_levels.append([78,"Pt","4","p","1/2",609.1])
core_levels.append([78,"Pt","4","p","3/2",519.4])
core_levels.append([78,"Pt","4","d","3/2",331.6])
core_levels.append([78,"Pt","4","d","5/2",314.6])
core_levels.append([78,"Pt","4","f","5/2",74.5])
core_levels.append([78,"Pt","4","f","7/2",71.2])
core_levels.append([78,"Pt","5","s","0",101.7])
core_levels.append([78,"Pt","5","p","1/2",65.3])
core_levels.append([78,"Pt","5","p","3/2",51.7])

core_levels.append([79,"Au","1","s","0",80725])
core_levels.append([79,"Au","2","s","0",14353])
core_levels.append([79,"Au","2","p","1/2",13734])
core_levels.append([79,"Au","2","p","3/2",11919])
core_levels.append([79,"Au","3","s","0",3425])
core_levels.append([79,"Au","3","p","1/2",3148])
core_levels.append([79,"Au","3","p","3/2",2743])
core_levels.append([79,"Au","3","d","3/2",2291])
core_levels.append([79,"Au","3","d","5/2",2206])
core_levels.append([79,"Au","4","s","0",762.1])
core_levels.append([79,"Au","4","p","1/2",642.7])
core_levels.append([79,"Au","4","p","3/2",546.3])
core_levels.append([79,"Au","4","d","3/2",353.2])
core_levels.append([79,"Au","4","d","5/2",335.1])
core_levels.append([79,"Au","4","f","5/2",87.6])
core_levels.append([79,"Au","4","f","7/2",83.9])
core_levels.append([79,"Au","5","s","0",107.2])
core_levels.append([79,"Au","5","p","1/2",74.2])
core_levels.append([79,"Au","5","p","3/2",57.2])

core_levels.append([80,"Hg","1","s","0",83102])
core_levels.append([80,"Hg","2","s","0",14839])
core_levels.append([80,"Hg","2","p","1/2",14209])
core_levels.append([80,"Hg","2","p","3/2",12284])
core_levels.append([80,"Hg","3","s","0",3562])
core_levels.append([80,"Hg","3","p","1/2",3279])
core_levels.append([80,"Hg","3","p","3/2",2847])
core_levels.append([80,"Hg","3","d","3/2",2385])
core_levels.append([80,"Hg","3","d","5/2",2295])
core_levels.append([80,"Hg","4","s","0",802.2])
core_levels.append([80,"Hg","4","p","1/2",680.2])
core_levels.append([80,"Hg","4","p","3/2",576.6])
core_levels.append([80,"Hg","4","d","3/2",378.2])
core_levels.append([80,"Hg","4","d","5/2",358.8])
core_levels.append([80,"Hg","4","f","5/2",104])
core_levels.append([80,"Hg","4","f","7/2",99.9])
core_levels.append([80,"Hg","5","s","0",127])
core_levels.append([80,"Hg","5","p","1/2",83.1])
core_levels.append([80,"Hg","5","p","3/2",64.5])
core_levels.append([80,"Hg","5","d","3/2",9.6])
core_levels.append([80,"Hg","5","d","5/2",7.8])

core_levels.append([81,"Tl","1","s","0",85530])
core_levels.append([81,"Tl","2","s","0",15347])
core_levels.append([81,"Tl","2","p","1/2",14698])
core_levels.append([81,"Tl","2","p","3/2",12658])
core_levels.append([81,"Tl","3","s","0",3704])
core_levels.append([81,"Tl","3","p","1/2",3416])
core_levels.append([81,"Tl","3","p","3/2",2957])
core_levels.append([81,"Tl","3","d","3/2",2485])
core_levels.append([81,"Tl","3","d","5/2",2389])
core_levels.append([81,"Tl","4","s","0",846.2])
core_levels.append([81,"Tl","4","p","1/2",720.5])
core_levels.append([81,"Tl","4","p","3/2",609.5])
core_levels.append([81,"Tl","4","d","3/2",405.7])
core_levels.append([81,"Tl","4","d","5/2",385])
core_levels.append([81,"Tl","4","f","5/2",122.2])
core_levels.append([81,"Tl","4","f","7/2",117.8])
core_levels.append([81,"Tl","5","s","0",136])
core_levels.append([81,"Tl","5","p","1/2",94.6])
core_levels.append([81,"Tl","5","p","3/2",73.5])
core_levels.append([81,"Tl","5","d","3/2",14.7])
core_levels.append([81,"Tl","5","d","5/2",12.5])

core_levels.append([82,"Pb","1","s","0",88005])
core_levels.append([82,"Pb","2","s","0",15861])
core_levels.append([82,"Pb","2","p","1/2",15200])
core_levels.append([82,"Pb","2","p","3/2",13035])
core_levels.append([82,"Pb","3","s","0",3851])
core_levels.append([82,"Pb","3","p","1/2",3554])
core_levels.append([82,"Pb","3","p","3/2",3066])
core_levels.append([82,"Pb","3","d","3/2",2586])
core_levels.append([82,"Pb","3","d","5/2",2484])
core_levels.append([82,"Pb","4","s","0",891.8])
core_levels.append([82,"Pb","4","p","1/2",761.9])
core_levels.append([82,"Pb","4","p","3/2",643.5])
core_levels.append([82,"Pb","4","d","3/2",434.3])
core_levels.append([82,"Pb","4","d","5/2",412.2])
core_levels.append([82,"Pb","4","f","5/2",141.7])
core_levels.append([82,"Pb","4","f","7/2",136.9])
core_levels.append([82,"Pb","5","s","0",147])
core_levels.append([82,"Pb","5","p","1/2",106.4])
core_levels.append([82,"Pb","5","p","3/2",83.3])
core_levels.append([82,"Pb","5","d","3/2",20.7])
core_levels.append([82,"Pb","5","d","5/2",18.1])

core_levels.append([83,"Bi","1","s","0",90526])
core_levels.append([83,"Bi","2","s","0",16388])
core_levels.append([83,"Bi","2","p","1/2",15711])
core_levels.append([83,"Bi","2","p","3/2",13419])
core_levels.append([83,"Bi","3","s","0",3999])
core_levels.append([83,"Bi","3","p","1/2",3696])
core_levels.append([83,"Bi","3","p","3/2",3177])
core_levels.append([83,"Bi","3","d","3/2",2688])
core_levels.append([83,"Bi","3","d","5/2",2580])
core_levels.append([83,"Bi","4","s","0",939])
core_levels.append([83,"Bi","4","p","1/2",805.2])
core_levels.append([83,"Bi","4","p","3/2",678.8])
core_levels.append([83,"Bi","4","d","3/2",464])
core_levels.append([83,"Bi","4","d","5/2",440.1])
core_levels.append([83,"Bi","4","f","5/2",162.3])
core_levels.append([83,"Bi","4","f","7/2",157])
core_levels.append([83,"Bi","5","s","0",159.3])
core_levels.append([83,"Bi","5","p","1/2",119])
core_levels.append([83,"Bi","5","p","3/2",92.6])
core_levels.append([83,"Bi","5","d","3/2",26.9])
core_levels.append([83,"Bi","5","d","5/2",23.8])

core_levels.append([92,"U","4","s","0",1131.4])
core_levels.append([92,"U","4","p","0",1004.3])
core_levels.append([92,"U","5","s","0",251.6])
core_levels.append([92,"U","4","d","0",766.1])
core_levels.append([92,"U","5","p","0",202.0])
core_levels.append([92,"U","6","s","0",40.3])
core_levels.append([92,"U","4","f","0",434.7])
core_levels.append([92,"U","5","d","0",114.8])
core_levels.append([92,"U","6","p","0",26.1])
core_levels.append([92,"U","5","f","0",16.1])

