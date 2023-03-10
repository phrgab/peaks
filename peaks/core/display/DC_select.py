#Functions used to extract DCs from data
#Phil King 24/04/2021
#Updated PK 27/4/21 to change the method of DC integration for speed-up with large integration ranges
#Brendan Edwards 03/12/2021

import matplotlib.pyplot as plt
from matplotlib import colors, cm
from cycler import cycler
import numpy as np
import xarray as xr

from peaks.utils.metadata import update_hist
from peaks.utils.OOP_method import add_methods
from peaks.utils.misc import ana_warn



#Function to extract a single MDC, with or without integration
@add_methods(xr.DataArray)
def MDC(dispersion, E0, dE=0.):
    '''Extract a single MDC from a dataarray

        Parameters
        ------------
        dispersion : xr.DataArray
             The dispersion (2D dataarray) to extract MDC from

        E0 : float
            Energy of MDC to extract

        dE : float, optional
            Integration range (total range, integrates +/- dE/2) <br>
            Defaults to single DC

        Returns
        ------------
        MDC : xr.DataArray
            xarray DataArray of MDC

        '''

    #Call function to extract relevant MDC from dispersion          
    MDC = DC(dispersion, 'eV', [E0], dE, plot=False)
    
    #If returning a single DC, want to remove the non-varying coorinate as a dimension
    try:
        MDC = MDC.squeeze('eV')
    except:
        pass
    
    #Update the analysis history
    hist = 'MDC extracted, integration window: '+str(dE)
    update_hist(MDC,hist)
    
    return MDC


#Function to extract a stack of MDCs with a defined spacing, with or without integration
@add_methods(xr.DataArray)
def MDCs(dispersion, E0, E1, Estep, dE=0, plot=False, **kwargs):
    '''Extract a stack of MDCs from a dataarray, and return this or optionally return a (stack) plot

    Parameters
    ------------
    dispersion : xr.DataArray
         The dispersion (2D dataarray) to extract MDC from

    E0 : float
        Energy of first MDC to extract

    E1 : float
        Energy of last MDC to extract

    Estep : float
        Energy step between extracted MDCs

    dE : float, optional
        Integration range (total range, integrates +/- dE/2) <br>
        Defaults to single DC

    plot : bool, optional
         Call with 'plot=True' to return a plot

    **kwargs : optional
        Additional arguments to pass to the plotting function <br>
        Offset to define offset between DCs, plus standard matplotlib calls

    Returns
    ------------
    MDCs : xr.DataArray
        xarray DataArray of MDC stack or plot if called with `plot=True`

    '''

    #Make array of requested MDCs
    En = np.arange(E0,E1+Estep,Estep)  # Energies of MDCs
    
    MDCs = DC(dispersion, 'eV', En, dE, plot, **kwargs)
    
    #Update the analysis history
    if not plot:  # Only do this if a dataarray is being returned, rather than a plot
        hist = 'MDCs extracted, integration window: '+str(dE)
        update_hist(MDCs,hist)
    
    return MDCs


#Function to extract a single EDC, with or without integration
@add_methods(xr.DataArray)
def EDC(dispersion, ang0, dang=0):
    '''Extract an EDC from a dataarray

    Parameters
    ------------
    dispersion : xr.DataArray
         The dispersion (2D dataarray) to extract EDC from

    ang0 : float
        Angle or k of first EDC to extract

    dang : float, optional
        Integration range (total range, integrates +/- dang/2) <br>
        Defaults to single DC

    Returns
    ------------
    EDCs : xr.DataArray
        xarray DataArray of EDC

    '''
      
    #Check data is 2D
    #if len(dispersion.dims) != 2:
    #    raise Exception("Function only acts on 2D data.")
    
    #Work out correct variable for dispersive direction (i.e. is data in angle or k-space)
    #coord = list(filter(lambda i: i != 'eV', dispersion.dims))[0]
    if dispersion.dims[-1] != 'eV':
        coord = dispersion.dims[-1]  # With ordering of the array now standerdised, should always be the last one
    else:
        coord = list(filter(lambda i: i != 'eV', dispersion.dims))[0]
    
    #Call function to extract relevant EDC from dispersion          
    EDC = DC(dispersion, coord, [ang0], dang, plot=False)
    
    #If returning a single DC, want to remove the non-varying coorinate as a dimension
    try:
        EDC = EDC.squeeze(coord)
    except:
        pass

    hist = 'EDC extracted, integration window: '+str(dang)
    update_hist(EDC,hist)
    
    return EDC


#Function to extract a stack of MDCs with a defined spacing, with or without integration
@add_methods(xr.DataArray)
def EDCs(dispersion, ang0, ang1, ang_step, dang=0, plot=False, **kwargs):
    '''Extract a stack of EDCs from a dataarray, and return this or optionally return a (stack) plot

    Parameters
    ------------
    dispersion : xr.DataArray
         The dispersion (2D dataarray) to extract EDCs from

    ang0 : float
        Angle or k of first EDC to extract

    ang1 : float
        Angle or k of last EDC to extract

    ang_step : float
        Angle/k step between extracted EDCs

    dang : float, optional
        Integration range (total range, integrates +/- dang/2) <br>
        Defaults to single DC

    plot : bool, optional
         Call with 'plot=True' to return a plot

    **kwargs : optional
        Additional arguments to pass to the plotting function <br>
        Offset to define offset between DCs, plus standard matplotlib calls

    Returns
    ------------
    EDCs : xr.DataArray
        xarray DataArray of MDC stack or plot if called with `plot=True`

    '''
    
    #Check data is 2D
    if len(dispersion.dims) != 2:
        raise Exception("Function only acts on 2D data.")
    
    #Work out correct variable for dispersive direction (i.e. is data in angle or k-space)
    coord = list(filter(lambda i: i != 'eV', dispersion.dims))[0]
    
    #Make array of requested MDCs
    ang_n = np.arange(ang0,ang1+ang_step,ang_step) #Angles/ks of EDCs 
    
    #Check if display definition given in kwarg, if not set a default value
    if 'y' not in kwargs:
        kwargs['y'] = 'eV'

    #Call function to extract relevant EDCs from dispersion          
    EDCs = DC(dispersion=dispersion, DC_sel=coord, DC_vals=ang_n, DC_delta=dang, plot=plot, **kwargs)
    
    #Update the analysis history
    if plot==False: #Only do this if a dataarray is being returned, rather than a plot
        hist = 'EDCs extracted, integration window: '+str(dang)      
        update_hist(EDCs,hist)
    
    return EDCs


# Function to extract a Fermi surface or constant energy contour, with or without integration
@add_methods(xr.DataArray)
def FS(cube, E0=0, dE=0):
    '''Extract a constant energy slice (e.g. a Fermi surface) from a 3D dataarray. <br>
    Thin wrapper around peaks.display.DC_select.MDC

    Parameters
    ------------
    cube : xr.DataArray
         The 3D data cube to extract FS from

    E0 : float
        Energy of slice to extract

    dE : float, optional
        Integration range (total range, integrates +/- dE/2) <br>
        Defaults to single slice

    Returns
    ------------
    FS : xr.DataArray
        xarray DataArray of constant energy slice

    '''


    # Check data is 3D
    if len(cube.dims) != 3:
        raise Exception("Function only acts on 3D data.")

    # Call function to extract relevant FS from data cube
    FS = cube.MDC(E0=E0, dE=dE)

    # Sequeze the xarray to remove the energy axis as a dimension
    FS = FS.squeeze()

    return FS

# Function to extract a stack plot of constant energy contours, with or without integration
@add_methods(xr.DataArray)
def FS_stack(cube, E0, E1, Estep, dE=0, plot=False, cmap='Purples', vmax=None, figsize=(8, 12), elev=10., azim=-60.,
             **kwargs):
    '''Extract a set of constant energy slices from a 3D dataarray. <br>
    Optionally returns a stack plot.

    Parameters
    ------------
    cube : xr.DataArray
         The 3D data cube to extract FS from

    E0 : float
        Energy of fist slice to extract

    E1 : float
        Energy of last slice to extract

    Estep : float
        Energy step between slices

    dE : float, optional
        Integration range (total range, integrates +/- dE/2) <br>
        Defaults to single slice

    plot : bool, optional
        Call with 'plot=True' to return a plot

    cmap : str, optional
        colourmap to use, default Purples

    vmax : float, optional
        colour scale maximum, defaults to full range

    figsize : tuple, optional
        spec figure size in (w,h) format <br>
        default: (8,12)

    elev : float, optional
        Elevation camera angle for 3D plot <br>
        default: 10 deg

    azim : float, optional
        Azimuthal camera angle for 3D plot <br>
        default: -60 deg

    **kwargs : optional
        Additional arguments to pass to the plotting <br>
        Must be suitable for passing in ax.set(**kwargs) format. <br>
        Note xlim and ylim don't clip the range for this type of plot.

    Returns
    ------------
    FS : xr.DataArray
        xarray DataArray of constant energy slices or alternatively a plot if called with plot=True

    '''

    # Check data is 3D
    if len(cube.dims) != 3:
        raise Exception("Function only acts on 3D data.")

    # # Work out correct variable for dispersive direction (i.e. is data in angle or k-space)
    # eV_dim = cube.dims.index('eV')  # Work out index of energy dimension
    # coords = list(cube.dims)  # List of coordinates
    # del (coords[eV_dim])  # Delete 'eV' coordinate to leave remaining angular/k-space coords
    #
    # # Make array of requested MDCs
    # En = np.arange(E0, E1 + Estep, Estep)  # Energies of MDCs
    #
    # # Call function to extract relevant FS from data cube
    # FS = DC(cube, coords[0], 'eV', En, dE, 0)

    FS = cube.MDCs(E0=E0, E1=E1, Estep=Estep, dE=dE, plot=False)

    # Update the analysis history
    hist = 'Energy slice extracted, integration window: ' + str(dE)
    update_hist(FS, hist)

    # Return plot if plot option selected
    if plot:
        #ToDo Speed this up: maybe something using holoviews?

        # Turn FS into regular xarray not Dask
        FS = FS.compute()

        plt.interactive(True)

        # Set up plot axes
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d')

        # Relevant coords
        coords = list(filter(lambda i: i != 'eV', FS.dims))

        # Define the grid
        X, Y = np.meshgrid(cube[coords[1]], cube[coords[0]])

        # Normalise the data
        if vmax is None:  # no max colour limit specified
            vmax = FS.max().data
        C = FS / vmax
        C = np.clip(C, 0, 1)

        # Make the planes
        Z = np.zeros_like(X)

        # Define colour map
        cmap_sel = cm.get_cmap(cmap)

        # Make the plot surfaces (downsample 2x2)
        for i in FS['eV'].data:
            ax.plot_surface(X, Y, Z + i, rstride=2, cstride=2, facecolors=cmap_sel(C.sel(eV=i).data), shade=False)

        # Colour scale
        m = cm.ScalarMappable(cmap=cmap)
        m.set_array(np.linspace(0, vmax, 100))
        plt.colorbar(m, shrink=0.2, aspect=8)

        # Some plot display settings
        # Label the axes
        ax.set_xlabel(coords[1])
        ax.set_ylabel(coords[0])
        ax.set_zlabel('eV')

        # Set overall view (no grids etc.)
        ax.grid(False)
        ax.xaxis.pane.set_edgecolor('black')
        ax.yaxis.pane.set_edgecolor('black')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Sort out axis label locations
        [t.set_va('center') for t in ax.get_yticklabels()]
        [t.set_ha('left') for t in ax.get_yticklabels()]
        [t.set_va('center') for t in ax.get_xticklabels()]
        [t.set_ha('right') for t in ax.get_xticklabels()]
        [t.set_va('center') for t in ax.get_zticklabels()]
        [t.set_ha('left') for t in ax.get_zticklabels()]
        ax.xaxis.set_rotate_label(False)
        ax.yaxis.set_rotate_label(False)
        ax.zaxis.set_rotate_label(False)

        # Set z-axis ticks to match the slices
        ax.zaxis.set_major_locator(plt.FixedLocator(np.round(C['eV'].data, 3)))

        # Set default elevation view
        ax.view_init(elev=elev, azim=azim)

        ax.set(**kwargs)

        # Show the plot
        plt.show()

        return
    return FS

#Function which does the relevant DC extraction (generalisable to any 2D array)
@add_methods(xr.DataArray)
def DC(dispersion, DC_sel, DC_vals, DC_delta, plot, **kwargs):
    '''Master DC extraction function

    Parameters
    ------------
    dispersion : xr.DataArray
         The dispersion (2D dataarray) to extract DCs from

    DC_sel : str
        co-ordinate to select along

    DC_vals : list
        array of DC values to select

    DC_delta : float
        Integration range (total range, integrates +/- DC_delta/2) <br>

    plot : bool
         Flag indicating whether to return a plot

    **kwargs : optional
        Additional arguments to pass to the plotting function <br>
        See peaks.display.DC_extract.plot_DCs.

    Returns
    ------------
    DC : xr.DataArray
        xarray DataArray of DC stack or plot if called with `plot=True`

    '''

    # Convert window to pixels
    pix_av = int((DC_delta/abs(dispersion[DC_sel].data[1] - dispersion[DC_sel].data[0]))+1)

    # Extract the DCs and apply extra binning if required
    DC_out = dispersion.sel({DC_sel: DC_vals}, method='nearest')

    if pix_av > 1:
        for i in DC_out[DC_sel]:
            DC_out.loc[{DC_sel: i}] = dispersion.sel({DC_sel: slice(i-DC_delta/2,i+DC_delta/2)}).mean(DC_sel, keep_attrs=True)
                
    #If plotting selected, need to return a plot:
    if plot:
        if len(DC_vals) == 1: #for a single DC, return as a simple matplotlib file, passing additional kwargs for plot definitions
            DC_plot = DC_out.plot(**kwargs)
        
        else: #set up a stack plot
            DC_out.plot_DCs(**kwargs)
            
        return   
    return DC_out

            

#Shortcut function to return an integrated spectrum across all but the energy axis, for DOS approximation
@add_methods(xr.DataArray)
def DOS(data):
    '''Integrate over all but the energy axis to return the best approximation to the DOS possible from the data

    Parameters
    ------------
    data : xr.DataArray
         Data to extract DOS from

    Returns
    ------------
    DOS : xr.DataArray
        xarray DataArray of DOS

    '''

    # Relevant dimensions to integrate over
    int_dim = list(filter(lambda i: i != 'eV', data.dims))

    # Calculate the DOS
    DOS = data.mean(int_dim, keep_attrs=True)

    #Update the history
    hist = 'Integrated along axes: ' + str(int_dim)
    update_hist(DOS, hist)
    
    return DOS

#Function to make a plot of multiple DCs
@add_methods(xr.DataArray)
def plot_DCs(DCs, titles=None, cmap=None, color='k', offset=None, norm=False, stack_dim='auto', figsize=(6,6), **kwargs):
    '''Plot a DC stack with the colours varying according to a colormap

    Parameters
    ------------
    DCs : xr.DataArray or list(xr.DataArray)
         DCs to plot, either as a list of single DC xr.DataArrays <br>
         or a single xr.DataArray with the stack of DCs included

    titles : list, optional
        list of subtitles to be supplied on the plots (length must match number of DCs to plot) <br>
        Defaults to None

    color : str, optional
        Matplotlib color, to use single color for all lines <br>
        Defaults to 'k' (black)

    cmap : str, optional
        name of matplotlib cmap to use <br>
        Takes presedence over color

    offset : float, optional
        Vertical offsets between subsequent DCs <br>
        Defaults to 5% of max peak height

    norm : bool, optional
        Whether to normalise the DCs to 1 <br>
        Defaults to False

    stack_dim : str, optional
        Which dimension to stack DCs along <br>
        Defaults to 'auto' which takes the smallest dimension <br>
        Ignored if a list of DCs is passed.

    figsize : tuple, optional
        Figsize for plot, in matplotlib style <br>
        Defaults to (6,6)

    **kwargs : optional
    Additional arguments to pass to the plotting function <br>
    Standard matplotlib calls

    Returns
    ------------
    Stack plot of DCs

    '''

    if isinstance(DCs, list):  # List of DCs supplied
        # Combine into single array
        DC_array = xr.concat(DCs, dim='DC_no')
        stack_dim = 'DC_no'
    else:
        DC_array = DCs.copy(deep=True)  # Make a copy so we can modify
        if stack_dim == 'auto':  # Assume the stack dim is the smallest dimension size
            if DC_array.shape[0] < DC_array.shape[1]:
                stack_dim = DC_array.dims[0]
            else:
                stack_dim = DC_array.dims[1]

    # Check correct dimension supplied
    if len(DC_array.shape) != 2:
        raise Exception('Incorrect dimension of data supplied. Please supply single 2D dataarray or list of 1D dataarrays')

    # Non stacking dimension
    other_dim = [dim for dim in DC_array.dims if dim != stack_dim][0]

    # Normalise if required
    if norm:
        DC_array = DC_array/DC_array.max(dim=other_dim)

    # Estimate offset if none supplied
    if not offset:
        # Make a guess at the offset as 5% of the max peak height
        offset = float(DC_array.max()) * 0.05

    # Define linear offset wave in steps of the supplied or guessed offset
    offset_data = xr.DataArray(data=np.arange(len(DC_array[stack_dim])) * offset,dims=stack_dim,coords={stack_dim: DC_array[stack_dim]})

    # Add offset to DC data
    DC_array += offset_data

    # Check a few style options and provide defaults if not already specified
    if 'linewidth' not in kwargs:
        kwargs['linewidth'] = 1
    if 'add_legend' not in kwargs and not titles:
        kwargs['add_legend'] = False
    if 'y' not in kwargs and 'x' not in kwargs:
        # If it seems to be an EDC plot, set vertical by default
        if other_dim == 'eV':
            kwargs['y'] = 'eV'
        else:
            kwargs['x'] = other_dim

    # Set up plot
    if 'ax' in kwargs:
        ax = kwargs['ax']
    else:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    if cmap:
        # Define the colour scheme
        cols = getattr(cm, cmap)(np.linspace(0, 1, len(DC_array[stack_dim])))
        cmap = colors.ListedColormap(cols)
        custom_cycler = cycler(color=cmap.colors)  # or simply color=colorlist
        ax.set_prop_cycle(custom_cycler)
        DC_array.plot.line(hue=stack_dim, **kwargs)
    else:
        DC_array.plot.line(hue=stack_dim, color=color, **kwargs)

    if 'y' in kwargs:
        plt.xlabel("Intensity [arb. units]")
    else:
        plt.ylabel("Intensity [arb. units]")

    if titles:
        if len(titles) == len(DC_array[stack_dim]):
            for ct, i in enumerate(ax.lines):
                i.set_label(titles[ct])
            ax.legend()
        else:
            warn_str = 'Supplied titles not used as incompatible length with DC number'
            ana_warn(warn_str, title='Plot info')

    plt.tight_layout()


@add_methods(xr.DataArray)
def radial_cuts(xarray, num_azi=60, num_points=100, radius=2, **kwargs):
    '''Function to extract radial cuts of Fermi surfaces, and plot them against azimuth

    Inputs:
        xarray - the data from which radial cuts will be extracted (xarray)
        **kwargs:
            num_azi - the number of evenly spaced azi values between 0 and 360 degrees to take radial cuts (integer)
            num_points - the number of evenly spaced points to sample along a cut (integer)
            radius - the maximum radius to take cuts up to (float)
            x - the coordinate to take as centre for x (float)
                e.g. k_par = 1 sets the k_par centre as 1
            y - the coordinate to take as centre for y (float)
                e.g. polar = 0.5 sets the polar centre as 0.5

    Returns:
        xarray_to_return - radial cuts against azi (xarray)
    '''

    #define the coordinate system
    x_coord = xarray.dims[1]
    y_coord = xarray.dims[0]
    try:
        x_centre = (kwargs[x_coord])
    except:
        x_centre = 0
    try:
        y_centre = (kwargs[y_coord])
    except:
        y_centre = 0

    #define coordinates to be sampled
    azi_angles = np.linspace(0,360,num_azi)
    k_values = np.linspace(0,radius,num_points)
    spectrum = []

    # for each azi angle, interpolate the data onto a radial cut and append result to spectrum
    for angle in azi_angles:
        x_values = np.linspace(0+x_centre,(np.cos(np.radians(angle))*radius)+x_centre,num_points)
        y_values = np.linspace(0+y_centre,(np.sin(np.radians(angle))*radius)+y_centre,num_points)
        x_xarray = xr.DataArray(x_values, dims='k')
        y_xarray = xr.DataArray(y_values, dims='k')
        interpolated_data = xarray.interp({x_coord:x_xarray, y_coord:y_xarray})
        spectrum.append(interpolated_data.data)

    #create xarray of radial cuts against azi
    xarray_to_return = xr.DataArray(np.array(spectrum).transpose(), dims=("k", "azi"), coords={"k": k_values, "azi":azi_angles})
    xarray_to_return.attrs = xarray.attrs

    #update history attribute
    update_hist(xarray_to_return, 'Radial cuts taken as a function of azi')
    return xarray_to_return
