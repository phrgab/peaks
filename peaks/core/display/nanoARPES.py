#A collection of functions designed for spatially-resolved ARPES data display/processing
#Phil King 15/5/21

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import xrft
from skimage.feature import peak_local_max
from tqdm.notebook import tqdm
from peaks.core.process import smooth
from peaks.core.display.DC_select import plot_DCs
from peaks.core.display.plot_2d import plot_grid
#from peaks.ML.denoise import denoise
from peaks.utils.OOP_method import add_methods
import xarray as xr

#Script for focussing scans in nano-ARPES (essentially looking for increased sharpness as the beam is scanned across some feature)
@add_methods(xr.DataArray)
def nanofocus(data, focus='defocus'):
    ''' Alingment (focussing) scan for nano-ARPES (essentially looking for increased sharpness as the beam is scanned across some feature)
    
    Input:
        data - the data from the focussing scan - 4D data with some spatial direction of scan, some focus direction, theta_par, and energy (xarray)
        focus (optional, default='defocus') - relevant co-ordinate of the focus direction
        
    Returns:
        Plot '''
        
    #Integrate in energy and angle
    data = data.mean(['theta_par', 'eV'], keep_attrs=True)
    
    #Determine non-focus dimension
    for i in data.dims:
        if i != focus:
            spatial_dim = i
    
    #Work out focus spacing
    focus_step = data[focus].data[1]-data[focus].data[0]
    
    #Set up plot axes
    fig, axes = plt.subplots(ncols=2,nrows=3, figsize=(12,14))
    
    #Plot the integated data as a heat map
    data.plot(x=focus, ax=axes[0,0])
    axes[0,0].set_title('Integrated intensity', fontsize=16) #Label plot
    
    #Stack plot (calling DC function)
    DCs = []
    for i in range(len(data[focus])):
        DCs.append(data.isel({focus:i}))
    plot_DCs(DCs, offset=(data.max().values/10), y=spatial_dim, ax=axes[0,1])
    axes[0,1].set_title('Line cuts', fontsize=16) #Label plot
    
    #Mean abs diff
    absdiff = abs(data.diff(spatial_dim)).mean(spatial_dim)
    absdiff.plot(ax=axes[1,0])
    #Estimate focus point
    maxat = absdiff.pipe(smooth, **{focus: 2*focus_step}).argmax()
    focus1 = absdiff[focus].data[maxat]
    #Label plot
    axes[1,0].set_title('Mean |diff|: focus est. '+str(focus1), fontsize=16)
    
    #Max diff
    maxdiff = abs(data.diff(spatial_dim)).max(spatial_dim)
    maxdiff.plot(ax=axes[1,1])
    #Estimate focus point
    maxat = maxdiff.pipe(smooth, **{focus: 2*focus_step}).argmax()
    focus2 = maxdiff[focus].data[maxat]
    #Label plot
    axes[1,1].set_title('Max |diff|: focus est. '+str(focus2), fontsize=16)
    
    #Variance
    var = data.var(spatial_dim)
    var.plot(ax=axes[2,0])
    #Estimate focus point
    maxat = var.pipe(smooth, **{focus: 2*focus_step}).argmax()
    focus3 = var[focus].data[maxat]
    #Label plot
    axes[2,0].set_title('Variance: focus est. '+str(focus3), fontsize=16)
    
    #FFT based analysis (look for increased higher frequencies indicative of a sharper step)
    data_new = data.copy(deep='True').compute()

    data_new[spatial_dim] = np.linspace(data[spatial_dim].data[0],data[spatial_dim].data[-1],len(data[spatial_dim].data))
    fft = xrft.fft(data_new, dim=spatial_dim, true_phase=True, true_amplitude=True).real
    #Determine non-focus dimension
    for i in fft.dims:
        if i != focus:
            spatial_dim_freq = i
    fft_out = fft.isel({spatial_dim_freq: fft.argmax('freq_x2').data[0]+1})
    fft_out.plot(ax=axes[2,1])
    #Estimate focus point
    maxat = fft_out.pipe(smooth, **{focus: 2*focus_step}).argmax()
    focus4 = fft_out[focus].data[maxat]
    #Label plot
    axes[2,1].set_title('FFT: focus est. '+str(focus4), fontsize=16)
    
    #Calculate mean and median focus, and add focus estimates as lines on plots
    mean_focus = np.mean([focus1,focus2,focus3,focus4])
    median_focus = np.median([focus1,focus2,focus3,focus4])
    
    axes[1,0].axvline(median_focus,color='lightgrey')
    axes[1,0].axvline(focus1)
    axes[1,1].axvline(median_focus,color='lightgrey')
    axes[1,1].axvline(focus2)
    axes[2,0].axvline(median_focus,color='lightgrey')
    axes[2,0].axvline(focus3)
    axes[2,1].axvline(median_focus,color='lightgrey')
    axes[2,1].axvline(focus4)

    #Plot title
    fig.suptitle('Focus estimate: '+str(median_focus)+' (median); '+str(mean_focus)+' (mean)', fontsize=20, y=1)

    #Layout
    plt.tight_layout(pad=3.)

# Integrate over spatial dimensions
@add_methods(xr.DataArray)
def Tot(data, spatial_int = False):
    ''' Quick helper tool for integrating spatial map data over energy and angle (default) or over spatial
    dimensions

    Input:
        data - the 4D data from the spatial map (xarray)
        spatial_int (optional, default = False) - flag to indicate integration should be performed over
         the spatial dimensions

    Output:
        data_tot - the integrated data (xarray). '''

    if spatial_int:
        return data.mean(['x1','x2'], keep_attrs=True)
    else:
        coords = list(filter(lambda n: n != 'x1' and n != 'x2', data.dims))
        return data.mean(coords, keep_attrs=True)



# Using de-noising
@add_methods(xr.DataArray)
def sharpness_dn(data, nx=50, ny=50, dn_selected = True, dn_selected_nx = 300, dn_selected_ny = 300, **kwargs):
    ''' Sharpness estimation from spatial map data, useful for determining where to measure, in particular from
    noisy map data

    Input:
        data - the 4D data from the spatial map (xarray)
        nx (optional, default: 50) - number of points in x to use for neural network size for denoising
        ny (optional, default: 50) - number of points in y to use for neural network size for denoising
         (nx and ny likely shouldn't be too large here, as we need a smoothing effect too)
        dn_selected (optional, default: True) - the spectra selected as possible sharpest will be denoised
         with higher quality, set False to not do this (for time saving)
        dn_selected_nx (optional, default: 300) - nx for this final denoising
        dn_selected_ny (optional, default: 300) - ny for this final denoising
        **kwargs - additional arguments to pass to the plotting function (standard matplotlib calls),
          all passed to displayed dispersions, if cmap specified, this is also passed to the var map

    Returns:
        Plot '''

    denoised_data = denoise(data, nx, ny)

# Script for estimating where in a spatial map the sharpest data can be found
# Using de-noising
@add_methods(xr.DataArray)
def sharpness_dn(data, nx=50, ny=50, dn_selected = True, dn_selected_nx = 300, dn_selected_ny = 300, **kwargs):
    ''' Sharpness estimation from spatial map data, useful for determining where to measure, in particular from
    noisy map data

    Input:
        data - the 4D data from the spatial map (xarray)
        nx (optional, default: 50) - number of points in x to use for neural network size for denoising
        ny (optional, default: 50) - number of points in y to use for neural network size for denoising
         (nx and ny likely shouldn't be too large here, as we need a smoothing effect too)
        dn_selected (optional, default: True) - the spectra selected as possible sharpest will be denoised
         with higher quality, set False to not do this (for time saving)
        dn_selected_nx (optional, default: 300) - nx for this final denoising
        dn_selected_ny (optional, default: 300) - ny for this final denoising
        **kwargs - additional arguments to pass to the plotting function (standard matplotlib calls),
          all passed to displayed dispersions, if cmap specified, this is also passed to the var map

    Returns:
        Plot '''

    denoised_data = denoise(data, nx, ny)

    # Calculate variance of this denoised spatial map
    var_map = denoised_data.var(dim=['eV','theta_par'])

    # Determine local maxima
    coords = peak_local_max(var_map.data, min_distance=2, exclude_border=0)

    # Convert coords from pixel to actual space
    coords_x1 = []
    coords_x2 = []
    titles = []
    data_select = []
    data_select_denoised = []


    for i in coords:
        x1t = float(data.x1[i[0]].data)
        x2t = float(data.x2[i[1]].data)
        coords_x1.append(x1t)
        coords_x2.append(x2t)
        titles.append('x1: '+str(x1t)+', x2: '+str(x2t))
        data_select.append(data.sel(x1=x1t, method='nearest').sel(x2=x2t, method='nearest'))



    # Make the plots
    fig, axes = plt.subplots(ncols=2, figsize=(14,5))

    # Plot the variance map and extracted local maxima
    if 'cmap' in kwargs:  # Pull colour map out if defined in kwargs
        cmap_var = kwargs['cmap']
    else:
        cmap_var = 'viridis'
    var_map.plot(x='x1', ax=axes[0], cmap=cmap_var)
    clist = cm.brg(np.linspace(0,1,len(coords_x1)))
    clist_tab = [clist,clist]
    axes[0].scatter(coords_x1,coords_x2,c=clist)
    axes[0].set_title('Variance map')

    # Plot the coordinate positions
    collabel = ('x1','x2')
    axes[1].axis('tight')
    axes[1].axis('off')
    cellText = np.dstack((coords_x1, coords_x2))[0]
    axes[1].table(cellText=cellText,colLabels=collabel,cellLoc='center',loc='center')
    axes[1].set_title('Suggested locations')
    plt.show()

    print('Suggested points:')
    plot_grid(data_select, ncols=5, titles=titles, titles_col=clist, **kwargs)
    plt.show()

    if dn_selected == True:  # Do a higher quality denoising step on the final selected data
        data_select_denoised = []
        for i in tqdm(data_select, desc='De-noising selected examples: ', leave=False):
            data_select_denoised.append(i.denoise(dn_selected_nx, dn_selected_ny))

        print('Suggested points with de-noising:')
        plot_grid(data_select_denoised, ncols=5, titles=titles, titles_col=clist, vmin=0, **kwargs)
        plt.show()

# Not using de-noising (much faster)
@add_methods(xr.DataArray)
def sharpness(data, smooth_theta_par=0.1, smooth_eV=0.02, **kwargs):
    ''' Sharpness estimation from spatial map data, useful for determining where to measure, in particular from
    noisy map data

    Input:
        data - the 4D data from the spatial map (xarray, in theta_par vs eV format)
        smooth_theta_par - FWHM for data smoothing along theta_par axis (default 0.1 deg)
        smooth_eV - FWHM for data smoothing along eV axis (default 0.02 eV)
        **kwargs - additional arguments to pass to the plotting function (standard matplotlib calls),
          all passed to displayed dispersions, if cmap specified, this is also passed to the var map

    Returns:
        Plot '''

    # Smooth the map
    data_sm = data.copy(deep=True)

    data_sm = data.smooth(theta_par=smooth_theta_par, eV=smooth_eV)

    # Calculate variance of the spatial map
    var_map = data_sm.var(dim=['eV','theta_par'])

    # Determine local maxima
    coords = peak_local_max(var_map.data, min_distance=2, exclude_border=0)

    # Convert coords from pixel to actual space
    coords_x1 = []
    coords_x2 = []
    titles = []
    data_select = []


    for i in coords:
        if data.dims.index('x1') < data.dims.index('x2'):
            x1t = float(data.x1[i[0]].data)
            x2t = float(data.x2[i[1]].data)
        else:
            x1t = float(data.x1[i[1]].data)
            x2t = float(data.x2[i[0]].data)
        coords_x1.append(x1t)
        coords_x2.append(x2t)
        titles.append('x1: '+str(x1t)+', x2: '+str(x2t))
        data_select.append(data.sel(x1=x1t, method='nearest').sel(x2=x2t, method='nearest'))

    # Make the plots
    fig, axes = plt.subplots(ncols=2, figsize=(14,5))

    # Plot the variance map and extracted local maxima
    if 'cmap' in kwargs:  # Pull colour map out if defined in kwargs
        cmap_var = kwargs['cmap']
    else:
        cmap_var = 'viridis'
    var_map.plot(x='x1', ax=axes[0], cmap=cmap_var)
    clist = cm.brg(np.linspace(0,1,len(coords_x1)))
    clist_tab = [clist,clist]
    axes[0].scatter(coords_x1,coords_x2,c=clist)
    axes[0].set_title('Variance map')

    # Plot the coordinate positions
    collabel = ('x1','x2')
    axes[1].axis('tight')
    axes[1].axis('off')
    cellText = np.dstack((coords_x1, coords_x2))[0]
    axes[1].table(cellText=cellText,colLabels=collabel,cellLoc='center',loc='center')
    axes[1].set_title('Suggested locations')
    plt.show()

    print('Suggested points:')
    plot_grid(data_select, ncols=5, titles=titles, titles_col=clist, **kwargs)
    plt.show()
