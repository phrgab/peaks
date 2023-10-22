#Functions used to apply Fermi level corrections to dispersions
#Brendan Edwards 26/04/2021
#PK: rejigged a few things

import matplotlib.pyplot as plt
import numpy as np
import warnings
import xarray as xr
from tqdm.notebook import tqdm
from peaks.core.utils.metadata import update_hist
from peaks.core.utils.misc import warning_simple, warning_standard
from peaks.core.display import disp_from_hv
from peaks.core.utils import estimate_EF
from peaks.core.utils.E_shift import Eshift_from_correction
from peaks.core.utils.OOP_method import add_methods



@add_methods(xr.DataArray)
def apply_fermi(dispersion, **kwargs):
    '''This function applies a Fermi level correction to the input dispersion/FM
    
    Input:
        dispersion - the dispersion/FM to be corrected (xarray)
        **kwargs:
            EF_correction - the Fermi level correction to be applied to the input dispersion/FM (float/int, dict, xarray)
    
    Returns:
        corrected_dispersion - the corrected dispersion/FM (xarray)'''
    
    #update attr EF_correction if provided
    try:
        dispersion.attrs['EF_correction'] = kwargs['EF_correction']
    except:
        pass
    #copy the input xarray
    dispersion_to_correct = dispersion.copy(deep=True)
    
    #get theta_par values of dispersion
    theta_par_values = dispersion_to_correct.coords['theta_par'].data
    
    # Extract relevant energy shifts and correction type from dataarray
    correction_to_apply, correction_type = Eshift_from_correction(dispersion)

    #if the correction type is a float/int
    if correction_type == 'value':
        correction_to_apply = float(correction_to_apply)
        dispersion_to_correct.coords['eV'] = dispersion_to_correct.coords['eV'] - correction_to_apply
        corrected_dispersion = dispersion_to_correct
    else:
        #need to shift coords so that our xarray plots the relevant eV section (not near the old Fermi level)
        shift = max(correction_to_apply) #results in only data above EF being cropped
        dispersion_to_correct.coords['eV'] = dispersion_to_correct.coords['eV'] - shift #shift xarray axis
        correction_to_apply = correction_to_apply - shift #shift correction
        
        #get eV coords as a list
        eV_values = dispersion_to_correct.coords['eV'].data    
        #relation between new and original coordinates which will be used for the interpolation
        eV_xarray = xr.DataArray(
            (eV_values[:, np.newaxis]) + correction_to_apply,
            dims=['eV', "theta_par"],
            coords={'eV': eV_values, "theta_par": theta_par_values},
        )
        #get theta_par values as an xarray
        theta_par_xarray = xr.DataArray(theta_par_values, dims=["theta_par"], coords={"theta_par": theta_par_values})
        #apply the interpolation
        corrected_dispersion = dispersion_to_correct.interp(theta_par=theta_par_xarray, eV=eV_xarray)
        #define theta_par units
        corrected_dispersion.coords['theta_par'].attrs = {'units' : 'deg'}
    
    #update metadata
    corrected_dispersion.attrs['eV_type'] = 'binding'
    
    #update the xarray analysis history
    if correction_type == 'fit':
        if dispersion.attrs['EF_correction']['c3'] == 0:
            update_hist(corrected_dispersion, 'Fermi level corrected by a quadratic fit')
        else:
            update_hist(corrected_dispersion, 'Fermi level corrected by a cubic fit')
    elif correction_type == 'array':
        update_hist(corrected_dispersion, 'Fermi level corrected by an array')
    elif correction_type == 'value':
        update_hist(corrected_dispersion, 'Fermi level corrected by a rigid shift')
    
    #return the corrected dispersion
    return corrected_dispersion

@add_methods(xr.DataArray)
def hv_align(data_in, hv_from=0, hv_to=0, hv_step=5, fit_polyorder=3, **kwargs):
    '''This function performs an automatic alignment of the data from an hv-dep cube trying to do a rough
    alignment of EF

    Input:
        data - hv-dep cube of data (xarray)

        Optional arguments for specifying fit:
            - hv_from (optional, defaults to first photon energy) - value of photon energy to start the alignment
              from (float)
            - hv_to (optional, defaults to last photon energy) - value of photon energy to perform alignment to
              (float)
            - hv_step (optional, default 5eV) - step size of how many photon energies to correct
            - fit_polyorder (optional, default=3) - set the order of the polynomial function used to fit the Fermi
              level data

        **kwargs (further optional arguments):
            Setting EF_correction method:
                - EF_correction = float, fit_coeffs, xarray (see fermi_level_corrections.py): EF_correction to
                  use (also overwrites that attribute in the original dataarray)
                - fit = boolean; if fit == False, then the Fermi level is estiamted from the peak in the first
                  derivative of the data (this is the deafult); if fit == True, then the Fermi level is fit
                  for each slice.


    Returns:
        data - hv-dep cube of data with updated EF alignment parameters (xarray)'''

    # Set up warnings display and formatting
    warnings.simplefilter('always', UserWarning)  # Give warnings every time, even on function re-run
    warnings.formatwarning = warning_simple  # Formatting of warnings for peaks user errors

    if 'EF_correction' in kwargs:  # If called with specific EF_correction
        data_in.attrs['EF_correction'] = kwargs['EF_correction']  # Write EF_correction to data attributes

    if 'fit' in kwargs:  # Check whether to estimate EF from peak in derivative or from fitting Fermi edge
        do_EF_fit = kwargs['fit']  # Use what is specified in call
    else:
        do_EF_fit = False  # Else default to False (i.e. use derivative method)

    # Copy the input array
    data = data_in.copy(deep=True)

    # Set proper hv range to loop over
    if hv_from == 0:
        hv_from = data.hv.data[0]
    if hv_to == 0:
        hv_to = data.hv.data[-1]
    n_hv = ((np.abs(hv_from - hv_to)) / np.abs(hv_step)).astype(np.int64) + 1
    hv_list = np.linspace(hv_from, hv_to, n_hv)

    EF_out = np.zeros(n_hv)

    # Check for whether we can correct EF curvature
    # determine if there is an angular-dependent correction defined
    if 'EF_correction' in data.attrs:
        if data.attrs['EF_correction'] != None:
            correct_EF_curve = True
        else:
            correct_EF_curve = False
    else:
        correct_EF_curve = False

    # Iterate through the hv list and estimate EF for each one from a simple derivative
    for count, i in tqdm(enumerate(hv_list), total=len(hv_list), desc='Estimating EF'):
        disp = disp_from_hv(data, i, correct_EF_curve)
        EF_out[count] = estimate_EF(disp, fit=do_EF_fit)
        hv_list[count] = float(disp.hv.data)

    # Make this a dataarray
    EF = xr.DataArray(EF_out, dims=['hv'], coords={'hv': hv_list}, name='E_F')

    # Perform linear fit to extracted data
    EF_fit = EF.polyfit(dim='hv', deg=fit_polyorder)
    # Calculate residuals
    residual = EF - xr.polyval(coord=EF.hv.sel(), coeffs=EF_fit.polyfit_coefficients)
    residual.name = 'Residual'

    # Calculate this over full hv range
    EF_out = xr.polyval(coord=data.hv, coeffs=EF_fit.polyfit_coefficients)
    EF_out.name = 'E_F'

    # Plot the fits and output
    fig, axes = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    EF.plot.line("o", ax=axes[0])
    EF_out.plot(ax=axes[0])
    residual.plot.line("x", ax=axes[1])
    plt.tight_layout()
    plt.show()

    # Add EF list to the hv cube as a non-dimension coordinate
    data.coords['EF'] = ('hv', EF_out.data)

    # Update history and provide a warning that this method is not so precise
    update_hist(data,
                "Performed automatic Fermi level alignment based on derivative method to find Fermi edge - check carefully for accuracy")
    warning_str = "Performed automatic Fermi level alignment based on derivative method to find Fermi edge. Check carefully - a more accurate method may be required for proper analysis."
    warnings.warn(warning_str)

    # Reset warnings display and formatting
    warnings.simplefilter('once', UserWarning)  # Give warnings first time only
    warnings.formatwarning = warning_standard  # Formatting of warnings for peaks user errors

    return data
