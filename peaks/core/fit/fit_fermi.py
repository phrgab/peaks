#Functions used to fit Fermi functions, used for Fermi level corrections
#Tommaso Antonelli 13/03/2021
#Brendan Edwards 26/04/2021

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
import xarray as xr
from lmfit import Model
from lmfit.models import LinearModel, PolynomialModel
from tqdm.notebook import tqdm
from peaks.utils.OOP_method import add_methods

def FermiFunction(x, EF, T):
    '''This function defines the fermi function used for fitting

    Input:
        x - current energy value
        EF - the Fermi level
        T - the temperature

    Returns:
        fermi_function_value - the calculated value of the Fermi function'''

    kb = 8.617333262145 * 10 ** (-5)  # Boltzmann constant in eV units
    fermi_function_value = 1 / (1 + np.exp((x - EF) / (kb * T)))  # fermi function
    return fermi_function_value


# ToDo: Get resolution in fit_fermi function
@add_methods(xr.DataArray)
def fit_fermi(dispersion, **kwargs):
    '''This function calculates a Fermi level correction for the input dispersion, assuming a fit as follows:
    (DOS*Fermi_function) + Bkg, where DOS and Bkg are linear functions

    Input:
        dispersion - the dispersion to be corrected (xarray)
        **kwargs - used to modify the method to calculate the the Fermi level correction:
            EF - user defined Fermi level estimate
            theta_par - user defined theta_par fitting range, supplied as slice(theta0, theta1)
            eV_range - user defined eV fitting range around EF, single value for symmetric range, tuple for
            asymmetric range
            sum_range - user defined EDC integration range over theta_par
            fit_type - change correction fit to quadratic if kwarg['fit_type'] = 'quadratic'
            output_type - change output to be the correction array if kwarg['output_type'] = 'array'

    Returns:
        if kwarg['output_type'] == 'array':
            EF_values_array - an array of the Fermi level corrections (np.ndarray)
        else:
            if a dispersion is passed:
                correction_fit_params - the Fermi level correction fit parameters (dict)
            if a single EDC is passed:
                fit parameters for the EDC (dict) '''

    # determine what paramteters user has defined
    user_defined_EF = False
    user_defined_theta_range = False
    user_defined_eV_range = False
    user_defined_sum_range = False
    user_defined_fit_type = False
    user_defined_output_type = False
    for kwarg in kwargs:
        if kwarg == 'EF':
            user_defined_EF = True
        elif kwarg == 'theta_par':
            user_defined_theta_range = True
        elif kwarg == 'eV':
            user_defined_eV_range = True
        elif kwarg == 'sum_range':
            user_defined_sum_range = True
        elif kwarg == 'fit_type':
            user_defined_fit_type = True
        elif kwarg == 'output_type':
            user_defined_output_type = True

    # define theta_par limits
    if user_defined_theta_range == True:
        # get user defined theta_par limits
        min_theta_par = kwargs['theta_par'].start
        max_theta_par = kwargs['theta_par'].stop
        theta_par_values = dispersion.sel(theta_par=slice(min_theta_par, max_theta_par)).coords['theta_par'].values
    elif 'theta_par' in dispersion.dims:  # Fit over a range if multiple theta_par values exist
        # take theta_par limits as that of the input dispersion
        min_theta_par = min(dispersion.coords['theta_par'].values)
        max_theta_par = max(dispersion.coords['theta_par'].values)
        theta_par_values = dispersion.sel(theta_par=slice(min_theta_par, max_theta_par)).coords['theta_par'].values

    # determine eV fitting range around EF
    if user_defined_eV_range == True:
        # get user defined eV fitting range
        if type(kwargs['eV_range']) == float or type(kwargs['eV_range']) == int:
            min_eV_range = -1 * kwargs['eV_range'] / 2
            max_eV_range = kwargs['eV_range'] / 2
        elif len(kwargs['eV_range']) == 2:
            min_eV_range = kwargs['eV_range'][0]
            max_eV_range = kwargs['eV_range'][1]
    else:
        # set a reasonable eV fitting range
        min_eV_range = -0.15
        max_eV_range = 0.15

    # determine EDC integration range over theta_par
    if user_defined_sum_range == True:
        # get user defined EDC integration range over theta_par
        theta_par_sum_range = kwargs['sum_range']
    else:
        # set a reasonable EDC integration range over theta_par
        theta_par_sum_range = 0.3

    # determine temperature value
    if dispersion.attrs['temp_sample'] == None:
        # set a reasonable temperature value if it is not defined
        T_value = 100
    else:
        # get sample temperature from input dispersion xarray attributes
        T_value = dispersion.attrs['temp_sample']

    # create models for background and DOS using built-in LinearModel from lmfit
    bkg = LinearModel(prefix='bkg')
    dos = LinearModel(prefix='dos')
    # create Fermi model from FermiFunction(E, EF, T)
    fermi = Model(FermiFunction, prefix='Fermi')
    # create composite model
    model_tot = bkg + (dos * fermi)

    # create a test EDC for getting inital paramaters for the Fermi level corrections
    if 'theta_par' in dispersion.dims:
        test_EDC = dispersion.sel(theta_par=slice(theta_par_values[0], theta_par_values[0] + theta_par_sum_range)).mean(
            'theta_par')
    else:  # If only a single EDC supplied
        test_EDC = dispersion

    # get EF estimate
    if user_defined_EF == True:
        # if user has defined EF
        EF_estimate = kwargs['EF']
    else:
        # estimate EF
        energy = test_EDC.coords['eV'].values
        filtered_data = gaussian_filter1d(test_EDC, 2)
        dev_data = np.gradient(filtered_data)
        EF_estimate = energy[dev_data.argmin()]

    # definie inital guesses for parameters
    dos_intercept = float(test_EDC.sel(eV=(EF_estimate + min_eV_range), method='nearest').values)
    bkg_intercept = float(test_EDC.sel(eV=max(test_EDC.coords['eV'].values), method='nearest').values)
    init_bkg = bkg.make_params(intercept=bkg_intercept, slope=0)
    init_dos = dos.make_params(intercept=dos_intercept, slope=0)
    init_fermi = fermi.make_params(T=T_value, EF=EF_estimate)
    init_params = init_fermi + init_bkg + init_dos

    # crop the test EDC to an energy region around the estimated EF
    test_EDC_cropped = test_EDC.sel(eV=slice(EF_estimate + min_eV_range, EF_estimate + max_eV_range))
    # do not hold the Temperature during the fitting (will want to hold in the future)
    init_params['FermiT'].vary = True
    # produce a fit to the cropped test EDC
    test_result = model_tot.fit(test_EDC_cropped, x=test_EDC_cropped.coords['eV'].values, params=init_params)
    # get the optimised fitting parameters
    opt_params = test_result.best_values

    if 'theta_par' not in dispersion.dims:  # If only single EDC supplied, we are done, so now return the fit
        return opt_params

    else:  # Otherwise, iterate through the full fit
        # now we have the inital parameters, we want to cycle through the theta_par_values, producing EDC fits at each point
        EF_values = []  # used to store calculated EF values
        for theta_par in tqdm(theta_par_values,
                              desc='EDC fitting progress'):  # Iterate through all theta_par values, put this in a progress bar

            # set the previous EDC fit parameters as the inital fit parameters for the current EDC fit
            init_bkg = bkg.make_params(intercept=opt_params['bkgintercept'], slope=opt_params['bkgslope'])
            init_dos = dos.make_params(intercept=opt_params['dosintercept'], slope=opt_params['dosslope'])
            init_fermi = fermi.make_params(T=opt_params['FermiT'], EF=opt_params['FermiEF'])
            init_params = init_fermi + init_bkg + init_dos

            # define the current EDC to fit
            current_EDC = dispersion.sel(
                theta_par=slice(theta_par - (theta_par_sum_range / 2), theta_par + (theta_par_sum_range / 2))).mean(
                'theta_par').sel(eV=slice(opt_params['FermiEF'] + min_eV_range, opt_params['FermiEF'] + max_eV_range))
            # fit the current EDC
            result = model_tot.fit(current_EDC, x=current_EDC.coords['eV'].values, params=init_params)
            # get the optimised fitting parameters
            opt_params = result.best_values
            # append the current EDC fitted EF value
            EF_values.append(opt_params['FermiEF'])

        # create a model for the polynomial (cubic) correction fit
        correction_fit = PolynomialModel(degree=3)
        # define inital fit parameters
        init_correction_fit = correction_fit.make_params(c0=opt_params['FermiEF'], c1=0, c2=0, c3=0)
        # if the user requires the fit type to be quadratic, fix the x^3 coefficient to 0
        if user_defined_fit_type == True and kwargs['fit_type'] == 'quadratic':
            init_correction_fit['c3'].vary = False
        # produce a correction fit for the calculated EF values
        correction_fit_result = correction_fit.fit(EF_values, x=theta_par_values, params=init_correction_fit)
        # get the optimised fitting parameters
        correction_fit_params = correction_fit_result.best_values
        # define the correction fit details
        correction_fit_details = correction_fit.make_params(c0=correction_fit_params['c0'],
                                                            c1=correction_fit_params['c1'],
                                                            c2=correction_fit_params['c2'],
                                                            c3=correction_fit_params['c3'])
        # get the correction at the theta_par values of the input dispersion
        correction_fit_to_plot = correction_fit.eval(params=correction_fit_details,
                                                     x=dispersion.coords['theta_par'].values)

        # plot the fit to the test EDC
        fig, axes = plt.subplots(ncols=3, figsize=(15, 4))
        test_result.plot_fit(ax=axes[0])
        axes[0].set_title("Test EDC fit")
        axes[0].set_xlabel("eV")
        axes[0].set_ylabel("Intensity")

        # plot the correction fit results
        correction_fit_result.plot_fit(ax=axes[1])
        axes[1].set_title("Correction")
        axes[1].set_xlabel("theta_par [deg]")
        axes[1].set_ylabel("eV")

        # plot the correction on top of the dispersion
        axes[2].plot(dispersion.coords['theta_par'].values, correction_fit_to_plot, color='orange')
        dispersion.sel(eV=slice(min(correction_fit_to_plot) - 0.1, max(correction_fit_to_plot) + 0.1)).plot(
            add_colorbar=False, cmap="Greys", ax=axes[2]);
        axes[2].set_title("Raw data + correction")
        axes[2].set_xlabel("theta_par [deg]")
        axes[2].set_ylabel("eV")
        plt.tight_layout()
        plt.show()

        # return the array of calculated EF_values of the fitting paramaters (returns fitting parameters by default)
        if user_defined_output_type == True:
            # Put the calculated EF_values into an xarray format
            EF_values_array = np.array(EF_values)
            EF_xarray = xr.DataArray(
                EF_values_array, dims=['theta_par'],
                coords={'theta_par': theta_par_values})
            return EF_xarray
        else:
            # return the the fitting paramaters
            return correction_fit_params