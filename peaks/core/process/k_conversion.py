# Functions used to apply k-space conversions
# Phil King 11/01/2021
# Brendan Edwards 26/04/2021
# PK 25/5/21 Updated to extend angular mapping to type I and II and to deflector-based analysers

import numpy as np
import xarray as xr
import warnings
from peaks.utils.metadata import update_hist
from peaks.core.process.fermi_level_corrections import apply_fermi
from peaks.utils.E_shift import Eshift_from_correction
from peaks.utils.get_angles import get_angles, set_normals
from peaks.utils.misc import warning_simple, warning_standard
from peaks.utils.consts import const
from peaks.core.fileIO.fileIO_opts import BL_angles

###################################################################################################################################################
# Mapping functions: angle -> k-space (in plane), following conventions and nomenclature of Ishida and Shin, Rev. Sci. Instrum. 89 (2018) 043903. #
###################################################################################################################################################
# In-plane mapping functions for converting between angle and k-space: angle --> k
def fI(alpha, beta, delta, xi, Ek):  # Type I, no deflector
    '''Convert angles to k-space for a type-I analyser with no deflector, following the conventions and nomenclature of Ishida and Shin, Rev. Sci. Instrum. 89 (2018) 043903.

    Input:
        alpha: theta_par angle (analyser slit angle, deg)
        beta: polar angle (deg)
        delta: azi angle (deg)
        xi: tilt angle (deg)
        Ek: Kinetic energy (eV)

    Output:
        kx: k-vector along analyser slit (1/A)
        ky: k-vector perp to analyser slit (1/A) '''

    # k_vacuum from KE
    kvac = const.kvac_const * np.sqrt(Ek)

    # Convert angles to radians
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    delta = np.radians(delta)
    xi = np.radians(xi)

    # Mapping functions
    kx = kvac * ((((np.sin(delta)*np.sin(beta))+(np.cos(delta)*np.sin(xi)*np.cos(beta)))*np.cos(alpha)) - (np.cos(delta)*np.cos(xi)*np.sin(alpha)))
    ky = kvac * ((((-np.cos(delta) * np.sin(beta)) + (np.sin(delta) * np.sin(xi) * np.cos(beta))) * np.cos(alpha)) - (np.sin(delta) * np.cos(xi) * np.sin(alpha)))

    return kx,ky

def fII(alpha,beta,delta,xi,Ek): # Type II, no deflector
    '''Convert angles to k-space for a type-II analyser with no deflector, following the conventions and nomenclature of Ishida and Shin, Rev. Sci. Instrum. 89 (2018) 043903.

        Input:
            alpha: theta_par angle (analyser slit angle, deg)
            beta: tilt angle (deg)
            delta: azi angle (deg)
            xi: polar angle (deg)
            Ek: Kinetic energy (eV)

        Output:
            kx: k-vector perp to analyser slit (1/A)
            ky: k-vector along analyser slit (1/A) '''

    # k_vacuum from KE
    kvac = const.kvac_const * np.sqrt(Ek)

    # Convert angles to radians
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    delta = np.radians(delta)
    xi = np.radians(xi)

    # Mapping functions
    kx = kvac * ((((np.sin(delta) * np.sin(xi)) + (np.cos(delta) * np.sin(beta) * np.cos(xi))) * np.cos(alpha)) - (((np.sin(delta)*np.cos(xi) - (np.cos(delta)*np.sin(beta)*np.sin(xi))))*np.sin(alpha)))
    ky = kvac * ((((-np.cos(delta) * np.sin(xi)) + (np.sin(delta) * np.sin(beta) * np.cos(xi))) * np.cos(alpha)) + (((np.cos(delta)*np.cos(xi) + (np.sin(delta)*np.sin(beta)*np.sin(xi))))*np.sin(alpha)))

    return kx,ky

def fIp(alpha,beta,delta,xi,chi,Ek): # Type I with deflector
    '''Convert angles to k-space for a type-I analyser with deflector, following the conventions and nomenclature of Ishida and Shin, Rev. Sci. Instrum. 89 (2018) 043903.

    Input:
        alpha: theta_par angle (analyser slit angle) + defl_par (deflector angle along the slit) (deg)
        beta: defl_perp (deflector angle perpendicular to the slit, deg)
        delta: azi angle (deg)
        xi: tilt angle (deg)
        chi: polar angle (deg)
        Ek: Kinetic energy (eV)

    Output:
        kx: k-vector along analyser slit (1/A)
        ky: k-vector perp to analyser slit (1/A) '''

    # k_vacuum from KE
    kvac = const.kvac_const * np.sqrt(Ek)

    # Convert angles to radians
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    delta = np.radians(delta)
    xi = np.radians(xi)
    chi = np.radians(chi)

    kx = kvac * ((((-alpha*np.cos(delta)*np.cos(xi))+(beta*np.sin(delta)*np.cos(chi))-(beta*np.cos(delta)*np.sin(xi)*np.sin(chi)))*np.sinc(np.sqrt(alpha**2 + beta**2)/np.pi))+(((np.sin(delta)*np.sin(chi))+(np.cos(delta)*np.sin(xi)*np.cos(chi)))*np.cos(np.sqrt(alpha**2 + beta**2))))
    ky = kvac * ((((-alpha*np.sin(delta)*np.cos(xi))-(beta*np.cos(delta)*np.cos(chi))-(beta*np.sin(delta)*np.sin(xi)*np.sin(chi)))*np.sinc(np.sqrt(alpha**2 + beta**2)/np.pi))-(((np.cos(delta)*np.sin(chi))-(np.sin(delta)*np.sin(xi)*np.cos(chi)))*np.cos(np.sqrt(alpha**2 + beta**2))))

    return kx,ky

def fIIp(alpha,beta,delta,xi,chi,Ek): # Type II with deflector
    '''Convert angles to k-space for a type-II analyser with deflector, following the conventions and nomenclature of Ishida and Shin, Rev. Sci. Instrum. 89 (2018) 043903.

    Input:
        alpha: theta_par angle (analyser slit angle) + defl_par (deflector angle along the slit) (deg)
        beta: defl_perp (deflector angle perpendicular to the slit, deg)
        delta: azi angle (deg)
        xi: tilt angle (deg)
        chi: polar angle (deg)
        Ek: Kinetic energy (eV)

    Output:
        kx: k-vector perp to analyser slit (1/A)
        ky: k-vector along analyser slit (1/A) '''

    # k_vacuum from KE
    kvac = const.kvac_const * np.sqrt(Ek)

    # Convert angles to radians
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    delta = np.radians(delta)
    xi = np.radians(xi)
    chi = np.radians(chi)

    kx = kvac * ((((-beta*np.cos(delta)*np.cos(xi))-(alpha*np.sin(delta)*np.cos(chi))+(alpha*np.cos(delta)*np.sin(xi)*np.sin(chi)))*np.sinc(np.sqrt(alpha**2 + beta**2)/np.pi))+(((np.sin(delta)*np.sin(chi))+(np.cos(delta)*np.sin(xi)*np.cos(chi)))*np.cos(np.sqrt(alpha**2 + beta**2))))
    ky = kvac * ((((-beta*np.sin(delta)*np.cos(xi))+(alpha*np.cos(delta)*np.cos(chi))+(alpha*np.sin(delta)*np.sin(xi)*np.sin(chi)))*np.sinc(np.sqrt(alpha**2 + beta**2)/np.pi))-(((np.cos(delta)*np.sin(chi))-(np.sin(delta)*np.sin(xi)*np.cos(chi)))*np.cos(np.sqrt(alpha**2 + beta**2))))

    return kx,ky

# Inverse mapping functions for converting between angle and k-space: k --> angle
def fI_inv(kx,ky,delta,xi,Ek,beta0=0.): #Type I, no deflector
    '''Convert k-space to angles for a type-I analyser with no deflector, following the conventions and nomenclature of Ishida and Shin, Rev. Sci. Instrum. 89 (2018) 043903.

            Input:
                kx: k-vector along analyser slit (1/A)
                ky: k-vector perp to analyser slit (1/A)
                delta: azi angle (deg)
                xi: tilt angle (deg)
                Ek: Kinetic energy (eV)
                beta0 (optional): polar angle offset, default=0

            Output:
                alpha: theta_par angle (analyser slit angle, deg)
                beta: polar angle  (deg) '''

    # k_vacuum from KE
    kvac = const.kvac_const * np.sqrt(Ek)

    # Convert angles to radians
    delta = np.radians(delta)
    xi = np.radians(xi)
    beta0 = np.radians(beta0)

    # Mapping function
    alpha = np.arcsin(((np.sin(xi)*np.sqrt(kvac**2 - kx**2 - ky**2))-(np.cos(xi)*((kx*np.cos(delta))+(ky*np.sin(delta)))))/kvac)
    beta = beta0 + np.arctan(((kx*np.sin(delta))-(ky*np.cos(delta)))/((kx*np.sin(xi)*np.cos(delta))+(ky*np.sin(xi)*np.sin(delta))+(np.cos(xi)*(np.sqrt(kvac**2 - kx**2 - ky**2)))))

    # Convert relevant angles to degrees
    alpha = np.degrees(alpha)
    beta = np.degrees(beta)

    return alpha,beta

def fII_inv(kx,ky,delta,xi,Ek,beta0=0.): # Type I, no deflector
    '''Convert k-space to angles for a type-II analyser with no deflector, following the conventions and nomenclature of Ishida and Shin, Rev. Sci. Instrum. 89 (2018) 043903.

            Input:
                kx: k-vector perp to analyser slit (1/A)
                ky: k-vector along analyser slit (1/A)
                delta: azi angle (deg)
                xi: polar angle (deg)
                Ek: Kinetic energy (eV)
                beta0 (optional): tilt angle offset, default=0

            Output:
                alpha: theta_par angle (analyser slit angle, deg)
                beta: tilt angle  (deg) '''

    # k_vacuum from KE
    kvac = const.kvac_const * np.sqrt(Ek)

    # Convert angles to radians
    delta = np.radians(delta)
    xi = np.radians(xi)
    beta0 = np.radians(beta0)

    # Mapping function
    alpha = np.arcsin(((np.sin(xi)*np.sqrt(kvac**2 - (((kx*np.sin(delta))-(ky*np.cos(delta)))**2)))-(np.cos(xi)*((kx*np.sin(delta))-(ky*np.cos(delta)))))/kvac)
    beta = beta0 + np.arctan(((kx*np.cos(delta))+(ky*np.sin(delta)))/(np.sqrt(kvac**2 - kx**2 - ky**2)))

    # Convert relevant angles to degrees
    alpha = np.degrees(alpha)
    beta = np.degrees(beta)

    return alpha,beta

def tij(ij,delta,xi,chi):
    '''Defines the inverse of the rotation matrix T_rot to obtain the elements of the inverse functions of type I' and II' manipulators, Eqn A9 of Ishida and Shin, Rev. Sci. Instrum. 89 (2018) 043903.
    Input:
        ij - index to return
        delta: azi angle (deg)
        xi: tilt angle (deg)
        chi: polar angle (deg)

    Output:
        tij - relevant element of T_rot^-1 '''

    tij = {}
    tij[11] = np.cos(xi)*np.cos(delta)
    tij[12] = np.cos(xi)*np.sin(delta)
    tij[13] = -np.sin(xi)
    tij[21] = (np.sin(chi)*np.sin(xi)*np.cos(delta)) - (np.cos(chi)*np.sin(delta))
    tij[22] = (np.sin(chi)*np.sin(xi)*np.sin(delta)) + (np.cos(chi)*np.cos(delta))
    tij[23] = np.sin(chi) * np.cos(xi)
    tij[31] = (np.cos(chi) * np.sin(xi) * np.cos(delta)) + (np.sin(chi) * np.sin(delta))
    tij[32] = (np.cos(chi) * np.sin(xi) * np.sin(delta)) - (np.sin(chi) * np.cos(delta))
    tij[33] = np.cos(chi) * np.cos(xi)

    return tij[ij]

def fIp_inv(kx,ky,delta,xi,chi,Ek): # Type I, with deflector
    '''Convert k-space to angles for a type-I analyser with deflector, following the conventions and nomenclature of Ishida and Shin, Rev. Sci. Instrum. 89 (2018) 043903.

            Input:
                kx: k-vector along analyser slit (1/A)
                ky: k-vector perp to analyser slit (1/A)
                delta: azi angle (deg), relative to 'normal' emission
                xi: tilt angle (deg), relative to normal emission
                chi: polar angle (deg), relative to normal emission
                Ek: Kinetic energy (eV)

            Output:
                alpha: theta_par angle (analyser slit angle, deg)
                beta: polar angle  (deg) '''

    # k_vacuum from KE
    kvac = const.kvac_const * np.sqrt(Ek)
    k2p = np.sqrt(kvac**2 - kx**2 - ky**2)

    # Convert angles to radians
    delta = np.radians(delta)
    xi = np.radians(xi)
    chi = np.radians(chi)

    # Mapping functions
    arg1 = (tij(31,delta,xi,chi)*kx)+(tij(32,delta,xi,chi)*ky)+(tij(33,delta,xi,chi)*k2p)
    arg2 = (tij(11,delta,xi,chi)*kx)+(tij(12,delta,xi,chi)*ky)+(tij(13,delta,xi,chi)*k2p)
    arg3 = (tij(21,delta,xi,chi)*kx)+(tij(22,delta,xi,chi)*ky)+(tij(23,delta,xi,chi)*k2p)

    alpha = -np.arccos(arg1/kvac) * arg2 / np.sqrt(kvac**2 - arg1**2)
    beta = -np.arccos(arg1/kvac) * arg3 / np.sqrt(kvac**2 - arg1**2)

    # Convert relevant angles to degrees
    alpha = np.degrees(alpha)
    beta = np.degrees(beta)

    return alpha,beta

def fIIp_inv(kx,ky,delta,xi,chi,Ek): # Type II, with deflector
    '''Convert k-space to angles for a type-II analyser with deflector, following the conventions and nomenclature of Ishida and Shin, Rev. Sci. Instrum. 89 (2018) 043903.

            Input:
                kx: k-vector perp to analyser slit (1/A)
                ky: k-vector along analyser slit (1/A)
                delta: azi angle (deg), relative to 'normal' emission
                xi: tilt angle (deg), relative to normal emission
                chi: polar angle (deg), relative to normal emission
                Ek: Kinetic energy (eV)

            Output:
                alpha: theta_par angle (analyser slit angle, deg)
                beta: polar angle  (deg) '''

    # k_vacuum from KE
    kvac = const.kvac_const * np.sqrt(Ek)
    k2p = np.sqrt(kvac**2 - kx**2 - ky**2)

    # Convert angles to radians
    delta = np.radians(delta)
    xi = np.radians(xi)
    chi = np.radians(chi)

    # Mapping functions
    arg1 = (tij(31,delta,xi,chi)*kx)+(tij(32,delta,xi,chi)*ky)+(tij(33,delta,xi,chi)*k2p)
    arg2 = (tij(11,delta,xi,chi)*kx)+(tij(12,delta,xi,chi)*ky)+(tij(13,delta,xi,chi)*k2p)
    arg3 = (tij(21,delta,xi,chi)*kx)+(tij(22,delta,xi,chi)*ky)+(tij(23,delta,xi,chi)*k2p)

    alpha = np.arccos(arg1/kvac) * arg3 / np.sqrt(kvac**2 - arg1**2)
    beta = -np.arccos(arg1/kvac) * arg2 / np.sqrt(kvac**2 - arg1**2)

    # Convert relevant angles to degrees
    alpha = np.degrees(alpha)
    beta = np.degrees(beta)

    return alpha,beta


######################################
#### k-space conversion functions ####
######################################

def k_conv(data_in, **kwargs):
    '''This function applies a k-space conversion by interpolating an xarray from angle space to k-space along a
    single cut along the slit direction, with some angular offsets.
    
    Input:
            data_in - the dispersion(s) to be converted (xarray)

            **kwargs - optional arguments:
            Data attributes:
                Setting manipulator angles:
                    - polar = ## (float) to set the manipulator polar angle (also overwrites that attribute
                    in the original dataarray)
                    - tilt = ## (float) to set the manipulator tilt angle (also overwrites that attribute
                    in the original dataarray)
                    - azi = ## (float) to set the manipulator azi angle (also overwrites that  attribute
                    in the original dataarray)
                Setting normal emissions:
                    - norm_polar = ## (float) to set the normal emission in the polar axis (also overwrites that
                    attribute in the original dataarray)
                    - norm_tilt = ## (float) to set the normal emission in the tilt axis (also overwrites that
                    attribute in the original dataarray)
                    - norm_azi = ## (float) to set the offset in the azimuthal axis (also overwrites that attribute
                    in the original dataarray)

                Setting EF_correction method:
                    - EF_correction = float, fit_coeffs, xarray (see fermi_level_corrections.py): EF_correction to
                    use (also overwrites that attribute in the original dataarray)
                    - KE2BE (boolean): If False, keeps everything in KE even if it is possible to convert to BE

                k-point spacings:
                    - dk = ## (float); to manually select k-point spacing (in-plane). Defaults to keeping same
                    number of points as in theta_par dimension.

                Data binning:
                    - binning = {'eV': 2, 'theta_par: 2} to apply a 2x2 binning on the energy and theta_par axis.
                    Can change factor, and bin on one or both axes
                    - bin_factor = #: shortcut to do # x # binning on the energy and theta_par axis, call this
                    instead of binning = {}

        Returns:
            data_out - the data that has been converted to k-space and/or binding energy (xarray)'''

    # Set up warnings display and formatting
    warnings.simplefilter('always', UserWarning)  # Give warnings every time, even on function re-run
    warnings.formatwarning = warning_simple  # Formatting of warnings for peaks user errors

    # Perform binning if required
    if 'binning' in kwargs:
        if 'bin_factor' in kwargs:  # Two types of binning requested - raise error
            err_str = 'Function called with two binning instructions. Use either bin_factor=# to apply #x# binning to the energy and theta_par dims, or call explicitly with binning=dict, where dict is a dictionary of dimension names and binning factors to bin over.'
            raise Exception(err_str)
        else:  # Apply the requested binning
            data_in = data_in.coarsen(kwargs['binning'], boundary='pad').mean(keep_attrs=True).copy(deep=True)
        # Update the history
        hist_str = 'Data binning applied: ' + str(kwargs['binning'])
        update_hist(data_in, hist_str)
    elif 'bin_factor' in kwargs:  # Apply bin_factor to energy and theta_par axes
        if kwargs['bin_factor'] == 1:  # Don't do any binning if called with bin_factor = 1
            pass
        else:  # Write binning into correct form and apply
            kwargs['binning'] = {'eV': kwargs['bin_factor'], 'theta_par': kwargs['bin_factor']}
            data_in = data_in.coarsen(kwargs['binning'], boundary='pad').mean(keep_attrs=True).copy(deep=True)
        # Update the history
        hist_str = 'Data binning applied: ' + str(kwargs['binning'])
        update_hist(data_in, hist_str)

    #################################
    #####     ENERGY SCALES     #####
    #################################
    # determine angular-dependent Fermi energy correction based on the EF_correction attribute
    if 'EF_correction' in kwargs:  # If passed with call, write that to original dataarray (also updates original array outside of function)
        data_in.attrs['EF_correction'] = kwargs['EF_correction']

    # Extract relevant energy shifts, correction type and raw correction parameters from dataarray
    E_shift, EF_correction_type = Eshift_from_correction(data_in)
    EF_correction = data_in.attrs['EF_correction']

    # Get beamline
    BL = data_in.attrs['beamline']

    # Workfunction
    if EF_correction_type == 'fit' or EF_correction_type == 'array':
        wf = data_in.attrs['hv'] - max(E_shift)
    elif EF_correction_type == 'value':
        wf = data_in.attrs['hv'] - E_shift

    # Convert to binding energy if possible. If not possible, or if asked not to, keep in KE
    if data_in.attrs['eV_type'] == 'kinetic':  # Data in KE
        if EF_correction_type == 'None' or kwargs['KE2BE']==False:  # No attributes to allow conversion to BE, or asked to keep in KE - in this case, we will stick with KE from the raw data
            data = data_in.copy(deep=True)
            KE_values = data.eV.data  # Get raw KE array

        else:  # An EF correction has been applied, so we can use this to convert the data to BE
            data = data_in.pipe(apply_fermi)  # Convert data to BE
            KE_values = data.attrs['hv'] - wf + data.eV.data  # Get KE array, shifting from BE to KE for use in k-conv

    # If array already in BE
    else:
        data = data_in.copy(deep=True)
        KE_values = data.attrs['hv'] - wf + data.eV.data


    ###################################
    #####   k-space conversions   #####
    ###################################
    # This follows the conventions and nomenclature of Ishida and Shin, Rev. Sci. Instrum. 89 (2018) 043903, although rather than using kx and ky in the final data output (which can be ambiguous) we use k_par and k_perp, for k parallel and perpendicular to analyser slit.
    # Set relevant normal emission angles
    set_normals(data_in, **kwargs)
    set_normals(data, **kwargs)

    # Get relevant angles from data_in
    angles = get_angles(data_in)


    # Extract relevant angles from angles
    alpha = angles['alpha']
    beta = angles['beta']
    delta = angles['delta']
    xi = angles['xi']
    if 'chi' in angles:
        chi = angles['chi']
    ana_type = angles['ana_type']

    # Work out relevant k-range, arrays as [angle, energy]
    if ana_type == 'I':  # Type I, no deflector
        kx, ky = fI(alpha[:, np.newaxis], beta, delta, xi, KE_values[np.newaxis, :])
        # Get k-range to interpolate over
        k0 = kx.min()
        k1 = kx.max()
        k_perp = ky  # Store this to keep track of values
    elif ana_type == 'II':  # Type II, no deflector
        kx, ky = fII(alpha[:, np.newaxis], beta, delta, xi, KE_values[np.newaxis, :])
        # Get k-range to interpolate over
        k0 = ky.min()
        k1 = ky.max()
        k_perp = kx  # Store this to keep track of values, NB negative sign
    elif ana_type == 'Ip':  # Type I, with deflector
        kx, ky = fIp(alpha[:, np.newaxis], beta, delta, xi, chi, KE_values[np.newaxis, :])
        # Get k-range to interpolate over
        k0 = kx.min()
        k1 = kx.max()
        k_perp = ky  # Store this to keep track of values
    elif ana_type == 'IIp':  # Type II, with deflector
        kx, ky = fIp(alpha[:, np.newaxis], beta, delta, xi, chi, KE_values[np.newaxis, :])
        # Get k-range to interpolate over
        k0 = kx.min()
        k1 = kx.max()
        k_perp = kx  # Store this to keep track of values, NB negative sign

    # Set k_values to interpolate over along the slit direction
    if 'dk' not in kwargs:  # Use default k-point spacing
        k_values = np.linspace(k0, k1, data.theta_par.size)
    else:
        k_values = np.arange(k0, k1, kwargs['dk'])

    # eV coords unchanged
    eV_xarray = xr.DataArray(data.eV.data, dims=['eV'], coords={'eV': data.eV.data})  # eV array not changed during the k-par interpolation

    # Get the relevant angle from k-space mappings and interpolate over these
    if ana_type == 'I':  # Type I, no deflector
        alpha_out, beta_out = fI_inv(k_values[:, np.newaxis], 0, delta, xi, KE_values[np.newaxis, :], angles['norm_polar'])
    elif ana_type == 'II':  # Type II, no deflector
        alpha_out, beta_out = fII_inv(0, k_values[:, np.newaxis], delta, xi, KE_values[np.newaxis, :], angles['norm_tilt'])
    elif ana_type == 'Ip':  # Type I, with deflector
        alpha_out, beta_out = fIp_inv(k_values[:, np.newaxis], 0, delta, xi, chi, KE_values[np.newaxis, :])
    elif ana_type == 'IIp':  # Type II, with deflector
        alpha_out, beta_out = fIIp_inv(0, k_values[:, np.newaxis], delta, xi, chi, KE_values[np.newaxis, :])

    #Make mapping from k-space to angle along slit.
    theta_from_k_xarray = xr.DataArray(
        alpha_out * BL_angles.angles[BL]['theta_par'],
        dims=['k_par', 'eV'],
        coords={'k_par': k_values, 'eV': data.eV.data}
    )

    # interpolate to k-space
    data_out = data.interp(eV=eV_xarray, theta_par=theta_from_k_xarray)
    data_out.coords['k_par'].attrs = {'units' : 'inv. ang.'}

    #Reindex array to have increasing k in the array
    if data_out.k_par.data[1] - data_out.k_par.data[0] < 0:  # Array currently has decreasing k-order
        data_out = data_out.reindex(k_par=data_out.k_par[::-1])

    # Check the k_perp value
    k_perp_diff = k_perp.max() - k_perp.min()
    if abs(k_perp_diff) > 0.01:  # Threshold to avoid numerical noise
        data_out.attrs['k_perp'] = k_perp.mean()  # If this is significant, write mean value to the attributes

    # update the xarray analysis history
    update_hist(data_out, 'Converted to k-space')

    # Reset warnings display and formatting
    warnings.simplefilter('once', UserWarning)  # Give warnings first time only
    warnings.formatwarning = warning_standard  # Formatting of warnings for peaks user errors

    return data_out

#Convert a FS map (cube) into BE and k-space
def kFS_conv(data_in, **kwargs):
    '''This function applies a k-space conversion by interpolating an xarray from two anglular directions into
    in-plane kx and ky. This works for manipulator or deflector maps, for vertical and horizontal slit geometries.

Input:
        data_in - the data cube to be converted to k-space, in [mapping angle, theta_par, eV] format (xarray).

        **kwargs - optional arguments:
        Data attributes:
            Setting manipulator angles:
                - polar = ## (float) to set the manipulator polar angle (also overwrites that attribute
                in the original dataarray)
                - tilt = ## (float) to set the manipulator tilt angle (also overwrites that attribute
                in the original dataarray)
                - azi = ## (float) to set the manipulator azi angle (also overwrites that  attribute
                in the original dataarray)
            Setting normal emissions:
                - norm_polar = ## (float) to set the normal emission in the polar axis (also overwrites that
                attribute in the original dataarray)
                - norm_tilt = ## (float) to set the normal emission in the tilt axis (also overwrites that
                attribute in the original dataarray)
                - norm_azi = ## (float) to set the offset in the azimuthal axis (also overwrites that attribute
                in the original dataarray)

            Setting EF_correction method:
                - EF_correction = float, fit_coeffs, xarray (see fermi_level_corrections.py): EF_correction to
                use (also overwrites that attribute in the original dataarray)
                - KE2BE (boolean): If False, keeps everything in KE even if it is possible to convert to BE

            k-point spacings:
                - dk = ## (float); to manually select k-point spacing (in-plane). Defaults to 0.01 inv. Ang.

            Binning/selection (these can provide substantial speed-up if converting a larger amount of data (e.g.
                full cube):
                    - binning = {'eV': 2, 'theta_par: 2} to apply a 2x2 binning on the energy and theta_par axis,
                    to save a lot of time if building the full cube. Can change factor, and bin on one or both axes
                    - bin_factor = #: shortcut to do #x# binning on the energy and theta_par axis, call this
                    instead of binning = {}
                    - eV=E0 or eV=slice(E0,E1) to return a constant energy slice (single value or range), (E0,E1
                     floats, ignored for single dispersion)
                    - FS = dE or FS=(dE, E0) as a shortcut to select a slice of width dE at energy E0 (default at
                     E0=0), and performs the mean over eV before returning (E0,dE floats, ignored for single
                     dispersion)

        Returns:
            data_out - the data that has been converted to k-space and/or binding energy (xarray)'''

    # Set up warnings display and formatting
    warnings.simplefilter('always', UserWarning)  # Give warnings every time, even on function re-run
    warnings.formatwarning = warning_simple  # Formatting of warnings for peaks user errors

    #Check data in correct format
    if len(data_in.dims) != 3:
        err_string = 'Data in incorrect format. kFS_conv expects a 3D cube of data with dimensions [mapping_angle, theta_par, eV]'
        raise Exception(err_string)
    #Check mapping angle
    for i in data_in.dims:
        if i == 'polar' or i == 'tilt' or i == 'defl_perp' or i == 'ana_polar': #Currently supported mapping dimensions
            mapping_dim = i
            if i == 'ana_polar':
                if data_in.attrs['ana_slit_angle'] != 90:
                    err_string = 'Mapping dimension with moveable polar axis of analyser only supported for \'Type I\' configurations, with polar axis along the slit.'
                    raise Exception(err_string)
        elif i != 'eV' and i != 'theta_par':
            err_string = 'Mapping angle of '+str(i)+' not currently supported; kFS_conv expects a 3D cube of data with dimensions [mapping_angle, theta_par, eV]'
            raise Exception(err_string)

    # Get beamline
    BL = data_in.attrs['beamline']

    # Check call for energy slice
    if 'FS' in kwargs:
        if 'eV' in kwargs:
            err_str = 'Function called with two energy selections. Use either eV=E0, eV=slice(E0,E1), FS=dE, or FS=(dE,E0).'
            raise Exception(err_str)

        if type(kwargs['FS']) == float or type(kwargs['FS']) == int: #At Fermi energy with some width
            if kwargs['FS'] == 0 or kwargs['FS'] == 0.: #Single slice requied
                kwargs['eV'] = 0  # Write that into appropriate eV call
            else:
                kwargs['eV'] = slice(-float(kwargs['FS'])/2,float(kwargs['FS'])/2)
        elif type(kwargs['FS']) == tuple: #Called with range and centre energy
            kwargs['eV'] = slice(kwargs['FS'][1]-kwargs['FS'][0]/2,kwargs['FS'][1]+kwargs['FS'][0]/2) #Write that into appropriate eV call
        else:
            err_str = 'FS should be a float to select a slice at the Fermi level (eV=0) with width dE, or a tuple (dE,E0) to select a slice of width dE centered at E0.'
            raise Exception(err_str)


    if 'KE2BE' not in kwargs:  # If not told not to, convert KE to BE as part of conversion
        kwargs['KE2BE'] = True


    #################################
    #####     ENERGY SCALES     #####
    #################################
    # determine angular-dependent Fermi energy correction based on the EF_correction attribute
    if 'EF_correction' in kwargs: #If passed with call, write that to original dataarray (also updates original array outside of function)
        data_in.attrs['EF_correction'] = kwargs['EF_correction']

    #Extract relevant energy shifts, correction type and raw correction parameters from dataarray
    E_shift, EF_correction_type = Eshift_from_correction(data_in)
    correction = data_in.attrs['EF_correction']

    #Workfunction
    if EF_correction_type == 'fit' or EF_correction_type == 'array':
        wf = data_in.attrs['hv'] - max(E_shift)
    elif EF_correction_type == 'value':
        wf = data_in.attrs['hv'] - E_shift

   #Determine binning (but don't run this yet, this should come later for speed)
    bin_flag = 0
    if 'binning' in kwargs:
        bin_flag = 1 #Flag to indicate if binning to be performed
        if 'bin_factor' in kwargs:
            err_str = 'Function called with two binning instructions. Use either bin_factor=# to apply #x# binning to the energy and theta_par dims, or call explicitly with binning=dict, where dict is a dictionary of dimension names and binning factors to bin over.'
            raise Exception(err_str)

    elif 'bin_factor' in kwargs: #Apply bin_factor to energy and theta_par axes
        if kwargs['bin_factor'] == 1: #Don't do any binning if called with bin_factor = 1
            bin_flag = 2
        else: #Write binning into correct form
            bin_flag = 1
            kwargs['binning'] = {'eV': kwargs['bin_factor'], 'theta_par': kwargs['bin_factor']}

    if 'binning' in kwargs and 'eV' in kwargs: #Don't allow binning if the data will end up with no energy slices
        if type(kwargs['eV']) != slice:
            if EF_correction_type != 'fit' or EF_correction_type != 'array':
                bin_flag=2
                warning_str = 'Binning not supported for data with this small a slice to be passed for k-conversion. Binning ignored here.'
                warnings.warn(warning_str)

    #Convert to binding energy if possible, restricting range to minimum required by any kwargs specified. If not possible, or if asked not to, keep in KE
    if data_in.attrs['eV_type'] == 'kinetic': #Data in KE
        if EF_correction_type == 'None' or kwargs['KE2BE']==False:  # No attributes to allow conversion to BE, or asked to keep in KE - in this case, we will stick with KE from the raw data
            if 'eV' in kwargs:  # Reduced eV range has been specified
                if type(kwargs['eV']) is slice: #reduced eV range requested
                    #Work out energy range required
                    eV_min = kwargs['eV'].start
                    eV_max = kwargs['eV'].stop
                    #Subselect the data to required range, making this a copy
                    if (eV_max - eV_min) > (data_in.eV.data[1] - data_in.eV.data[0]):  # slice is more than one pixel wide
                        data = data_in.sel(eV=slice(eV_min, eV_max)).copy(deep=True)
                    else: #If range < pixel size
                        data = data_in.sel(eV=eV_min, method='nearest').copy(deep=True)
                elif type(kwargs['eV']) is int or float: #If only a single pixel requested
                    data = data_in.sel(eV=kwargs['eV'], method='nearest').copy(deep=True)

            #If no energy slice specified, keep the full array, and make a copy
            else:
                data = data_in.copy(deep=True)

            #Perform binning if required
            if bin_flag == 1:
                data = data.coarsen(kwargs['binning'], boundary='pad').mean(keep_attrs=True)
            elif data.size > 10000000: #Automatically 2x2 bin if array is too large, unless explicitly told not to
                if bin_flag == 2:  # Told no binning
                    pass
                else: #Run auto binning
                    bin_flag = 1
                    kwargs['binning'] = {'eV': 2, 'theta_par': 2}
                    data = data.coarsen(kwargs['binning'], boundary='pad').mean(keep_attrs=True)
                    warning_str = 'Data array is large, 2x2 binning on energy and theta_par axis has been automatically applied. To run without binning, call function with bin_factor=1.'
                    warnings.warn(warning_str)

            #Get KE array
            KE_values = data.eV.data

        else: #An EF correction has been applied, so we can use this to convert the data to BE
            if 'eV' in kwargs:  # Reduced eV range has been specified
                if type(kwargs['eV']) is slice:  # reduced eV range requested
                    # Work out energy range required
                    eV_min0 = kwargs['eV'].start
                    eV_max0 = kwargs['eV'].stop
                elif type(kwargs['eV']) is int or float:  # If only a single pixel requested
                    eV_min0 = kwargs['eV']
                    eV_max0 = kwargs['eV']

                if EF_correction_type == 'value': #Just need to offset the requested energies by a constant
                    eV_min = eV_min0 + E_shift
                    eV_max = eV_max0 + E_shift
                else: #There is curvature of the Fermi energy, so we will need to keep a slightly extended range to account for that
                    eV_min = eV_min0 + min(E_shift)
                    eV_max = eV_max0 + max(E_shift)

                # Subselect the data to required range, and then run a BE conversion on this
                if (eV_max - eV_min) > (data_in.eV.data[1] - data_in.eV.data[0]):  # slice is more than one pixel wide
                    if (eV_max0 - eV_min0) > (data_in.eV.data[1] - data_in.eV.data[0]):  # output slice is more than one pixel wide
                        # Include binning if required
                        if bin_flag == 1:
                            data = data_in.sel(eV=slice(eV_min, eV_max)).coarsen(kwargs['binning'],  boundary='pad').mean(keep_attrs=True).pipe(apply_fermi).sel(eV=slice(eV_min0, eV_max0))
                        else:
                            data = data_in.sel(eV=slice(eV_min, eV_max))
                            if data.size > 10000000:  # Automatically 2x2 bin if array is too large, unless explicitly told not to
                                if bin_flag == 2:  # Told no binning
                                    data = data.pipe(apply_fermi).sel(eV=slice(eV_min0, eV_max0))
                                else: #Run auto-binning
                                    bin_flag = 1
                                    kwargs['binning'] = {'eV': 2, 'theta_par': 2}
                                    data = data.coarsen(kwargs['binning'], boundary='pad').mean(keep_attrs=True).pipe(apply_fermi).sel(eV=slice(eV_min0, eV_max0))
                                    warning_str = 'Data array is large, 2x2 binning on energy and theta_par axis has been automatically applied. To run without binning, call function with bin_factor=1.'
                                    warnings.warn(warning_str)
                            else: #Array is not too large, so just do BE conversion without further binning
                                data = data.pipe(apply_fermi).sel(eV=slice(eV_min0, eV_max0))
                    else:
                        if bin_flag == 1: #binning required
                            data = data_in.sel(eV=slice(eV_min, eV_max)).coarsen(kwargs['binning'],  boundary='pad').mean(keep_attrs=True).pipe(apply_fermi).sel(eV=eV_min0, method='nearest')
                        else: #Single slice, so no automatic binning should be required
                            data = data_in.sel(eV=slice(eV_min, eV_max)).pipe(apply_fermi).sel(eV=eV_min0, method='nearest')
                else:  # If range < pixel size
                    if bin_flag == 1:  # binning required
                        data = data_in.sel(eV=eV_min, method='nearest').coarsen(kwargs['binning'],  boundary='pad').mean(keep_attrs=True).pipe(apply_fermi)
                    else:  # Single slice, so no automatic binning should be required
                        data = data_in.sel(eV=eV_min, method='nearest').pipe(apply_fermi)

                # If returning a single pixel slice, want to remove the non-varying coorinate as a dimension
                try:
                    data = data.squeeze('eV')
                except:
                    pass

            # If no energy slice specified, keep the full array, and convert to BE, with binning as required
            else:
                if bin_flag == 1:  # binning required
                    data = data_in.coarsen(kwargs['binning'],  boundary='pad').mean(keep_attrs=True).pipe(apply_fermi)
                else:
                    if data_in.size > 10000000:  # Automatically 2x2 bin if array is too large, unless explicitly told not to
                        if bin_flag == 2: #Told no binning
                            data = data_in.pipe(apply_fermi)
                        else: #Run auto binning
                            bin_flag = 1
                            kwargs['binning'] = {'eV': 2, 'theta_par': 2}
                            data = data_in.coarsen(kwargs['binning'], boundary='pad').mean().pipe(apply_fermi)
                            warning_str = 'Data array is large, 2x2 binning on energy and theta_par axis has been automatically applied. To run without binning, call function with bin_factor=1.'
                            warnings.warn(warning_str)
                    else:
                        data = data_in.pipe(apply_fermi)


            # Get KE array, shifting from BE to KE for use in k-conv
            KE_values = data.attrs['hv'] - wf + data.eV.data


    #If array already in BE
    else:
        if 'eV' in kwargs:  # Reduced eV range has been specified
            if type(kwargs['eV']) is slice:  # reduced eV range requested
                # Work out energy range required
                eV_min = kwargs['eV'].start
                eV_max = kwargs['eV'].stop
                # Subselect the data to required range, making this a copy
                if (eV_max - eV_min) > (data_in.eV.data[1] - data_in.eV.data[0]):  # slice is more than one pixel wide
                    data = data_in.sel(eV=slice(eV_min, eV_max)).copy(deep=True)
                else:  # If range < pixel size
                    data = data_in.sel(eV=eV_min, method='nearest').copy(deep=True)
            elif type(kwargs['eV']) is int or float:  # If only a single pixel requested
                data = data_in.sel(eV=kwargs['eV'], method='nearest').copy(deep=True)

        # If no energy slice specified, keep the full array, and make a copy
        else:
            data = data_in.copy(deep=True)

        # Perform binning if required
        if bin_flag == 1:
            data = data.coarsen(kwargs['binning'],  boundary='pad').mean(keep_attrs=True)
        elif data.size > 100000000:  # Automatically 2x2 bin if array is too large, unless explicitly told not to
            if bin_flag == 2:  # Told no binning
                pass
            else:  # Run auto binning
                bin_flag = 1
                kwargs['binning'] = {'eV': 2, 'theta_par': 2}
                data = data.coarsen(kwargs['binning'], boundary='pad').mean(keep_attrs=True)
                warning_str = 'Data array is large, 2x2 binning on energy and theta_par axis has been automatically applied. To run without binning, call function with bin_factor=1.'
                warnings.warn(warning_str)

        # Get KE array, shifting from BE to KE for use in k-conv
        KE_values = data.attrs['hv'] - wf + data.eV.data


    ###################################
    #####   k-space conversions   #####
    ###################################
    #This follows the conventions and nomenclature of Ishida and Shin, Rev. Sci. Instrum. 89 (2018) 043903, although rather than using kx and ky in the final data output (which can be ambiguous) we use k_par and k_perp, for k parallel and perpendicular to analyser slit.

    # Set relevant normal emission angles
    set_normals(data_in, **kwargs)
    set_normals(data, **kwargs)

    # Get relevant angles from data_in
    angles = get_angles(data_in)

    # Extract relevant angles from angles
    alpha = angles['alpha']
    beta = angles['beta']
    delta = angles['delta']
    xi = angles['xi']
    if 'chi' in angles:
        chi = angles['chi']
    ana_type = angles['ana_type']
    if ana_type == 'I':
        if mapping_dim == 'ana_polar':  # Special case for moving analyser - need to add the manipulator angle to the offset
            norm_offset = angles['norm_polar'] - angles['polar']
        else:
            norm_offset = angles['norm_polar']
    elif ana_type == 'II':
        norm_offset = angles['norm_tilt']


    # Work out relevant k-range
    # Make arrays to store k-points at extemes of angular range
    kx_lim = np.zeros(4)
    ky_lim = np.zeros(4)

    # Work out relevant k-range, arrays as [angle, energy]
    if ana_type == 'I':  # Type I, no deflector
        kx_lim[0], ky_lim[0] = fI(alpha.min(), beta.min(), delta, xi, KE_values.max())
        kx_lim[1], ky_lim[1] = fI(alpha.min(), beta.max(), delta, xi, KE_values.max())
        kx_lim[2], ky_lim[2] = fI(alpha.max(), beta.min(), delta, xi, KE_values.max())
        kx_lim[3], ky_lim[3] = fI(alpha.max(), beta.max(), delta, xi, KE_values.max())
    elif ana_type == 'II':  # Type II, no deflector
        kx_lim[0], ky_lim[0] = fII(alpha.min(), beta.min(), delta, xi, KE_values.max())
        kx_lim[1], ky_lim[1] = fII(alpha.min(), beta.max(), delta, xi, KE_values.max())
        kx_lim[2], ky_lim[2] = fII(alpha.max(), beta.min(), delta, xi, KE_values.max())
        kx_lim[3], ky_lim[3] = fII(alpha.max(), beta.max(), delta, xi, KE_values.max())
    elif ana_type == 'Ip':  # Type I, with deflector
        kx_lim[0], ky_lim[0] = fIp(alpha.min(), beta.min(), delta, xi, chi, KE_values.max())
        kx_lim[1], ky_lim[1] = fIp(alpha.min(), beta.max(), delta, xi, chi, KE_values.max())
        kx_lim[2], ky_lim[2] = fIp(alpha.max(), beta.min(), delta, xi, chi, KE_values.max())
        kx_lim[3], ky_lim[3] = fIp(alpha.max(), beta.max(), delta, xi, chi, KE_values.max())
    elif ana_type == 'IIp':  # Type II, with deflector
        kx_lim[0], ky_lim[0] = fIIp(alpha.min(), beta.min(), delta, xi, chi, KE_values.max())
        kx_lim[1], ky_lim[1] = fIIp(alpha.min(), beta.max(), delta, xi, chi, KE_values.max())
        kx_lim[2], ky_lim[2] = fIIp(alpha.max(), beta.min(), delta, xi, chi, KE_values.max())
        kx_lim[3], ky_lim[3] = fIIp(alpha.max(), beta.max(), delta, xi, chi, KE_values.max())

    if 'dk' in kwargs:  # If function called with explicit k-point spacing
        dk = kwargs['dk']
    else:  # Set a default value
        dk = 0.01

    #Make k-arrays
    kx_values = np.arange(kx_lim.min(),kx_lim.max()+dk, dk)
    ky_values = np.arange(ky_lim.min(), ky_lim.max()+dk, dk)

    #Get the relevant angle from k-space mappings and interpolate over these
    if KE_values.size == 1:  # Only single value in KE selection
        # Get angle scales - array order [kx,ky]
        if ana_type == 'I':  # Type I, no deflector
            alpha_out, beta_out = fI_inv(kx_values[:, np.newaxis], ky_values[np.newaxis,:], delta, xi, KE_values, norm_offset)
        elif ana_type == 'II':  # Type II, no deflector
            alpha_out, beta_out = fII_inv(kx_values[:, np.newaxis], ky_values[np.newaxis,:], delta, xi, KE_values, norm_offset)
        elif ana_type == 'Ip':  # Type I, with deflector
            alpha_out, beta_out = fIp_inv(kx_values[:, np.newaxis], ky_values[np.newaxis,:], delta, xi, chi, KE_values)
        elif ana_type == 'IIp':  # Type II, with deflector
            alpha_out, beta_out = fIIp_inv(kx_values[:, np.newaxis], ky_values[np.newaxis,:], delta, xi, chi, KE_values)
        # Make the relevant xarrays to use for interpolation (NB use k_par and k_perp for output data rather than kx and ky)
        if ana_type == 'I' or ana_type == 'Ip':
            theta_par_from_k_xarray = xr.DataArray(alpha_out  * BL_angles.angles[BL]['theta_par'], dims=['k_par', 'k_perp'], coords={'k_par': kx_values, 'k_perp': ky_values})
            mapping_angle_from_k_xarray = xr.DataArray(beta_out, dims=['k_par', 'k_perp'], coords={'k_par': kx_values, 'k_perp': ky_values})
        else:
            theta_par_from_k_xarray = xr.DataArray(alpha_out * BL_angles.angles[BL]['theta_par'], dims=['k_perp', 'k_par'], coords={'k_par': ky_values, 'k_perp': kx_values})
            mapping_angle_from_k_xarray = xr.DataArray(beta_out, dims=['k_perp', 'k_par'], coords={'k_par': ky_values, 'k_perp': kx_values})
        # Do the interpolation
        data_out = data.interp({mapping_dim: mapping_angle_from_k_xarray, 'theta_par': theta_par_from_k_xarray})

    else:  # Array of KE values
        # Get angle scales - array order [kx,ky,eV]
        if ana_type == 'I':  # Type I, no deflector
            alpha_out, beta_out = fI_inv(kx_values[:, np.newaxis, np.newaxis], ky_values[np.newaxis, :, np.newaxis], delta, xi, KE_values[np.newaxis, np.newaxis, :], norm_offset)
        elif ana_type == 'II':  # Type II, no deflector
            alpha_out, beta_out = fII_inv(kx_values[:, np.newaxis, np.newaxis], ky_values[np.newaxis, :, np.newaxis], delta, xi, KE_values[np.newaxis, np.newaxis, :], norm_offset)
        elif ana_type == 'Ip':  # Type I, with deflector
            alpha_out, beta_out = fIp_inv(kx_values[:, np.newaxis, np.newaxis], ky_values[np.newaxis, :, np.newaxis], delta, xi, chi, KE_values[np.newaxis, np.newaxis, :])
        elif ana_type == 'IIp':  # Type II, with deflector
            alpha_out, beta_out = fIIp_inv(kx_values[:, np.newaxis, np.newaxis], ky_values[np.newaxis, :, np.newaxis], delta, xi, chi, KE_values[np.newaxis, np.newaxis, :])

        # Make the relevant xarrays to use for interpolation
        if ana_type == 'I' or ana_type == 'Ip':
            theta_par_from_k_xarray = xr.DataArray(alpha_out * BL_angles.angles[BL]['theta_par'], dims=['k_par', 'k_perp', 'eV'], coords={'k_par': kx_values, 'k_perp': ky_values, 'eV': data.eV.data})
            mapping_angle_from_k_xarray = xr.DataArray(beta_out, dims=['k_par', 'k_perp', 'eV'], coords={'k_par': kx_values, 'k_perp': ky_values, 'eV': data.eV.data})
        else:
            theta_par_from_k_xarray = xr.DataArray(alpha_out * BL_angles.angles[BL]['theta_par'], dims=['k_perp', 'k_par', 'eV'], coords={'k_par': ky_values, 'k_perp': kx_values, 'eV': data.eV.data})
            mapping_angle_from_k_xarray = xr.DataArray(beta_out, dims=['k_perp', 'k_par', 'eV'], coords={'k_par': ky_values, 'k_perp': kx_values, 'eV': data.eV.data})
        eV_xarray = xr.DataArray(data.eV.data, dims=['eV'], coords={'eV': data.eV.data})

        # Do the interpolation
        data_out = data.interp({mapping_dim: mapping_angle_from_k_xarray, 'theta_par': theta_par_from_k_xarray, 'eV': eV_xarray})

    # Reindex array to ensure increasing k in the array
    if data_out.k_par.data[1] - data_out.k_par.data[0] < 0 and data_out.k_perp.data[1] - data_out.k_perp.data[0] < 0:  # Both k-arrays currently have decreasing k-order
        data_out = data_out.reindex(k_par=data_out.k_par[::-1],k_perp=data_out.k_perp[::-1])
    elif data_out.k_par.data[1] - data_out.k_par.data[0] < 0:  # k_par array currently has decreasing k-order
        data_out = data_out.reindex(k_par=data_out.k_par[::-1])
    elif data_out.k_perp.data[1] - data_out.k_perp.data[0] < 0:  # k_perp array currently has decreasing k-order
        data_out = data_out.reindex(k_perp=data_out.k_perp[::-1])

    # Update units of axes
    data_out.coords['k_par'].attrs = {'units': 'inv. ang.'}
    data_out.coords['k_perp'].attrs = {'units': 'inv. ang.'}

    #Update analysis history and take mean of output array if required
    norm_em = {'norm_polar': angles['norm_polar'], 'norm_tilt': angles['norm_tilt'], 'norm_azi': angles['norm_azi']}
    hist_str = 'Converted FS map to k-space, with normal emissions: '+str(norm_em)

    if bin_flag == 1:
        hist_str += ', binning applied: '+str(kwargs['binning'])

    if 'FS' in kwargs: #If called with FS way to specify energy, take mean along eV (if >1 slice) in here
        try: #May fail if only one slice
            data_out = data_out.mean('eV', keep_attrs=True)
            #Add to the history
            hist_str += ', FS taken at eV=' + str(kwargs['FS'][0]) + ' with integration window dE=' + str(kwargs['FS'][1]) + ' eV'
            update_hist(data_out, hist_str)
        except: #Fails if only one energy in call, in which case just update the history
            update_hist(data_out, hist_str)
    else:
        update_hist(data_out, hist_str)

    # Reset warnings display and formatting
    warnings.simplefilter('once', UserWarning)  # Give warnings first time only
    warnings.formatwarning = warning_standard  # Formatting of warnings for peaks user errors

    return data_out

# Convert photon energy-dependent data to k-space using free-electron final state model with defined inner potential
def kz_conv(data_in,**kwargs):
    '''This function applies a k-space conversion by interpolating an xarray from angle (along the slit direction)
    and hv space to k_par/kz-space, with angular offsets which can vary with hv.
    
    Input:
            data_in - the dispersion to be converted (xarray)

            **kwargs - optional arguments:
            Data attributes:
                Setting manipulator angles:
                    - polar = ## (float) to set the manipulator polar angle (also overwrites that attribute
                    in the original dataarray)
                    - tilt = ## (float) to set the manipulator tilt angle (also overwrites that attribute
                    in the original dataarray)
                    - azi = ## (float) to set the manipulator azi angle (also overwrites that  attribute
                    in the original dataarray)
                Setting normal emissions:
                    - norm_polar = ## (float) to set the normal emission in the polar axis (also overwrites that
                    attribute in the original dataarray)
                    - norm_tilt = ## (float) to set the normal emission in the tilt axis (also overwrites that
                    attribute in the original dataarray)
                    - norm_azi = ## (float) to set the offset in the azimuthal axis (also overwrites that attribute
                    in the original dataarray)

                Setting EF_correction method:
                    - EF_correction = float, fit_coeffs, xarray (see fermi_level_corrections.py): EF_correction to
                    use (also overwrites that attribute in the original dataarray)
                    - BE_only (boolean): If True, convert only to BE, and do not do the k-space conversions

                Inner potential:
                    - V0: set inner potential (float, in eV, defaults to 12 eV if not specified)

                k-point spacings:
                    - dk = ## (float); to manually select k-point spacing (in-plane). Defaults to keeping same
                    number of points as in theta_par dimension for dispersions, or to 0.01 for Fermi maps.
                    - dkz = ## (float); to manually select k-point spacing (out-of-plane). Defaults
                    to keeping same number of points as in hv dimension.

                Binning/selection (these can provide substantial speed-up if converting a larger amount of data (e.g.
                full cube):
                    - binning = {'eV': 2, 'theta_par: 2} to apply a 2x2 binning on the energy and theta_par axis,
                    to save a lot of time if building the full cube. Can change factor, and bin on one or both axes
                    - bin_factor = #: shortcut to do #x# binning on the energy and theta_par axis, call this
                    instead of binning = {}
                    - eV=E0 or eV=slice(E0,E1) to return a constant energy slice (single value or range), (E0,E1
                     floats, ignored for single dispersion)
                    - FS = dE or FS=(dE, E0) as a shortcut to select a slice of width dE at energy E0 (default at
                     E0=0), and performs the mean over eV before returning (E0,dE floats, ignored for single
                     dispersion)
                    - k_par = k0 or k_par = slice(k0,k1) to return a constant k_par slice (single
                     value or range, k0,k1 floats)

        Returns:
            data_out - the data that has been converted to k-space and/or binding energy (xarray)'''

    # Set up warnings display and formatting
    warnings.simplefilter('always', UserWarning)  # Give warnings every time, even on function re-run
    warnings.formatwarning = warning_simple  # Formatting of warnings for peaks user errors

    #Check supplied format is correct
    if 'EF' not in data_in.coords or 'KE_delta' not in data_in.coords:
        raise Exception("Data not supplied in correct format for hv-scan (missing EF or KE_delta info)")

    # Check call for energy slice
    if 'FS' in kwargs:
        if 'eV' in kwargs:
            err_str = 'Function called with two energy selections. Use either eV=E0, eV=slice(E0,E1), FS=dE, or FS=(dE,E0).'
            raise Exception(err_str)

        if type(kwargs['FS']) == float or type(kwargs['FS']) == int:  # At Fermi energy with some width
            if kwargs['FS'] == 0 or kwargs['FS'] == 0.:  # Single slice requied
                kwargs['eV'] = 0  # Write that into appropriate eV call
            else:
                kwargs['eV'] = slice(-float(kwargs['FS']) / 2, float(kwargs['FS']) / 2)
        elif type(kwargs['FS']) == tuple:  # Called with range and centre energy
            kwargs['eV'] = slice(kwargs['FS'][1] - kwargs['FS'][0] / 2,
                                 kwargs['FS'][1] + kwargs['FS'][0] / 2)  # Write that into appropriate eV call
        else:
            err_str = 'FS should be a float to select a slice at the Fermi level (eV=0) with width dE, or a tuple (dE,E0) to select a slice of width dE centered at E0.'
            raise Exception(err_str)

    # Set inner potential
    if 'V0' in kwargs: # Set value passed as a kwarg
        data_in.attrs['V0'] = kwargs['V0']
    elif 'V0' not in data_in.attrs:  # If not already in attributes, set a default value and trigger a warning
        data_in.attrs['V0'] = 12
        warning_str = 'No inner potential specified. Default value of 12 eV used. To use specific value call with argument V0=float.'
        warnings.warn(warning_str)
    V0 = data_in.attrs['V0']  # Set inner potential to use

    # Get beamline
    BL = data_in.attrs['beamline']

    #################################
    #####     ENERGY SCALES     #####
    #################################
    # determine angular-dependent Fermi energy correction based on the EF_correction attribute
    if 'EF_correction' in kwargs:  # If passed with call, write that to original dataarray (also updates original array outside of function)
        data_in.attrs['EF_correction'] = kwargs['EF_correction']

    # Extract relevant energy shifts, correction type and raw correction parameters from dataarray
    E_shift, EF_correction_type = Eshift_from_correction(data_in)

    # Determine binning (but don't run this yet, this should come later for speed)
    bin_flag = 0
    if 'binning' in kwargs:
        bin_flag = 1  # Flag to indicate if binning to be performed
        if 'bin_factor' in kwargs:
            err_str = 'Function called with two binning instructions. Use either bin_factor=# to apply #x# binning to the energy and theta_par dims, or call explicitly with binning=dict, where dict is a dictionary of dimension names and binning factors to bin over.'
            raise Exception(err_str)

    elif 'bin_factor' in kwargs:  # Apply bin_factor to energy and theta_par axes
        if kwargs['bin_factor'] == 1:  # Don't do any binning if called with bin_factor = 1
            bin_flag = 2
        else:  # Write binning into correct form
            bin_flag = 1
            kwargs['binning'] = {'eV': kwargs['bin_factor'], 'theta_par': kwargs['bin_factor']}

    if 'binning' in kwargs and 'eV' in kwargs:  # Don't allow binning if the data will end up with no energy slices
        if type(kwargs['eV']) != slice:
            if EF_correction_type != 'fit' or EF_correction_type != 'array':
                bin_flag = 2
                warning_str = 'Binning not supported for data with this small a slice to be passed for k-conversion. Binning ignored here.'
                warnings.warn(warning_str)

    #hv axis
    hv_values = data_in.hv.data
    hv_min = hv_values.min()
    hv_max = hv_values.max()
    hv_xarray = xr.DataArray(hv_values, dims=["hv"], coords={"hv": hv_values})

    #Estimate work function from first photon energy, and then take that as fixed (uncertainty here from errors in photon energy, but this should have only a very small effect on the k conversion)
    wf = data_in.hv.data[0] - data_in.EF.data[0]

    #Current energy axis scaling
    eV_values = data_in.coords['eV'].data

    #################################
    #####     ANGLE SCALES      #####
    #################################
    # Set relevant normal emission angles
    set_normals(data_in, **kwargs)

    # Get relevant angles from data_in
    angles = get_angles(data_in)

    # Extract relevant angles from angles
    alpha = angles['alpha']
    beta = angles['beta']
    delta = angles['delta']
    xi = angles['xi']
    if 'chi' in angles:
        chi = angles['chi']
    ana_type = angles['ana_type']
    if ana_type == 'I':
        norm_offset = angles['norm_polar']
    elif ana_type == 'II':
        norm_offset = angles['norm_tilt']

    # theta_par axis xarray
    theta_par_xarray = xr.DataArray(alpha, dims=["theta_par"], coords={"theta_par": alpha})

    ################################
    #Convert data to binding energy
    ################################

    # PK 18/3/22: Need to be careful with which value to take for the E-shift and BE indexing depending on whether hv is increasing or decreasing
    if data_in.KE_delta[0].data < -0.01:
        E_index = -1
    else:
        E_index = 0

    #if the correction type is fitting parameters (apply EF curvature correction, further shifted by Fermi levels in EF coordinate)
    if EF_correction_type == 'fit' or EF_correction_type == 'array': #Something that includes curvature of the Fermi energy on detector
        #Convert E_shift into an angle-dependent shift that needs to be applied to straighten the Fermi edge up
        E_shift = max(E_shift) - E_shift
        #Add a hv-dep dimension to this, taking acount of both EF shifts and relevant KE shifts (EF stored in coordinate taken to be relevant value for the maximum of the Au dispersion curve)
        E_shift = E_shift[:,np.newaxis] - (data_in.EF.data - data_in.EF.data[E_index]) + (data_in.KE_delta.data)
        #Determine relevant KE matrix to interpolate over
        eV_values_tointerp = eV_values[:, np.newaxis, np.newaxis] - E_shift

        #Set BE scale for full cube - note that this is negative for energies below Fermi level, need to be careful with sign conventions below
        BE_values = data_in.eV.data-data_in.EF.data[E_index]

        eV_xarray = xr.DataArray(eV_values_tointerp, dims=["eV", "theta_par", "hv"], coords={"theta_par": alpha, "eV": BE_values, "hv": hv_values})
        

    #if the correction type is a float/int or not specified (just shift by the relevant values in EF coordinate)
    elif EF_correction_type == 'value' or EF_correction_type == 'None':
        E_shift = data_in.EF.data[E_index] - data_in.EF.data + data_in.KE_delta.data
        eV_values_tointerp = eV_values[:, np.newaxis] - E_shift

        #Set BE scale for full cube - note that this is negative for energies below Fermi level, need to be careful with sign conventions below
        BE_values = data_in.eV.data-data_in.EF.data[E_index]
        eV_xarray = xr.DataArray(eV_values_tointerp, dims=["eV", "hv"], coords={"eV": BE_values, "hv": hv_values})

    #If an energy slice is requested, subselect only relevant portion of this cube for interpolation
    if 'eV' in kwargs:
        if type(kwargs['eV']) == float or type(kwargs['eV']) == int: #single slice requested
            eV_xarray=eV_xarray.sel(eV=kwargs['eV'], method='nearest')
        else: #range supplied
            eV_xarray=eV_xarray.sel(eV=kwargs['eV'])
            
    #if k-slice is requested, select relevant portion of cube for interpolation
    if 'k_par' in kwargs:   
        #Get k-range required
        if type(kwargs['k_par']) == float or type(kwargs['k_par']) == int: #single slice requested
            k_par_min = float(kwargs['k_par']) - 0.02  # Add a small amount to range to help with interpolation
            k_par_max = float(kwargs['k_par']) + 0.02  # Add a small amount to range to help with interpolation
        else: #range supplied
            k_par_min = kwargs['k_par'].start - 0.02  # Add a small amount to range to help with interpolation
            k_par_max = kwargs['k_par'].stop + 0.02  # Add a small amount to range to help with interpolation

        # ToDo: Make this work for variable normal emission angles

        # Work out angle range
        if ana_type == 'I':  # Type I, no deflector
            alpha1, beta1 = fI_inv(k_par_min, 0, delta, xi, data_in.hv.data.min()-wf+BE_values.max(), norm_offset)  # Min values
            alpha2, beta2 = fI_inv(k_par_max, 0, delta, xi, data_in.hv.data.max() - wf + BE_values.min(), norm_offset)  # Max values
        elif ana_type == 'II':  # Type II, no deflector
            alpha1, beta1 = fII_inv(0, k_par_min, delta, xi, data_in.hv.data.min() - wf + BE_values.max(), norm_offset)  # Min values
            alpha2, beta2 = fII_inv(0, k_par_max, delta, xi, data_in.hv.data.max() - wf + BE_values.min(), norm_offset)  # Max values
        elif ana_type == 'Ip':  # Type I, with deflector
            alpha1, beta1 = fIp_inv(k_par_min, 0, delta, xi, chi, data_in.hv.data.min() - wf + BE_values.max())  # Min values
            alpha2, beta2 = fIp_inv(k_par_max, 0, delta, xi, chi, data_in.hv.data.max() - wf + BE_values.min())  # Max values
        elif ana_type == 'IIp':  # Type II, with deflector
            alpha1, beta1 = fIIp_inv(0, k_par_min, delta, xi, chi, data_in.hv.data.min() - wf + BE_values.max())  # Min values
            alpha2, beta2 = fIIp_inv(0, k_par_max, delta, xi, chi, data_in.hv.data.max() - wf + BE_values.min())  # Max values

        #Get angle range to keep along analyser slit
        theta_min = min([alpha1* BL_angles.angles[BL]['theta_par'],alpha2* BL_angles.angles[BL]['theta_par']])
        theta_max = max([alpha1* BL_angles.angles[BL]['theta_par'],alpha2* BL_angles.angles[BL]['theta_par']])

        #theta_par_xarray = theta_par_xarray * BL_angles.angles[BL]['theta_par']

        if theta_par_xarray['theta_par'].data[1] - theta_par_xarray['theta_par'].data[0] < 0:  # Array currently has decreasing order
            theta_par_xarray = theta_par_xarray.reindex({'theta_par': theta_par_xarray['theta_par'][::-1]})
        if eV_xarray['theta_par'].data[1] - eV_xarray['theta_par'].data[0] < 0:  # Array currently has decreasing order
            eV_xarray = eV_xarray.reindex({'theta_par': eV_xarray['theta_par'][::-1]})

        #Subselect theta_par range
        if np.abs(theta_max-theta_min) > data_in.theta_par.data[1]-data_in.theta_par.data[0]: #Integration range larger than step size
            theta_par_xarray = theta_par_xarray.sel(theta_par=slice(theta_min,theta_max))
            if 'theta_par' in eV_xarray.dims:
                eV_xarray = eV_xarray.sel(theta_par=slice(theta_min,theta_max))
        else: #select single pixel
            theta_par_xarray = theta_par_xarray.sel(theta_par=theta_min,method='nearest')
            if 'theta_par' in eV_xarray.dims:
                eV_xarray = eV_xarray.sel(theta_par=theta_min, method='nearest')


    # Binning
    if bin_flag == 0:
        if eV_xarray.eV.size * theta_par_xarray.size * data_in.hv.size > 10000000:  # Automatically 2x2 bin if array is too large, unless explicitly told not to
            data = data_in.coarsen(eV=2, theta_par=2, boundary='pad').mean(keep_attrs=True).copy(deep=True)
            if 'theta_par' in eV_xarray.dims:
                eV_xarray = eV_xarray.coarsen(eV=2, theta_par=2, boundary='pad').mean(keep_attrs=True)
            else:
                eV_xarray = eV_xarray.coarsen(eV=2, boundary='pad').mean(keep_attrs=True)
            theta_par_xarray = theta_par_xarray.coarsen(theta_par=2, boundary='pad').mean(keep_attrs=True)
            warning_str = 'Data array is large, 2x2 binning on energy and theta_par axis has been automatically applied. To run without binning, call function with bin_factor=1.'
            warnings.warn(warning_str)
        else:
            data = data_in.copy(deep=True)
    elif bin_flag == 1:
        data = data_in.coarsen(kwargs['binning'], boundary='pad').mean(keep_attrs=True).copy(deep=True)
        if 'theta_par' in eV_xarray.dims:
            eV_xarray = eV_xarray.coarsen(kwargs['binning'], boundary='pad').mean(keep_attrs=True)
        else:
            if 'eV' in kwargs['binning']:
                eV_xarray = eV_xarray.coarsen({'eV': kwargs['binning']['eV']}, boundary='pad').mean(keep_attrs=True)
        if 'theta_par' in kwargs['binning']:
            theta_par_xarray = theta_par_xarray.coarsen({'theta_par': kwargs['binning']['theta_par']}, boundary='pad').mean(keep_attrs=True)
    else:
        data = data_in.copy(deep=True)

    # Refresh BE values for new range
    BE_values = eV_xarray.eV.data
    BE_values = np.atleast_1d(BE_values)

    BE_min = BE_values.min()
    BE_max = BE_values.max()
    BE_xarray = xr.DataArray(BE_values, dims=["eV"], coords={"eV": BE_values})

    ################################
    # k_par conversion
    ################################

    if 'k_par' in kwargs: #if range specified
        if k_par_min == k_par_max: #Only single value required
            k_values = np.atleast_1d(k_par_min)
        else: #range required
            k_values = np.linspace(k_par_min,k_par_max, len(theta_par_xarray.theta_par))
        k_perp = 0  # Assume k_perp = 0

    else: #work out full range
        if ana_type == 'I':  # Type I, no deflector
            kx1, ky1 = fI(theta_par_xarray.theta_par.data[0], beta, delta, xi, hv_values-wf+BE_max)
            kx2, ky2 = fI(theta_par_xarray.theta_par.data[-1], beta, delta, xi, hv_values-wf+BE_max)
            # Get k-range to interpolate over
            k0 = kx1.min()
            k1 = kx2.max()
            k_perp = -ky1.mean()  # Keep track of approximate k_perp
            if ky1.var() > 0.1:
                warning_str = 'k_perp (in-plane k, perpendicular to the slit) appears to be varying significantly throughout the kz-map.'
                warnings.warn(warning_str)

        elif ana_type == 'II':  # Type II, no deflector
            kx1, ky1 = fII(theta_par_xarray.theta_par.data[0], beta, delta, xi, hv_values - wf + BE_max)
            kx2, ky2 = fII(theta_par_xarray.theta_par.data[-1], beta, delta, xi, hv_values - wf + BE_max)
            # Get k-range to interpolate over
            k0 = ky1.min()
            k1 = ky2.max()
            k_perp = -kx1.mean()  # Keep track of approximate k_perp
            if kx1.var() > 0.1:
                warning_str = 'k_perp (in-plane k, perpendicular to the slit) appears to be varying significantly throughout the kz-map.'
                warnings.warn(warning_str)
        elif ana_type == 'Ip':  # Type I, with deflector
            kx1, ky1 = fIp(theta_par_xarray.theta_par.data[0], beta, delta, xi, chi, hv_values-wf+BE_max)
            kx2, ky2 = fIp(theta_par_xarray.theta_par.data[-1], beta, delta, xi, chi, hv_values-wf+BE_max)
            # Get k-range to interpolate over
            k0 = kx1.min()
            k1 = kx2.max()
            k_perp = -ky1.mean()  # Keep track of approximate k_perp
            if ky1.var() > 0.1:
                warning_str = 'k_perp (in-plane k, perpendicular to the slit) appears to be varying significantly throughout the kz-map.'
                warnings.warn(warning_str)
        elif ana_type == 'IIp':  # Type II, with deflector
            kx1, ky1 = fIIp(theta_par_xarray.theta_par.data[0], beta, delta, xi, chi, hv_values - wf + BE_max)
            kx2, ky2 = fIIp(theta_par_xarray.theta_par.data[-1], beta, delta, xi, chi, hv_values - wf + BE_max)
            # Get k-range to interpolate over
            k0 = ky1.min()
            k1 = ky2.max()
            k_perp = -kx1.mean()  # Keep track of approximate k_perp
            if kx1.var() > 0.1:
                warning_str = 'k_perp (in-plane k, perpendicular to the slit) appears to be varying significantly throughout the kz-map.'
                warnings.warn(warning_str)

        # Set k_values to interpolate over along the slit direction
        if 'dk' not in kwargs:  # Use default k-point spacing
            k_values = np.linspace(k0, k1, data.theta_par.size)
        else:
            k_values = np.arange(k0, k1, kwargs['dk'])


    # Get relevant KE array for conversion
    KE_array = hv_values[:,np.newaxis] - wf + BE_values
    KE_array = KE_array[:,:,np.newaxis]  # Array ordered (hv,eV,theta_par)

    # Get the relevant angle from k-space mappings and interpolate over these
    if ana_type == 'I':  # Type I, no deflector
        alpha_out, beta_out = fI_inv(k_values[np.newaxis,np.newaxis,:], 0, delta, xi, KE_array, norm_offset)
    elif ana_type == 'II':  # Type II, no deflector
        alpha_out, beta_out = fII_inv(0, k_values[np.newaxis,np.newaxis,:], delta, xi, KE_array, norm_offset)
    elif ana_type == 'Ip':  # Type I, with deflector
        alpha_out, beta_out = fIp_inv(k_values[np.newaxis,np.newaxis,:], 0, delta, xi, chi, KE_array)
    elif ana_type == 'IIp':  # Type II, with deflector
        alpha_out, beta_out = fIIp_inv(0, k_values[np.newaxis, np.newaxis, :], delta, xi, chi, KE_array)

    # Make mapping from k-space to angle along slit.
    theta_from_k_xarray = xr.DataArray(
        alpha_out * BL_angles.angles[BL]['theta_par'],
        dims = ['hv', 'eV', 'k_par'],
        coords={'hv': hv_values, 'k_par': k_values, 'eV': BE_values}
    )

    #Interpolate onto BE/k_||/hv grid
    if kwargs['BE_only'] == True:  # In this case, only do BE conversion
        data_BE = data.interp(eV=eV_xarray, theta_par=theta_par_xarray, hv=hv_xarray)

        # Update attributes to note in BE
        data_BE.attrs['eV_type'] = 'binding'

        # Take any relevant energy slices
        if 'eV' in kwargs:
            try:
                data_BE = data_BE.sel(eV=kwargs['eV'])
            except:
                data_BE = data_BE.sel(eV=kwargs['eV'], method='nearest')

        # Update analysis history and take mean of output array if required
        hist_str = "Data converted to binding energy"
        if bin_flag == 1:
            hist_str += ', binning applied: ' + str(kwargs['binning'])

        if 'FS' in kwargs:  # If called with FS way to specify energy, take mean along eV (if >1 slice) in here
            try:  # May fail if only one slice
                data_BE = data_BE.mean('eV', keep_attrs=True)
                # Add to the history
                if type(kwargs['FS']) == float or type(kwargs['FS']) == int:  # At Fermi energy with some width
                    hist_str += ', FS taken at eV=0 with integration window dE=' + str(kwargs['FS']) + ' eV'
                else:
                    hist_str += ', FS taken at eV=' + str(kwargs['FS'][1]) + ' with integration window dE=' + str(kwargs['FS'][0]) + ' eV'
                update_hist(data_BE, hist_str)
            except:  # Fails if only one energy in call, in which case just update the history
                update_hist(data_BE, hist_str)

        else:
            update_hist(data_BE, hist_str)

        return data_BE

    elif EF_correction_type == 'value' or EF_correction_type == 'None': #For this correction type, can convert to BE and kp in one step
        data_BEkp = data.interp(eV=eV_xarray, theta_par=theta_from_k_xarray, hv=hv_xarray)
    else: #need 2 steps
        data_BE = data.interp(eV=eV_xarray, theta_par=theta_par_xarray, hv=hv_xarray)
        if 'eV' not in data_BE.dims: #If a single slice has been selected
            data_BEkp = data_BE.interp(theta_par=theta_from_k_xarray.squeeze("eV"), hv=hv_xarray)
            data_BEkp = data_BEkp.expand_dims('eV') #convert eV back into a dimension for forwards compatibility
        else:
            data_BEkp = data_BE.interp(eV=BE_xarray, theta_par=theta_from_k_xarray, hv=hv_xarray)

    if 'k_par' in kwargs:  # Select slice in k_par if function called with k_par specification
        try:  # May fail for a single slice
            data_BEkp = data_BEkp.sel(k_par=kwargs['k_par'])
        except:
            data_BEkp = data_BEkp.sel(k_par=kwargs['k_par'], method='nearest')

        if 'k_par' not in data_BEkp.dims:  # Make sure that k_par is a dimension still
            data_BEkp = data_BEkp.expand_dims('k_par')

    ################################
    #kz conversion
    ################################
    
    #work out some limits
    #max k_par vales at minimum hv
    if ana_type == 'I':  # Type I, no deflector
        kx1, ky1 = fI(theta_par_xarray.theta_par.data[0], beta, delta, xi, hv_values.min() - wf + BE_min)
        kx2, ky2 = fI(theta_par_xarray.theta_par.data[-1], beta, delta, xi, hv_values.min() - wf + BE_min)
        k_par_at_hv_min = max([abs(kx1.min()), abs(kx2.max())])
    elif ana_type == 'II':  # Type II, no deflector
        kx1, ky1 = fII(theta_par_xarray.theta_par.data[0], beta, delta, xi, hv_values.min() - wf + BE_min)
        kx2, ky2 = fII(theta_par_xarray.theta_par.data[-1], beta, delta, xi, hv_values.min() - wf + BE_min)
        k_par_at_hv_min = max([abs(ky1.min()), abs(ky2.max())])
    elif ana_type == 'Ip':  # Type I, with deflector
        kx1, ky1 = fIp(theta_par_xarray.theta_par.data[0], beta, delta, xi, chi, hv_values.min() - wf + BE_min)
        kx2, ky2 = fIp(theta_par_xarray.theta_par.data[-1], beta, delta, xi, chi, hv_values.min() - wf + BE_min)
        k_par_at_hv_min = max([abs(kx1.min()), abs(kx2.max())])
    elif ana_type == 'IIp':  # Type II, with deflector
        kx1, ky1 = fIp(theta_par_xarray.theta_par.data[0], beta, delta, xi, chi, hv_values.min() - wf + BE_min)
        kx2, ky2 = fIp(theta_par_xarray.theta_par.data[-1], beta, delta, xi, chi, hv_values.min() - wf + BE_min)
        k_par_at_hv_min = max([abs(ky1.min()), abs(ky2.max())])

    # Get kz range
    kz_max = const.kvac_const*np.sqrt(V0 + hv_max - wf + BE_max)
    kz_min = const.kvac_const*np.sqrt(V0 + hv_min - wf + BE_min - ((k_par_at_hv_min/const.kvac_const)**2))

    if 'dkz' in kwargs: #Manual density of kz-points specified
        kz_values = np.arange(kz_min,kz_max, kwargs['dkz'])
    else: #Take default based on number of photon energies
        kz_values = np.linspace(kz_min,kz_max, len(hv_values), endpoint=True)
        
    #Make kpar xarray for interpolation
    k_par_xarray = xr.DataArray(data_BEkp.k_par.data, dims=["k_par"], coords={"k_par": data_BEkp.k_par.data})

    hv_to_kz = ((kz_values[:,np.newaxis,np.newaxis]/const.kvac_const)**2) + ((data_BEkp.k_par.data[np.newaxis,:,np.newaxis]/const.kvac_const)**2) + wf - BE_values[np.newaxis,np.newaxis,:] - V0

    hv_to_kz_xarray = xr.DataArray(
        hv_to_kz,
        dims = ['kz', 'k_par', 'eV'],
        coords={'kz': kz_values, 'k_par': data_BEkp.k_par.data, 'eV': BE_values}
    )

    #Interpolate onto BE/k_||/kz grid
    if len(data_BEkp.eV.data) == 1: #Only single slice in energy being returned
        data_out = data_BEkp.squeeze('eV').interp(k_par=k_par_xarray, hv=hv_to_kz_xarray.squeeze('eV'))
    elif len(data_BEkp.k_par.data) == 1: #Only single slice in k_par being returned
        data_out = data_BEkp.squeeze('k_par').interp(eV=BE_xarray, hv=hv_to_kz_xarray.squeeze('k_par'))
    else:
        data_out = data_BEkp.interp(eV=BE_xarray, k_par=k_par_xarray, hv=hv_to_kz_xarray)

    # Reindex array to have increasing k in the array
    try:  # Will fail for single-width k_par slice
        if data_out.k_par.data[1] - data_out.k_par.data[0] < 0:  # Array currently has decreasing k-order
            data_out = data_out.reindex(k_par=data_out.k_par[::-1])
    except:
        pass
    if data_out.kz.data[1] - data_out.kz.data[0] < 0:  # Array currently has decreasing k-order
        data_out = data_out.reindex(kz=data_out.kz[::-1])

    # Check the k_perp value
    if abs(k_perp) > 0.01:  # Threshold to avoid numerical noise
        data_out.attrs['k_perp'] = k_perp  # If this is significant, write mean value to the attributes

    #Update attributes to note in BE
    data_out.attrs['eV_type'] = 'binding'
    
    #Update units
    data_out.coords['k_par'].attrs = {'units' : 'inv. ang.'}
    data_out.coords['kz'].attrs = {'units' : 'inv. ang.'}
    
    # Update analysis history and take mean of output array if required
    hist_str = "hv-dep data converted to k-space using free-electron final state model, V0="+str(V0)+" eV"
    if bin_flag == 1:
        hist_str += ', binning applied: ' + str(kwargs['binning'])

    if 'FS' in kwargs: #If called with FS way to specify energy, take mean along eV (if >1 slice) in here
        try: #May fail if only one slice
            data_out = data_out.mean('eV', keep_attrs=True)
            #Add to the history
            if type(kwargs['FS']) == float or type(kwargs['FS']) == int:  # At Fermi energy with some width
                hist_str += ', FS taken at eV=0 with integration window dE=' + str(kwargs['FS']) + ' eV'
            else:
                hist_str += ', FS taken at eV=' + str(kwargs['FS'][1]) + ' with integration window dE=' + str(kwargs['FS'][0]) + ' eV'
            update_hist(data_out, hist_str)
        except: #Fails if only one energy in call, in which case just update the history
            update_hist(data_out, hist_str)
    else:
        update_hist(data_out, hist_str)

    # Reset warnings display and formatting
    warnings.simplefilter('once', UserWarning)  # Give warnings first time only
    warnings.formatwarning = warning_standard  # Formatting of warnings for peaks user errors

    return data_out







