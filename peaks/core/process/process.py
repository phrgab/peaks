# Functions used to apply k-space conversions
# Phil King 09/06/2021
# make_hv_scan function update Brendan Edwards 28/06/21
# Lewis Hart q_conv modification 15/06/2021

from peaks.core.process.q_conversion import *
from peaks.core.process.merging import merge_data
from peaks.core.process.fermi_level_corrections import apply_fermi, hv_align
from peaks.utils.estimate_EF import estimate_EF
from peaks.utils.metadata import update_hist
from peaks.utils.misc import warning_simple, warning_standard
from peaks.utils.OOP_method import add_methods



#####################################################
# Master conversion function to convert to k and BE #
#####################################################
@add_methods(xr.DataArray)
def proc(data_in, **kwargs):
    '''Master function for data processing k-space and binding energy conversions. Accepts multiple types of data
    (dispersions, hv-scans, angle maps etc.) and returns data in binding energy, k, or both depending on the args
    specified.  Multiple options exist for selecting data, binning data etc.

    Parameters
    ------------
    data_in : xr.DaraArray or list(xr.DataArray)
        The data to be converted or a list of dataArrays for batch processing

    **kwargs - optional arguments:
        Conversion options; optional calls to specify which conversions should be performed
         (if not supplied, defaults to BE and k conversion if possible with supplied attributes):
            - convert = 'k'; specify to convert only to k-space, leaving in kinetic energy (assumes data
              already in KE, does not work for kz-scans as KE varies through the scan)
            - convert = 'BE'; specify to convert only to binding energy (keep in angle)
            - merge = True (boolean); specify whether merging into a single spectrum is to be
              performed if a list of xarrays is passed in data_in (defaults to false, i.e. batch
              processing)

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
        Merging option:
            - flip = True/False (boolean) to set whether to 'flip' the manipulator axis sign when merging
              multiple scans onto an enhanced theta_par range (depends on sign conversion between theta_par
              and the relevant manipulator angle (polar or tilt depending on Type I or II configuration).
              Defaults to False.

        Setting EF_correction method:
            - EF_correction = float, fit_coeffs, xarray (see fermi_level_corrections.py): EF_correction to
              use (also overwrites that attribute in the original dataarray)

        Inner potential (considered for hv-scans only):
            - V0: set inner potential (float, in eV, defaults to 12 eV if not specified)

        Output options:
            k-point spacings:
                - dk = ## (float); to manually select k-point spacing (in-plane). Defaults to keeping same
                  number of points as in theta_par dimension for dispersions, or to 0.01 for Fermi maps.
                - dkz = ## (float); to manually select k-point spacing (out-of-plane) for a kz scan. Defaults
                  to keeping same number of points as in hv dimension.

            Binning/selection (these can provide substantial speed-up if converting a larger data set (e.g.
            kz-map, FS map):
                - binning = {'eV': 2, 'theta_par: 2} to apply a 2x2 binning on the energy and theta_par axis,
                  to save a lot of time if building the full cube. Can change factor, and bin on one or both
                  axes
                - bin_factor = #: shortcut to do #x# binning on the energy and theta_par axis, call this
                  instead of binning = {}
                - eV=E0 or eV=slice(E0,E1) to return a constant energy slice (single value or range), (E0,E1
                   floats, ignored for single dispersion)
                - FS = dE or FS=(dE, E0) as a shortcut to select a slice of width dE at energy E0 (default at
                   E0=0), and performs the mean over eV before returning (E0,dE floats, ignored for single
                   dispersion)
                - k_par = k0 or k_par = slice(k0,k1) to return a constant k_par slice from a kz-map (single
                   value or range, k0,k1 floats)
                - theta_par = slice(th0,th1) to crop the data to a given theta_par range when merging multiple
                  scans together (ignored otherwise)

    Returns:
        data_out - the data that has been converted to k-space and/or binding energy (xarray)'''

    if type(data_in) == list:  # xarray data can be passed in a list format if batch processing (conversion of each one) or merging is to be performed
        # Check if merging is to be performed
        if 'merge' in kwargs:
            do_merge = kwargs['merge']
        else:
            do_merge = False

        # Batch processing mode
        if do_merge == False:
            print('Batch processing:')  # Flag that this is running as batch processing
            data_out = batch_process(data_in, **kwargs)

        # Merge mode
        elif do_merge == True:
            # Merging works as a multi-step process

            # Perform binning if required - do this here, as otherwise the merge step can be very slow
            if 'binning' in kwargs:
                if 'bin_factor' in kwargs:  # Two types of binning requested - raise error
                    err_str = 'Function called with two binning instructions. Use either bin_factor=# to apply #x# binning to the energy and theta_par dims, or call explicitly with binning=dict, where dict is a dictionary of dimension names and binning factors to bin over.'
                    raise Exception(err_str)
                else:  # Apply the requested binning
                    for count, i in enumerate(data_in):
                        data_in[count] = i.coarsen(kwargs['binning'], boundary='pad').mean(keep_attrs=True).copy(deep=True)
                        # Update the history
                        hist_str = 'Data binning applied: ' + str(kwargs['binning'])
                        update_hist(i, hist_str)
                kwargs.pop('binning') # Remove from kwargs to avoid double binning later
            elif 'bin_factor' in kwargs:  # Apply bin_factor to energy and theta_par axes
                if kwargs['bin_factor'] == 1:  # Don't do any binning if called with bin_factor = 1
                    pass
                else:  # Write binning into correct form and apply
                    kwargs['binning'] = {'eV': kwargs['bin_factor'], 'theta_par': kwargs['bin_factor']}
                    for count, i in enumerate(data_in):
                        data_in[count] = i.coarsen(kwargs['binning'], boundary='pad').mean(keep_attrs=True).copy(deep=True)
                        # Update the history
                        hist_str = 'Data binning applied: ' + str(kwargs['binning'])
                        update_hist(i, hist_str)
                kwargs.pop('bin_factor')  # Remove from kwargs to avoid double binning later
            else:  # If not called, check the data size to be processed
                if data_in[0].size > 10000000: #Automatically 2x2 bin if array is too large, unless explicitly told not to
                    #Run auto binning
                    for count, i in enumerate(data_in):
                        data_in[count] = i.coarsen({'eV': 2, 'theta_par': 2}, boundary='pad').mean(keep_attrs=True).copy(deep=True)
                        # Update the history
                        hist_str = 'Data binning applied: ' + str({'eV': 2, 'theta_par': 2})
                        update_hist(i, hist_str)
                    warning_str = 'Data array is large, 2x2 binning on energy and theta_par axis has been automatically applied. To run without binning, call function with bin_factor=1.'
                    warnings.warn(warning_str)

            # Batch convert to BE
            print('Converting to binding energy:')
            data_BE = batch_process(data_in, convert='BE', **kwargs)

            # Perform any energy slicing
            if 'FS' in kwargs:
                if 'eV' in kwargs:
                    err_str = 'Function called with two energy selections. Use either eV=E0, eV=slice(E0,E1), FS=dE, or FS=(dE,E0).'
                    raise Exception(err_str)

                if type(kwargs['FS']) == float or type(kwargs['FS']) == int:  # At Fermi energy with some width
                    if kwargs['FS'] == 0 or kwargs['FS'] == 0.:  # Single slice requied
                        kwargs['eV'] = 0  # Write that into appropriate eV call
                        kwargs.pop['FS']  # Remove from kwargs to avoid double conversions later
                    else:
                        kwargs['eV'] = slice(-float(kwargs['FS']) / 2, float(kwargs['FS']) / 2)
                elif type(kwargs['FS']) == tuple:  # Called with range and centre energy
                    kwargs['eV'] = slice(kwargs['FS'][1] - kwargs['FS'][0] / 2,
                                         kwargs['FS'][1] + kwargs['FS'][0] / 2)  # Write that into appropriate eV call
                    kwargs.pop['FS']  # Remove from kwargs to avoid double conversions later
                else:
                    err_str = 'FS should be a float to select a slice at the Fermi level (eV=0) with width dE, or a tuple (dE,E0) to select a slice of width dE centered at E0.'
                    raise Exception(err_str)
            if 'eV' in kwargs:
                if type(kwargs['eV']) == slice:
                    for count, i in enumerate(data_BE):
                        data_BE[count] = i.sel(eV=kwargs['eV'])
                else:
                    for count, i in enumerate(data_BE):
                        data_BE[count] = i.sel(eV=kwargs['eV'], method='nearest')
                kwargs.pop('eV')  # Remove from kwargs to avoid double binning

            # Crop the data
            if 'theta_par' in kwargs:
                for count, i in enumerate(data_BE):
                    hist_str = 'Data cropped to theta_par range : (' + str(kwargs['theta_par'].start) + ', ' + str(kwargs['theta_par'].stop) + ')'
                    update_hist(i, hist_str)
                    data_BE[count] = i.sel(theta_par=kwargs['theta_par'])
                print(hist_str)

            # Merge the data in BE
            print('Merging data')
            # Work out offsets
            if data_in[0].attrs['ana_slit_angle'] == 90:  # Type I analyser, need tilt angle to move along slit
                if 'tilt' in kwargs:  # If supplied in kwargs, read tilt from there
                    offsets = kwargs['tilt']
                else:  # Read it from the data attributes
                    offsets = []
                    for i in data_BE:
                        offsets.append(i.attrs['tilt'])
            else:  # Type II, need polar angle
                if 'polar' in kwargs:  # If supplied in kwargs, read polar from there
                    offsets = kwargs['polar']
                else:  # Read it from the data attributes
                    offsets = []
                    for i in data_BE:
                        offsets.append(i.attrs['polar'])

            if 'flip' in kwargs:
                if kwargs['flip'] == True:
                    for count, j in enumerate(offsets):
                        offsets[count] = -j

            merged_BE = merge_data(data_BE, coord = 'theta_par',  offsets = offsets)

            # Set the relevant 'manipulator' angle to zero (as this has already been accounted for in the theta_par scale
            if data_in[0].attrs['ana_slit_angle'] == 90:  # Type I analyser
                merged_BE.attrs['tilt'] = 0
                if 'tilt' in kwargs:
                    kwargs.pop('tilt')
            else:  # Type II analyser
                merged_BE.attrs['polar'] = 0
                if 'polar' in kwargs:
                    kwargs.pop('polar')

            # Do k-space conversion on merged data
            print('Converting to k-space')
            data_out = proc_single(merged_BE, convert='k', **kwargs)

        else:  # Merging call supplied in incorrect form
            err_str = 'Pass argument merge=True to function call to request merging.'
            raise Exception(err_str)


    # Single scan mode
    else:
        data_out = proc_single(data_in, **kwargs)

    gc.collect()

    return data_out


# Helper function for batch processing
def batch_process(data_in, **kwargs):
    '''Helper function used in batch processing, which iterates through the scans in the supplied list and applies
    an appropriate process to each, taking care of any angular definitions that need to be split up and applied
    seperately to each scan.
        Input:
            data_in - the list of data to be converted (list of xarrays)
            **kwargs - optional arguments, for full definitions, see the master `proc` procedure

        Returns:
            data_out - list of processed xarrays '''

    # Make an empty list for storing the relevant processed xarrays
    data_out = []

    # Check format of supplied manipulator formats and angular offsets
    axis_list = ['polar', 'tilt', 'azi', 'norm_polar', 'norm_tilt', 'norm_azi']
    axis_list_values = {}

    for i in axis_list:
        if i in kwargs:
            if type(kwargs[i]) == list:  # If supplied as a list, one is supplied for each scan in called list
                axis_list_values[i] = kwargs[i]
                if len(kwargs[i]) != len(data_in):  # Incompatible formats
                    err_str = 'Different number of values supplied for ' + i + 'as number of arrays passed for batch processing. Either specify a single value or a list of equal length to the number of dataarraus'
                    raise Exception(err_str)
                kwargs.pop(i)  # Remove axis from the general kwargs list

    # Iterate through each scan in the list and perform k-conversion for each
    for count, i in enumerate(data_in):
        print(i.name)  # Print scan name to keep track of batch processing
        # Add in any kwargs as required for angular offsets and manipulator values
        kwargs_temp = kwargs
        for j in axis_list_values:
            kwargs_temp[j] = axis_list_values[j][count]

        # Do the processing of this scan
        data_out.append(proc_single(i, **kwargs_temp))
        time.sleep(0.05)

    return data_out


def proc_single(data_in, **kwargs):
    '''Function for data processing k-space and binding energy conversions of a single xarray. Accepts multiple types of data
        (dispersions, hv-scans, angle maps etc.) and returns data in binding energy, k, or both depending on the args
        specified.  Multiple options exist for selecting data, binning data etc.

            Input:
                data_in - the data to be converted (xarray)

                **kwargs - optional arguments:
                    Conversion options; optional calls to specify which conversions should be performed
                     (if not supplied, defaults to BE and k conversion if possible with supplied attributes):
                        - convert = 'k'; specify to convert only to k-space, leaving in kinetic energy (assumes data
                          already in KE, does not work for kz-scans as KE varies through the scan)
                        - convert = 'BE'; specify to convert only to binding energy (keep in angle)

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

                    Inner potential (considered for hv-scans only):
                        - V0: set inner potential (float, in eV, defaults to 12 eV if not specified)

                Output options:
                    k-point spacings:
                        - dk = ## (float); to manually select k-point spacing (in-plane). Defaults to keeping same
                          number of points as in theta_par dimension for dispersions, or to 0.01 for Fermi maps.
                        - dkz = ## (float); to manually select k-point spacing (out-of-plane) for a kz scan. Defaults
                          to keeping same number of points as in hv dimension.

                    Binning/selection (these can provide substantial speed-up if converting a larger data set (e.g.
                    kz-map, FS map):
                        - binning = {'eV': 2, 'theta_par: 2} to apply a 2x2 binning on the energy and theta_par axis,
                          to save a lot of time if building the full cube. Can change factor, and bin on one or both
                          axes
                        - bin_factor = #: shortcut to do #x# binning on the energy and theta_par axis, call this
                          instead of binning = {}
                        - eV=E0 or eV=slice(E0,E1) to return a constant energy slice (single value or range), (E0,E1
                           floats, ignored for single dispersion)
                        - FS = dE or FS=(dE, E0) as a shortcut to select a slice of width dE at energy E0 (default at
                           E0=0), and performs the mean over eV before returning (E0,dE floats, ignored for single
                           dispersion)
                        - k_par = k0 or k_par = slice(k0,k1) to return a constant k_par slice from a kz-map (single
                           value or range, k0,k1 floats)

            Returns:
                data_out - the data that has been converted to k-space and/or binding energy (xarray)'''

    if 'EF_correction' in kwargs:  # If called with specific EF_correction
        data_in.attrs['EF_correction'] = kwargs['EF_correction']  # Write EF_correction to data attributes

    # Initiate relevant conversion
    if 'hv' in data_in.dims:  # This should be a photon energy scan, this works a bit different to the others as 'k' only conversion isn't allowed
        if 'convert' in kwargs:  # Call specifies which conversion to perform
            if kwargs['convert'] == 'BE':  # BE only conversion
                BE_only = True  # Set flag
            elif kwargs['convert'] == 'k':  # k only conversion requested - this is not possible for a kz-map
                warning_str = 'kz conversion requires conversion to BE at same time as k-conversion; flag convert=\'k\' has been ignored.'
                warnings.warn(warning_str)
                BE_only = False  # Set BE_only flag to false
        else:
            BE_only = False  # Set BE_only flag to false

        kwargs['BE_only'] = BE_only  # Add flag to kwargs

        if 'EF' not in data_in.coords:  # No EF non-dimension coordinate is present
            # Run hv_align to try and estimate EF variation with hv
            temp = hv_align(data_in)
            data_in.coords['EF'] = ('hv', temp.coords['EF'].data)
            del (temp)

        # Trigger kz-conversion
        data_out = kz_conv(data_in, **kwargs)

    # All other k- and BE-conversions can be handled together
    else:
        if 'convert' in kwargs:  # Call specifies which conversion to perform
            if kwargs['convert'] == 'BE':  # Call to convert to BE only
                if 'EF_correction' not in data_in.attrs:  # If no suitable Fermi correction specified
                    add_estimated_EF(data_in)  # Estimate this and add to the attributes
                elif data_in.attrs['EF_correction'] == None:  # If no suitable Fermi correction specified
                    add_estimated_EF(data_in)  # Estimate this and add to the attributes
                if 'hv' not in data_in.attrs:  # If no hv specified
                    add_estimated_hv(data_in)  # Estimate this and add to the attributes
                elif data_in.attrs['hv'] == None:  # If no hv specified
                    add_estimated_hv(data_in)  # Estimate this and add to the attributes

                # Apply Fermi level correction of data
                data_out = apply_fermi(data_in)
            elif kwargs['convert'] == 'k':  # Call to convert to k only
                kwargs['KE2BE'] = False  # Set flag to say that energy conversion should not be applied
                
            elif kwargs['convert'] == 'q':  # Call to convert to q only
                pass

            else:
                err_str = 'Only valid specified conversion options are \'k\', \'q\' or \'BE\'; to convert both, do not include \'convert\' in the call.'
                raise Exception(err_str)

        if 'convert' not in kwargs or 'KE2BE' in kwargs:  # Call includes k-conversion
            if 'KE2BE' not in kwargs:  # If not already set to false, then KE2BE conversion is required; KE2BE should be made true
                kwargs['KE2BE'] = True
                if 'EF_correction' not in data_in.attrs:  # If no suitable Fermi correction specified
                    add_estimated_EF(data_in)  # Estimate this and add to the attributes
                elif data_in.attrs['EF_correction'] == None:  # If no suitable Fermi correction specified
                    add_estimated_EF(data_in)  # Estimate this and add to the attributes
                if 'hv' not in data_in.attrs:  # If no hv specified
                    add_estimated_hv(data_in)  # Estimate this and add to the attributes
                elif data_in.attrs['hv'] == None:  # If no hv specified
                    add_estimated_hv(data_in)  # Estimate this and add to the attributes

            # Call correct k-conv function based on type of data
            if 'polar' in data_in.dims or 'tilt' in data_in.dims or 'defl_perp' in data_in.dims or 'ana_polar' in data_in.dims:  # Currently supported angular mapping dimensions
                data_out = kFS_conv(data_in, **kwargs)  # Fermi surface k-conversion
            
            else:  # Do k-space conversion just for theta_par / E dimensions
                data_out = k_conv(data_in, **kwargs)
        
        # The case for q conversion
        elif kwargs['convert'] == 'q' :
            if 'BE' in kwargs:  # If not already set to false, then KE2BE conversion is required; KE2BE should be made true
                if kwargs['BE'] == True:
                    kwargs['KE2BE'] = True
                    if 'EF_correction' not in data_in.attrs:  # If no suitable Fermi correction specified
                        add_estimated_EF(data_in)  # Estimate this and add to the attributes
                    elif data_in.attrs['EF_correction'] == None:  # If no suitable Fermi correction specified
                        add_estimated_EF(data_in)  # Estimate this and add to the attributes
                    if 'hv' not in data_in.attrs:  # If no hv specified
                        add_estimated_hv(data_in)  # Estimate this and add to the attributes
                    elif data_in.attrs['hv'] == None:  # If no hv specified
                        add_estimated_hv(data_in)  # Estimate this and add to the attributes
                    
                    data_out = q_conv(data_in, **kwargs)
                        
            else:
                kwargs['KE2BE'] = False
                data_out = q_conv(data_in, **kwargs)

    # Clean up the memory
    gc.collect()

    return data_out


# Helper function to estimate EF from the data itself if missing, and add that to the attributes
def add_estimated_EF(data):
    ''' Function to estimate the Fermi energy from an xarray, and add that to the EF_correction attribute.

    Input:
        data - the relevant data (xarray)

    Returns:
        nothing; EF_correction attribute is updated in data  '''
    # Set up warnings display and formatting
    warnings.simplefilter('always', UserWarning)  # Give warnings every time, even on function re-run
    warnings.formatwarning = warning_simple  # Formatting of warnings for peaks user errors

    # Estimate Fermi level and add that to the data attributes
    try:
        EF_est = float(estimate_EF(data))  # Try to fit the Fermi level from the data
    except:
        EF_est = float(estimate_EF(data, fit=False))  # If the fit fails, estimate this simply from the first (negative) peak in the derivative
    data.attrs['EF_correction'] = EF_est

    hist_str = 'EF_correction attribute set from automatic estimation to ' + str(EF_est) + ' eV. NB may not be very accurate; check this carefully.'
    update_hist(data, hist_str)

    warning_str = 'Fermi level set from automatic estimation to ' + str(EF_est) + ' eV. NB may not be very accurate; check this carefully. To set specific correction, write suitable \'EF_correction\' attribute to data, or call this function with argument \'EF_correction = ##\' where ## is a suitable correction.'
    warnings.warn(warning_str)

    # Reset warnings display and formatting
    warnings.simplefilter('once', UserWarning)  # Give warnings first time only
    warnings.formatwarning = warning_standard  # Formatting of warnings for peaks user errors

# Helper function to estimate hv from the data itself if missing, and add that to the attributes
def add_estimated_hv(data):
    ''' Function to estimate the photon energy from an xarray, and add that to the hv attribute.

        Input:
            data - the relevant data (xarray)

        Returns:
            nothing; hv attribute is updated in data  '''

    # Set up warnings display and formatting
    warnings.simplefilter('always', UserWarning)  # Give warnings every time, even on function re-run
    warnings.formatwarning = warning_simple  # Formatting of warnings for peaks user errors

    # Estimate Fermi level if not specified
    EF_est = None
    if 'EF_correction' in data.attrs:
        if 'EF_correction' != None:
            EF_est = data.attrs['EF_correction']
    if EF_est == None:  # EF not specified
        try:
            EF_est = float(estimate_EF(data))  # Try to fit the Fermi level from the data
        except:
            EF_est = float(estimate_EF(data, fit=False))  # If the fit fails, estimate this simply from the first (negative) peak in the derivative

    if EF_est > 16 and EF_est < 17.5:  # Likely to be He-I
        hv_est = 21.2182
        hv_est_str = str(hv_est) + ' eV (He I).'
    elif EF_est > 1 and EF_est < 2:  # Likely to be 6.05 eV laser
        hv_est = 6.05
        hv_est_str = str(hv_est) + ' eV (6 eV laser).'
    elif EF_est > 6 and EF_est < 7:  # Likely to be 11 eV laser
        hv_est = 10.897
        hv_est_str = str(hv_est) + ' eV (11 eV laser).'
    else:  # Estimate from EF_est
        hv_est = EF_est + 4.4  # Taking 4.4 eV as a reasonable work function
        hv_est_str = str(hv_est) + ' eV.'

    data.attrs['hv'] = hv_est

    hist_str = 'Photon energy set from automatic estimation to ' + hv_est_str
    update_hist(data, hist_str)

    warning_str = hist_str + ' To set specific correction, write suitable \'hv\' attribute to data.'
    warnings.warn(warning_str)

    # Reset warnings display and formatting
    warnings.simplefilter('once', UserWarning)  # Give warnings first time only
    warnings.formatwarning = warning_standard  # Formatting of warnings for peaks user errors
    
def make_hv_scan(dispersions):
    ''' Function to combine multiple dispersions scans taken at different hv, into a single hv scan xarray

        Input:
            dispersions - list of the dipsersions taken at different hv (list of xarrays)

        Returns:
            data - resulting hv scan (xarray)  '''
    
    #get photon energies and sort them into increasing order
    hv = []
    for i in range(len(dispersions)):
        hv.append([float(dispersions[i].attrs['hv']),i])
    hv.sort()
    hv_values = np.array(hv)[:,0]
    hv_indexes = np.array(hv)[:,1]
    
    #Extract the spectrum, kinetic energies and theta_par values
    theta_par_values = dispersions[0].coords['theta_par'].data
    spectrum = []
    KE_values = []
    for index in hv_indexes:
        dispersion = dispersions[int(index)]
        spectrum.append(dispersion.data)
        hv.append(float(dispersion.attrs['hv']))
        KE_values.append(dispersion.coords['eV'].data)

    KE0 = np.array(KE_values)[:,0] #Determine max of kinetic energy scan as a function of photon energy
    KE_delta = KE0 - KE0[0] #Change in KE value of detector as a function of hv

    #create a DataArray with the measured hv map
    #eV coordinate kept as that of first scan, with then the relative shift in KE from scan to scan recorded as a separate (non-dimenison) coordinate
    data = xr.DataArray(spectrum, dims=("hv","theta_par","eV"), coords={"hv": hv_values, "theta_par": theta_par_values, "eV": KE_values[0]})
    data.coords['theta_par'].attrs = {'units' : 'deg'}
    data.coords['hv'].attrs = {'units': 'eV'}  # Set units
    #Add KE_delta coordinate
    data.coords['KE_delta'] = ('hv', KE_delta)
    data.coords['KE_delta'].attrs = {'units': 'eV'}  # Set units

    #add metadata
    dispersion1 = dispersions[0].copy(deep=True)
    data.attrs = dispersion1.attrs
    data.name = 'hv_scan'
    data.attrs['scan_name'] = 'hv_scan'
    data.attrs['scan_type'] = 'hv scan'
    hv_first = np.round((hv_values[0]),1)
    hv_last = np.round((hv_values[-1]),1)
    hv_step = np.round((hv_last-hv_first)/(len(hv_values)-1),1)
    data.attrs['hv'] = str(hv_first) + ':' + str(hv_last) + ' ('+ str(hv_step) + ')'
    update_hist(data, 'Combined single dispersions into a hv scan')
    
    #Give a notification that this is how the data is stored
    warnings.simplefilter('always', UserWarning)  # Give warnings every time, even on function re-run
    warnings.formatwarning = warning_simple  # Formatting of warnings for peaks user errors
    warn = "Note: single xarray dataarray returned. The kinetic energy coordinate is that of the first scan; corresponding offsets for successive scans are included in the KE_delta coordinate. Run .pipe(disp_from_hv, hv) where hv is the relevant photon energy to extract a dispersion with the proper KE scaling for that photon energy."
    warnings.warn(warn) 
    
    return data




