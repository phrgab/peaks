import warnings
from peaks.core.utils.misc import warning_simple, warning_standard
from peaks.core.fileIO.fileIO_opts import BL_angles

def get_angles(data,warn_flag=['polar','tilt','azi','norm_polar','norm_tilt','norm_azi'],defaults={}):
    '''Function to extract the relevant angles from the dimensions, coordinates and attributes of a dataarray, setting default values if no angles are present.

    Inputs:
        - data: data to extract angles from (xarray)
        - warn_flag: list of axes to give a warning for if not in data
        - defaults: dictionary of default values, for user specified defaults

    Returns:
        - angles: dictionary of angles in both peaks notation and in convention used for k-space transformations'''

    # Set up warnings display and formatting
    warnings.simplefilter('always', UserWarning)  # Give warnings every time, even on function re-run
    warnings.formatwarning = warning_simple  # Formatting of warnings for peaks user errors

    # Get manipulator angles, including warnings if not present
    axes = ['polar', 'tilt', 'azi', 'norm_polar', 'norm_tilt', 'norm_azi', 'theta_par', 'ana_polar', 'defl_par', 'defl_perp']
    missing_axes = {}
    angles_out = {}

    # Norm azi should default to axis value if unspecified
    if 'azi' not in defaults:  # Check no specific default has been set for azi
        if 'azi' in data.attrs:  # Check that an azi axis exists
            if data.attrs['azi'] != None:  # Only want to set the default if azi has a set value
                defaults['norm_azi'] = data.attrs['azi']  # Set azi norm_default to existing azi value

    #Search the dataarray. The heirarchy is coordinates then attributes
    for i in axes:  # Iterate through the axes
        if i in data.coords.dims:  # If this is a coordinate, read the array direct from the coordinate
            axis_temp = data[i].data
        elif i in data.attrs:  # If this exists as an attribute catagory
            if data.attrs[i] != None:  # If there is an actual entry there
                axis_temp = data.attrs[i]  # Read the axis entry from the attributes
                # Sometimes this is a string, which isn't useable
                if type(axis_temp) == str:
                    try:
                        (float(axis_temp))  # Try converting to a float
                    except:  # If this doesn't work, give a warning
                        try:
                            axis_temp = defaults[i]  # See if there is anything specified for this in defaults
                        except:  # If not, take a default of zero
                            axis_temp = 0.
                        warning_str = i + ' could not be parsed (likely it was moving during the scan). From metadata, ' + i + '=' + data.attrs[i] + '. This has been set to ' + str(axis_temp) + ' here. To use a defined value, set this in the attributes of the original data with \'.attrs[' + i + '] = ##\' where ## is a float or an array of appropriate length.'
                        warnings.warn(warning_str)
                    try:
                        if type(defaults['norm_azi']) == str:  # Also need to reset the default for norm_azi so that is not a string
                            defaults['norm_azi'] = axis_temp
                    except:
                        pass
            else:  # No supplied angle, set a default
                try:
                    axis_temp = defaults[i]  # See if there is anything specified for this in defaults
                except:  # If not, take a default of zero
                    axis_temp = 0.
                missing_axes[i] = axis_temp  # Append used axis default to dictionary
        else:  # This axis wasn't included as either a dimension or an attribute - set defaults straight away
            try:
                axis_temp = defaults[i]  # See if there is anything specified for this in defaults
            except:  # If not, take a default of zero
                axis_temp = 0.
            missing_axes[i] = axis_temp  # Append used axis default to dictionary

        # Add axis to dictionary of output angles
        angles_out[i] = axis_temp

    # Filter out only those for warning about
    for i in list(missing_axes.keys()):  # Iterate through the missing axes
        if i not in warn_flag:  # Check if this is in the warn_flag list
            missing_axes.pop(i)  # If not, remove the axis from missing_axes (this is just used for the warning to the user)

    if len(missing_axes) != 0:
        warning_str = 'Some manipulator and/or normal emission values missing. These have been set to default values: ' + str(missing_axes)
        warnings.warn(warning_str)

    # Identify analyser type (nomenclature from Ishida and Shin, Rev. Sci. Instrum. 89 (2018) 043903)
    if data.attrs['ana_slit_angle'] == 90:  # Type I
        if 'defl_par' in data.coords.dims or 'defl_perp' in data.coords.dims:  # Deflector-type analyser
            ana_type = 'Ip'
        elif 'defl_par' in data.attrs and 'defl_perp' in data.attrs:
            if data.attrs['defl_par'] == None or data.attrs['defl_par'] == 0:
                if data.attrs['defl_par'] == None or data.attrs['defl_par'] == 0:
                    ana_type = 'I'
                else:
                    ana_type = 'Ip'
            else:
                ana_type = 'Ip'
        elif 'defl_par' in data.attrs:
            if data.attrs['defl_par'] == None or data.attrs['defl_par'] == 0:
                ana_type = 'I'
            else:
                ana_type = 'Ip'
        elif 'defl_perp' in data.attrs:
            if data.attrs['defl_perp'] == None or data.attrs['defl_perp'] == 0:
                ana_type = 'I'
            else:
                ana_type = 'Ip'
        else:
            ana_type = 'I'

    elif data.attrs['ana_slit_angle'] == 0:  # Type II
        if 'defl_par' in data.coords.dims or 'defl_perp' in data.coords.dims:  # Deflector-type analyser
            ana_type = 'IIp'
        elif 'defl_par' in data.attrs:
            if data.attrs['defl_par'] != None:
                ana_type = 'IIp'
            else:
                ana_type = 'II'
        elif 'defl_perp' in data.attrs:
            if data.attrs['defl_perp'] != None:
                ana_type = 'IIp'
            else:
                ana_type = 'II'
        else:
            ana_type = 'II'
    else:  # Some other type, not yet supported
        err_str = 'Analyser slit angle should be 0 (polar moves along slit) or 90 (tilt moves along slit). Analyser type not supported, or ama_slit_angle attribute not set. Set with \'.attrs[\'ana_slit_angle\'=##]\'.'
        raise Exception(err_str)

    angles_out['ana_type'] = ana_type

    # Get beamline
    BL = data.attrs['beamline']

    # Determine relevant angles using nomenclature from Ishida and Shin, Rev. Sci. Instrum. 89 (2018) 043903)
    if ana_type == 'I':  # Type I
        angles_out['alpha'] = angles_out['theta_par'] * BL_angles.angles[BL]['theta_par']
        angles_out['beta'] = ((angles_out['polar'] - angles_out['norm_polar'])*BL_angles.angles[BL]['polar']) + angles_out['ana_polar'] * BL_angles.angles[BL]['ana_polar']
        angles_out['delta'] = (angles_out['azi'] - angles_out['norm_azi']) * BL_angles.angles[BL]['azi'] * BL_angles.angles[BL]['theta_par']
        angles_out['xi'] = (angles_out['tilt'] - angles_out['norm_tilt']) * BL_angles.angles[BL]['tilt']   # Return array

    elif ana_type == 'II':  # Type II
        angles_out['alpha'] = angles_out['theta_par'] * BL_angles.angles[BL]['theta_par']
        angles_out['beta'] = (angles_out['tilt'] - angles_out['norm_tilt']) * BL_angles.angles[BL]['tilt']   # Return array
        angles_out['delta'] = (angles_out['azi'] - angles_out['norm_azi']) * BL_angles.angles[BL]['azi'] * BL_angles.angles[BL]['theta_par']
        angles_out['xi'] = ((angles_out['polar'] - angles_out['norm_polar']) * BL_angles.angles[BL]['polar']) + angles_out['ana_polar'] * BL_angles.angles[BL]['ana_polar']

    elif ana_type == 'Ip' or ana_type == 'IIp':  # Deflector-type
        angles_out['alpha'] = angles_out['theta_par']  * BL_angles.angles[BL]['theta_par'] + angles_out['defl_par'] * BL_angles.angles[BL]['defl_par']
        angles_out['beta'] = angles_out['defl_perp'] * BL_angles.angles[BL]['defl_perp']
        angles_out['delta'] = (angles_out['azi'] - angles_out['norm_azi']) * BL_angles.angles[BL]['azi'] * BL_angles.angles[BL]['theta_par']
        angles_out['xi'] = (angles_out['tilt'] - angles_out['norm_tilt']) * BL_angles.angles[BL]['tilt']   # Return array
        angles_out['chi'] = ((angles_out['polar'] - angles_out['norm_polar']) * BL_angles.angles[BL]['polar']) + angles_out['ana_polar'] * BL_angles.angles[BL]['ana_polar']

    # Reset warnings display and formatting
    warnings.simplefilter('once', UserWarning)  # Give warnings first time only
    warnings.formatwarning = warning_standard  # Formatting of warnings for peaks user errors

    return angles_out

def set_normals(data, **kwargs):
    '''Function to set normal emissions into the attributes of the xarray data array based on arguments passed in kwargs.

        Inputs:
            - data: data to write normal emission angles to (xarray)
            - **kwargs:
                - polar = float, to set polar angle
                - tilt = float, to set tilt angle
                - azi = float, to set azi angle
                - norm_polar = float, to set norm_polar
                - norm_tilt = float, to set norm_tilt
                - norm_azi = float, to set norm_azi
                - any other kwargs ignored

        Returns:
            Nothing, just overwrite attributes in data that is passed. '''

    if 'polar' in kwargs:
        data.attrs['polar'] = kwargs['polar']
    if 'tilt' in kwargs:
        data.attrs['tilt'] = kwargs['tilt']
    if 'azi' in kwargs:
        data.attrs['azi'] = kwargs['azi']
    if 'norm_polar' in kwargs:
        data.attrs['norm_polar'] = kwargs['norm_polar']
    if 'norm_tilt' in kwargs:
        data.attrs['norm_tilt'] = kwargs['norm_tilt']
    if 'norm_azi' in kwargs:
        data.attrs['norm_azi'] = kwargs['norm_azi']

    return