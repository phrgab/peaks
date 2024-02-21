"""Classes to store data- or beamline-specific loading options.

"""

# Phil King 13/06/2021
# Brendan Edwards 21/02/2024


class file(object):
    """Class used to store and print user-defined file paths (optionally including some of the file name for
    disambiguation e.g. path_to_data/i05-123), file extensions, and location (e.g. beamline). Note: multiple file paths
    or extensions can be defined as lists to allow for the use of data saved at different folders, with different
    extensions.

    Examples
    ------------
    Example usage is as follows::

        from peaks import *

        # Define file paths
        file.path = ['sample1/i05-1-12', 'sample2/i05-1-12']

        # Define file extensions
        file.ext = ['nxs', 'zip']

        # Define location
        file.loc = ['Diamond I05-nano']

        # Print file options
        file()

    """

    def __init__(self):
        """This function simply prints the current file variables.

        """

        # Print the currently supported locations
        print('file.path: {path}'.format(path=self.path))
        print('file.ext: {ext}'.format(ext=self.ext))
        print('file.loc: {loc}'.format(loc=self.loc))

    # Initialise variables as None
    path = None  # Path(s) to files
    ext = None  # Extension(s) of files
    loc = None  # Location where files were measured


class loc_opts(object):
    """Class used to store and print the currently supported locations (typically beamlines).

    Examples
    ------------
    Example usage is as follows::

        from peaks import *

        # Print currently supported locations
        loc_opts()

    """

    def __init__(self):
        """This function simply prints the currently supported locations when loc_opts() is called.

        """

        # Print the currently supported locations
        print(self.locs)

    # List to store the currently supported locations
    locs = ['ALBA LOREA',
            'CLF Artemis',
            'Diamond I05-HR',
            'Diamond I05-nano',
            'Elettra APE',
            'MAX IV Bloch',
            'MAX IV Bloch-spin',
            'SOLEIL CASSIOPEE',
            'StA-MBS',
            'StA-Phoibos',
            'StA-Bruker',
            'StA-LEED',
            'StA-RHEED',
            'Structure',
            'ibw',
            'NetCDF',
            ]


class _BL_angles(object):
    """Class used to store definitions of relative angle signs between manipulator and detector for different beamlines.
    Useful in functions involving angle conversions, merging and normal emission extractions.

    Examples
    ------------
    Example usage is as follows::

        from peaks.core.fileIO.fileIO_opts import _BL_angles

        # Get the name of the tilt coordinate at Diamond I05-nano
        tilt_name = _BL_angles.angles['Diamond I05-nano']['tilt_name']

    """

    # Create angles dictionary to store information in
    angles = {}

    # Default values for unknown beamlines
    angles['default'] = {}
    angles['default']['ana_type'] = 'Ip'
    angles['default']['theta_par'] = -1
    angles['default']['defl_par'] = 1
    angles['default']['defl_perp'] = 1
    angles['default']['polar'] = 1
    angles['default']['tilt'] = 1
    angles['default']['azi'] = -1
    angles['default']['ana_polar'] = 0
    angles['default']['polar_name'] = 'polar'
    angles['default']['tilt_name'] = 'tilt'
    angles['default']['azi_name'] = 'azi'
    angles['default']['defl_par_name'] = 'Defl_par'
    angles['default']['defl_perp_name'] = 'Defl_perp'
    angles['default']['x1 name'] = 'x1'
    angles['default']['x2 name'] = 'x2'
    angles['default']['x3 name'] = 'x3'

    # Values for ALBA LOREA
    angles['ALBA LOREA'] = {}
    angles['ALBA LOREA']['ana_type'] = 'IIp'
    angles['ALBA LOREA']['theta_par'] = -1
    angles['ALBA LOREA']['defl_par'] = 1
    angles['ALBA LOREA']['defl_perp'] = 1
    angles['ALBA LOREA']['polar'] = 1
    angles['ALBA LOREA']['tilt'] = 1
    angles['ALBA LOREA']['azi'] = -1
    angles['ALBA LOREA']['ana_polar'] = 0
    angles['ALBA LOREA']['polar_name'] = 'sapolar'
    angles['ALBA LOREA']['tilt_name'] = 'satilt'
    angles['ALBA LOREA']['azi_name'] = 'saazimuth'
    angles['ALBA LOREA']['defl_par_name'] = 'defl_x'
    angles['ALBA LOREA']['defl_perp_name'] = 'defl_y'
    angles['ALBA LOREA']['x1 name'] = 'sax'
    angles['ALBA LOREA']['x2 name'] = 'saz'
    angles['ALBA LOREA']['x3 name'] = 'say'

    # Values for CLF Artemis
    angles['CLF Artemis'] = {}
    angles['CLF Artemis']['ana_type'] = 'II'
    angles['CLF Artemis']['theta_par'] = -1
    angles['CLF Artemis']['defl_par'] = 0
    angles['CLF Artemis']['defl_perp'] = 0
    angles['CLF Artemis']['polar'] = 1
    angles['CLF Artemis']['tilt'] = 1
    angles['CLF Artemis']['azi'] = -1
    angles['CLF Artemis']['ana_polar'] = 0
    angles['CLF Artemis']['polar_name'] = 'theta'
    angles['CLF Artemis']['tilt_name'] = None
    angles['CLF Artemis']['azi_name'] = 'azi'
    angles['CLF Artemis']['defl_par_name'] = None
    angles['CLF Artemis']['defl_perp_name'] = None
    angles['CLF Artemis']['x1 name'] = 'X'
    angles['CLF Artemis']['x2 name'] = 'Z'
    angles['CLF Artemis']['x3 name'] = 'Y'

    # Values for Diamond I05-HR
    angles['Diamond I05-HR'] = {}
    angles['Diamond I05-HR']['ana_type'] = 'Ip'
    angles['Diamond I05-HR']['theta_par'] = -1
    angles['Diamond I05-HR']['defl_par'] = 1
    angles['Diamond I05-HR']['defl_perp'] = 1
    angles['Diamond I05-HR']['polar'] = 1
    angles['Diamond I05-HR']['tilt'] = 1
    angles['Diamond I05-HR']['azi'] = -1
    angles['Diamond I05-HR']['ana_polar'] = 0
    angles['Diamond I05-HR']['polar_name'] = 'sapolar'
    angles['Diamond I05-HR']['tilt_name'] = 'satilt'
    angles['Diamond I05-HR']['azi_name'] = 'saazimuth'
    angles['Diamond I05-HR']['defl_par_name'] = None
    angles['Diamond I05-HR']['defl_perp_name'] = 'deflector_x'
    angles['Diamond I05-HR']['x1 name'] = 'sax'
    angles['Diamond I05-HR']['x2 name'] = 'saz'
    angles['Diamond I05-HR']['x3 name'] = 'say'

    # Values for Diamond I05-nano
    angles['Diamond I05-nano'] = {}
    angles['Diamond I05-nano']['ana_type'] = 'Ip'
    angles['Diamond I05-nano']['theta_par'] = -1
    angles['Diamond I05-nano']['defl_par'] = 1
    angles['Diamond I05-nano']['defl_perp'] = 1
    angles['Diamond I05-nano']['polar'] = 1
    angles['Diamond I05-nano']['tilt'] = 1
    angles['Diamond I05-nano']['azi'] = -1
    angles['Diamond I05-nano']['ana_polar'] = 1
    angles['Diamond I05-nano']['polar_name'] = 'smpolar'
    angles['Diamond I05-nano']['tilt_name'] = None
    angles['Diamond I05-nano']['azi_name'] = 'smazimuth'
    angles['Diamond I05-nano']['defl_par_name'] = 'theta_x'
    angles['Diamond I05-nano']['defl_perp_name'] = 'theta_y'
    angles['Diamond I05-nano']['x1 name'] = 'smx'
    angles['Diamond I05-nano']['x2 name'] = 'smy'
    angles['Diamond I05-nano']['x3 name'] = 'smz'

    # Values for Elettra APE
    angles['Elettra APE'] = {}
    angles['Elettra APE']['ana_type'] = 'Ip'
    angles['Elettra APE']['theta_par'] = -1
    angles['Elettra APE']['defl_par'] = 1
    angles['Elettra APE']['defl_perp'] = 1
    angles['Elettra APE']['polar'] = 1
    angles['Elettra APE']['tilt'] = 1
    angles['Elettra APE']['azi'] = -1
    angles['Elettra APE']['ana_polar'] = 0
    angles['Elettra APE']['polar_name'] = 'polar'
    angles['Elettra APE']['tilt_name'] = 'tilt'
    angles['Elettra APE']['azi_name'] = None
    angles['Elettra APE']['defl_par_name'] = 'theta_x'
    angles['Elettra APE']['defl_perp_name'] = 'theta_y'
    angles['Elettra APE']['x1 name'] = 'temp'
    angles['Elettra APE']['x2 name'] = 'temp'
    angles['Elettra APE']['x3 name'] = 'temp'

    # Values for MAX IV Bloch
    angles['MAX IV Bloch'] = {}
    angles['MAX IV Bloch']['ana_type'] = 'Ip'
    angles['MAX IV Bloch']['theta_par'] = -1
    angles['MAX IV Bloch']['defl_par'] = 1
    angles['MAX IV Bloch']['defl_perp'] = 1
    angles['MAX IV Bloch']['polar'] = 1
    angles['MAX IV Bloch']['tilt'] = 1
    angles['MAX IV Bloch']['azi'] = -1
    angles['MAX IV Bloch']['ana_polar'] = 0
    angles['MAX IV Bloch']['polar_name'] = 'P'
    angles['MAX IV Bloch']['tilt_name'] = 'T'
    angles['MAX IV Bloch']['azi_name'] = 'A'
    angles['MAX IV Bloch']['defl_par_name'] = 'theta_x'
    angles['MAX IV Bloch']['defl_perp_name'] = 'theta_y'
    angles['MAX IV Bloch']['x1 name'] = 'X'
    angles['MAX IV Bloch']['x2 name'] = 'Y'
    angles['MAX IV Bloch']['x3 name'] = 'Z'

    # Values for MAX IV Bloch-spin
    angles['MAX IV Bloch-spin'] = {}
    angles['MAX IV Bloch-spin']['ana_type'] = 'Ip'
    angles['MAX IV Bloch-spin']['theta_par'] = -1
    angles['MAX IV Bloch-spin']['defl_par'] = 1
    angles['MAX IV Bloch-spin']['defl_perp'] = 1
    angles['MAX IV Bloch-spin']['polar'] = 1
    angles['MAX IV Bloch-spin']['tilt'] = 1
    angles['MAX IV Bloch-spin']['azi'] = -1
    angles['MAX IV Bloch-spin']['ana_polar'] = 0
    angles['MAX IV Bloch-spin']['polar_name'] = 'P'
    angles['MAX IV Bloch-spin']['tilt_name'] = 'T'
    angles['MAX IV Bloch-spin']['azi_name'] = 'A'
    angles['MAX IV Bloch-spin']['defl_par_name'] = 'theta_x'
    angles['MAX IV Bloch-spin']['defl_perp_name'] = 'theta_y'
    angles['MAX IV Bloch-spin']['x1 name'] = 'X'
    angles['MAX IV Bloch-spin']['x2 name'] = 'Y'
    angles['MAX IV Bloch-spin']['x3 name'] = 'Z'

    # Values for SOLEIL CASSIOPEE
    angles['SOLEIL CASSIOPEE'] = {}
    angles['SOLEIL CASSIOPEE']['ana_type'] = 'I'
    angles['SOLEIL CASSIOPEE']['theta_par'] = -1
    angles['SOLEIL CASSIOPEE']['defl_par'] = 1
    angles['SOLEIL CASSIOPEE']['defl_perp'] = 1
    angles['SOLEIL CASSIOPEE']['polar'] = 1
    angles['SOLEIL CASSIOPEE']['tilt'] = 1
    angles['SOLEIL CASSIOPEE']['azi'] = -1
    angles['SOLEIL CASSIOPEE']['ana_polar'] = 0
    angles['SOLEIL CASSIOPEE']['polar_name'] = 'theta'
    angles['SOLEIL CASSIOPEE']['tilt_name'] = 'tilt'
    angles['SOLEIL CASSIOPEE']['azi_name'] = 'phi'
    angles['SOLEIL CASSIOPEE']['defl_par_name'] = None
    angles['SOLEIL CASSIOPEE']['defl_perp_name'] = None
    angles['SOLEIL CASSIOPEE']['x1 name'] = 'x'
    angles['SOLEIL CASSIOPEE']['x2 name'] = 'z'
    angles['SOLEIL CASSIOPEE']['x3 name'] = 'y'

    # Values for StA-MBS
    angles['StA-MBS'] = {}
    angles['StA-MBS']['ana_type'] = 'IIp'
    angles['StA-MBS']['theta_par'] = -1
    angles['StA-MBS']['defl_par'] = 1
    angles['StA-MBS']['defl_perp'] = 1
    angles['StA-MBS']['polar'] = 1
    angles['StA-MBS']['tilt'] = 1
    angles['StA-MBS']['azi'] = 1
    angles['StA-MBS']['ana_polar'] = 0
    angles['StA-MBS']['polar_name'] = 'polar'
    angles['StA-MBS']['tilt_name'] = 'tilt'
    angles['StA-MBS']['azi_name'] = 'azi'
    angles['StA-MBS']['defl_par_name'] = 'defl_y'
    angles['StA-MBS']['defl_perp_name'] = 'defl_x'
    angles['StA-MBS']['x1 name'] = 'temp'
    angles['StA-MBS']['x2 name'] = 'temp'
    angles['StA-MBS']['x3 name'] = 'temp'

    # Values for StA-Phoibos
    angles['StA-Phoibos'] = {}
    angles['StA-Phoibos']['ana_type'] = 'II'
    angles['StA-Phoibos']['theta_par'] = -1
    angles['StA-Phoibos']['defl_par'] = 0
    angles['StA-Phoibos']['defl_perp'] = 0
    angles['StA-Phoibos']['polar'] = 1
    angles['StA-Phoibos']['tilt'] = 1
    angles['StA-Phoibos']['azi'] = 1
    angles['StA-Phoibos']['ana_polar'] = 0
    angles['StA-Phoibos']['polar_name'] = 'Theta'
    angles['StA-Phoibos']['tilt_name'] = 'Phi'
    angles['StA-Phoibos']['azi_name'] = 'Azi'
    angles['StA-Phoibos']['defl_par_name'] = None
    angles['StA-Phoibos']['defl_perp_name'] = None
    angles['StA-Phoibos']['x1 name'] = 'temp'
    angles['StA-Phoibos']['x2 name'] = 'temp'
    angles['StA-Phoibos']['x3 name'] = 'temp'


class _coords(object):
    """Class used to store the units for different coordinates.

    Examples
    ------------
    Example usage is as follows::

        from peaks.core.fileIO.fileIO_opts import _coords

        # Get the units associated with x1 coordinates
        x1_units = _coords.units['x1']

    """

    # Dictionary to store the units for different coordinates
    units = {'theta_par': 'deg', 'polar': 'deg', 'tilt': 'deg', 'ana_polar': 'deg', 'defl_perp': 'deg', 'x1': 'um',
             'x2': 'um', 'x3': 'um', 'x': 'um', 'y': 'um', 'z': 'um', 'location': 'um', 'hv': 'eV', 'KE_delta': 'eV'}
