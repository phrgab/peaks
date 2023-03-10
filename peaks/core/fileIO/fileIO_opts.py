# Beamline angle conventions
# Phil King 13/06/2021

# Currently supported locations for file loading
loc_opts = ['ALBA LOREA',
            'Artemis',
            'Bruker XRD',
            'Diamond I05-HR',
            'Diamond I05-nano',
            'Elettra APE',
            'MAX IV Bloch',
            'netCDF file',
            'SOLEIL CASSIOPEE',
            'St Andrews - Phoibos',
            'St Andrews - MBS',
            ]

class fname:
    '''
    Define path to folder containing data in .path (optionally including some of the file name for disambiguation).
    Define allowed file extension (or list of allowed extensions).
    '''

    path = None
    ext = None

class BL_angles:
    '''Definitions of relative angle signs between manipulator and detector, useful in angle conversions, merging
    and extracting normal emissions'''
    angles = {}

    # Default values
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

    # ALBA LOREA
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

    # Artemis
    angles['Artemis'] = {}
    angles['Artemis']['ana_type'] = 'II'
    angles['Artemis']['theta_par'] = -1
    angles['Artemis']['defl_par'] = 0
    angles['Artemis']['defl_perp'] = 0
    angles['Artemis']['polar'] = 1
    angles['Artemis']['tilt'] = 1
    angles['Artemis']['azi'] = -1
    angles['Artemis']['ana_polar'] = 0
    angles['Artemis']['polar_name'] = 'theta'
    angles['Artemis']['tilt_name'] = 'None'
    angles['Artemis']['azi_name'] = 'azi'
    angles['Artemis']['defl_par_name'] = 'None'
    angles['Artemis']['defl_perp_name'] = 'None'
    angles['Artemis']['x1 name'] = 'X'
    angles['Artemis']['x2 name'] = 'Z'
    angles['Artemis']['x3 name'] = 'Y'

    # Diamond I05-HR
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
    angles['Diamond I05-HR']['defl_par_name'] = 'None'
    angles['Diamond I05-HR']['defl_perp_name'] = 'deflector_x'
    angles['Diamond I05-HR']['x1 name'] = 'sax'
    angles['Diamond I05-HR']['x2 name'] = 'saz'
    angles['Diamond I05-HR']['x3 name'] = 'say'

    # Diamond I05-nano
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
    angles['Diamond I05-nano']['tilt_name'] = 'None'
    angles['Diamond I05-nano']['azi_name'] = 'smazimuth'
    angles['Diamond I05-nano']['defl_par_name'] = 'theta_x'
    angles['Diamond I05-nano']['defl_perp_name'] = 'theta_y'
    angles['Diamond I05-nano']['x1 name'] = 'smx'
    angles['Diamond I05-nano']['x2 name'] = 'smy'
    angles['Diamond I05-nano']['x3 name'] = 'smz'

    # St Andrews - Phoibos
    angles['St Andrews - Phoibos'] = {}
    angles['St Andrews - Phoibos']['ana_type'] = 'II'
    angles['St Andrews - Phoibos']['theta_par'] = -1
    angles['St Andrews - Phoibos']['defl_par'] = 0
    angles['St Andrews - Phoibos']['defl_perp'] = 0
    angles['St Andrews - Phoibos']['polar'] = 1
    angles['St Andrews - Phoibos']['tilt'] = 1
    angles['St Andrews - Phoibos']['azi'] = 1
    angles['St Andrews - Phoibos']['ana_polar'] = 0
    angles['St Andrews - Phoibos']['polar_name'] = 'Theta'
    angles['St Andrews - Phoibos']['tilt_name'] = 'Phi'
    angles['St Andrews - Phoibos']['azi_name'] = 'Azi'
    angles['St Andrews - Phoibos']['defl_par_name'] = 'None'
    angles['St Andrews - Phoibos']['defl_perp_name'] = 'None'
    angles['St Andrews - Phoibos']['x1 name'] = 'temp'
    angles['St Andrews - Phoibos']['x2 name'] = 'temp'
    angles['St Andrews - Phoibos']['x3 name'] = 'temp'

    # St Andrews - MBS
    angles['St Andrews - MBS'] = {}
    angles['St Andrews - MBS']['ana_type'] = 'IIp'
    angles['St Andrews - MBS']['theta_par'] = -1
    angles['St Andrews - MBS']['defl_par'] = 1
    angles['St Andrews - MBS']['defl_perp'] = 1
    angles['St Andrews - MBS']['polar'] = 1
    angles['St Andrews - MBS']['tilt'] = 1
    angles['St Andrews - MBS']['azi'] = 1
    angles['St Andrews - MBS']['ana_polar'] = 0
    angles['St Andrews - MBS']['polar_name'] = 'polar'
    angles['St Andrews - MBS']['tilt_name'] = 'tilt'
    angles['St Andrews - MBS']['azi_name'] = 'azi'
    angles['St Andrews - MBS']['defl_par_name'] = 'defl_y'
    angles['St Andrews - MBS']['defl_perp_name'] = 'defl_x'
    angles['St Andrews - MBS']['x1 name'] = 'temp'
    angles['St Andrews - MBS']['x2 name'] = 'temp'
    angles['St Andrews - MBS']['x3 name'] = 'temp'

    # Elettra APE
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
    angles['Elettra APE']['azi_name'] = 'None'
    angles['Elettra APE']['defl_par_name'] = 'theta_x'
    angles['Elettra APE']['defl_perp_name'] = 'theta_y'
    angles['Elettra APE']['x1 name'] = 'temp'
    angles['Elettra APE']['x2 name'] = 'temp'
    angles['Elettra APE']['x3 name'] = 'temp'

    # MAX IV Bloch
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

    # SOLEIL CASSIOPEE
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
    angles['SOLEIL CASSIOPEE']['defl_par_name'] = 'None'
    angles['SOLEIL CASSIOPEE']['defl_perp_name'] = 'None'
    angles['SOLEIL CASSIOPEE']['x1 name'] = 'x'
    angles['SOLEIL CASSIOPEE']['x2 name'] = 'z'
    angles['SOLEIL CASSIOPEE']['x3 name'] = 'y'



