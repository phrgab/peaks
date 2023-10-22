# Feed data from xarray to the FS align panel
# Phil King (based on code from Edgar Abarca Morales)
# 19/6/21

from peaks.core.utils.estimate_EF import estimate_EF
from peaks.core.utils.get_angles import get_angles
from peaks.core.fileIO.fileIO_opts import BL_angles

# Create align panel input dictionary from xarray data
def feedfromLoaders(a):

    # If data is 3D
    if len(a.dims) == 3:  # For FS map data
        # Work out mapping dimension
        for i in a.dims:
            if i != 'theta_par' and i != 'eV':
                mapping_dim = i

        # Check the dispersion has the correct axis ordering (this will be important for interpreting the raw np array below)
        a = a.transpose(mapping_dim, 'theta_par', 'eV')

        # Reindex array to ensure it has increasing angles with array pixel
        if a.theta_par.data[1] - a.theta_par.data[0] < 0:  # Array currently has decreasing order
            a = a.reindex(theta_par=a.theta_par[::-1])
        if a[mapping_dim].data[1] - a[mapping_dim].data[0] < 0:  # Array currently has decreasing order
            a = a.reindex({mapping_dim: a[mapping_dim][::-1]})
        if a.eV.data[1] - a.eV.data[0] < 0:  # Array currently has decreasing order
            a = a.reindex(eV=a.eV[::-1])

        # Extract (x,y,z) dimensions
        xdim = a.dims[0] # mapping angle
        ydim = a.dims[1] # theta_par
        zdim = a.dims[2] # Energy

        # Define Int dimension
        Idim='counts'

        # Extract x units
        if 'units' in a.coords[xdim].attrs:
            xuts=a.coords[xdim].attrs['units']
        else:
            xuts=''

        # Extract y units
        if 'units' in a.coords[ydim].attrs:
            yuts=a.coords[ydim].attrs['units']
        else:
            yuts=''

        # Extract z units
        if 'units' in a.coords[zdim].attrs:
            zuts = a.coords[zdim].attrs['units']
        else:
            zuts = ''

        # Define Int units
        Iuts = None

        # Extract data
        x = a.coords[xdim].data
        y = a.coords[ydim].data
        z = a.coords[zdim].data
        Int = a.data

    elif len(a.dims) == 2:  # For dispersion data
        # Check the dispersion has the correct axis ordering (this will be important for interpreting the raw np array below)
        a = a.transpose('theta_par', 'eV')

        # Reindex array to ensure it has increasing angles with array pixel
        if a.theta_par.data[1] - a.theta_par.data[0] < 0:  # Array currently has decreasing order
            a = a.reindex(theta_par=a.theta_par[::-1])
        if a.eV.data[1] - a.eV.data[0] < 0:  # Array currently has decreasing order
            a = a.reindex(eV=a.eV[::-1])

        # Extract (x,y) dimensions
        xdim = a.dims[0]  # theta_par
        ydim = a.dims[1]  # Energy

        # Define Int dimension
        Idim = 'counts'

        # Extract x units
        if 'units' in a.coords[xdim].attrs:
            xuts = a.coords[xdim].attrs['units']
        else:
            xuts = ''

        # Extract y units
        if 'units' in a.coords[ydim].attrs:
            yuts = a.coords[ydim].attrs['units']
        else:
            yuts = ''

        # Define Int units
        Iuts = None

        # Extract data
        x = a.coords[xdim].data
        y = a.coords[ydim].data
        Int = a.data

    # Extract Fermi level
    if a.attrs['eV_type'] == 'kinetic':
        if 'EF_correction' in a.attrs:
            if a.attrs['EF_correction'] != None:
                EF = a.attrs['EF_correction']
            else:
                EF = estimate_EF(a, fit=False)
        else:
            EF = estimate_EF(a, fit=False)
    else:
        EF = 0

    # Extract current manipulator positions and axis labels
    angles = get_angles(a, warn_flag=[])

    # Get beamline conventions
    if a.attrs['beamline'] in BL_angles.angles:
        BL_conventions = BL_angles.angles[a.attrs['beamline']]
    else:
        BL_conventions = BL_angles.angles['default']

    if len(a.dims) == 3:  # For FS map data
        data_dict={'x':x, 'y':y, 'z':z, 'Int':Int, 'xdim':xdim, 'ydim':ydim, 'zdim':zdim, 'Idim':Idim, 'xuts':xuts, 'yuts':yuts, 'zuts':zuts, 'Iuts':Iuts, 'EF':EF, 'angles': angles, 'BL_conv': BL_conventions}
    elif len(a.dims) == 2:  # For dispersion data
        data_dict = {'x': x, 'y': y, 'Int': Int, 'xdim': xdim, 'ydim': ydim, 'Idim': Idim, 'xuts': xuts, 'yuts': yuts, 'Iuts': Iuts, 'EF': EF, 'angles': angles, 'BL_conv': BL_conventions}

    return data_dict
