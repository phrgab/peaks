"""Functions to load data into DataArray format.

"""

# Brendan Edwards 28/06/2021
# Phil King 14/05/2022
# Brendan Edwards 14/02/2024

import os
import natsort
import h5py
import xarray as xr
import dask
import dask.array as da
from tqdm import tqdm
from peaks.core.utils.misc import analysis_warning
from peaks.core.fileIO.fileIO_opts import file, loc_opts, _coords
from peaks.core.fileIO.loaders.SES import _load_SES_metalines, _SES_find


def load(fname, lazy='auto', loc='auto', metadata=True, parallel=False):
    """Shortcut function to load_data where much of the file path can be set by the file global options:

        file.path : str, list
            Path (or list of paths) to folder(s) where data is stored, e.g. file.path = 'C:/User/Documents/i05-1-123'.

        file.ext : str, list
            Extension (or list of extensions) of data, e.g. file.ext = ['ibw', 'zip'].

        file.loc : str
            Location (typically a beamline) where data was acquired, e.g. file.loc = 'MAX IV Bloch' (call class
            loc_opts() to see currently supported locations). Note: setting this prevents any automatic location
            detection.

    Parameters
    ------------
    fname : str, list
        Either the full file name(s), or the remainder of the file name(s) not already specified in the file global
        options.

    lazy : str, Boolean (optional)
        Whether to load data in a lazily evaluated dask format. Set explicitly using True/False Boolean. Defaults
        to 'auto' where a file is only loaded in the dask format if its spectrum is above 500 MB.

    loc : str (optional)
        The name of the location (typically a beamline). Defaults to 'auto', where the location will be attempted to be
        automatically determined, unless a value is defined for file.loc. If loc is not the default 'auto' value, the
        parameter loc will take priority over the value in file.loc.

    metadata : Boolean (optional)
        Whether to attempt to load metadata into the attributes of the xr.DataArray. Defaults to True.

    parallel : Boolean (optional)
        Whether to load data in parallel when multiple files are being loaded. Only compatible with certain file types
        such as those based on the h5py format, e.g. nxs files. Takes priority over lazy, enforcing that all data is
        computed and loaded into memory. Defaults to False.

    Returns
    ------------
    loaded_data : xr.DataArray, xr.DataSet, list
        The loaded data.

    Examples
    ------------
    from peaks import *

    # Load data using a complete data path
    disp1 = load('C:/User/Documents/disp1.ibw')
    FM1 = load('C:/User/Documents/FM1.ibw')

    # Define global options in the class file to aid loading
    file.path = 'C:/User/Documents/Data/'
    file.ext = ['ibw', 'zip']

    # Load data without needing to define data path or extension (determined from global options defined in file)
    disp2 = load('disp2')
    FM2 = load('FM2')

    # Define global options in the class file to aid loading, and include a partial part of the scan name
    file.path = 'C:/User/Documents/Data/i05-1-123'
    file.ext = 'nxs'

    # Load data without needing to define data path or extension, nor the repeated part of the scan name (determined
    # from global options defined in file)
    disp3 = load(456)
    disp4 = load(457)

    # Load multiple files at once
    disps = load([456, 457, 458])

    # Still can load data using a complete data path, and global options defined in file will be ignored
    disp1 = load('C:/User/Documents/Data/disp1.ibw')

    # Load data in a lazily evaluated dask format
    disp1 = load('C:/User/Documents/Data/disp1.ibw', lazy=True)

    # Load data without metadata
    disp1 = load('C:/User/Documents/Data/disp1.ibw', metadata=False)

    # Load data file a pre-defined location (use if automatic location identification fails)
    disp1 = load('C:/User/Documents/Data/disp1.ibw', loc='MAX IV Bloch')

    # Alternatively could define location using global options. Here loc will be defined as 'MAX IV Bloch'
    file.loc = 'MAX IV Bloch'
    disp1 = load('C:/User/Documents/Data/disp1.ibw')

    """

    # Set placeholder file path and extension
    path = [None]
    ext = ['']

    # If file.path is defined, make a list of the inputted path(s)
    if file.path:
        if not isinstance(file.path, list):
            path = [str(file.path)]
        else:
            path = [str(item) for item in file.path]

    # If file.ext is defined, make a list of the inputted extension(s)
    if file.ext:
        if not isinstance(file.ext, list):
            ext = [str(file.ext)]
        else:
            ext = [str(item) for item in file.ext]

    # If the parameter loc is 'auto' and file.loc is defined (for a valid location defined in loc_opts.locs), update loc
    if loc == 'auto':
        if file.loc in loc_opts.locs:
            loc = file.loc

    # Ensure that fname is a list of strings
    if not isinstance(fname, list):
        fname = [str(fname)]
    else:
        fname = [str(item) for item in fname]

    # Obtain all possible file addresses
    possible_file_addresses = []
    # Loop through file names
    for file_name in fname:
        # Remove any extensions from fname, adding them to ext
        file_name, fname_ext = os.path.splitext(file_name)
        if fname_ext != '' and fname_ext[1:] not in ext:
            ext.append(fname_ext[1:])

        # Loop through file paths
        for file_path in path:
            # If file path is not None, add address file_path/file_name to possible_file_addresses
            if file_path:
                possible_file_addresses.append(file_path + file_name)
            # Add address file_name to valid_file_addresses
            possible_file_addresses.append(file_name)

    # Obtain all valid file addresses
    file_list = []
    # Loop through the non-duplicated addresses in possible_file_addresses
    for address in set(possible_file_addresses):
        for extension in ext:
            current_file = address + '.' + extension
            # If file path is valid append current_file to file_list
            if os.path.exists(current_file):
                file_list.append(current_file)

    # Load data by calling load_data if a valid file has been found. If not raise an error
    if len(file_list) > 0:
        loaded_data = load_data(file_list, lazy=lazy, loc=loc, metadata=metadata, parallel=parallel)
    else:
        raise Exception('No valid file paths could be found.')

    return loaded_data


def load_data(fname, lazy='auto', loc='auto', metadata=True, parallel=False):
    """Function to load data files in the xr.DataArray format.

    Parameters
    ------------
    fname : str, list
        Path(s) to the file(s) to be loaded.

    lazy : str, Boolean (optional)
        Whether to load data in a lazily evaluated dask format. Set explicitly using True/False Boolean. Defaults
        to 'auto' where a file is only loaded in the dask format if its spectrum is above 500 MB.

    loc : str (optional)
        The name of the location (typically a beamline). Defaults to 'auto', where the location will be attempted to be
        automatically determined.

    metadata : Boolean (optional)
        Whether to attempt to load metadata into the attributes of the xr.DataArray. Defaults to True.

    parallel : Boolean (optional)
        Whether to load data in parallel when multiple files are being loaded. Only compatible with certain file types
        such as those based on the h5py format, e.g. nxs files. Takes priority over lazy, enforcing that all data is
        computed and loaded into memory. Defaults to False.

    Returns
    ------------
    loaded_data : xr.DataArray, xr.DataSet, list
        The loaded data.

    Examples
    ------------
    from peaks import *

    # Load data
    disp1 = load_data('disp1.ibw')
    FM1 = load_data('C:/User/Documents/FM1.ibw')

    # Load multiple data files at once
    disps = load(['disp1.ibw', 'disp2.ibw', 'disp3.ibw'])

    # Load data in a lazily evaluated dask format
    FM1 = load('FM1.ibw', lazy=True)

    # Load data without metadata
    FM1 = load('FM1.ibw', metadata=False)

    # Load data file a pre-defined location (use if automatic location identification fails)
    FM1 = load('FM1.ibw', loc='MAX IV Bloch')

    """

    # Ensure fname is of type list
    if not isinstance(fname, list):
        fname = [fname]

    # Define an empty list to store loaded data
    loaded_data = []

    # If the files have been requested to be loaded in parallel
    if parallel:
        # Loop through and set up the dask delayed function which will facilitate data loading in parallel
        for single_fname in fname:
            loaded_data.append(
                dask.delayed(_load_single_data(fname=single_fname, lazy=lazy, loc=loc, metadata=metadata)))

        # Perform the data loading in parallel
        loaded_data = dask.compute(*loaded_data)

        # If Lazy is not False, display message informing the user that setting parallel to True means all data is
        # loaded into memory and cannot be lazily evaluated
        if lazy:
            analysis_warning(
                'By setting parallel=True, the data has been computed and loaded into memory. This means that the '
                'data is unable to be lazily evaluated in the dask format. If a lazy evaluation is required, set '
                'parallel=False', title='Loading info', warn_type='danger')

    # If not, load files sequentially
    else:
        for single_fname in tqdm(fname, desc='Loading data', disable=len(fname) == 1, colour='CYAN'):
            loaded_data.append(_load_single_data(fname=single_fname, lazy=lazy, loc=loc, metadata=metadata))

    # Check if any of the loaded data have been lazily evaluated. If so, inform the user
    for data in loaded_data:
        if isinstance(data.data, dask.array.core.Array):
            analysis_warning(
                'DataArray has been lazily evaluated in the dask format (set lazy=False to load as DataArray in xarray '
                'format). Use the .compute() method to load DataArray into RAM in the xarray format, or the .persist() '
                'method to instead load DataArray into RAM in the dask format. Note: these operations load all of the '
                'data into memory, so large files may require an initial reduction in size through either a slicing or '
                'binning operation.',
                title='Loading info', warn_type='info')
            break

    # If there is only one loaded item in loaded_data, return the xr.DataArray (xr.DataSet) entry instead of a list
    if len(loaded_data) == 1:
        loaded_data = loaded_data[0]

    return loaded_data


def _load_single_data(fname, lazy='auto', loc='auto', metadata=True):
    """Function to load a single data file as an xr.DataArray

    Parameters
    ------------
    fname : str
        Path to the file to be loaded.

    lazy : str, Boolean (optional)
        Whether to load data in a lazily evaluated dask format. Set explicitly using True/False Boolean. Defaults
        to 'auto' where a file is only loaded in the dask format if its spectrum is above 500 MB.

    loc : str (optional)
        The name of the location (typically a beamline). Defaults to 'auto', where the location will be attempted to be
        automatically determined.

    metadata : Boolean (optional)
        Whether to attempt to load metadata into the attributes of the xr.DataArray. Defaults to True.

    Returns
    ------------
    DataArray : xr.DataArray
        The loaded DataArray.

    Examples
    ------------
    from peaks.core.fileIO.data_loading import _load_single_data

    # Load single data file
    FM1 = _load_single_data('C:/User/Documents/Data/FM1.ibw')

    # Load single data file in a lazily evaluated dask format
    FM1 = _load_single_data('C:/User/Documents/Data/FM1.ibw', lazy=True)

    # Load single data file without metadata
    FM1 = _load_single_data('C:/User/Documents/Data/FM1.ibw', metadata=False)

    # Load single data file with a pre-defined location (use if automatic location identification fails)
    FM1 = _load_single_data('C:/User/Documents/Data/FM1.ibw', loc='MAX IV Bloch')

    """

    # Identify which location (beamline) the scan was obtained at if it has not already been specified
    if loc == 'auto':
        loc = _get_loc(fname)

    # Load the data in the format of a dictionary containing the file scan type, spectrum, and coordinates (except for
    # NetCDF files where the data will be already loaded into xr.DataArray format)
    if loc == 'ALBA LOREA':
        from .loaders.LOREA import _load_LOREA_data
        data = _load_LOREA_data(fname)

    elif loc == 'CLF Artemis':
        from .loaders.Artemis import _load_Artemis_data
        data = _load_Artemis_data(fname)

    elif loc == 'Diamond I05-HR' or loc == 'Diamond I05-nano':
        from .loaders.I05 import _load_I05_data
        data = _load_I05_data(fname)

    elif loc == 'Elettra APE':
        from .loaders.APE import _load_APE_data
        data = _load_APE_data(fname)

    elif loc == 'MAX IV Bloch':
        from .loaders.Bloch import _load_Bloch_data
        data = _load_Bloch_data(fname)

    elif loc == 'MAX IV Bloch-spin':
        from .loaders.Bloch import _load_Bloch_spin_data
        data = _load_Bloch_spin_data(fname)

    elif loc == 'SOLEIL CASSIOPEE':
        from .loaders.CASSIOPEE import _load_CASSIOPEE_data
        data = _load_CASSIOPEE_data(fname)

    elif loc == 'StA-MBS':
        from .loaders.StA_MBS import _load_StA_MBS_data
        data = _load_StA_MBS_data(fname)

    elif loc == 'StA-Phoibos':
        from .loaders.StA_Phoibos import _load_StA_Phoibos_data
        data = _load_StA_Phoibos_data(fname)

    elif loc == 'StA-Bruker':
        from .loaders.StA_Bruker import _load_StA_Bruker_data
        data = _load_StA_Bruker_data(fname)

    elif loc == 'StA-LEED':
        from .loaders.StA_LEED import _load_StA_LEED_data
        data = _load_StA_LEED_data(fname)

    elif loc == 'StA-RHEED':
        from .loaders.StA_RHEED import _load_StA_RHEED_data
        data = _load_StA_RHEED_data(fname)

    elif loc == 'Structure':
        raise Exception('Structure is not currently supported.')
        # from ... import ...
        # data = ...(fname)
        # Return data (don't want to mess up later code operations in this function so return here)

    elif loc == 'ibw':
        from .loaders.ibw import _load_ibw_data
        data = _load_ibw_data(fname)

    elif loc == 'NetCDF':
        from .loaders.NetCDF import _load_NetCDF_data
        data = _load_NetCDF_data(fname)

    # If loc is not a valid location, raise an error
    else:
        raise Exception(
            'Data source is not supported or could not be identified. Currently supported options are: '
            '{locs}.'.format(locs=str(loc_opts.locs)))

    # If location is NetCDF, data is already loaded in xr.DataArray format
    if loc == 'NetCDF':
        # Convert the data to dask format if the user has requested to load data in a lazily evaluated dask format, or
        # if the data spectrum is above 500 MB (and lazy='auto')
        if lazy is True or (lazy == 'auto' and data.nbytes > 500000000):
            DataArray = da.from_array(data['spectrum'], chunks='auto')
        else:
            DataArray = data

    # If location is not NetCDF, we must convert the data to xr.DataArray format and add metadata (if requested)
    else:
        # Generate an xr.DataArray from the loaded data (converting to dask format if required)
        DataArray = _make_DataArray(data, lazy=lazy)

        # Add metadata to the loaded DataArray if requested
        if metadata:
            _add_metadata(DataArray, fname, loc, scan_type=data['scan_type'])

    # Ensure that all the DataArray coordinates are ordered low to high
    for dim in DataArray.dims:
        if len(DataArray[dim]) > 1:  # If there are multiple coordinates along the given dimension
            if DataArray[dim].data[1] - DataArray[dim].data[0] < 0:  # If the DataArray currently has decreasing order
                # Flip the order of the coordinates along the given dimension
                DataArray = DataArray.reindex({dim: data[dim][::-1]})

    # Ensure that the dimensions of the DataArray are arranged in the standard order
    DataArray = DataArray.transpose('scan_no', 't', 'hv', 'temp_sample', 'temp_cryo', 'defocus', 'focus', 'da30_z',
                                    'x1', 'x2', 'x3', 'dim0', 'dim1', 'y_scale', 'two_th', 'polar', 'ana_polar', 'tilt',
                                    'phi', 'defl_perp', 'k_perp', 'azi', 'defl_par', 'eV', 'theta_par', 'k_par',
                                    missing_dims='ignore')

    return DataArray


def _make_DataArray(data, lazy='auto'):
    """This function makes an xr.DataArray from the inputted data.

    Parameters
    ------------
    data : dict
        Dictionary containing the file scan type, spectrum, and coordinates.

    lazy : str, Boolean (optional)
        Whether to load data in a lazily evaluated dask format. Set explicitly using True/False Boolean. Defaults to
        'auto' where a file is only loaded in the dask format if its spectrum is above 500 MB.

    Returns
    ------------
    DataArray : xr.DataArray
        The inputted data in xr.DataArray format.

    Examples
    ------------
    from peaks.core.fileIO.data_loading import _make_DataArray
    from peaks.core.fileIO.loaders.I05 import _load_I05_data

    fname = 'C:/User/Documents/Research/i05-12345.nxs'

    # Extract data from file obtained at the I05 beamline
    data = _load_I05_data(fname)

    # Get data in xr.DataArray format
    DataArray = _make_DataArray(data)

    # Get data in a lazily evaluated dask format
    DataArray = _make_DataArray(data, lazy=True)

    """

    # Extract the keys of the data dictionary, providing information on the coordinates present
    data_keys = list(data)

    # Extract the scan type of the data
    scan_type = data['scan_type']

    # Convert the data spectrum to dask format if the user has requested to load data in a lazily evaluated dask format,
    # or if the data spectrum is above 500 MB (and lazy='auto')
    if lazy is True or (lazy == 'auto' and data['spectrum'].nbytes > 500000000):
        data['spectrum'] = da.from_array(data['spectrum'], chunks='auto')

    # If the scan type is a dispersion, there is only one possible set of coordinates
    if scan_type == 'dispersion':
        DataArray = xr.DataArray(data['spectrum'], dims=("theta_par", "eV"),
                                 coords={"theta_par": data['theta_par'], "eV": data['eV']})

    # If the scan type is a Fermi map, the mapping coordinate could be tilt, polar, ana_polar or defl_perp
    elif scan_type == 'FS map':
        if 'tilt' in data_keys:
            DataArray = xr.DataArray(data['spectrum'], dims=('tilt', 'theta_par', 'eV'),
                                     coords={'tilt': data['tilt'], 'theta_par': data['theta_par'], 'eV': data['eV']})
        elif 'polar' in data_keys:
            DataArray = xr.DataArray(data['spectrum'], dims=('polar', 'theta_par', 'eV'),
                                     coords={'polar': data['polar'], 'theta_par': data['theta_par'], 'eV': data['eV']})
        elif 'ana_polar' in data_keys:
            DataArray = xr.DataArray(data['spectrum'], dims=('ana_polar', 'theta_par', 'eV'),
                                     coords={'ana_polar': data['ana_polar'], 'theta_par': data['theta_par'],
                                             'eV': data['eV']})
        elif 'defl_perp' in data_keys:
            DataArray = xr.DataArray(data['spectrum'], dims=('defl_perp', 'theta_par', 'eV'),
                                     coords={'defl_perp': data['defl_perp'], 'theta_par': data['theta_par'],
                                             'eV': data['eV']})

    # If the scan type is a spatial map, there is only one possible set of coordinates
    elif scan_type == 'spatial map':
        DataArray = xr.DataArray(data['spectrum'], dims=('x1', 'x2', 'theta_par', 'eV'),
                                 coords={'x1': data['x1'], 'x2': data['x2'], 'theta_par': data['theta_par'],
                                         'eV': data['eV']})

    # If the scan type is a line scan, the mapping coordinate could be x1, x2 or x3
    elif scan_type == 'line scan':
        if 'x1' in data_keys:
            DataArray = xr.DataArray(data['spectrum'], dims=('x1', 'theta_par', 'eV'),
                                     coords={'x1': data['x1'], 'theta_par': data['theta_par'], 'eV': data['eV']})
        elif 'x2' in data_keys:
            DataArray = xr.DataArray(data['spectrum'], dims=('x2', 'theta_par', 'eV'),
                                     coords={'x2': data['x2'], 'theta_par': data['theta_par'], 'eV': data['eV']})
        elif 'x3' in data_keys:
            DataArray = xr.DataArray(data['spectrum'], dims=('x3', 'theta_par', 'eV'),
                                     coords={'x3': data['x3'], 'theta_par': data['theta_par'], 'eV': data['eV']})

    # If the scan type is a da30_z scan, there is only one possible set of coordinates
    elif scan_type == 'da30_z scan':
        DataArray = xr.DataArray(data['spectrum'], dims=('da30_z', 'location', 'eV'),
                                 coords={'da30_z': data['da30_z'], 'location': data['location'], "eV": data['eV']})

    # If the scan type is a Focus scan, the mapping coordinate could be x1 or x2
    elif scan_type == 'Focus scan':
        if 'x1' in data_keys:
            DataArray = xr.DataArray(data['spectrum'], dims=('defocus', 'x1', 'theta_par', 'eV'),
                                     coords={'defocus': data['defocus'], 'x1': data['x1'],
                                             'theta_par': data['theta_par'], "eV": data['eV']})
        elif 'x2' in data_keys:
            DataArray = xr.DataArray(data['spectrum'], dims=('defocus', 'x2', 'theta_par', 'eV'),
                                     coords={'defocus': data['defocus'], 'x2': data['x2'],
                                             'theta_par': data['theta_par'], 'eV': data['eV']})

    # If the scan type is a 1D Focus scan, the focussing coordinate could be da30_z, defocus or focus
    elif scan_type == '1D focus scan':
        if 'da30_z' in data_keys:
            DataArray = xr.DataArray(data['spectrum'], dims=('da30_z', 'theta_par', 'eV'),
                                     coords={'da30_z': data['da30_z'], 'theta_par': data['theta_par'],
                                             'eV': data['eV']})
        elif 'defocus' in data_keys:
            DataArray = xr.DataArray(data['spectrum'], dims=('defocus', 'theta_par', 'eV'),
                                     coords={'defocus': data['defocus'], 'theta_par': data['theta_par'],
                                             'eV': data['eV']})
        elif 'focus' in data_keys:
            DataArray = xr.DataArray(data['spectrum'], dims=('focus', 'theta_par', 'eV'),
                                     coords={'focus': data['focus'], 'theta_par': data['theta_par'], "eV": data['eV']})

    # If the scan type is an hv scan, there is only one possible set of coordinates
    elif scan_type == 'hv scan':
        DataArray = xr.DataArray(data['spectrum'], dims=('hv', 'theta_par', 'eV'),
                                 coords={'hv': data['hv'], 'theta_par': data['theta_par'], 'eV': data['eV']})
        # Add KE_delta coordinate
        DataArray.coords['KE_delta'] = ('hv', KE_delta)

    # Loop through all the possible coordinates in saved in _coords.units. If the coordinate is present in the
    # DataArray, add the relevant unit to the coordinate attributes
    for coord in _coords.units:
        if coord in list(DataArray.coords):
            DataArray.coords[coord].attrs = {'units': _coords.units[coord]}

    return DataArray


def _add_metadata(DataArray, fname, loc, scan_type):
    """This function adds metadata to an inputted xr.DataArray.

    Parameters
    ------------
    DataArray : xr.DataArray
        The inputted DataArray.

    fname : str
        Path to the file to be loaded.

    loc : str
        The name of the location (typically a beamline).

    scan_type : str
        The scan type of the data.

    Examples
    ------------
    from peaks.core.fileIO.data_loading import _make_DataArray, _add_metadata, _get_loc
    from peaks.core.fileIO.loaders.StA_Phoibos_metadata import _load_StA_Phoibos_data

    fname = 'C:/User/Documents/Research/i05-12345.nxs'

    # Determine location qat which data was obtained
    loc = _get_loc(fname)

    # Extract data from file obtained using the ARPES system (Phoibos analyser) at St Andrews
    data = _load_StA_Phoibos_data(fname)

    # Get data in xr.DataArray format
    DataArray = _make_DataArray(data)

    # Add metadata to DataArray
    add_metadata(DataArray, fname, loc, scan_type=data['scan_type'])

    """

    # Load the metadata in the format of a dictionary where each key is an attribute of the data, e.g. pass energy
    if loc == 'ALBA LOREA':
        from .loaders.LOREA import _load_LOREA_metadata
        metadata = _load_LOREA_metadata(fname, scan_type)

    elif loc == 'CLF Artemis':
        from .loaders.Artemis import _load_Artemis_metadata
        metadata = _load_Artemis_metadata(fname, scan_type)

    elif loc == 'Diamond I05-HR' or loc == 'Diamond I05-nano':
        from .loaders.I05 import _load_I05_metadata
        metadata = _load_I05_metadata(fname, scan_type, loc)

    elif loc == 'Elettra APE':
        from .loaders.APE import _load_APE_metadata
        metadata = _load_APE_metadata(fname, scan_type)

    elif loc == 'MAX IV Bloch':
        from .loaders.Bloch import _load_Bloch_metadata
        metadata = _load_Bloch_metadata(fname, scan_type, loc)

    elif loc == 'MAX IV Bloch-spin':
        from .loaders.Bloch import _load_Bloch_spin_metadata
        metadata = _load_Bloch_spin_metadata(fname, scan_type, loc)

    elif loc == 'SOLEIL CASSIOPEE':
        from .loaders.CASSIOPEE import _load_CASSIOPEE_metadata
        metadata = _load_CASSIOPEE_metadata(fname, scan_type)

    elif loc == 'StA-MBS':
        from .loaders.StA_MBS import _load_StA_MBS_metadata
        metadata = _load_StA_MBS_metadata(fname, scan_type)

    elif loc == 'StA-Phoibos':
        from peaks.core.fileIO.loaders.StA_Phoibos import _load_StA_Phoibos_metadata
        metadata = _load_StA_Phoibos_metadata(fname, scan_type)

    elif loc == 'StA-Bruker':
        from .loaders.StA_Bruker import _load_StA_Bruker_metadata
        metadata = _load_StA_Bruker_metadata(fname, scan_type)

    elif loc == 'StA-LEED':
        from .loaders.StA_LEED import _load_StA_LEED_metadata
        metadata = _load_StA_LEED_metadata(fname, scan_type)

    elif loc == 'StA-RHEED':
        from .loaders.StA_RHEED import _load_StA_RHEED_metadata
        metadata = _load_StA_RHEED_metadata(fname, scan_type)

    elif loc == 'ibw':
        from .loaders.ibw import _load_ibw_metadata
        data = _load_ibw_metadata(fname)

    # Assign the metadata information to attributes of the DataArray
    DataArray.attrs = metadata

    # Give the DataArray a name equal to the scan_name attribute
    DataArray.name = metadata['scan_name']


def _get_loc(fname):
    """This function determines the location at which the data was obtained.

    Parameters
    ------------
    fname : str
        Path to the file to be loaded.

    Returns
    ------------
    loc : str
        The name of the location (typically a beamline).

    Examples
    ------------
    from peaks.core.fileIO.data_loading import _get_loc

    fname = 'C:/User/Documents/Research/disp1.xy'

    # Determine the location at which the data was obtained
    loc = _get_loc(fname)

    """

    # Get file_extension to determine the file type
    filename, file_extension = os.path.splitext(fname)

    # Set loc to Default None value
    loc = None

    # If there is no extension, the data is in a folder. This is consistent with SOLEIL CASSIOPEE Fermi maps or
    # CLF Artemis data
    if file_extension == '':
        # Extract identifiable data from the file to determine if the location is SOLEIL CASSIOPEE
        file_list = natsort.natsorted(os.listdir(filename))
        file_list_ROI = [item for item in file_list if 'ROI1_' in item and isfile(join(file, item))]
        if len(file_list_ROI) > 1:  # Must be SOLEIL CASSIOPEE
            loc = 'SOLEIL CASSIOPEE'
        else:  # Likely CLF Artemis
            # Extract identifiable data from the file to determine if the location is CLF Artemis
            file_list_Neq = [item for item in file_list if 'N=' in item]
            if len(file_list_Neq) > 0:  # Must be CLF Artemis
                loc = 'CLF Artemis'

    # If the file is .xy format, the location must be either MAX IV Bloch-spin, StA-Phoibos or StA-Bruker
    elif file_extension == '.xy':
        # Open the file and load the first line
        with open(fname) as f:
            line0 = f.readline()

        # If measurement was performed using Specs analyser, location must be MAX IV Bloch-spin or StA-Phoibos
        if 'SpecsLab' in line0:
            # By default, assume StA-Phoibos
            loc = 'StA-Phoibos'
            # Open the file and load the first 30 lines to to check if this is instead MAX IV Bloch-spin
            with open(fname) as f:
                for i in range(30):
                    # If the 'PhoibosSpin' identifier is present in any of the lines, location must be MAX IV Bloch-spin
                    if 'PhoibosSpin' in f.readline():
                        loc = 'MAX IV Bloch-spin'

        # If not (and the identifier 'Anode' is present), then measurement was XRD using StA-Bruker
        elif 'Anode' in line0:
            loc = 'StA-Bruker'

    # If the file is .sp2 format, the location must be MAX IV Bloch-spin
    elif file_extension == '.sp2':
        loc = 'MAX IV Bloch-spin'

    # If the file is .krx format, the location must be StA-MBS
    elif file_extension == '.krx':
        loc = 'StA-MBS'

    # If the file is .txt format, the location must be StA-MBS, MAX IV Bloch, Elettra APE or SOLEIL CASSIOPEE
    elif file_extension == '.txt':
        # Open the file and load the first line
        with open(fname) as f:
            line0 = f.readline()

        # MAX IV Bloch, Elettra APE or SOLEIL CASSIOPEE .txt files follow the same SES data format, so we can identify
        # the location from the location line in the file
        if line0 == '[Info]\n':  # Identifier of the SES data format
            # Extract metadata information from file using the _load_SES_metalines function
            metadata_lines = _load_SES_metalines(fname)
            # Extract and determine the location using the _SES_find function
            location = _SES_find(metadata_lines, 'Location=')
            if 'bloch' in location.lower() or 'maxiv' in location.lower():
                loc = 'MAX IV Bloch'
            elif 'ape' in location.lower() or 'elettra' in location.lower():
                loc = 'Elettra APE'
            elif 'cassiopee' in location.lower() or 'soleil' in location.lower():
                loc = 'SOLEIL CASSIOPEE'

        # If the file does not follow the SES format, it must be StA-MBS
        else:
            loc = 'StA-MBS'

    # If the file is .zip format, the file must be of SES format. Thus, the location must be MAX IV Bloch, Elettra APE,
    # SOLEIL CASSIOPEE or Diamond I05-nano (defl map)
    elif file_extension == '.zip':
        # Extract metadata information from file using the _load_SES_metalines function
        metadata_lines = _load_SES_metalines(fname)
        # Extract and determine the location using the _SES_find function
        location = _SES_find(metadata_lines, 'Location=')
        if 'bloch' in location.lower() or 'maxiv' in location.lower():
            loc = 'MAX IV Bloch'
        elif 'ape' in location.lower() or 'elettra' in location.lower():
            loc = 'Elettra APE'
        elif 'cassiopee' in location.lower() or 'soleil' in location.lower():
            loc = 'SOLEIL CASSIOPEE'
        elif 'i05' in location.lower() or 'diamond' in location.lower():
            loc = 'Diamond I05-nano'

    # If the file is .ibw format, the file is likely SES format. Thus location should be MAX IV Bloch, Elettra APE or
    # SOLEIL CASSIOPEE
    elif file_extension == '.ibw':
        # Extract metadata information from file using the _load_SES_metalines function
        metadata_lines = _load_SES_metalines(fname)
        # Determine the location
        if 'bloch' in metadata_lines.lower() or 'maxiv' in metadata_lines.lower():
            loc = 'MAX IV Bloch'
        elif 'ape' in metadata_lines.lower() or 'elettra' in metadata_lines.lower():
            loc = 'Elettra APE'
        elif 'cassiopee' in metadata_lines.lower() or 'soleil' in metadata_lines.lower():
            loc = 'SOLEIL CASSIOPEE'
        # If we are unable to find a location, define location as a generic ibw file
        else:
            loc = 'ibw'

    # If the file is .nxs format, the location should be Diamond I05-nano, Diamond I05-HR or ALBA LOREA
    elif file_extension == '.nxs':
        # Open the file (read only)
        f = h5py.File(fname, 'r')
        # .nxs files at Diamond and Alba contain approximately the same identifier format
        identifier = _h5py_str(f, 'entry1/instrument/name')
        # From the identifier, determine the location
        if 'i05-1' in identifier:
            loc = 'Diamond I05-nano'
        elif 'i05' in identifier:
            loc = 'Diamond I05-HR'
        elif 'lorea' in identifier:
            loc = 'ALBA LOREA'

    # If the file is .nc format, it is a NetCDF file. Set location to NetCDF, since the location information should be
    # in the NetCDF attributes
    elif file_extension == '.nc':
        loc = 'NetCDF'

    # If the file is .cif format, it should be a structure file
    elif file_extension == '.cif':
        loc = 'Structure'

    # If the file is standard image format, it should be a LEED file
    elif file_extension == '.bmp' or file_extension == '.jpeg' or file_extension == '.png':
        loc = 'StA-LEED'

    # If the file is .iso format, it should be a LEED file
    elif file_extension == '.iso':
        loc = 'StA-RHEED'

    return loc


def _h5py_str(file, file_handle):
    """Parses string or binary strings into the correct format when reading from a h5py file

    Parameters
    ------------
    file : h5py._hl.files.File
        An open h5py format file.

    file_handle : str, bin, list
         File handle of h5py file object. Can either be str, bin or list of str/bin

    Returns
    ------------
    contents : str
        String of field contents

    Examples
    ------------
    from peaks.core.fileIO.data_loading import _h5py_str

    disp1 = load('disp1.ibw')

    # Acts as a shortcut for accessing h5py file data. e.g. f['entry1/instrument/name'][()].decode()
    location = _h5py_str(f, 'entry1/instrument/name')

    """

    # Attempt to obtain field contents
    try:
        if not isinstance(file[file_handle][()], str):
            try:
                contents = file[file_handle][0]
            except ValueError:
                contents = file[file_handle][()]
        else:
            contents = file[file_handle][()]

    # Return None if unable to obtain field contents
    except TypeError:
        contents = None

    # Decode contents if required
    try:
        contents = contents.decode()
    except AttributeError:
        pass

    return contents


def _extract_mapping_metadata(mapping_coordinates, num_dp):
    """Utility function used for metadata loading which extracts a string to describe the inputted mapping coordinates
    in the format: 'min:max (step)'.

    Parameters
    ------------
    mapping_coordinates : list, np.array
         The coordinates of a mapping variable

    num_dp : int
        The number of decimal places round to.

    Returns
    ------------
    mapping_metadata : str
        String describing the mapping coordinates in the format: 'min:max (step)'.

    Examples
    ------------
    from peaks.core.fileIO.data_loading import _extract_mapping_metadata

    mapping_coordinates = [0, 0.5, 1, 1.5, 2, 2.5, 3]

    # Extract the mapping coordinates in the format: 'min:max (step)'. For this example, we would get: '0:3 (0.5)'
    mapping_metadata = _extract_mapping_metadata(mapping_coordinates)

    """

    # Extract the minimum and maximum of mapping_coordinates, and round to
    min_value = round(min(mapping_coordinates), num_dp)
    max_value = round(max(mapping_coordinates), num_dp)

    # Extract the step size of mapping_coordinates
    step_size = round((max_value - min_value) / (len(mapping_coordinates) - 1), num_dp)

    # Describe the inputted mapping coordinates in the format: 'min:max (step)'.
    mapping_metadata = str(min_value) + ':' + str(max_value) + ' (' + str(step_size) + ')'

    return mapping_metadata
