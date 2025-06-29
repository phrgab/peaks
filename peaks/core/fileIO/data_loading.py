"""Functions to load data into DataArray format."""

import glob
import os

import dask
import pint
from tqdm.notebook import tqdm

from peaks.core.fileIO.base_data_classes.base_data_class import BaseDataLoader
from peaks.core.options import opts
from peaks.core.utils.datatree_utils import _dataarrays_to_datatree
from peaks.core.utils.misc import analysis_warning


def load(
    fpath,
    lazy=None,
    loc=None,
    metadata=True,
    parallel=False,
    names=None,
    quiet=False,
    **kwargs,
):
    """Core function to load data.

    Parameters
    ------------
    fpath : str, list
        Either the full file path(s), or the remainder of the file name(s) not already
        specified in the file global options (see Notes).

    lazy : str, bool, optional
        Whether to load data in a lazily evaluated dask format. Set explicitly using
        True/False Boolean. Defaults to `None` where a file is only loaded in the dask
        format if its spectrum is above threshold set in `opts.FileIO.lazy_size` or is a
        Zarr store, where it is lazily loaded by default.

    loc : str, optional
        Location identifier for where data was acquired. Defaults to `None`, where the
        location will be attempted to be automatically determined, unless a value is
        defined in `opts.FileIO.loc`. If `loc` is specified in the function call, this
        takes priority over the value in `opts.FileIO.loc`.

    metadata : bool, optional
        Whether to attempt to load metadata into the attributes of the
        :class:`xarray.DataArray`. Defaults to True.

    parallel : bool, optional
        Whether to load data in parallel when multiple files are being loaded. Only
        compatible with certain file types such as those based on the h5py format, e.g.
        .nxs files. Takes priority over lazy, enforcing that all data is computed and
        loaded into memory. Defaults to False.

    names : list, optional
        List of names to assign to the branches of the :class:xarray.DataTree when
        mutliple scans loaded simultaneously. If provided, should be a list of unique
        strings of the same length as the number of scans being loaded. If not, or if
        not provided, the names will be the file names. Defaults to None.

    quiet : bool, optional
        Whether to suppress analysis warnings when loading data. Defaults to False.

    kwargs : dict
        Additional keyword arguments to pass to the data loader.

    Returns
    ------------
    loaded_data : xarray.DataArray, xarray.DataSet, xarray.DataTree
        The loaded data.

    Notes
    ------------
    Much of file path can be set by :class:`peaks.core.options.FileIO` global options:

        opts.FileIO.path : str, list
            Path (or list of paths) to folder(s) where data is stored,
            e.g. `opts.FileIO.path = 'C:/User/Documents/i05-1-123'`.

        opts.FileIO.ext : str, list
            Extension (or list of) of data, e.g. `opts.FileIO.ext = ['ibw', 'zip']`.

        opts.FileIO.loc : str
            Location identifier for where data was acquired, e.g. `opts.FileIO.loc = 'MAX IV Bloch'`.
            Current supported options can be obtained using `pks.locs()`.

        opts.FileIO.lazy_size : int
            Size in bytes above which data is loaded in a lazily evaluated dask format.
            Defaults to 1 GB.

    Examples
    ------------
    Example usage is as follows::

        import peaks as pks

        # Load data using a complete data path
        disp1 = pks.load('C:/User/Documents/disp1.ibw')
        FM1 = pks.load('C:/User/Documents/FM1.ibw')

        # Define global options to define file loading
        pks.opts.FileIO.set(path = 'C:/User/Documents/Data/',
                            ext = ['ibw', 'zip'])

        # Load data without needing to define data path or extension
        disp2 = pks.load('disp2')
        FM2 = pks.load('FM2')

        # Define global options to define file structure, including part of the scan name
        pks.opts.FileIO.path = 'C:/User/Documents/Data/i05-1-123'
        pks.opts.FileIO.ext = 'nxs'

        # Load data without needing to define data path or extension, nor the repeated
        # part of the scan name
        disp3 = load(456)
        disp4 = load(457)

        # Provide part of the scan name using a glob character
        disps = load('45*')

        # Load multiple files at once
        disps = load([456, 457, 458])

        # Still can load data using a complete data path
        # global options defined in `pks.opts.FileIO` will be ignored
        disp1 = load('C:/User/Documents/Data/disp1.ibw')

        # Load data in a lazily evaluated dask format
        disp1 = load('C:/User/Documents/Data/disp1.ibw', lazy=True)

        # Load data without metadata
        disp1 = load('C:/User/Documents/Data/disp1.ibw', metadata=False)

        # Load data file for a defined location (use if automatic location ID fails)
        disp1 = load('C:/User/Documents/Data/disp1.ibw', loc='MAX IV Bloch')

        # Alternatively could define location using global options.
        # Here loc will be defined as 'MAX IV Bloch'
        pks.opts.FileIO.loc = 'MAXIV_Bloch_A'
        disp1 = load('C:/User/Documents/Data/disp1.ibw')

        # Set the data size to trigger lazy loading by default to 500 MB
        pks.opts.FileIO.lazy_size = 500000000
        FM3 = pks.load('C:/User/Documents/FM3.ibw')

        # Can also use with a context manager to temporarily set the global options
        with pks.opts as opts:
            opts.FileIO.path = 'C:/User/Documents/Data/'
            opts.FileIO.ext = ['ibw', 'zip']
            opts.FileIO.loc = 'MAXIV_Bloch_A'
            opts.FileIO.lazy_size = 500000000

            # Load data without needing to define data path or extension
            disp2 = pks.load('disp2')
            FM2 = pks.load('FM2')
    """

    load_opts = {
        "lazy": lazy,
        "loc": loc,
        "metadata": metadata,
        "parallel": parallel,
        "names": names,
        "quiet": quiet,
    }
    load_opts.update(kwargs)

    # If a full file path provided, always load only this
    if isinstance(fpath, str) and os.path.exists(fpath) and "." in fpath:
        return _load_data(fpath, **load_opts)

    # Set placeholder file path and extension
    base_path = [None]
    ext = [""]

    # If FileIO.path is defined, make a list of the inputted path(s)
    if opts.FileIO.path:
        base_path = (
            [str(path) for path in opts.FileIO.path]
            if isinstance(opts.FileIO.path, list)
            else [str(opts.FileIO.path)]
        )

    # If file.ext is defined, make a list of the inputted extension(s)
    if opts.FileIO.ext:
        ext = [opts.FileIO.ext] if isinstance(opts.FileIO.ext, str) else opts.FileIO.ext

    # If the parameter loc is not defined and opts.FileIO.loc is defined, update loc
    if loc is None:
        load_opts["loc"] = opts.FileIO.loc

    # Ensure that fpath is a list of strings
    fpath = [str(path) for path in fpath] if isinstance(fpath, list) else [str(fpath)]

    # Obtain all possible file addresses
    possible_file_addresses = []
    # Loop through file names
    for file_name in fpath:
        # Remove any extensions from fname, adding them to ext
        file_name, fname_ext = os.path.splitext(file_name)
        if fname_ext != "" and fname_ext[1:] not in ext:
            ext.append(fname_ext[1:])

        # Add raw address file_name to valid_file_addresses
        possible_file_addresses.append(file_name)

        # Loop through base file paths and add file_path/file_name combinations
        for file_path in base_path:
            if file_path:
                # Check if file_path is already an existing folder
                if os.path.exists(file_path) and os.path.isdir(file_path):
                    # If so, join properly so it doesn't matter if a terminator was given
                    possible_file_addresses.append(os.path.join(file_path, file_name))
                else:
                    # Otherwise, must have had a partial file name, so just append
                    possible_file_addresses.append(file_path + file_name)

    # Obtain all valid file addresses
    file_list = []
    for address in possible_file_addresses:
        for extension in ext:
            # Construct the path pattern
            if extension:
                pattern = f"{address}.{extension}"
            else:
                pattern = address

            # Use glob to expand wildcards
            matched_files = glob.glob(pattern)

            # Filter for existing paths and add them
            for file in matched_files:
                if os.path.exists(file):
                    file_list.append(file)

    # Remove duplicates from file_list
    file_list = list(set(file_list))

    # Load data by calling load_data if a valid file has been found. If not raise an error
    if len(file_list) > 0:
        return _load_data(file_list, **load_opts, **kwargs)

    raise Exception("No valid file paths could be found.")


def _load_data(fpath, lazy, loc, metadata, parallel, names, quiet, **kwargs):
    """Function to handle loading of single or multiple data files into the xarray DataArray format.

    Returns
    ------------
    loaded_data : xarray.DataArray, xarray.DataSet, list
        The loaded data.

    See Also
    ------------
    load : Public function to load data, which sets much of the fpath and defines the options to pass to `_load_data`.

    """

    load_opts = {"lazy": lazy, "loc": loc, "metadata": metadata, "quiet": quiet}
    load_opts.update(kwargs)

    # Ensure fname is of type list
    if not isinstance(fpath, list):
        fpath = [fpath]

    # Define an empty list to store loaded data
    loaded_data = []

    # If the files have been requested to be loaded in parallel
    if parallel:
        # Loop through and set up the dask delayed function which will facilitate data loading in parallel
        for single_fpath in fpath:
            loaded_data.append(
                dask.delayed(BaseDataLoader.load(fpath=single_fpath, **load_opts))
            )

        # Perform the data loading in parallel
        loaded_data = dask.compute(*loaded_data)

        # If Lazy is not False, display message informing the user that setting parallel to True means all data is
        # loaded into memory and cannot be lazily evaluated
        if lazy:
            analysis_warning(
                "By setting parallel=True, the data has been computed and loaded into memory. This means that the "
                "data is unable to be lazily evaluated in the dask format. If a lazy evaluation is required, set "
                "parallel=False",
                title="Loading info",
                warn_type="danger",
            )

    # If not, load files sequentially
    else:
        for single_fpath in tqdm(
            fpath,
            desc="Loading data",
            disable=len(fpath) == 1,
        ):
            loaded_data.append(BaseDataLoader.load(fpath=single_fpath, **load_opts))

    # Check if any of the loaded data have been lazily evaluated. If so, inform the user
    for data in loaded_data:
        try:
            if isinstance(data.data, dask.array.core.Array) or (
                isinstance(data.data, pint.Quantity)
                and isinstance(data.data.magnitude, dask.array.core.Array)
            ):
                analysis_warning(
                    "DataArray has been lazily evaluated in the dask format (set lazy=False to load as DataArray in xarray "
                    "format). Use the .compute() method to load DataArray into RAM in the xarray format, or the .persist() "
                    "method to instead load DataArray into RAM in the dask format. Note: these operations load all of the "
                    "data into memory, so large files may require an initial reduction in size through either a slicing or "
                    "binning operation.",
                    title="Loading info",
                    warn_type="info",
                )
                break
        except AttributeError:
            pass

    # If there is only one loaded item in loaded_data, return the xr.DataArray (xr.DataSet) entry instead of a list
    if len(loaded_data) == 1:
        return loaded_data[0]

    # Otherwise, parse these into a DataTree
    return _dataarrays_to_datatree(loaded_data, names)
