"""Functions to load Igor binary wave (ibw) files.

"""

# Phil King 24/07/2022
# Brendan Edwards 15/02/2024

import os
import numpy as np


def _load_ibw_data(fname):
    """*** NOT COMPLETED - IGOR IMPORT OF FUNCTION binarywave MUST BE FIXED. ***

    This function loads data stored in ibw files.

    Parameters
    ------------
    fname : str
        Path to the file to be loaded.

    Returns
    ------------
    data : dict
        Dictionary containing the file scan type, spectrum, and coordinates.

    Examples
    ------------
    Example usage is as follows::

        from peaks.core.fileIO.loaders.ibw import _load_ibw_data

        fname = 'C:/User/Documents/Research/disp1.ibw'

        # Extract data from an ibw file
        data = _load_NetCDF_data(fname)

    """

    # Open the file and load its contents
    file_contents = None
    # file_contents = binarywave.load(fname)

    # Extract spectrum
    spectrum = file_contents['wave']['wData']

    # Extract relevant information on the dimensions of the data
    dim_size = file_contents['wave']['bin_header']['dimEUnitsSize']
    dim_units = file_contents['wave']['dimension_units'].decode()

    # Extract scales of dimensions
    dim_start = file_contents['wave']['wave_header']['sfB']  # Initial value
    dim_step = file_contents['wave']['wave_header']['sfA']  # Step size
    dim_points = file_contents['wave']['wave_header']['nDim']  # Number of points

    # Loop through dimensions and determine the relevant dimension waves and names, as labelled in file
    dims = []
    coords = {}
    counter = 0
    for i in range(spectrum.ndim):
        # Determine dimension units/name as labelled in file
        dims.append(dim_units[counter: (counter + dim_size[i])])
        coords[dims[i]] = np.linspace(dim_start[i], dim_start[i] + dim_step[i] * dim_points[i], dim_points[i],
                                      endpoint=False)
        counter += dim_size[i]


def _load_ibw_wavenote(fname):
    """This function will load the wavenote (which contains metadata) from an Igor binary wave (ibw) file.

    Parameters
    ------------
    fname : str
        Path to the file to be loaded.

    Returns
    ------------
    wavenote : str
        The wavenote containing the metadata.

    Examples
    ------------
    Example usage is as follows::

        from peaks.core.fileIO.loaders.ibw import _load_ibw_wavenote

        fname = 'C:/User/Documents/Research/disp1.ibw'

        # Extract the wavenote containing metadata
        wavenote = _load_ibw_wavenote(fname)

    """

    # Define maximum number of dimensions
    max_dims = 4

    # Read the ibw bin header segment of the file (IBW version 2,5 only)
    with open(fname, 'rb') as f:
        # Determine file version and extract file information
        version = np.fromfile(f, dtype=np.dtype('int16'), count=1)[0]
        if version == 2:
            # The size of the WaveHeader2 data structure plus the wave data plus 16 bytes of padding.
            wfmSize = np.fromfile(f, dtype=np.dtype('uint32'), count=1)[0]
            # The size of the note text.
            noteSize = np.fromfile(f, dtype=np.dtype('uint32'), count=1)[0]
            # Reserved. Write zero. Ignore on read.
            pictSize = np.fromfile(f, dtype=np.dtype('uint32'), count=1)[0]
            # Checksum over this header and the wave header.
            checksum = np.fromfile(f, dtype=np.dtype('int16'), count=1)[0]
        elif version == 5:
            # Checksum over this header and the wave header.
            checksum = np.fromfile(f, dtype=np.dtype('short'), count=1)[0]
            # The size of the WaveHeader5 data structure plus the wave data.
            wfmSize = np.fromfile(f, dtype=np.dtype('int32'), count=1)[0]
            # The size of the dependency formula, if any.
            formulaSize = np.fromfile(f, dtype=np.dtype('int32'), count=1)[0]
            # The size of the note text.
            noteSize = np.fromfile(f, dtype=np.dtype('int32'), count=1)[0]
            # The size of optional extended data units.
            dataEUnitsSize = np.fromfile(f, dtype=np.dtype('int32'), count=1)[0]
            # The size of optional extended dimension units.
            dimEUnitsSize = np.fromfile(f, dtype=np.dtype('int32'), count=max_dims)
            # The size of optional dimension labels.
            dimLabelsSize = np.fromfile(f, dtype=np.dtype('int32'), count=4)
            # The size of string indices if this is a text wave.
            sIndicesSize = np.fromfile(f, dtype=np.dtype('int32'), count=1)[0]
            # Reserved. Write zero. Ignore on read.
            optionsSize1 = np.fromfile(f, dtype=np.dtype('int32'), count=1)[0]
            # Reserved. Write zero. Ignore on read.
            optionsSize2 = np.fromfile(f, dtype=np.dtype('int32'), count=1)[0]

    # Open the file and read the wavenote
    with open(fname, 'rb') as f:
        # Move the cursor to the end of the file
        f.seek(0, os.SEEK_END)
        # Get the current position of pointer
        pointer_location = f.tell()

        # Determine file version-dependent offset
        if version == 2:
            offset = noteSize
        elif version == 5:
            # Work out file location of wavenote
            offset = (noteSize + dataEUnitsSize.sum() + dimEUnitsSize.sum() + dimLabelsSize.sum() + sIndicesSize.sum()
                      + optionsSize1.sum() + optionsSize2.sum())

        # Move the file pointer to the location pointed by pointer_location, considering the offset
        f.seek(pointer_location - offset)
        # read that bytes/characters to determine the wavenote
        wavenote = f.read(offset).decode()

    return wavenote
