import os

import numpy as np
from igor2 import binarywave

from peaks.core.fileIO.base_data_classes.base_data_class import BaseDataLoader
from peaks.core.fileIO.loc_registry import register_loader


@register_loader
class BaseIBWDataLoader(BaseDataLoader):
    """Base class for data loaders for Igor Binary Wave files."""

    _loc_name = "ibw"
    _loc_description = "General loader for Igor Binary Wave files"
    _loc_url = "https://www.wavemetrics.com/products"
    _metadata_parsers = ["_parse_wavenote_metadata"]

    @classmethod
    def _load_data(cls, fpath, lazy):
        """Load the data from an Igor Binary Wave file."""

        # Open the file and load its contents
        file_contents = binarywave.load(fpath)

        # Extract spectrum
        spectrum = file_contents["wave"]["wData"]

        # Extract relevant information on the dimensions of the data
        dim_size = file_contents["wave"]["bin_header"]["dimEUnitsSize"]
        dim_units = file_contents["wave"]["dimension_units"].decode()

        # Extract scales of dimensions
        dim_start = file_contents["wave"]["wave_header"]["sfB"]  # Initial value
        dim_step = file_contents["wave"]["wave_header"]["sfA"]  # Step size
        dim_points = file_contents["wave"]["wave_header"]["nDim"]  # Number of points
        dim_end = dim_start + (dim_step * (dim_points - 1))

        # Loop through the dimensions, extract relevant dimension names and coordinates
        dims = []
        coords = {}
        counter = 0
        for i in range(spectrum.ndim):
            dim = dim_units[counter : counter + dim_size[i]]
            dims.append(dim)
            coords[dim] = np.linspace(
                dim_start[i], dim_end[i], dim_points[i], endpoint=False
            )
            counter += dim_size[i]

        return {"spectrum": spectrum, "dims": dims, "coords": coords, "units": {}}

    @classmethod
    def _load_metadata(cls, fpath):
        """Load metadata from an Igor Binary Wave file."""
        # Load just the wavenote
        # Define maximum number of dimensions
        max_dims = 4

        # Read the ibw bin header segment of the file (IBW version 2,5 only)
        with open(fpath, "rb") as f:
            # Determine file version and extract file information
            version = np.fromfile(f, dtype=np.dtype("int16"), count=1)[0]
            if version == 2:
                # The size of the WaveHeader2 data structure plus the wave data plus 16
                # bytes of padding.
                wfmSize = np.fromfile(f, dtype=np.dtype("uint32"), count=1)[0]
                # The size of the note text.
                noteSize = np.fromfile(f, dtype=np.dtype("uint32"), count=1)[0]
                # Reserved. Write zero. Ignore on read.
                pictSize = np.fromfile(f, dtype=np.dtype("uint32"), count=1)[0]  # noqa: F841
                # Checksum over this header and the wave header.
                checksum = np.fromfile(f, dtype=np.dtype("int16"), count=1)[0]  # noqa: F841
            elif version == 5:
                # Checksum over this header and the wave header.
                checksum = np.fromfile(f, dtype=np.dtype("short"), count=1)[0]  # noqa: F841
                # The size of the WaveHeader5 data structure plus the wave data.
                wfmSize = np.fromfile(f, dtype=np.dtype("int32"), count=1)[0]  # noqa: F841
                # The size of the dependency formula, if any.
                formulaSize = np.fromfile(f, dtype=np.dtype("int32"), count=1)[0]  # noqa: F841
                # The size of the note text.
                noteSize = np.fromfile(f, dtype=np.dtype("int32"), count=1)[0]
                # The size of optional extended data units.
                dataEUnitsSize = np.fromfile(f, dtype=np.dtype("int32"), count=1)[0]
                # The size of optional extended dimension units.
                dimEUnitsSize = np.fromfile(f, dtype=np.dtype("int32"), count=max_dims)
                # The size of optional dimension labels.
                dimLabelsSize = np.fromfile(f, dtype=np.dtype("int32"), count=4)
                # The size of string indices if this is a text wave.
                sIndicesSize = np.fromfile(f, dtype=np.dtype("int32"), count=1)[0]
                # Reserved. Write zero. Ignore on read.
                optionsSize1 = np.fromfile(f, dtype=np.dtype("int32"), count=1)[0]
                # Reserved. Write zero. Ignore on read.
                optionsSize2 = np.fromfile(f, dtype=np.dtype("int32"), count=1)[0]

        # Open the file and read the wavenote
        with open(fpath, "rb") as f:
            # Move the cursor to the end of the file
            f.seek(0, os.SEEK_END)
            # Get the current position of pointer
            pointer_location = f.tell()

            # Determine file version-dependent offset
            if version == 2:
                offset = noteSize
            elif version == 5:
                # Work out file location of wavenote
                offset = (
                    noteSize
                    + dataEUnitsSize.sum()
                    + dimEUnitsSize.sum()
                    + dimLabelsSize.sum()
                    + sIndicesSize.sum()
                    + optionsSize1.sum()
                    + optionsSize2.sum()
                )

            # Move the file pointer to the location pointed by pointer_location,
            # considering the offset
            f.seek(pointer_location - offset)
            # read that bytes/characters to determine the wavenote
            wavenote = f.read(offset).decode()
            # NB No not cache the wavenote, as other loaders rely on this method and
            # that screws things up

        return {"wavenote": wavenote}

    @classmethod
    def _parse_wavenote_metadata(cls, metadata_dict):
        """Parse metadata specific to the wavenote."""
        return {"wavenote": metadata_dict.get("wavenote")}, []
