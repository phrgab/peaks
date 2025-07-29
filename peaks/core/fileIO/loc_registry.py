"""Registry for file loaders and locations."""

import os
from os.path import isfile, join

import h5py
import natsort

LOC_REGISTRY = {}


def locs():
    """Return the list of available locations."""
    return set(LOC_REGISTRY.copy().keys())


def register_loader(loader_class):
    """Decorator to register a loader class in the LOC_REGISTRY."""
    LOC_REGISTRY[loader_class._loc_name] = loader_class
    return loader_class


class IdentifyLoc:
    """Class to identify the location based on the file name and its contents.

    The methods in this class follow a specific naming convention to handle different file formats.
    Except for the _default and _no_extension cases, each method name should start with `_handler_`
    followed by the file extension it handles. For example, `_handler_txt` handles `.txt` files,
    `_handler_zip` handles `.zip` files, etc.
    """

    @staticmethod
    def _default_handler(fname):
        raise ValueError(
            f"Location of file {fname} could not be determined. Please specify the location in the call to `load`"
        )

    @staticmethod
    def _no_extension(fname):
        """Handle case for no extension (normally folder supplied)"""
        # If there is no extension, the data is in a folder. This is consistent with SOLEIL CASSIOPEE Fermi maps or
        # CLF Artemis data

        # Extract identifiable data from the file to determine if the location is SOLEIL CASSIOPEE
        file_list = natsort.natsorted(os.listdir(fname))
        file_list_ROI = [
            item for item in file_list if "ROI1_" in item and isfile(join(fname, item))
        ]
        if len(file_list_ROI) > 1:  # Must be SOLEIL CASSIOPEE
            return "Soleil_Casiopee_ARPES"
        else:  # Likely CLF Artemis
            # Extract identifiable data from the file to determine if the location is CLF Artemis
            file_list_Neq = [item for item in file_list if "N=" in item]
            if len(file_list_Neq) > 0:  # Must be CLF Artemis
                return "CLF_Artemis"
        # If we are unable to determine the location, fallback to the default handler
        IdentifyLoc._default_handler(fname)

    @staticmethod
    def _handler_xy(fname):
        # If the file is .xy format, the location must be either MAX IV Bloch-spin, StA-Phoibos or StA-Bruker

        # Open the file and load the first line
        with open(fname) as f:
            line0 = f.readline()

        # If measurement was performed using Specs analyser, location must be MAX IV Bloch-spin or StA-Phoibos
        if "SpecsLab" in line0:
            with open(fname) as f:
                for _i in range(30):
                    # If the 'PhoibosSpin' identifier is present in any of the lines, location must be MAX IV Bloch-spin
                    if "PhoibosSpin" in f.readline():
                        return "MAX IV Bloch-spin"
            # Otherwise by default, assume StA-Phoibos
            return "StA_Phoibos"

        IdentifyLoc._default_handler(fname)

    @staticmethod
    def _handler_sp2(fname):
        # If the file is .sp2 format, the location must be MAX IV Bloch-spin
        return "MAX IV Bloch-spin"

    @staticmethod
    def _handler_krx(fname):
        # If the file is .krx format, the location must be StA-MBS
        return "StA_MBS"

    @staticmethod
    def _handler_txt(fname):
        # If the file is .txt format, the location must be StA-MBS, MAX IV Bloch, Elettra APE or SOLEIL CASSIOPEE

        # Open the file and load the first line
        with open(fname) as f:
            line0 = f.readline()

        # MAX IV Bloch, Elettra APE or SOLEIL CASSIOPEE .txt files follow the same SES data format, so we can identify
        # the location from the location line in the file
        if line0 == "[Info]\n":  # Identifier of the SES data format
            from peaks.core.fileIO.base_data_classes.base_data_class import (
                BaseDataLoader,
            )

            # Extract metadata dictionary from file
            metadata = BaseDataLoader.load_metadata(fname, loc="SES", quiet=True)
            location_identifier = metadata.get("local_location_identifier")
            if (
                "bloch" in location_identifier.lower()
                or "maxiv" in location_identifier.lower()
            ):
                return "MAXIV_Bloch_A"
            elif (
                "ape" in location_identifier.lower()
                or "elettra" in location_identifier.lower()
            ):
                return "Elettra_APE_LE"
            elif (
                "cassiopee" in location_identifier.lower()
                or "soleil" in location_identifier.lower()
            ):
                return "Soleil_Casiopee_ARPES"

        elif "Lines" in line0:
            # This should be MBS format, default to StA loader
            return "StA_MBS"

    @staticmethod
    def _handler_zip(fname):
        # If the file is .zip format, the file must be of SES format. Thus, the location must be MAX IV Bloch,
        # Elettra APE, SOLEIL CASSIOPEE or Diamond I05-nano (defl map)
        from peaks.core.fileIO.base_data_classes.base_data_class import BaseDataLoader

        # Extract metadata dictionary from file
        metadata = BaseDataLoader.load_metadata(fname, loc="SES", quiet=True)
        location_identifier = metadata.get("local_location_identifier")
        if (
            "bloch" in location_identifier.lower()
            or "maxiv" in location_identifier.lower()
        ):
            return "MAXIV_Bloch_A"
        elif (
            "ape" in location_identifier.lower()
            or "elettra" in location_identifier.lower()
        ):
            return "Elettra_APE_LE"
        elif (
            "cassiopee" in location_identifier.lower()
            or "soleil" in location_identifier.lower()
        ):
            return "Soleil_Casiopee_ARPES"
        elif (
            "i05" in location_identifier.lower()
            or "diamond" in location_identifier.lower()
        ):
            return "Diamond I05-nano"

    @staticmethod
    def _handler_ibw(fname):
        from peaks.core.fileIO.base_data_classes.base_ibw_class import BaseIBWDataLoader

        # Read the wavenote
        wavenote = BaseIBWDataLoader._load_metadata(fname)["wavenote"]

        # If the file is .ibw format, the file is likely SES format.
        if "SES" in wavenote:
            from peaks.core.fileIO.base_arpes_data_classes.base_ses_class import (
                SESDataLoader,
            )

            metadata_dict_SES_keys = SESDataLoader._SES_metadata_to_dict_w_SES_keys(
                wavenote.split("\r")
            )
            location = metadata_dict_SES_keys.get("Location")
            if "bloch" in location.lower() or "maxiv" in location.lower():
                return "MAXIV_Bloch_A"
            elif "ape" in location.lower() or "elettra" in location.lower():
                return "Elettra_APE_LE"
            elif "cassiopee" in location.lower() or "soleil" in location.lower():
                return "Soleil_Casiopee_ARPES"
            else:  # Return general SES loader
                return "SES"
        # If we are unable to find a location, define location as a generic ibw file
        return "ibw"

    @staticmethod
    def _handler_nxs(fname):
        # If the file is .nxs format, the location should be Diamond I05-nano or Diamond I05-HR

        from peaks.core.fileIO.base_data_classes.base_hdf5_class import (
            BaseHDF5DataLoader,
        )

        # Open the file (read only)
        with h5py.File(fname, "r") as f:
            # .nxs files at Diamond and Alba contain approximately the same identifier format
            identifier = BaseHDF5DataLoader._extract_hdf5_value(
                f, "entry1/instrument/name"
            )
        # From the identifier, determine the location
        if "i05-1" in identifier:
            return "Diamond_I05_Nano-ARPES"
        elif "i05" in identifier:
            return "Diamond_I05_ARPES"

    @staticmethod
    def _handler_h5(fname):
        # Use CLF_Artemis loader which will in turn call the FeSuMa loader for the key things
        return "CLF_Artemis"

    @staticmethod
    def _handler_nc(fname):
        # If the file is .nc format, the location should be NetCDF
        return "NetCDF"

    @staticmethod
    def _handler_zarr(fname):
        # If the file is .zarr format, the location should be Zarr
        return "Zarr"

    @staticmethod
    def _handler_cif(fname):
        # If the file is .cif format, the location should be cif
        return "cif"
