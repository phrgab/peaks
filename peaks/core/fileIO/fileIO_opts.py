"""Classes to store data- or beamline-specific loading options.

"""


class File:
    """Class used to store and print user-defined file paths (optionally including some of the file name for
    disambiguation e.g. path_to_data/i05-123), file extensions, and location (e.g. beamline). Note: multiple file paths
    or extensions can be defined as lists to allow for the use of data saved at different folders, with different
    extensions. To clear the variables, set the entry to `None` or use the clear_path, clear_ext, or clear_loc methods.

    Examples
    ------------
    Example usage is as follows::

        import peaks as pks

        # Define file paths
        pks.File.path = ['sample1/i05-1-12', 'sample2/i05-1-12']

        # Define file extensions
        pks.File.ext = ['nxs', 'zip']

        # Define location
        pks.File.loc = ['Diamond I05-nano']

        # Print file options
        pks.File()

        # Clear the location
        pks.File.clear_loc()

    Methods
    ------------
    __init__:
        Prints the current file variables.
    clear_path:
        Clears the path variable.
    clear_ext:
        Clears the ext variable.
    clear_loc:
        Clears the loc variable.

    """

    def __init__(self):
        """This function prints the current file variables."""

        print(f"File.path: {self.path};")
        print(f"File.ext: {self.ext};")
        print(f"File.loc: {self.loc};")
        print(f"File.lazy_size: {self.lazy_size}.")

    @classmethod
    def clear_path(cls):
        """This function clears the path variable."""
        cls.path = None

    @classmethod
    def clear_ext(cls):
        """This function clears the ext variable."""
        cls.ext = None

    @classmethod
    def clear_loc(cls):
        """This function clears the loc variable."""
        cls.loc = None

    @classmethod
    def clear_lazy_size(cls):
        """This function clears the lazy_size variable."""
        cls.lazy_size = None

    # Initialise fpath variables as None
    path = None  # Path(s) to files
    ext = None  # Extension(s) of files
    loc = None  # Location where files were measured
    # Initialise lazy_size variable as 1 Gb
    lazy_size = 1000000000


class LocOpts:
    """Class used to store and print the currently supported locations (often beamlines) and to retrieve the relevant
     data conventions (axis names, sign conventions etc.).

    Examples
    ------------
    Example usage is as follows::

        import peaks as pks

        # Print currently supported locations
        pks.LocOpts()

        # Get the conventions class for Diamond I05-nano
        conventions = pks.LocOpts.get_conventions('Diamond I05-nano')

    """

    locs = []
    loc_conventions = {}

    def __init__(self):
        """This function prints the currently supported locations when LocOpts() is called."""

        # Print the currently supported locations
        print(sorted(self.locs))

    @classmethod
    def get_conventions(cls, loc):
        """Get the conventions class for the specified location.

        Parameters
        ------------
        loc : str
            The location (beamline) for which the conventions class is required.

        Returns
        ------------
        conventions : class
            The conventions class for the specified location.
        """

        # Return the conventions class for the specified location
        return cls.loc_conventions.get(loc, {})

    # List to store the currently supported locations
    # locs = [
    #     "ALBA LOREA",
    #     "CLF Artemis",
    #     "Diamond I05-HR",
    #     "Diamond I05-nano",
    #     "Elettra APE",
    #     "MAX IV Bloch",
    #     "MAX IV Bloch-spin",
    #     "SOLEIL CASSIOPEE",
    #     "StA-MBS",
    #     "StA-Phoibos",
    #     "StA-Bruker",
    #     "StA-LEED",
    #     "StA-RHEED",
    #     "Structure",
    #     "ibw",
    #     "NetCDF",
    # ]


def _register_location(cls):
    """This function registers a location (beamline) in the LocOpts class.

    Parameters
    ------------
    cls : class
        Class containing the location information defined by a class attribute loc_name.


    Examples
    ------------
    Example usage is as follows::

        # In the loader file for the beamline, import the _register_location function
        from peaks.core.fileIO.fileIO_opts import _register_location

        # Register the location when the location class is defined
        @_register_location
        class _I05NanoConventions(_ARPESConventions):  # Subclass the base class
            loc_name = "Diamond I05-nano"
            x1_name = "smx"
            # Overwrite other attributes as needed

        # Add the loader function to the import statement in the __init__.py file in the loaders folder

    """

    LocOpts.locs.append(cls.loc_name)
    LocOpts.loc_conventions[cls.loc_name] = cls
    return cls


# Base classes for data conventions (names and any angle etc. sign conventions)
class _BaseDataConventions:
    """Base Class used to store default definitions of loaders"""

    loc_name = "Default"
    loader = None
    metadata_loader = None

    @classmethod
    def get(cls, key, default=None):
        return getattr(cls, key, default)


class _BaseManipulatorConventions(_BaseDataConventions):
    """Base Data Conventions Class used to store default names and relative sign conventions for manipulator axes."""

    loc_name = "Default Manipulator"
    polar = 1
    tilt = 1
    azi = -1
    x1 = 1
    x2 = 1
    x3 = 1
    polar_name = "polar"
    tilt_name = "tilt"
    azi_name = "azi"
    x1_name = "x1"
    x2_name = "x2"
    x3_name = "x3"


class _BaseARPESConventions(_BaseManipulatorConventions):
    """Base Class used to store default definitions of relative angle signs between manipulator and detector and
    manipulator etc. axis names for ARPES BLs."""

    loc_name = "Default ARPES"
    ana_type = "Ip"
    theta_par = 1
    defl_par = 1
    defl_perp = 1
    ana_polar = 0
    ana_polar_name = None
    defl_par_name = None
    defl_perp_name = None


class _Coords(object):
    """Class used to store the units for different coordinates.

    Examples
    ------------
    Example usage is as follows::

        from peaks.core.fileIO.fileIO_opts import _coords

        # Get the units associated with x1 coordinates
        x1_units = _coords.units['x1']

    """

    # Dictionary to store the units for different coordinates
    units = {
        "theta_par": "deg",
        "polar": "deg",
        "tilt": "deg",
        "ana_polar": "deg",
        "defl_perp": "deg",
        "two_theta": "deg",
        "spin_rot_angle": "deg",
        "x1": "um",
        "x2": "um",
        "x3": "um",
        "x": "um",
        "y": "um",
        "z": "um",
        "location": "um",
        "y_scale": "mm",
        "hv": "eV",
        "KE_delta": "eV",
        "temp_sample": "K",
        "temp_cryo": "K",
    }
