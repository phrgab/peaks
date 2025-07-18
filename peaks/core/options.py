"""Classes to store peaks options."""

from peaks.core.utils.misc import format_colored_dict


class FileIOOptions:
    """Class used to set default options for file loading user-defined file paths, file extensions, and location.

    This class allows setting file paths, extensions, and location through its properties. Multiple paths and extensions
    can be provided as lists. The reset methods are used to reset specific options.

    Examples
    --------
    Example usage is as follows::

        import peaks as pks

        # Define file paths
        pks.opts.FileIO.path = ['sample1/i05-1-12', 'sample2/i05-1-12']

        # Define file extensions
        pks.opts.FileIO.ext = ['nxs', 'zip']

        # Define location
        pks.opts.FileIO.loc = 'Diamond I05-nano'

        # Show file options
        pks.opts.FileIO

        # Reset all FileIO optsions
        pks.opts.FileIO.reset()

        # Reset all options
        pks.opts.reset()
    """

    def __init__(self):
        self._path = None
        self._ext = None
        self._loc = None
        self._lazy_size = 1000000000  # Default lazy load size

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        if value is None:
            self._path = None
        elif isinstance(value, str) or (
            isinstance(value, list) and all(isinstance(p, str) for p in value)
        ):
            self._path = value
        else:
            raise TypeError(
                "Path must be a string or a list of strings pointing to desired file paths."
            )

    @path.deleter
    def path(self):
        self._path = None

    @property
    def ext(self):
        return self._ext

    @ext.setter
    def ext(self, value):
        if value is None:
            self._ext = None
        elif isinstance(value, str) or (
            isinstance(value, list) and all(isinstance(e, str) for e in value)
        ):
            self._ext = value
        else:
            raise TypeError(
                "Extension must be a string or a list of strings pointing to desired file extensions."
            )

    @ext.deleter
    def ext(self):
        self._ext = None

    @property
    def loc(self):
        return self._loc

    @loc.setter
    def loc(self, value):
        if value is None:
            self._loc = None
        elif isinstance(value, str):
            from peaks.core.fileIO.loc_registry import LOC_REGISTRY

            if value in LOC_REGISTRY:
                self._loc = value
            else:
                raise ValueError(
                    f"Location '{value}' is not in the list of available locations: {set(LOC_REGISTRY.keys())}"
                )
        else:
            raise TypeError(
                f"Location must be a string specifying an available file loader from {set(LOC_REGISTRY.keys())}."
            )

    @loc.deleter
    def loc(self):
        self._loc = None

    @property
    def lazy_size(self):
        return self._lazy_size

    @lazy_size.setter
    def lazy_size(self, value):
        if isinstance(value, int):
            self._lazy_size = value
        else:
            raise TypeError(
                "Lazy size must be an integer representing the file size (in bytes) for lazy loading."
            )

    @lazy_size.deleter
    def lazy_size(self):
        self._lazy_size = 1000000000  # Reset to default

    def reset(self):
        """Reset the file path, extension, location, and lazy size."""
        self._path = None
        self._ext = None
        self._loc = None
        self._lazy_size = 1000000000

    def __repr__(self):
        """Return a string representation of the current file variables."""
        return format_colored_dict(self.dict())

    def set(self, **kwargs):
        """Set the file path, extension, location, and lazy size.

        Parameters
        ----------
        kwargs : dict
            A dictionary of keyword arguments to set the file path, extension, location, and lazy size.
        """
        if "path" in kwargs:
            self.path = kwargs.pop("path")
        if "ext" in kwargs:
            self.ext = kwargs.pop("ext")
        if "loc" in kwargs:
            self.loc = kwargs.pop("loc")
        if "lazy_size" in kwargs:
            self.lazy_size = kwargs.pop("lazy_size")

        if kwargs:
            raise ValueError(
                f"Invalid keyword argument(s): {set(kwargs.keys())}. "
                f"Expected options from {set(self.dict().keys())}"
            )

    def dict(self):
        """Return a dictionary representation of the current file variables."""
        raw_dict = vars(self)
        return {k.lstrip("_"): raw_dict[k] for k in raw_dict}


class Options:
    """
    Singleton class to hold all fixed option groups, such as FileIO.

    This class provides a centralized way to manage various options used in the `peaks` package.
    It provides methods to reset, display, and access these options, and can be used with a context manager.

    Attributes
    ----------
    FileIO : FileIOOptions
        An instance of the FileIOOptions class to manage file input/output settings.

    Methods
    -------
    reset()
        Reset all option groups to their default values.

    Examples
    --------
    Example usage is as follows::

        import peaks as pks

        # Set some FileIO options
        pks.opts.FileIO.path = ['sample1/i05-1-12', 'sample2/i05-1-12']  # Default paths
        pks.opts.FileIO.ext = ['nxs', 'zip']  # Default extensions
        pks.opts.FileIO.loc = 'Diamond_I05_nano-ARPES'  # loc to use
        pks.opts.FileIO.lazy_size = 500000000  # Set lazy size to 500 Mb

        # Display all the current options
        pks.opts

        # Clear the location
        del pks.opts.FileIO.loc

        # Reset all options
        pks.opts.reset()

    Can also be used as a context manager to temporarily set options::

            import peaks as pks

            with pks.opts as opts:
                opts.FileIO.path = 'sads'
                opts.FileIO.loc = None
                opts.FileIO.ext = ['nxs', 'zip']
                opts.FileIO.lazy_size = 500000000

                # Display all the current options
                print(pks.opts)

            # Options are reset to their original state
            pks.opts
    """

    _instance = None
    _fileio_old_opts = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Options, cls).__new__(cls)
            cls._instance.FileIO = FileIOOptions()  # Initialize FileIO options here
        return cls._instance

    def reset(self):
        """Reset all option groups."""
        self.FileIO.reset()

    def dict(self):
        """Return a dictionary representation of the current options."""
        return {k: v.dict() for k, v in vars(self).items() if not k.startswith("_")}

    def __enter__(self):
        """Enter the context manager, storing the current state."""
        self._fileio_old_opts = self.FileIO.dict().copy()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager, resetting the state to the stored state."""
        if self._fileio_old_opts is not None:
            self.FileIO.set(**self._fileio_old_opts)
        self._fileio_old_opts = None

    def __repr__(self):
        return format_colored_dict(self.dict())


# Create the global `opts` instance
opts = Options()
