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
        """Return the file path(s)."""
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
        """Return the file extension(s)."""
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
        """Return the location where the data were collected."""
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
        """Return the lazy loading file size threshold (in bytes)."""
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


_DEFAULT_MAX_VIEWERS = 1


class GuiOptions:
    """Options controlling interactive display panel GUIs.

    This class allows setting the maximum number of display panels or other GUIs
    that may be open simultaneously. Setting ``max_viewers`` to ``None`` disables
    the limit. The reset method restores the default display options.

    Examples
    --------
    Example usage is as follows::

        import peaks as pks

        # Set the maximum number of simultaneously open display panels
        pks.opts.gui.max_viewers = 5

        # Disable the display panel limit
        pks.opts.gui.max_viewers = None

        # Show display options
        pks.opts.gui

        # Reset all Gui options
        pks.opts.gui.reset()

        # Reset all options
        pks.opts.reset()
    """

    def __init__(self):
        self._max_viewers = _DEFAULT_MAX_VIEWERS

    @property
    def max_viewers(self):
        """Return the maximum number of simultaneously open display panels."""
        return self._max_viewers

    @max_viewers.setter
    def max_viewers(self, value):
        if value is None:
            self._max_viewers = None
        elif isinstance(value, bool):
            raise TypeError("Maximum viewers must be a positive integer or None.")
        elif isinstance(value, int):
            if value < 1:
                raise ValueError("Maximum viewers must be greater than or equal to 1.")
            self._max_viewers = value
        else:
            raise TypeError("Maximum viewers must be a positive integer or None.")

    @max_viewers.deleter
    def max_viewers(self):
        self._max_viewers = _DEFAULT_MAX_VIEWERS

    def reset(self):
        """Reset display options to their defaults."""
        self._max_viewers = _DEFAULT_MAX_VIEWERS

    def __repr__(self):
        """Return a string representation of the display options."""
        return format_colored_dict(self.dict())

    def set(self, **kwargs):
        """Set display options."""
        if "max_viewers" in kwargs:
            self.max_viewers = kwargs.pop("max_viewers")

        if kwargs:
            raise ValueError(
                f"Invalid keyword argument(s): {set(kwargs.keys())}. "
                f"Expected options from {set(self.dict().keys())}"
            )

    def dict(self):
        """Return a dictionary representation of the display options."""
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

    gui : GuiOptions
        Options controlling interactive display panels.

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

        # Allow a fixed number of display panels to be open simultaneously
        pks.opts.gui.max_viewers = 3

        # Disable the display-panel limit
        pks.opts.gui.max_viewers = None

        # Reset display options
        pks.opts.gui.reset()  # Disables multiple panels by defualt


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

                opts.gui.max_viewers = 3

                # Display all the current options
                print(pks.opts)

            # Options are reset to their original state
            pks.opts
    """

    _instance = None
    _old_opts = None

    def __new__(cls):
        """Return the singleton ``Options`` instance, creating it on first call."""
        if cls._instance is None:
            cls._instance = super(Options, cls).__new__(cls)
            cls._instance.FileIO = FileIOOptions()  # Initialize FileIO options
            cls._instance.gui = GuiOptions()  # Initialize GUI options
        return cls._instance

    def reset(self):
        """Reset all option groups."""
        self.FileIO.reset()
        self.gui.reset()

    def dict(self):
        """Return a dictionary representation of the current options."""
        return {k: v.dict() for k, v in vars(self).items() if not k.startswith("_")}

    def __enter__(self):
        """Enter the context manager, storing the current state."""
        self._old_opts = {
            name: option_group.dict().copy()
            for name, option_group in vars(self).items()
            if not name.startswith("_")
        }
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager, restoring the previous state."""
        if self._old_opts is not None:
            for name, values in self._old_opts.items():
                getattr(self, name).set(**values)

        self._old_opts = None

    def __repr__(self):
        return format_colored_dict(self.dict())


# Create the global `opts` instance
opts = Options()
