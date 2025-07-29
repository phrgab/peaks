import importlib
import sys
from functools import wraps

import xarray as xr

from peaks.core.utils.misc import analysis_warning


class LazyAccessorDescriptor:
    """Descriptor for lazy-loading accessors, providing documentation and lazy execution."""

    def __init__(self, func_name, module_name, accessor_class):
        self.func_name = func_name
        self.module_name = module_name
        self.accessor_class = accessor_class
        self._func = None  # Cached function after loading

    def __get__(self, instance, owner):
        if instance is None:
            return self  # Accessed on the class, return the descriptor itself
        return self.accessor_class(instance, self.module_name, self.func_name)

    @property
    def __doc__(self):
        """Provide the docstring of the target function."""
        if self._func is None:
            self._load_function()
        return self._func.__doc__

    def _load_function(self):
        """Load the function and cache it."""
        module = importlib.import_module(self.module_name)
        self._func = getattr(module, self.func_name)

    def __dir__(self):
        """Provide autocompletion for the target function's attributes."""
        if self._func is None:
            self._load_function()
        return dir(self._func)

    def __getattr__(self, name):
        """Forward attribute access to the target function."""
        if self._func is None:
            self._load_function()
        return getattr(self._func, name)


class PeaksDirectCallAccessor:
    """Lazy-loading accessor for functions applied to xarray DataArrays."""

    def __init__(self, data_array, module_name, func_name):
        self.data_array = data_array
        self.module_name = module_name
        self.func_name = func_name
        self._func = None  # Cache for the function

    def _load_function(self):
        """Loads the function from the specified module, updating the cache if necessary."""
        # Always reload the module to ensure we get the latest version (compatible with %autoreload)
        module = importlib.import_module(self.module_name)
        if module.__name__ in sys.modules:
            importlib.reload(sys.modules[module.__name__])
        self._func = getattr(module, self.func_name)
        # Update the __doc__ attribute
        self.__doc__ = (
            ":::{note} "
            f"This is an accessor to the function `{self.module_name}.{self.func_name}` acting "
            f"directly on the xarray object. (Note, the :class:`xarray.DataArray`, :class:`xarray.Dataset` "
            f"or :class:`xarray.DataTree` does not now need to be explicitly passed to the function)."
            f":::\n\n"
            f"{self._func.__doc__ or ''}"
        )

    @property
    def func(self):
        """Property to access the loaded function, loading it if necessary."""
        if self._func is None:
            self._load_function()
        return self._func

    def __call__(self, *args, **kwargs):
        """Calls the loaded function with the DataArray and provided arguments."""
        return self.func(self.data_array, *args, **kwargs)

    def __dir__(self):
        """Provide autocompletion for the original function's attributes."""
        return dir(self.func)

    def __getattr__(self, name):
        """Forward attribute access to the loaded function if available."""
        return getattr(self.func, name)


class PeaksDataTreeIteratorAccessor(PeaksDirectCallAccessor):
    """Accessor that applies the function to each node in a DataTree.

    Assumes:
    - A hollow tree structure.
    - Leaves hold `xarray.Dataset` objects.
    - If the Dataset is the primary data container, it will have metadata attributes.
    - Otherwise, the Dataset will hold a single `xarray.DataArray` with the data variable `data`.
    """

    def __init__(self, data_tree, module_name, func_name):
        super().__init__(data_tree, module_name, func_name)
        self.data_tree = data_tree  # Overwrite data_array with data_tree

    def __call__(self, *args, **kwargs):
        """Iterates over DataTree nodes and applies the function."""
        func = self.func  # Ensure the function is loaded

        def apply_func(node):
            if hasattr(node, "_scan"):
                # Apply the function to the node directly
                return func(node, *args, **kwargs)
            elif hasattr(node, "data") and len(node) == 1:
                # Map over the Dataset which contains only a single DataArray
                return node.map(lambda da: func(da, *args, **kwargs))
            elif len(node) == 0:
                pass
            else:
                raise ValueError(
                    "For use of this `peaks` accessor, leaf nodes of the xr.DataTree must be a xr.Dataset with "
                    "a single xr.DataArray with key `data`, or the Dataset itself must be the primary data container, "
                    f"itself holding metadata. The current node does not meet these requirements: {node}."
                )

        # Use DataTree's map_over_datasets method to apply the function
        return self.data_tree.map_over_datasets(apply_func)

    def __dir__(self):
        """Provide autocompletion for the original function's attributes."""
        return dir(self.func)

    def __getattr__(self, name):
        """Forward attribute access to the loaded function if available."""
        return getattr(self.func, name)


def _pass_function_to_xarray_class_accessor(func_name, module_name):
    # Import the module and get the function once
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)

    @wraps(func)
    def method(self, *args, **kwargs):
        # Check if the object is a DataTree and use PeaksDataTreeIteratorAccessor
        if isinstance(self._obj, xr.DataTree):
            accessor = PeaksDataTreeIteratorAccessor(self._obj, module_name, func_name)
            return accessor(*args, **kwargs)
        else:
            # Call the original function with self._obj
            return func(self._obj, *args, **kwargs)

    # Modify the docstring to include the custom note
    note = (
        ":::{note} "
        f"This is an accessor to the function `{module_name}.{func_name}` acting "
        f"directly on the :class:`xarray` object. (Note, the :class:`xarray.DataArray`, :class:`xarray.Dataset` "
        f"or :class:`xarray.DataTree` does not now need to be explicitly passed to the function)."
        f":::\n\n"
    )
    method.__doc__ = note + (func.__doc__ or "")
    return method


def register_lazy_accessor(
    func_name, module_name, cls, accessor_class=PeaksDirectCallAccessor
):
    """Register a lazy-loading accessor as a descriptor on the class."""

    # Check if the accessor already exists on cls and issue a warning if so
    if hasattr(cls, func_name):
        analysis_warning(
            f"Registration of accessor under name '{func_name}' for type "
            f"{cls.__module__}.{cls.__name__} is overriding an existing attribute with the same name.",
            title="Registration of accessor",
            warn_type="warning",
        )

    # Register the accessor descriptor on cls
    setattr(
        cls, func_name, LazyAccessorDescriptor(func_name, module_name, accessor_class)
    )


def register_accessor(cls):
    """Decorator (function wrapper) used to allow a function to be used as an object-oriented programming method.

    Parameters
    ------------
    cls : object
        Here, this will be an:class:`xarray.DataArray` or similar. This allows a function to be applied to that
        class as a method.

    Examples
    ------------
    Example usage is as follows::

        import peaks as pks
        from peaks.core.accessors.accessor_methods import register_accessor

        my_data = pks.load('my_file.ibw')

        # Turn the function (data_plus_1) into a xarray.DataArray method
        @register_accessor(xr.DataArray)
        def data_plus_1(data):
            return data+1

        # Create a new xarray equal to my_data + 1
        my_data_plus_1 = my_data.data_plus_1()

    """

    # decorator for the function - func
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Give a warning if already exists
        if hasattr(cls, func.__name__):
            analysis_warning(
                f"Registration of accessor for `peaks` under name {func.__name__!r} for type "
                f"{cls.__module__}.{cls.__name__} is overriding a preexisting attribute with the same name.",
                title=" Registration of accessor",
                warn_type="warning",
            )

        setattr(cls, func.__name__, wrapper)
        # Note we are not binding func, but instead the wrapper which accepts self but does exactly the same as func
        return func  # returning func means func can still also be used normally (i.e. not as a method)

    return decorator
