"""Allows a function to act as an OOP method.

"""

# Lewis Hart 20/09/2021
# Brendan Edwards 16/10/2023

from functools import wraps
from peaks.utils import analysis_warning


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
        from peaks.utils.OOP_method import add_methods

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
