"""Allows a function to act as an OOP method.

"""

# Lewis Hart 20/09/2021
# Brendan Edwards 16/10/2023

from functools import wraps


def add_methods(cls):
    """Decorator (function wrapper) used to allow a function to be used as an object-oriented programming method.

    Parameters
    ------------
    cls : object
        Here, this will be xr.DataArray. This allows a function to be applied to an xarray as a method.

    Examples
    ------------
    from peaks import *

    my_data = load('my_file.ibw')

    @add_methods(xr.DataArray)  # Turns the following function (data_plus_1) into a xr.DataArray method

    def data_plus_1(data):

        return data+1

    my_data_plus_1 = my_data.data_plus_1()  # Create a new xarray equal to my_data + 1

    """

    # decorator for the function - func
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        setattr(cls, func.__name__, wrapper)
        # Note we are not binding func, but instead the wrapper which accepts self but does exactly the same as func
        return func  # returning func means func can still also be used normally (i.e. not as a method)

    return decorator
