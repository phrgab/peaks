# Functions to process time-resolved ARPES data
# Phil King 14/10/2022

import numpy as np
import xarray as xr

from peaks.utils.metadata import update_hist
from peaks.utils.OOP_method import add_methods

# Set t0
@add_methods(xr.DataArray)
def set_t0(data, t0):
    '''Set a new t0 in a TR-ARPES data set

    Parameters
    -----------
    data : chunked(xr.DataArray)
        raw data (t x E x ang)

    t0 : float
        New time zero (fs)

    Retruns :
        chunked(xr.DataArray)
    '''

    if t0:
        # Get current t0 in stage position
        t0_old = data.attrs['t0']

        # Work out different in stage position for new t0
        t0_delta = t0/(2/299704645*1e12)

        t0_new = t0_old + t0_delta

        # Convert to time in fs
        delay_time = (data.delay_pos - t0_new) * 2/299704645*1e12

        data.attrs['t0'] = np.round(t0_new,4)

        # Update the analysis history
        hist = f'stage t0 position updated - old: {t0_old}, new: {data.attrs["t0"]}'
        update_hist(data, hist)

        return data.assign_coords({'t': delay_time})

    else:  # No correction to do if t0 = 0
        return data

# Static ARPES
@add_methods(xr.DataArray)
def static(data, t_static=-250.):
    '''Calculate the static ARPES from a time-resolved ARPES experiment.
    Assumes that all data points recorded for a time < t_static are equilibrium.

    Parameters
    -----------
    data : xr.DataArray
        raw data (t x E x ang)

    t_static : float, optional
        time point to assume static up to (fs)
        default = -250 fs
    '''

    # Calculate the static data
    static = data.where(data.t < t_static).mean('t', keep_attrs=True)

    # Update the analysis history
    hist = f'Static data, for t<{t_static} fs'
    update_hist(static, hist)

    return static


# Difference plots
@add_methods(xr.DataArray)
def t_diff(data, t_select, t_static=-250):
    '''Calculate the difference spectrum of the ARPES at some time point
    or time window and the static ARPES

    Parameters
    -----------
    data : xr.DataArray
        raw data (E vs ang vs t)

    t_select : float or slice
        time to select data over

    t_static : float, optional
        time point to assume static up to (fs)
        default = -250 fs
    '''

    static = data.static(t_static)

    if isinstance(t_select, (int,float)):
        excited = data.sel(t=t_select, method='nearest')
    else:
        excited = data.sel(t=t_select).mean('t', keep_attrs=True)

    # Calculate the difference
    diff_data = (excited-static)

    diff_data.attrs = data.attrs

    # Update the analysis history
    hist = f'Difference data: {t_select} - static data (t<{t_static}) fs'
    update_hist(diff_data, hist)

    return diff_data

