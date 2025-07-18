import pint
import pint_xarray

ureg = pint_xarray.unit_registry


def _ensure_units(da, t, unit=None):
    """Ensure that the time is in units, taking the units of the time axis of the data by default."""
    if isinstance(t, (int, float)):
        unit = unit or da.t.units
        return t * ureg(unit)
    elif isinstance(t, pint.Quantity) and unit:
        return t.to(unit)
    return t


def mean(da):
    """Calculate the mean of the data over the time dimension."""
    mean = da.mean("t", keep_attrs=True)
    return mean.history.assign(
        f"Integrated over time from {da.t.min().data * da.t.units} to {da.t.max().data * da.t.units}"
    )


def static(da, t_static=None):
    """Calculate the static spectrum from a time-resolved experiment.
    Assumes that all data points recorded for a time < t_static are equilibrium.

    Parameters
    -----------
    da : xarray.DataArray
        time-resolved data, with a time axis with dimension "t".

    t_static : pint.Quantity or float, optional
        time point to assume static up to. If no units are provided, the units of the data are assumed.
        default = -250 fs
    """
    if t_static is None:
        t_static = -250.0 * ureg("fs")
    t_static = _ensure_units(da, t_static)
    t_static_axis_units = t_static.to(da.t.units).magnitude
    return (
        da.sel(t=slice(None, t_static_axis_units))
        .mean("t", keep_attrs=True)
        .history.assign(f"Static ARPES calculated from data up to {t_static}")
    )


def diff(da, t_select=None, t_static=None):
    """Calculate the difference spectrum of the data at some time point or time window and the static data.

    Parameters
    -----------
    da : xarray.DataArray
        time-resolved data, with a time axis with dimension "t".

    t_select : pint.Quantity or float or slice() or None, optional
        time to select data over. If no units are provided, the units of the data are assumed.
        If a slice is provided, the mean over the slice is calculated.
        If None, the difference of the full data cube is calculated against the static data.
        default = None

    t_static : pint.Quantity or float, optional
        time point to assume static up to. If no units are provided, the units of the data are assumed.
        default = -250 fs
    """
    if t_static is None:
        t_static = -250.0 * ureg("fs")

    axis_units = da.t.units
    t_select_ = t_select
    if isinstance(t_select, slice):
        t0 = (
            t_select.start.to(axis_units).magnitude
            if isinstance(t_select.start, pint.Quantity)
            else t_select.start
        )
        t1 = (
            t_select.stop.to(axis_units).magnitude
            if isinstance(t_select.stop, pint.Quantity)
            else t_select.stop
        )
        t_select = slice(t0, t1)
    elif isinstance(t_select, pint.Quantity):
        t_select = t_select.to(axis_units).magnitude

    static_ = static(da, t_static)
    if isinstance(t_select, slice):
        excited = da.sel(t=t_select).mean("t", keep_attrs=True)
    elif t_select is not None:
        excited = da.sel(t=t_select, method="nearest")
    else:
        excited = da
    diff = excited - static_
    return diff.history.assign(
        f"Difference spectrum: {t_select_ or 'all'} - static data (t<{t_static})"
    )
