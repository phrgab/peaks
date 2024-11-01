import xarray as xr
import pint
import pint_xarray

ureg = pint_xarray.unit_registry


def _ensure_units(t, unit=None):
    """Ensure that the time is in units, taking the units of the time axis of the data by default."""
    if isinstance(t, (int, float)):
        unit = unit or self._obj.t.units
        return t * ureg(unit)
    elif isinstance(t, pint.Quantity) and unit:
        return t.to(unit)
    return t


def mean(da):
    """Calculate the mean of the data over the time dimension."""
    mean = da.mean("t", keep_attrs=True)
    return mean.history.assign(
        f"Integrated over time from {da.t.min().data} to {self._obj.t.max().data}"
    )


def static(da, t_static=-250.0 * ureg("fs")):
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
    t_static = _ensure_units(t_static)
    static = da.pint.sel(t=slice(None, t_static)).mean("t", keep_attrs=True)
    return static.history.assign(f"Static ARPES calculated from data up to {t_static}")


def diff(da, t_select, t_static=-250.0 * ureg("fs")):
    """Calculate the difference spectrum of the data at some time point or time window and the static data.

    Parameters
    -----------
    da : xarray.DataArray
        time-resolved data, with a time axis with dimension "t".

    t_select : pint.Quantity or float or slice()
        time to select data over. If no units are provided, the units of the data are assumed.
        If a slice is provided, the mean over the slice is calculated.

    t_static : pint.Quantity or float, optional
        time point to assume static up to. If no units are provided, the units of the data are assumed.
        default = -250 fs
    """

    if isinstance(t_select, slice):
        units = [
            t.units
            for t in [t_select.start, t_select.stop]
            if isinstance(t, pint.Quantity)
        ]
        if not units:
            unit = None
        else:
            unit = units[0]
        t_select = [_ensure_units(t, unit) for t in [t_select.start, t_select.stop]]
        t_select = slice(*t_select)
    else:
        t_select = _ensure_units(t_select)
    static = self.static(da, t_static)
    if isinstance(t_select, slice):
        excited = da.pint.sel(t=t_select).mean("t", keep_attrs=True)
    else:
        excited = da.pint.sel(t=t_select, method="nearest")
    diff = excited - static
    return diff.history.assign(
        f"Difference spectrum: {t_select} - static data (t<{t_static})"
    )
