import pint_xarray

ureg = pint_xarray.unit_registry


def _set_t0(da, t0, delay_line_roundtrips=2, assign=True):
    """Set a new t0 in a TR-ARPES data set

    Parameters
    -----------
    da : xarray.DataArray
        da : xarray.DataArray
        time-resolved data, with a time axis with dimension "t", and a metadata attribute pump.t0_position defining
        the delay line position corresponding to t0.

    t0 : pint.Quantity or float
        New time zero, expressed as a time based on the current time axis scaling.

    delay_line_roundtrips : int, optional
        Number of round trips in the delay line. Default is 2.

    assign : bool, optional
        If True, the new t0 is assigned to the :class:xarray.DataArray.
        If False, the :class:xarray.DataArray is updated in place

    Retruns
    -------
    xarray.DataArray : optional
        If assign is False, the updated xarray.DataArray is returned.

    See Also
    --------
    set_t0
    assign_t0
    """

    if not t0:  # No correction to do if t0 = 0
        return da

    original_t_units = da.t.units
    original_stage_units = da.metadata.pump.t0_position.units

    if isinstance(t0, (float, int)):
        t0 = t0 * original_t_units

    # Get current t0 in stage position
    t0_old_position = da.metadata.pump.t0_position

    # Work out different in stage position for new t0
    t0_delta = (t0 / (delay_line_roundtrips / constants.c)).to(original_stage_units)
    t0_new_position = t0_old_position + t0_delta

    # Convert to time in original units
    delay_time = ((da.delay_pos - t0_new_position) * delay_line_roundtrips / ureg.c).to(
        original_t_units
    )
    hist_str = (
        f"New t0 defined as {t0}. Stage position corresponding to t0 updated - "
        f'old: {t0_old_position}, new: {da.attrs["t0"]}'
    )

    if assign:
        new_da = da.assign_coords({"t": delay_time})
        new_da.metadata.pump.t0_position.set(t0_new_position, add_history=False)
        new_da.history.add(hist_str)
        return new_da

    da.metadata.pump.t0_position.set(t0_new_position, add_history=False)
    da.history.add(hist_str)
    da[t] = delay_time


def set_t0(da, t0, delay_line_roundtrips=2):
    """Set a new t0 for a time-resolved data set

    Parameters
    -----------
    da : xarray.DataArray
        da : xarray.DataArray
        time-resolved data, with a time axis with dimension "t", and a metadata attribute pump.t0_position defining
        the delay line position corresponding to t0.

    t0 : pint.Quantity or float
        New time zero, expressed as a time based on the current time axis scaling.

    delay_line_roundtrips : int, optional
        Number of round trips in the delay line. Default is 2.

    Retruns
    -------
    None
        The :class:xarray.DataArray is updated in place
    """

    _set_t0(da, t0, delay_line_roundtrips, assign=False)


def assign_t0(da, t0, delay_line_roundtrips=2):
    """Set a new t0 in a TR-ARPES data set

    Parameters
    -----------
    da : xarray.DataArray
        da : xarray.DataArray
        time-resolved data, with a time axis with dimension "t", and a metadata attribute pump.t0_position defining
        the delay line position corresponding to t0.

    t0 : pint.Quantity or float
        New time zero, expressed as a time based on the current time axis scaling.

    delay_line_roundtrips : int, optional
        Number of round trips in the delay line. Default is 2.

    Retruns
    -------
    xarray.DataArray
        The updated xarray.DataArray
    """

    return _set_t0(da, t0, delay_line_roundtrips, assign=True)
