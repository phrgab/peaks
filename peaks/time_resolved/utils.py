import pint_xarray

ureg = pint_xarray.unit_registry


def _set_t0(da, t0, delay_line_no_trips=2, assign=True):
    """Set a new t0 in a TR-ARPES data set

    Parameters
    -----------
    da : xarray.DataArray
        da : xarray.DataArray
        time-resolved data, with a time axis with dimension "t", and a metadata attribute pump.t0_position defining
        the delay line position corresponding to t0.

    t0 : pint.Quantity or float
        New time zero, expressed as a time based on the current time axis scaling.

    delay_line_no_trips : int, optional
        Number of times delay line is traversed. Default is 2 (there and back).

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
    delay_position = da.delay_pos.copy(deep=True)

    if isinstance(t0, (float, int)):
        t0 = t0 * original_t_units

    # Get current t0 in stage position
    t0_old_position = da.metadata.pump.t0_position
    if t0.check("[length]"):  # If provided as a length, set that directly
        t0_new_position = t0
    else:  # Otherwise convert the given t0 time into a stage position
        # Work out different in stage position for new t0
        t0_delta = (t0 / (delay_line_no_trips / ureg.c)).to(original_stage_units)
        t0_new_position = t0_old_position + t0_delta
    # Convert to time in original units
    delay_time = (
        (delay_position - t0_new_position) * delay_line_no_trips / ureg.c
    ).pint.to(original_t_units)
    hist_str = (
        f"New t0 defined as {t0}. Stage position corresponding to t0 updated - "
        f"old: {t0_old_position}, new: {t0_new_position}"
    )

    if assign:  # Assign rather than update in place
        # Make new da with updated delay dim
        new_da = da.copy(deep=True).assign_coords(
            {"t": delay_time.pint.dequantify(), "delay_pos": ("t", delay_position.data)}
        )
        new_da.t.attrs["units"] = delay_time.data.units
        # Update the metadata
        new_da.metadata.pump.set("t0_position", t0_new_position, add_history=False)
        new_da.history.add(hist_str)
        return new_da

    # Update the metadata in place
    da.metadata.pump.set("t0_position", t0_new_position, add_history=False)
    da.history.add(hist_str)
    # Update the time axis
    da["t"] = delay_time.data.magnitude
    da.t.attrs["units"] = delay_time.data.units
    da["delay_pos"] = ("t", delay_position.data)


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


def set_t0_like(da, da_ref):
    """Set t0 of a time-resolved data set to match another data set. Assumes the default delay line round trips of 2.

    Parameters
    -----------
    da : xarray.DataArray
        time-resolved data, with a time axis with dimension "t", and a metadata attribute pump.t0_position defining
        the delay line position corresponding to t0.

    da_ref : xarray.DataArray
        Reference time-resolved data set, with the t0 already calibtrated
    """

    t0 = da_ref.metadata.pump.t0_position
    set_t0(da, t0)


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


def assign_t0_like(da, da_ref):
    """Set t0 of a time-resolved data set to match another data set. Assumes the default delay line round trips of 2.

    Parameters
    -----------
    da : xarray.DataArray
        time-resolved data, with a time axis with dimension "t", and a metadata attribute pump.t0_position defining
        the delay line position corresponding to t0.

    da_ref : xarray.DataArray
        Reference time-resolved data set, with the t0 already calibtrated

    Returns
    -------
    xarray.DataArray
        The updated xarray.DataArray
    """

    return assign_t0(da, da_ref.metadata.pump.t0_position)
