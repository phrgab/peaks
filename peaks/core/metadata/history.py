"""Helper functions for acting on metadata."""

import copy
import inspect
from datetime import datetime
from pprint import pprint
from typing import List, Union

import pandas as pd
import xarray as xr
from IPython.display import display
from pydantic import BaseModel, Field

from peaks import __version__


# Define the structure of a single analysis history record
class AnalysisHistoryRecord(BaseModel):
    time: datetime
    peaks_version: str = Field(..., alias="peaks version")
    record: Union[str, dict]
    fn_name: str = Field(..., alias="function name")

    class Config:
        populate_by_name = (
            True  # Allow aliases for both serialization and deserialization
        )


# A model to hold the list of records
class AnalysisHistoryRecordCollection(BaseModel):
    records: List[AnalysisHistoryRecord] = []

    # Method to append new records
    def add_record(self, new_record: dict):
        record = AnalysisHistoryRecord(**new_record)
        self.records.append(record)


def _update_hist(data, record_text, fn_name=None, update_in_place=True):
    """Updates the analysis history metadata of the supplied DataArray. Can be used as a standalone function
    or via the :class:`xarray.DataArray` accessors `.history.add()` to update the history of the passed
    :class:`xarray.DataArray` in place or `history.assign()` to return a copy of the :class:`xarray.DataArray` with
    the updated history.

    Parameters
    ------------
    data : xarray.DataArray
        The :class:`xarray.DataArray` for which the analysis history is to be updated.

    record_text : string
        Descriptive string to append to the DataArray's history metadata record.

    fn_name : string, optional
        An optional function name to include in the history record. If not specified, the function name of the caller
        function will be attempted to be automatically parsed.

    update_in_place : bool, optional
        If True, updates the analysis history metadata in place. Defaults to True.


    Returns
    ------------
    xarray.DataArray, optional
        The DataArray with the updated analysis history metadata. If `update_in_place` is set `True`, returns None.

    Examples
    ------------
    Example usage is as follows::

        import peaks as pks

        # Load some data
        disp = pks.load('disp.ibw')

        # Update the analysis history metadata of the dispersion, modifiying the existing array in place
        disp.history.add('From inspection, seems the counts are a factor of 2 too high', update_in_place=True)
        # Display the current history metadata, which should now include the new record
        disp.history()

        disp = disp/2
        # Update the analysis history metadata of the dispersion
        disp2 = disp.history.add('Dispersion data divided by 2')

        # The history metadata of the original dispersion should not have been modified
        disp.history()
        # But the history metadata of the new dispersion should include the new record
        disp2.history()

        # Use within a function to update the history metadata, including a reference to the called function
        from peaks.utils.metadata import _update_hist
        def add_one(data):
            data += 1
            _update_hist(data, 'Data incremented by 1', fn_name='add_one', update_in_place=True)
            return data
    """

    # Get current analysis history metadata list, creating if it doesn't exist
    if "_analysis_history" not in data.attrs:
        data.attrs["_analysis_history"] = AnalysisHistoryRecordCollection()
    analysis_history = data.attrs.get("_analysis_history")

    # If updating in place, modify the existing history list, otherwise make a deep copy to avoid in-place modification
    if not update_in_place:
        analysis_history = copy.deepcopy(analysis_history)

    # If not provided, try to automatically parse the function name of the caller
    if fn_name is None:
        try:
            fn_name = inspect.stack()[1].function
        except IndexError:
            fn_name = ""

    # Format the new history entry as a JSON-style payload with additional metadata and add to the existing history
    new_record = {
        "time": datetime.now().isoformat(),
        "peaks version": str(__version__),
        "record": record_text,
        "function name": fn_name,
    }

    analysis_history.add_record(new_record)

    if not update_in_place:
        # Return the DataArray with the updated analysis history metadata,
        # using assign_attrs to avoid in-place modification
        return data.assign_attrs(_analysis_history=analysis_history)


def update_history_decorator(record_text):
    """Convenience decorator to make a function automatically add a fixed string to the analysis metadata record of an
    :class:`xarray.DataArray`.

    This decorator assumes that the first argument of the function is the :class:`xarray.DataArray` object on which
    the function is to act, and that the function returns a modified :class:`xarray.DataArray` object.

    Parameters
    ------------
    record_text : string
        Descriptive string to append to the DataArray's history metadata record.

    Examples
    ------------
    Example usage is as follows::

        import peaks as pks
        from peaks.utils.metadata import update_history_decorator

        @update_history_decorator('Data incremented by 1')
        def add_one(data):
            data += 1
            return data
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            if not isinstance(args[0], xr.DataArray):
                raise TypeError(
                    "Update history decorator assumes the first argument of the funciton is "
                    "the xarray.DataArray object on which the function is to act."
                )
            # Update the history, including passing the function name
            updated_dataarray = args[0].history.assign(
                record_text, fn_name=func.__name__
            )
            return func(updated_dataarray, *args[1:], **kwargs)

        return wrapper

    return decorator


@xr.register_dataset_accessor("history")
@xr.register_dataarray_accessor("history")
class History:
    """Custom accessor for the analysis history metadata of a :class:`xarray.DataArray`.

    This class provides methods to interact with the analysis history metadata of an `xarray.DataArray`.
    Access via the `.history` method.

    Methods
    -------
    __call__(return_history=False)
        Displays (and optionally returns) the history metadata of the :class:`xarray.DataArray`.
    add(record_text, fn_name=None)
        Updates the analysis history metadata of the :class:`xarray.DataArray`.
    """

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __repr__(self):
        """Make a nice in-line display of the history metadata and return a timestamp string."""
        analysis_history = self._obj.attrs.get("_analysis_history")
        if analysis_history:
            analysis_history = analysis_history.dict(by_alias=True)["records"]
        df = pd.DataFrame(analysis_history)
        with pd.option_context(
            "display.max_colwidth", None, "display.colheader_justify", "left"
        ):
            display(df.fillna(""))
        return f"History as of {datetime.now().strftime('%y-%m-%d %H-%M-%S')}"

    def __call__(self, index=None):
        """
        Return the history metadata of the DataArray.

        Returns
        -------
        list of dict
            List of the current history metadata entries

        index : int, optional
            The index of the history metadata item to display. Defaults to most recent entry.


        Examples
        --------
        Example usage is as follows::
            import peaks as pks

            disp = pks.load('disp.ibw')
            disp_k = disp.k_convert()  # Convert the data to k-space

            # Display the current history metadata
            disp_k.history()

        """

        record = (
            self._obj.attrs.get("_analysis_history")
            .records[index if index is not None else -1]
            .dict(by_alias=True)
            .copy()
        )
        pprint(record)

    def add(self, record_text, fn_name=None):
        """Update the analysis history metadata of the :class:`xarray.DataArray` in place.

        See Also
        --------
        _update_hist : The underlying function that updates the history metadata of the DataArray. This is called
        with `update_in_place=True` to update the history metadata in place.

        assign : Alternative method to update the history metadata in a copy of the DataArray.
        """
        # If not provided, try to automatically parse the function name of the caller
        if fn_name is None:
            try:
                fn_name = inspect.stack()[1].function
            except IndexError:
                fn_name = ""
        return _update_hist(self._obj, record_text, fn_name, True)

    def assign(self, record_text, fn_name=None):
        """Assign a new history item to a copy of self.

        See Also
        --------
        _update_hist : The underlying function that updates the history metadata of the DataArray. This is called
        with `update_in_place=False` to return a copy of the DataArray with the updated history metadata.

        add : Alternative method to update the history metadata in place.
        """
        # If not provided, try to automatically parse the function name of the caller
        if fn_name is None:
            try:
                fn_name = inspect.stack()[1].function
            except IndexError:
                fn_name = ""
        return _update_hist(self._obj, record_text, fn_name, False)

    def get(self, index=None, return_history=False):
        """Return all of or an item of the history metadata of the DataArray.

        Parameters
        ----------
        index : int, optional
            The index of the history metadata item to return. Defaults to returning a list of all entries.

        Returns
        -------
        list or dict
            The relevant history metadata item(s)
        """

        return (
            self._obj.attrs.get("_analysis_history")
            .dict(by_alias=True)
            .copy()["records"][index if index is not None else slice(None)]
        )

    def json(self):
        """Return the history metadata as a JSON-formatted string.

        Returns
        -------
        str
            The history metadata as a JSON-formatted string.

        Examples
        --------
        Example usage is as follows::

            import peaks as pks

            disp = load('disp.ibw')
            disp_k = disp.k_convert()

            # Get the history metadata as a JSON-formatted string
            disp_k.history.json()
        """
        return self._obj.attrs.get("_analysis_history").json(by_alias=True)

    def save(self, fname):
        """Save the history metadata as a JSON-formatted string.

        Parameters
        ----------
        fname : str
            The filename to save the history metadata to.

        Examples
        --------
        Example usage is as follows::

            import peaks as pks

            disp = load('disp.ibw')
            disp_k = disp.k_convert()

            # Save the history metadata to a JSON file
            disp_k.history.save('disp_k_history.json')
        """
        with open(fname, "w") as f:
            f.write(self.json())

    add.__doc__ = _update_hist.__doc__
