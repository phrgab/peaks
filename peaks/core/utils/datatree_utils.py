import functools
import re

import xarray as xr
from termcolor import colored

from peaks.core.utils.misc import analysis_warning


# Helper functions
def _ensure_empty_node(func):
    """Decorator to check if the DataTree node is empty before calling the function."""

    @functools.wraps(func)
    def wrapper(dt, *args, **kwargs):
        if not dt.is_empty:
            raise ValueError(
                f"DataTree node already appears to contain data. "
                f"`.{func.__name__}` should only be called on an empty node of the DataTree"
            )
        return func(dt, *args, **kwargs)

    return wrapper


def _ensure_name_does_not_exist(func):
    """Decorator to enforce passed name does not exist in the DataTree before calling the function,
    to avoid overwriting existing entries."""

    @functools.wraps(func)
    def wrapper(dt, *args, **kwargs):
        if kwargs.get("name") and kwargs.get("name") in dt:
            raise ValueError(
                f"Name '{kwargs.get('name')}' already exists as a child node in the DataTree. Please provide a unique name."
            )
        return func(dt, *args, **kwargs)

    return wrapper


def _make_name_callable(name):
    """Replace any character that is not a letter, digit, or underscore with an underscore"""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def _make_name_unique(name, dt):
    """Make a name unique by appending a number to it.

    Parameters
    ----------
    name : str
        Name to make unique.

    dt : DataTree
        DataTree to check for name uniqueness.

    Returns
    -------
    str
        Unique name.
    """
    index = 0
    base_name = name
    name = f"{base_name}_{index}"
    while name in dt:
        index += 1
        name = f"{base_name}_{index}"
    return name


def _dataarrays_to_datatree(data, names=None):
    """Parse one or a `list` of :class:`xarray.DataArray`'s into a :class:`xarray.DataTree` structure

    Parameters
    ----------
    data : xarray.DataArray or list of xarray.DataArray
        The data to be parsed into a DataTree structure.

    names : list of str, optional
        Names to assign to each DataArray. If not provided, the names will be automatically generated.
    """

    # Parse the names automatically if the names list not provided or not suitable
    if names and (len(names) != len(data) or len(set(names)) != len(names)):
        analysis_warning(
            "The names provided are not suitable for the number of files loaded. Pass a list of unique strings for "
            "each file loaded. Automatically generating names.",
            title="Loading info",
            warn_type="info",
        )
        names = None
    if not names:
        names = [_make_name_callable(da.name) for da in data]

    ds_dict = {
        da_name: (
            da.to_dataset(name="data", promote_attrs=False)
            if isinstance(da, xr.DataArray)
            else da  # Pass a DataSet directly if it is already a DataSet
        )
        for da_name, da in zip(names, data, strict=True)
    }

    return xr.DataTree.from_dict(ds_dict)


# General accessor methods for DataTree
def view(dt):
    """Make a nice view of all branches and scans in the DataTree.

    Parameters
    ----------
    dt : DataTree
        The DataTree to list scans from.
    """

    # Function to recursively iterate over DataTree and display in a nice structure
    def display_datatree(node, level=0, is_last=True):
        indent = ""

        if level > 0:
            if is_last:
                indent = "└── "
            else:
                indent = "├── "

        if node.name:
            name = node.name
        else:
            name = ""

        # Color the name if the node has a dataset
        if not node.is_empty:
            name = colored(name, "green")

        output = f"{'   ' * (level - 1) + indent}{name}\n"

        # Iterate over children, marking last child appropriately
        children = list(node.children.values())
        for i, child in enumerate(children):
            output += display_datatree(child, level + 1, i == len(children) - 1)

        return output

    # Display the DataTree structure
    print(display_datatree(dt))


@_ensure_empty_node
@_ensure_name_does_not_exist
def add_scan_group(dt, name=None):
    """Add a new group (branch) to the DataTree.

    Parameters
    ----------
    dt : DataTree
        The DataTree to add the group to.

    name : str, optional
        Name of the group to be added to the DataTree. Defaults to `scan_group_#` where # is a number to make the
        name unique.

    Examples
    --------
    Example usage is as follows::

        import peaks as pks

        # Create a new DataTree
        dt = xr.DataTree()

        # Add a new group to the DataTree
        dt.add_scan_group()
    """

    if not name:
        name = _make_name_unique("scan_group", dt)
    new_name = _make_name_callable(name)
    new_dt = xr.DataTree(name=new_name)
    dt[new_name] = new_dt


@_ensure_empty_node
@_ensure_name_does_not_exist
def add(dt, data_source, name=None, add_at_root=False, **kwargs):
    """Add data into an existing DataTree. Can be either loaded data, or arguments to pass to the main `peaks.load`
    function.

    Parameters
    ----------
    dt : DataTree
        The DataTree to add data into.

    data_source : str or int or xarray.DataArray or xarray.Dataset or xarray.DataTree
        File path identifier to load data from, or data to add to the DataTree.

    name : str, optional
        Name of the branch to be added to the :class:`xarray.DataTree`. Defaults to `scan_group_#` if a group of scans
        are added, or the name of the file if a single scan is added. Note, if loading multuple scans, the `names`
        argument can also be provided to specify the names of each scan within the added scan group.

    add_at_root : bool, optional
        If `True`, and if a :class:`xarray.DataTree` is passed, the children of the added :class:`xarray.DataTree`
         will all be be added at the root level of the original :class:`xarray.DataTree`. Defaults to `False`, in which
         case the new :class:`xarray.DataTree` is added as a new branch of the tree, with the passed name or an
         automatically generated name. If set `True`, the `name` argument is ignored.

    **kwargs : other keyword arguments to pass to `peaks.core.fileIO.data_loading.load`

    See Also
    --------
    peaks.core.fileIO.data_loading.load : The function that is called to load the data.

    Examples
    --------
    Example usage is as follows::

        import peaks as pks

        # Create a new DataTree by loading a set of data
        disps = pks.load(['disp1.ibw','disp2.ibw'])

        # Add a new dispersion into the DataTree
        disps.add('disp3.ibw')
    """

    if isinstance(data_source, xr.DataTree):
        # Group of scans passed.
        if add_at_root:
            # Add children of the DataTree to the root of the original DataTree
            for child in data_source.children.values():
                dt[child.name] = child
            return

        # Otherwise, add them as a new group with the given name or a default name
        name = _make_name_callable(name) if name else _make_name_unique("scan_group", dt)
        data_source.name = name
        dt[name] = data_source
        return

    if isinstance(data_source, xr.DataArray):
        # Single da passed - parse it to DataTree format and add to the tree
        data = _dataarrays_to_datatree([data_source], names=[name] if name else None)
        data = data[list(data)[0]]
        dt[data.name] = data
        return

    if isinstance(data_source, xr.Dataset):
        # Single ds passed - parse it to DataTree format and add to the tree
        if not name:
            raise ValueError(
                "A name must be provided when adding a Dataset to the DataTree."
            )
        data = xr.DataTree(name=_make_name_callable(name), dataset=data_source)
        dt[data.name] = data
        return

    if isinstance(data_source, (str, int, list)):
        from peaks.core.fileIO.data_loading import load as pks_load

        # This should be a file path or a file identifier - load the data
        loaded_data = pks_load(data_source, **kwargs)
        # Add loaded data to the DataTree
        add(dt=dt, data_source=loaded_data, name=name, add_at_root=add_at_root)


def get_DataArray(dt):
    """Get a :class:`xarray.DataArray` stored as the only entry of a :class:`xarray.DataSet` with
     data variable `data` which is the leaf node of a :class:`xarray.DataTree`.

    Parameters
    ----------
    dt : DataTree leaf
        The DataTree leaf to return the :class:`xarray.DataArray` from.

    Returns
    -------
    xarray.DataArray
        The :class:`xarray.DataArray` stored in the leaf node.
    """
    if not dt.is_empty and hasattr(dt, "data") and isinstance(dt.data, xr.DataArray):
        da = dt.data
        da.name = dt.name
        return da


def get_list_of_DataArrays_from_DataTree(dt):
    """Get a list of :class:`xarray.DataArray` stored as the only entry of a :class:`xarray.DataSet` with
     data variable `data` which are the leaf nodes of a :class:`xarray.DataTree`.

    Parameters
    ----------
    dt : DataTree
        The DataTree to return the :class:`xarray.DataArray` from.

    Returns
    -------
    list of xarray.DataArray
        The :class:`xarray.DataArray`s stored in the leaf nodes.
    """
    data_list = []
    for node in dt.subtree:
        if not node.is_empty:
            data_list.append(get_DataArray(node))
    return data_list


def _map_over_dt_containing_single_das(data_tree, func, *args, **kwargs):
    """Applies a function to each leaf node in a DataTree assuming that any leaf with data contains a single
    :class:`xarray.Dataset` with a single data variable `data` containing the :class:`xarray.DataArray`.

    Parameters
    ----------
    data_tree : xarray.DataTree
        The DataTree to apply the function to.
    func : callable
        The function to apply to each leaf node.
    *args, **kwargs
        Additional arguments to pass to the function.

    Returns
    -------
    xarray.DataTree
        The DataTree with the function applied to each leaf node.
    """

    def map_over_ds(node):
        if hasattr(node, "data") and len(node) == 1:
            return node.map(lambda da: func(da, *args, **kwargs) or da)
        else:
            return node

    return data_tree.map_over_datasets(map_over_ds)
