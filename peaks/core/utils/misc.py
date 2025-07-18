"""Miscellaneous helper functions."""

import xarray as xr
from IPython.display import Javascript, Markdown, display
from termcolor import colored


def analysis_warning(text, warn_type="info", title="Analysis info", quiet=False):
    """Tool to display a string as a warning in a formatted box.

    Parameters
    ------------
    text : str
        Text string to be displayed.

    warn_type : str, optional
        Warning box type:
            info : blue box (default)
            warning : yellow box
            success : green box
            danger : red box

    title : str, optional
        Title to be displayed in box. Defaults to `Analysis info`.

    quiet : bool, optional
        Determines whether the warning is displayed. Defaults to False.

    Examples
    ------------
    Example usage is as follows::

        import peaks as pks

        # Display a blue warning box informing the user that the data is being loaded
        pks.analysis_warning('Loading file into DataArray format.', warn_type='info', title='Loading info')

        # Display a red warning box informing the user that the fitting analysis failed
        pks.analysis_warning('Fitting result could not converge.', warn_type='danger', title='Analysis info')

    """

    # Display warning
    if not quiet:
        display(
            Markdown(
                '<div class="alert alert-block alert-%s"><b>%s: </b> %s </div>'
                % (warn_type, title, text)
            )
        )


def dequantify_quantify_wrapper(func):
    def wrapper(*args, **kwargs):
        # Apply dequantify to the first argument
        args = (args[0].pint.dequantify(),) + args[1:]

        # Call the original function
        result = func(*args, **kwargs)

        # Apply quantify to the result
        if isinstance(result, (xr.DataArray, xr.Dataset)):
            result = result.pint.quantify()
            # Check if there is a single-length coord with Units that should be dequantified
            for dim, coord in result.coords.items():
                if coord.size == 1:
                    result = result.assign_coords({dim: coord.pint.dequantify()})
        return result

    return wrapper


def format_colored_dict(d, indent_level=0, col_cycle=0):
    """Recursive function to format dictionary with colored keys."""
    colours = ["green", "blue", "red", "yellow"]
    indent = "    " * indent_level
    current_color = colours[col_cycle % len(colours)]  # Cycle through colors
    formatted_str = ""
    for key, value in d.items():
        if isinstance(value, dict):  # Nested dictionary (recursive case)
            formatted_str += f"{indent}{colored(key, current_color)}:\n"
            formatted_str += format_colored_dict(value, indent_level + 1, col_cycle + 1)
        else:  # Base case (simple value)
            formatted_str += (
                f"{indent}{colored(key, current_color)}: {colored(value, 'black')}\n"
            )
    return formatted_str


def make_cell(text, below=True, execute=True):
    """Generate a new cell in the Jupyter notebook, directly below or above the currently running cell.

    Parameters
    ------------
    text : str
        Code to be generated in new cell.

    below : bool, optional
        Determines whether the new cell is generated below the currently running cell (as opposed to above).
        Defaults to True.

    execute : bool, optional
        Determines whether the newly generated cell is automatically executed. Defaults to True.

    Examples
    ------------
    Example usage is as follows::

        from peaks import *

        # Generate a cell below which creates variable a that is equal to 2, and execute it
        make_cell('a=2')

        # Generate a cell above which calculates 5+7, but do not execute it
        make_cell('5+7', below=False, execute=False)

    """

    # Ensure text is in an appropriate format
    text = text.replace("\n", "\\n").replace("'", "\\'")

    if below:
        if execute:
            # Generate and execute new cell below
            display(
                Javascript(
                    """
            var cell = IPython.notebook.insert_cell_below('code')
            cell.set_text('"""
                    + text
                    + """')
            cell.execute()
            """
                )
            )
        else:
            # Generate but do not execute new cell below
            display(
                Javascript(
                    """
            var cell = IPython.notebook.insert_cell_below('code')
            cell.set_text('"""
                    + text
                    + """')
            """
                )
            )
    else:
        if execute:
            # Generate and execute new cell above
            display(
                Javascript(
                    """
            var cell = IPython.notebook.insert_cell_above('code')
            cell.set_text('"""
                    + text
                    + """')
            cell.execute()
            """
                )
            )
        else:
            # Generate but do not execute new cell above
            display(
                Javascript(
                    """
            var cell = IPython.notebook.insert_cell_above('code')
            cell.set_text('"""
                    + text
                    + """')
            """
                )
            )
