"""Miscellaneous helper functions.

"""

# Phil King 15/05/2021
# Brendan Edwards 16/10/2023

import os
from IPython.display import display, Javascript, Markdown


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
