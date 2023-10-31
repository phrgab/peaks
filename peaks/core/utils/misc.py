"""Miscellaneous helper functions

"""

# Phil King 15/05/2021
# Brendan Edwards 16/10/2023

import os
import pdoc
from IPython.display import display, Javascript, Markdown


def analysis_warning(text, warn_type='info', title='Analysis info'):
    """Warning tool for use within code, to display string in formatted box.

    Parameters
    ------------
    text : str
        Text string to be displayed

    warn_type : str (optional)
        Warning box type:
            info : blue box (default)
            warning : yellow box
            success : green box
            danger : red box

    title : str (optional)
        Title to be displayed in box. Defaults to `Analysis info`

    Examples
    ------------
    from peaks import *

    analysis_warning('Fitting result could not converge', warn_type='danger', title='Analysis info')  # Displays a red
        warning box informing the user that the fitting analysis failed

    """

    # Display warning
    display(Markdown('<div class="alert alert-block alert-%s"><b>%s: </b> %s </div>' % (warn_type, title, text)))


def make_docs(folder=None):
    """Automatically create module documentation in html format in the specified or default folder.

    Parameters
    ------------
    folder : str (optional)
        Defines the folder where the documentation is created. Defaults to a folder `docs` in the parent
        directory of the code

    Examples
    ------------
    from peaks import *

    make_docs(folder='C:/User/Documents/docs')  # Create module documentation in the docs folder at 'C:/User/Documents'

    """

    # Get peaks modules
    modules = ['peaks']
    context = pdoc.Context()
    modules = [pdoc.Module(mod, context=context, skip_errors=True) for mod in modules]
    pdoc.link_inheritance(context)

    # Helper function to extract module information
    def recursive_htmls(mod):
        yield mod.name, mod.html(), mod.url()
        for submod in mod.submodules():
            yield from recursive_htmls(submod)

    # If no user-defined folder, select default
    if folder is None:
        folder = os.path.dirname(__file__) + '/../../docs'

    # Go through modules and write documentation
    for mod in modules:
        for module_name, html, url in recursive_htmls(mod):
            # Get directory name
            fname = folder + '/' + url
            dir_name = os.path.dirname(fname)
            # Create directory if it doesn't already exist
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            # Write documentation
            with open(fname, "w") as f:
                f.write(html)
                f.close()


def cell_below(text, execute=True):
    """Generate a new cell in the Jupyter notebook, directly below the currently running cell.

    Parameters
    ------------
    text : str
        Code to be generated in new cell

    execute : Boolean (optional)
        Determines whether the newly generated cell is automatically executed. Defaults to True

    Examples
    ------------
    from peaks import *

    cell_below('a=2')  # Generate a cell which creates a variable a

    """

    # Ensure text is in an appropriate format
    text = text.replace("\n", "\\n").replace("'", "\\'")

    if execute == True:
        # Generate and execute new cell
        display(Javascript("""
        var cell = IPython.notebook.insert_cell_below('code')
        cell.set_text('""" + text + """')
        cell.execute()
        """))
    else:
        # Generate but do not execute new cell
        display(Javascript("""
        var cell = IPython.notebook.insert_cell_below('code')
        cell.set_text('""" + text + """')
        """))
