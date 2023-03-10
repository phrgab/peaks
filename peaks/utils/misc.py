#Miscelaneous helper functions
#Phil King 15/05/2021

import warnings
import os
from IPython.display import display, Javascript, Markdown

# Make a simple warning style when using warnings to return analysis-related warnings rather than warnings related to the code
#TODO Depreciate
def warning_simple(message, category, filename, lineno, file=None, line=None):
    return ' Analysis warning: %s\n' % (message)

# Make a more standard warning call
#TODO Depreciate
def warning_standard(message, category, filename, lineno, file=None, line=None):
    return ' %s:%s: %s:%s\n' % (filename, lineno, category.__name__, message)

# Standard analysis warning tool
def ana_warn(text, warn_type='info', title='Analysis info'):
    ''' Warning tool for use within code, to display string in formatted box.

    text : str
        text string to be displayed

    warn_type : str, optional
        Warning box type:
            info (defualt) : blue box
            warning : yellow box
            success : green box
            danger : red box

    title : str, optional
        Title to be displayed in box.
        Defaults to `Analysis info`

    '''

    display(Markdown('<div class="alert alert-block alert-%s"><b>%s: </b> %s </div>' % (warn_type, title, text) ))

# Make module documentation
def make_docs(folder=None):
    '''Make module documentation in html format.

    Input:
        `folder` (optional, str) - defaults to a folder `docs` in the parent directory of the code

    Output:
        Series of .html files containing the docs created in the specified or default folder. '''

    import pdoc3
    modules = ['peaks']
    context = pdoc.Context()

    modules = [pdoc.Module(mod, context=context, skip_errors=True) for mod in modules]
    pdoc.link_inheritance(context)

    def recursive_htmls(mod):
        yield mod.name, mod.html(), mod.url()
        for submod in mod.submodules():
            yield from recursive_htmls(submod)

    if folder is None:
        folder = os.path.dirname(__file__)+'/../../docs'
    for mod in modules:
        for module_name, html, url in recursive_htmls(mod):
            # Get directory name
            fname = folder+'/'+ url
            dir_name = os.path.dirname(fname)
            # Create directory if it doesn't already exist
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            with open(fname, "w") as f:
                f.write(html)
                f.close()

# Add a new cell in the Jupyter notebook
def cell_below(text):
    text = text.replace("\n", "\\n").replace("'", "\\'")
    display(Javascript("""
    var cell = IPython.notebook.insert_cell_below('code')
    cell.set_text('""" + text + """')
    cell.execute()
    """))

def cell_above(text):
    text = text.replace("\n", "\\n").replace("'", "\\'")
    display(Javascript("""
    var cell = IPython.notebook.insert_cell_above('code')
    cell.set_text('""" + text + """')
    cell.execute()
    """))