# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import re
import os
from datetime import date

__name__ = "peaks"

# Get current version
vfpath = os.path.join(os.path.dirname(__file__), "..", "..", "peaks", "__version__.py")
with open(vfpath, "r") as f:
    contents = f.read()
__version__ = re.search(r'__version__ = "(.*?)"', contents).group(1)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = str(__name__)
version = str(__version__)
release = str(__version__)
author = "King group @ St Andrews"
copyright = f"{date.today().year}, {author}"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    "autoapi.extension",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_togglebutton",
    "sphinx_inline_tabs",
    "myst_nb",
]
autoapi_dirs = ["../../peaks"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pyserial": ("https://pyserial.readthedocs.io/en/latest/", None),
    "xarray": ("https://xarray.pydata.org/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "dask": ("https://docs.dask.org/en/latest/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "lmfit": ("https://lmfit.github.io/lmfit-py/index.html", None),
}

source_suffix = [".rst", ".md"]
exclude_patterns = []

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

# myst-nb settings
nb_execution_mode = "off"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "furo"

myst_enable_extensions = [
    "colon_fence",
    "amsmath",
    "dollarmath",
    "attrs_inline",
    "tasklist",
    "fieldlist",
    "substitution",
]

myst_substitutions = {
    "version": version,
    "release": release,
    "date": date.today().strftime("%d.%m.%Y"),
}

html_css_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/fontawesome.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/solid.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/brands.min.css",
]

html_theme_options = {
    "footer_icons": [
        {
            "name": "Gitlab",
            "url": "https://gitlab.st-andrews.ac.uk/physics-and-astronomy/king-group/peaks",
            "html": "",
            "class": "fa-brands fa-solid fa-gitlab fa-2x",
        },
    ],
}
