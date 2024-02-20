# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import date

import xarray

xarray.DataArray.__module__ = "xarray"

# Add the project root to the path
sys.path.insert(0, os.path.abspath("../../"))

from peaks import __name__
from peaks.__version__ import __version__

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
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_togglebutton",
    "sphinx_inline_tabs",
    "myst_parser",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pyserial": ("https://pyserial.readthedocs.io/en/latest/", None),
    "xarray": ("https://xarray.pydata.org/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "dask": ("https://docs.dask.org/en/latest/", None),
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
            "name": "Github",
            "url": "https://github.com/phrgab/peaks",
            "html": "",
            "class": "fa-brands fa-solid fa-github fa-2x",
        },
    ],
}
