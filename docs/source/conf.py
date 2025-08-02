# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import re
from datetime import date

__name__ = "peaks"

# Get current version
vfpath = os.path.join(os.path.dirname(__file__), "..", "..", "peaks", "__init__.py")
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
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
    "private-members=False",
    "autoapi_add_toctree_entry=False",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pyserial": ("https://pyserial.readthedocs.io/en/latest/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "dask": ("https://docs.dask.org/en/latest/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "lmfit": ("https://lmfit.github.io/lmfit-py/index.html", None),
    "peaks": ("../build/html", "../build/html/objects.inv"),
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
nb_execution_mode = "force" if os.getenv("FORCE_NB_EXECUTION") == "1" else "off"
nb_execution_timeout = 300


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "pydata_sphinx_theme"

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

html_sidebars = {
    "getting_started": [],
}

html_theme_options = {
    "external_links": [
        {"name": "Kinggroup@StA", "url": "https://www.quantummatter.co.uk/king"},
    ],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/phrgab/peaks",
            "icon": "fa-brands fa-solid fa-github fa-2x",
            "type": "fontawesome",
        }
    ],
    "navbar_persistent": ["search-button", "version-switcher"],
    "check_switcher": False,
    "switcher": {
        "json_url": "https://research.st-andrews.ac.uk/kinggroup/peaks/switcher.json",
        "version_match": version,
    },
    "show_version_warning_banner": True,
}
