(contributing_section)=
# Contributing

## Installation
If you are actively developing the package, first clone the repo:
```{tab} Using HTTPS
    git clone https://gitlab.st-andrews.ac.uk/physics-and-astronomy/king-group/peaks.git
```
```{tab} Using SSH:
    git clone git@gitlab.st-andrews.ac.uk:physics-and-astronomy/king-group/peaks.git
```

Then create a conda environment and install the package in "editable" mode, including optional development dependencies:
```bash
conda create -n peaks-dev python=3.12
conda activate peaks-dev
pip install -e peaks\[dev\]
```

## Documentation
The hosted documentation is automatically built by Gitlab CI/CD, and updated on each commit. To make a local build of the documentation, first install [`sphinx`](https://www.sphinx-doc.org/en/master/) and the required additional packages:
```bash
pip install sphinx sphinx-autoapi sphinx-copybutton sphinx-togglebutton sphinx-inline-tabs myst-parser furo
```

To build the documentation, navigate to the `docs` directory and run:
```bash
cp -f ../tutorials/* source/_tutorials/
make html
```

```{note}
We use sphinx and MyST-parser for the documentation. See the [MyST documentation](https://myst-parser.readthedocs.io/en/latest/) for more information on the MyST markdown format. We use the furo theme for the documentation. See the [Furo documentation](https://pradyunsg.me/furo/) for more information on the theme and helpful guides for the markdown formatting.
```

## Making changes
Guidelines:
- Open an issue to discuss the change.
- Make changes in a new branch.
- Thoroughly test the changes before making a pull request. 
- Ensure the code is formatted with [`black`](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html) code style. This can be done automatically within your IDE. 
- Document new features and changes in the changelog. We use [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
- Ensure the documentation is updated. For new features, consider updating the tutorials with relevant examples. Ensure to write informative docstrings for new functions and classes. 

```{tip}
We use the [NumPy docstring format](https://numpydoc.readthedocs.io/en/latest/format.html). Specify Parameters and give relevant code examples. To make a code block, insert `::` at the end of the line before, and then leave a space and indent the text. 

We are using `intersphinx` to make links to other documentation, which can be quite helpful. But this only works if we put in the full name not the abbreviation for the other package. so use e.g. `xarray.DataArray` and not `xr.DataArray` in type annotations in the docstring. This works out of the box if this is given as the type for a parameter or returns. If you want to include this within part of the general text, need to do `:class:xarray.DataArray`. See other modules for examples.
```