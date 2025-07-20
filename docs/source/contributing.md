(contributing)=
# Contributing

We welcome contributions to the advancement of `peaks` and hope for it to develop into a community package for analysis of ARPES and related spectroscopic data. Contributions are welcome in all areas, and particularly at present in:
- implementing new [file loaders](#file_loaders)
- testing functionality and bug-checking
- developing a proper suite of unit tests
- enhancing and correcting the project documentation, including relevant doc-strings
- contributing feature enhancements.

If you wish to discuss a contribution or to report a bug, please open an issue on our [GitHub repository](https://github.com/phrgab/peaks/issues).

## Installation
If you are actively developing the package, we recommend cloning or forking the repository and installing an editable version from source, including with the optional development (and any other desired) dependencies:

```bash
conda create -n peaks-dev python=3.12
conda activate peaks-dev
# From the project root
pip install -e .\[dev\]
```

## Making changes
Guidelines:
- Open an issue to discuss the change.
- Make changes in a new branch/fork.
- Thoroughly [test](#testing) the changes before making a pull request. 
- Ensure the code is formatted following our [conventions](#linting-and-formatting-with-ruff). 
- Document new features and changes in the [changelog](#changelog). 
- Ensure the [documentation is updated](#documentation). 

## Testing
As well as thorougly testing the new feature, all of the existing tutorials (which act as a basic form of codebase test at present) should run without error. To test locally, from the parent directory of `peaks`, run:
```bash
bash ./tests/test_tutorials.sh
```
This will also be run in CI for merge requests, and only passing tests will usually be merged.


## Linting and Formatting with Ruff

We use [`Ruff`](https://docs.astral.sh/ruff/) for linting, import sorting, and formatting (largely following [black](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html) code style.). `Ruff` is installed as part of the developer optional dependencies and can be run on the code base manually:
```bash
ruff check . --fix               # Auto-fix lint issues
ruff format .                    # Format code (like Black)
```

### Pre-commit hook
A pre-commit hook is avaialble for checking and fixing lint issues where possible and formatting the code. From the project root, to register the hook:
```bash
pre-commit install
```
Now Ruff will automatically check and fix code on each commit. For manual use:
```bash
# Run all checks on all files
pre-commit run --all-files
```

### CI enforcement
Ruff also runs in CI. Commits and merge requests will fail if:

- Code is not formatted
- Lint errors are found except for line-length errors

Bundled ipython notebooks are automatically excluded from the linting and checking.

## Changelog
All notable changes to the project should be documented in the `CHANGELOG.md` file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Documentation
We use sphinx and MyST-parser for building the project documentation. See the [MyST documentation](https://myst-parser.readthedocs.io/en/latest/) for more information on the MyST markdown format. We use the [pydata-sphinx theme](https://pydata-sphinx-theme.readthedocs.io/en/stable/index.html).

The core user documentation comes from a combination of the tutorials (Jupyter notebooks), which are automatically built to form a [User Guide](https://research.st-andrews.ac.uk/kinggroup/peaks/user_guide.html), and the docstrings, which form an [API reference](https://research.st-andrews.ac.uk/kinggroup/peaks/autoapi/index.html). For new features, please ensure to write informative docstrings for new functions and classes, including usage examples. Consider also updating the tutorials with relevant examples where appropriate.

:::{tip}
We use the [NumPy docstring format](https://numpydoc.readthedocs.io/en/latest/format.html). Specify Parameters and give relevant code examples. To make a code block, insert `::` at the end of the line before, and then leave a space and indent the text. 

We are using `intersphinx` to make links to other documentation, which can be quite helpful. But this only works if we put in the full name not the abbreviation for the other package. so use e.g. `xarray.DataArray` and not `xr.DataArray` in type annotations in the docstring. This works out of the box if this is given as the type for a parameter or returns. If you want to include this within part of the general text, need to do `:class:xarray.DataArray`. See other modules for examples.
:::

The hosted documentation is automatically built by Gitlab CI, and updated on each merge to the main branch and on the release of a new tagged version. It is important that all of the tutorials can run without any local data files. If specific example data is required which is not already avaialble in the `peaks.core.utils.sample_data` module, raise an Issue to discuss adding a new example dataset there.

:::{tip}
To make a local build of the documentation, install `peaks` including the optional `[docs]` dependency. To build the documentation, navigate to the `docs` directory and run:
```bash
cp -f -r ../tutorials/* source/_tutorials/  # Copy the latest tutorials
make clean  # Clean the old documentation
make html  # Build the documentation
```

 Note, the version selector will not work for a local build of the documentation.
:::





