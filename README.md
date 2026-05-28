# peaks

[![git](https://img.shields.io/badge/repo-github-blue?logo=github)](https://github.com/phrgab/peaks)
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/phrgab/peaks/test-installs.yml?logo=github-actions&logoColor=white&label=install%20tests)](https://github.com/phrgab/peaks/actions/workflows/test-installs.yml)
[![Latest version on pypi](https://img.shields.io/pypi/v/peaks-arpes?logo=pypi&logoColor=white&color=green)](https://pypi.org/project/peaks-arpes/)
[![Latest version on conda-forge](https://img.shields.io/conda/vn/conda-forge/peaks-arpes?logo=conda-forge&logoColor=white&color=green)](https://anaconda.org/channels/conda-forge/packages/peaks-arpes/overview)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/peaks-arpes?logo=python&logoColor=white)](https://www.python.org)
[![docs](https://img.shields.io/badge/docs-research.st--andrews-blueviolet)](https://research.st-andrews.ac.uk/kinggroup/peaks)
[![Code Style: Ruff (Black-compatible)](https://img.shields.io/badge/code%20style-ruff-black)](https://docs.astral.sh/ruff/formatter/)

`peaks`: **P**ython **E**lectron Spectroscopy **A**nalysis by **K**ing Group **S**t Andrews.

<!-- overview-start -->

`peaks` provides a collection of analysis tools for the loading, processing and visualisation of spectroscopic data, with a core focus on tools for angle-resolved photoemission.

`peaks` is an evolution of the `PyPhoto` package originally developed by Phil King, Brendan Edwards, Tommaso Antonelli, Edgar Abarca Morales, Lewis Hart, and Liam Trzaska from the [King group](https://www.quantummatter.co.uk/king) at the [University of St Andrews](http://www.st-andrews.ac.uk). This version of `peaks` is the result of a major restructuring of the package in 2023-2025 by Brendan Edwards, Phil King, and Shu Mo.

<!-- overview-end -->

## Citation

<!-- citation-start -->

If you use `peaks` in your work, please cite:

_`peaks`: a Python package for analysis of angle-resolved photoemission and related spectroscopies_ \
Phil D. C. King, Brendan Edwards, Shu Mo, Tommaso Antonelli,
Edgar Abarca Morales, Lewis Hart, and Liam Trzaska \
[arXiv:2508.04803](https://arxiv.org/abs/2508.04803) (2025)

<!-- citation-end -->

<!-- main-installation-start -->

## Installation

`peaks` is registered on [PyPI](https://pypi.org/project/peaks-arpes/) under the name `peaks-arpes`, and is also distributed through the [conda-forge](https://anaconda.org/channels/conda-forge/packages/peaks-arpes/overview) channel under the same name `peaks-arpes` where the version may lag slightly behind that on [PyPI](https://pypi.org/project/peaks-arpes/).

`peaks` has been tested with Python 3.11 through 3.14. It is recommended to install `peaks` in its own isolated environment. E.g.

- **Using [`uv`](https://docs.astral.sh/uv) with a managed project:**

  ```bash
  uv init peaks -p 3.13
  cd peaks
  uv add peaks-arpes
  ```

- **Using [`conda`](https://conda.org) with an installer such as [Miniforge](https://conda-forge.org/download/) or [Miniconda](https://docs.anaconda.com/miniconda/):**

  ```bash
  conda create -n peaks python=3.13
  conda activate peaks
  conda install -c conda-forge peaks-arpes # or install from PyPI: pip install peaks-arpes
  ```

- **Using Python virtual environments** (This works with the default Python version installed on the OS. To use a specific Python version, see, e.g. [uv-managed virtual environments](https://docs.astral.sh/uv/pip/environments/)):

  ```bash
  python -m venv peaks
  source peaks/bin/activate  # Windows PowerShell: peaks\Scripts\activate
  pip install peaks-arpes
  ```

`peaks` will then be installed together with its core dependencies.

<!-- main-installation-end -->

<!-- intel-mac-tip-start -->

**Intel Mac users:**
`peaks` depends on `numba` which is no longer distributed via pip on Intel macOS. conda-forge still provides `numba` builds for this platform. Installing `peaks` from conda-forge is recommended on Intel Macs.

<!-- intel-mac-tip-end -->

<!-- extras-installation-start -->

### Optional dependencies

`peaks` can be installed with optional additional dependencies, as `uv add "peaks-arpes[dep1, dep2, ...]"` if using uv or `pip install "peaks-arpes[dep1, dep2, ...]"` with pip, where `dep` is the name of the dependency. The following options can currently be specified:

- **structure** - required for the use of the `bz` module, for e.g. plotting Brillouin zones on the data;
- **ML** - required for the use of the machine learning module;
- **dev** - optional development dependencies, used for e.g. linting the code and installing pre-commit hooks.
- **docs** - optional dependencies for building local copies of the documentation.

<!-- extras-installation-end -->

<!-- conda-extras-note-start -->

**Conda users:**
`conda` does not support the `[extras]` syntax. To install with optional dependencies using `conda`, list them explicitly alongside `peaks`. E.g. for the `structure` group:

```bash
conda install -c conda-forge peaks-arpes ase trimesh
```

The full list of optional dependency groups and their packages is defined under `[project.optional-dependencies]` in [pyproject.toml](https://github.com/phrgab/peaks/blob/main/pyproject.toml).

<!-- conda-extras-note-end -->

<!-- source-installation-start -->

### Installing from source

`peaks` can be installed directly from source:

```bash
pip install git+https://github.com/phrgab/peaks.git
```

By default, this installs the latest development version of `peaks`. To install a stable release from source, append `@<tag>` to the end of the git link where `<tag>` is the tag name of [the release](https://github.com/phrgab/peaks/releases).

<!-- source-installation-end -->

<!-- basic-usage-start -->

## Basic Usage

`peaks` is typically run in a Jupyter notebook or equivalent. To import peaks run:

```python
import peaks as pks
```

See the [User Guide](https://research.st-andrews.ac.uk/kinggroup/peaks/latest/user_guide.html) for more information on the package and its use.

<!-- basic-usage-end -->

## Documentation

The peaks documentation can be found at [research.st-andrews.ac.uk/kinggroup/peaks](https://research.st-andrews.ac.uk/kinggroup/peaks).

<!-- support-start -->

## Support

For general questions (e.g. installation, usage, contributing), please use our [GitHub Discussions](https://github.com/phrgab/peaks/discussions) page. If that is not possible, please e-mail [Phil](mailto:pdk6@st-andrews.ac.uk).

<!-- support-end -->

## Contributing

Contributions to the package are welcome. Please see the [contributing guide](https://research.st-andrews.ac.uk/kinggroup/peaks/latest/contributing.html) in the documentation for more information.

## License

Copyright 2019-2026, peaks developers

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

peaks also makes extensive use of many other packages - see dependencies in pyproject.toml and their relevant licenses in the source control of those packages.
