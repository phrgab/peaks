# peaks

[![git](https://img.shields.io/badge/repo-github-blue?logo=github)](https://github.com/phrgab/peaks)
[![Latest version on pypi](https://img.shields.io/pypi/v/peaks-arpes?logo=pypi&logoColor=white&color=brightgree)](https://pypi.org/project/peaks-arpes/)
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/phrgab/peaks/test-installs.yml?logo=github-actions&logoColor=white&label=Install%20tests)](https://github.com/phrgab/peaks/actions/workflows/test-installs.yml)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/peaks-arpes?logo=python&logoColor=white)](https://www.python.org)
[![docs](https://img.shields.io/badge/docs-research.st--andrews-blueviolet)](https://research.st-andrews.ac.uk/kinggroup/peaks)
[![Code Style: Ruff (Black-compatible)](https://img.shields.io/badge/code%20style-ruff-black)](https://docs.astral.sh/ruff/formatter/)

`peaks`: **P**ython **E**lectron Spectroscopy **A**nalysis by **K**ing Group **S**t Andrews.

<!-- overview-start -->

`peaks` provides a collection of analysis tools for the loading, processing and visualisation of spectroscopic data, with a core focus on tools for angle-resolved photoemission.

`peaks` is an evolution of the `PyPhoto` package originally developed by Phil King, Brendan Edwards, Tommaso Antonelli, Edgar Abarca Morales, Lewis Hart, and Liam Trzaska from the [King group](https://www.quantummatter.co.uk/king) at the [University of St Andrews](http://www.st-andrews.ac.uk). This version of `peaks` is the result of a major restructuring of the package in 2023-2025 by Brendan Edwards, Phil King, and Shu Mo.

Contact [pdk6@st-andrews.ac.uk](pdk6@st-andrews.ac.uk).

<!-- overview-end -->

## Citation

<!-- citation-start -->

If you use `peaks` in your work, please cite:

*`peaks`: a Python package for analysis of angle-resolved photoemission and related spectroscopies* \
Phil D. C. King, Brendan Edwards, Shu Mo, Tommaso Antonelli,
Edgar Abarca Morales, Lewis Hart, and Liam Trzaska \
[arXiv:2508.04803](https://arxiv.org/abs/2508.04803) (2025)

<!-- citation-end -->

<!-- pypi-installation-start -->

## Installation

`peaks` is registed on [PyPI](https://pypi.org/project/peaks-arpes/) under the name `peaks-arpes`.

`peaks` has been tested with Python 3.11 through 3.14. It is recommended to install `peaks` in its own isolated environment. E.g. using conda:

```bash
conda create -n peaks python=3.12
conda activate peaks
pip install peaks-arpes
```

`peaks` will then be installed together with its core dependencies.

### Optional dependencies

To install optional dependencies, append `\[dep1, dep2, ...\]` to the end of the `pip install ...` command, where `dep` is the name of the dependency. The following options can currently be specified:

- **structure** - required for the use of the `bz` module, for e.g. plotting Brillouin zones on the data;
- **ML** - required for the use of the machine learning module;
- **dev** - optional development dependencies, used for e.g. linting the code and installing pre-commit hooks.
- **docs** - optional dependencies for building local copies of the documentation.
<!-- pypi-installation-end -->

<!-- intel-mac-tip-start -->

**Intel mac users:**
`peaks` depends on `numba`, which cannot be installed via pip on Intel macOS [since `v0.63.0`](https://github.com/numba/numba/issues/10187) as pre-built wheels for this platform are no longer provided on PyPI. Install `numba` via conda first (or [build from source](https://numba.readthedocs.io/en/stable/user/installing.html#installing-from-source)) before installing `peaks`, e.g.

```bash
conda create -n peaks python=3.12
conda activate peaks
conda install -c conda-forge numba
pip install peaks-arpes
```

<!-- intel-mac-tip-end -->

<!-- source-installation-start -->

### Installing from source

The latest version of `peaks` can be installed directly from source:

```bash
pip install git+https://github.com/phrgab/peaks.git
```

To install a specific tagged version, append `@<tag>` to the end of the git link where `<tag>` is the tag name.

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

## Contributing

Contributions to the package are welcome. Please see the [contributing guide](https://research.st-andrews.ac.uk/kinggroup/peaks/latest/contributing.html) in the documentation for more information.

## License

Copyright 2019-2026, peaks developers

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

peaks also makes extensive use of many other packages - see dependencies in pyproject.toml and their relevant licenses in the source control of those packages.
