# peaks

[![git](https://img.shields.io/badge/repo-github-orange)](https://github.com/phrgab/peaks)
[![docs](https://img.shields.io/badge/docs-research.st--andrews-green?style=flat-square)](https://research.st-andrews.ac.uk/kinggroup/peaks)
[![Code Style: Ruff (Black-compatible)](https://img.shields.io/badge/code%20style-ruff-black?style=flat-square)](https://docs.astral.sh/ruff/formatter/)

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

<!-- installation-start -->
## Installation

`peaks` is registed on [PyPI](https://pypi.org/project/peaks-arpes/) under the name `peaks-arpes`.

It is recommended to install `peaks` in its own isolated environment. E.g. using conda:

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

### Installing from source

The latest version of `peaks` can be installed directly from source:
```bash
pip install git+https://github.com/phrgab/peaks.git
```


To install a specific tagged version, append `@<tag>` to the end of the git link where `<tag>` is the tag name.
<!-- installation-end -->

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
Copyright 2019-2025, peaks developers

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

peaks also makes extensive use of many other packages - see dependencies in pyproject.toml and their relevant licenses in the source control of those packages. 