## peaks

[![git](https://img.shields.io/badge/repo-gitlab-orange)](https://gitlab.st-andrews.ac.uk/physics-and-astronomy/king-group/peaks)
[![docs](https://img.shields.io/badge/docs-research.st--andrews-green?style=flat-square)](https://research.st-andrews.ac.uk/kinggroup/peaks)
[![Code Style: Ruff (Black-compatible)](https://img.shields.io/badge/code%20style-ruff-black?style=flat-square)](https://docs.astral.sh/ruff/formatter/)

`peaks`: **P**ython **E**lectron Spectroscopy **A**nalysis by **K**ing Group **S**t Andrews.

A collection of analysis tools for the loading, processing and display of spectroscopic data, with a core focus on tools for angle-resolved photoemission.

`peaks` is an evolution of the `PyPhoto` package originally developed by Phil King, Brendan Edwards, Tommaso Antonelli, Edgar Abarca Morales, and Lewis Hart from the King group at the University of St Andrews. This version of `peaks` is the result of a major restructuring of the package in 2023-2025 by Brendan Edwards and Phil King, with significant additional contributions from Shu Mo.

Copyright the above authors. Contact pdk6@st-andrews.ac.uk for further information, to contribute, and for bug reporting.

## Installation
It is recommended to install peaks in its own environment. Using conda:

```bash
conda create -n peaks python=3.12
conda activate peaks
```

Then install peaks using pip, directly from the Gitlab repo:
```{tab} Using HTTPS
    pip install git+https://gitlab.st-andrews.ac.uk/physics-and-astronomy/king-group/peaks.git
```

```{tab} Using SSH
    pip install git+ssh://git@gitlab.st-andrews.ac.uk:physics-and-astronomy/king-group/peaks.git
```

To install a specific tagged version, append `@<tag>` to the end of the git link where `<tag>` is the tag name.

### Optional dependencies
To install optional dependencies, append `\[dep1, dep2, ...\]` to the end of the `pip install ...` command, where `dep` is the name of the dependency. The following options can be used:

- **structure** - required for the use of the `bz` module, for e.g. plotting Brillouin zones on the data;
- **ML** - required for the use of the machine learning module;
- **dev** - optional development dependencies, used for e.g. formatting the code or building local copies of the documentation. 

## Basic Usage
`peaks` is typically run in a Jupyter notebook or equivalent. To import peaks run:
```python
import peaks as pks
```

See the <project:#Documentation> and associated tutorials for more information on the package and its modules.

## Documentation
The peaks documentation can be found at [research.st-andrews.ac.uk/kinggroup/peaks](https://research.st-andrews.ac.uk/kinggroup/peaks).

## Contributing
Contributions to the package are welcome. Please see the [contributing guide](#contributing_section) in the documentation for more information.

## License
Copyright 2019-2025, peaks developors

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

peaks also makes extensive use of many other packages - see dependencies in pyproject.toml and their relevant licenses in the source control of those packages. 