# `peaks` Documentation

**P**ython **E**lectron Spectroscopy **A**nalysis by **K**ing Group **S**t Andrews.

Version: {{ version }}

Documentation built: {{ date }}

```{include} ../../README.md
:start-after: <!-- overview-start -->
:end-before: <!-- overview-end -->
```

See [Getting Started](#getting_started) guide for installation instructions. The [User Guide](user_guide) gives an overview of the core usage, while detailed instructions are given in the [API documentation](autoapi/index). `peaks` builds heavily on [Xarray](https://xarray.dev), and it is strongly recommended to follow the Xarray [documentation](https://docs.xarray.dev/en/stable/) and [tutorials](https://tutorial.xarray.dev/intro.html) before starting to work with `peaks`.  

```{toctree}
:maxdepth: 2

getting_started
```

```{toctree}
:maxdepth: 2
:glob:

user_guide
```

```{toctree}
:maxdepth: 2
:glob:

explanations
```

```{toctree}
:maxdepth: 1

autoapi/index
```

```{toctree}
:maxdepth: 2

contributing
```

```{toctree}
:maxdepth: 1

changelog
```