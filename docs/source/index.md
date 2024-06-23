# `peaks` Documentation

**P**ython **E**lectron Spectroscopy and Diffraction **A**nalysis by **K**ing Group **S**t Andrews.

Version: {{ version }}

Documentation built: {{ date }}

See [Getting Started](#getting_started) guide for installation instructions. The tutorials give an overview of the core usage, while detailed instructions are given in the [API documentation](_apidoc/peaks). `peaks` builds heavily on [Xarray](https://xarray.dev), and it is strongly recommended to follow the Xarray [documentation](https://docs.xarray.dev/en/stable/) and [tutorials](https://tutorial.xarray.dev/intro.html) before starting to work with `peaks`.  

```{toctree}
:maxdepth: 3

getting_started
```

```{toctree}
:maxdepth: 2
:caption: Tutorials
:glob:

_tutorials/*
```

```{toctree}
:caption: API Reference
:maxdepth: 1

_apidoc/peaks
```

```{toctree}
:caption: Development
:maxdepth: 1

contributing
changelog
```
