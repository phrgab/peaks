# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.3.dev]

### Added

- Progress bar support for running in marimo notebooks. `analysis_warning`s now render as native marimo callouts when running inside marimo ([PR#56](https://github.com/phrgab/peaks/pull/56))

### Fixed

- `plot_grid` renders with a distorted aspect ratio in marimo notebooks when the figure is wider than the notebook column ([PR#56](https://github.com/phrgab/peaks/pull/56))
- `ZeroDivisionError` in `disp` colour bar for constant-value data slices ([PR#57](https://github.com/phrgab/peaks/pull/57))
- Metadata reports `[nan, nan]` when a scan contains a few NaN readbacks ([PR#57](https://github.com/phrgab/peaks/pull/57))
- Hard-coded `focus` dimension in `plot_nanofocus` ([PR#58](https://github.com/phrgab/peaks/pull/58))
- Non-monotonic and repeated data at end of travel in i05 nano ana_polar maps ([PR#58](https://github.com/phrgab/peaks/pull/58))
- Clicking _Copy_ in a `disp` panel raises `PyperclipException` on systems without a clipboard mechanism; text is now printed for manual copying instead ([PR#59](https://github.com/phrgab/peaks/pull/59))
- `.disp` now raises a clear error when a multi-leaf `DataTree` contains no suitable data ([PR#65](https://github.com/phrgab/peaks/pull/65))

### Changed

- Limit font size and max lines of titles in `plot_fit` outputs ([PR#56](https://github.com/phrgab/peaks/pull/56))

### Removed

## [0.5.2] - 2026-07-07

### Added

- Unit tests for `core/fileIO`: `data_saving` ([PR#48](https://github.com/phrgab/peaks/pull/48))
- Minimum version cap for `numba` (`>=0.61.0`) to fix a dependency resolution issue when installing with uv ([PR#52](https://github.com/phrgab/peaks/pull/52))

### Fixed

- Data loading bug for SES `zip` files where the loc identification handler reads metadata in the parsed pydantic-model format rather than the flat raw dictionary it requires ([PR#47](https://github.com/phrgab/peaks/pull/47))
- Error on Windows when clearing up lazily loaded sample data ([PR#50](https://github.com/phrgab/peaks/pull/50))
- Cross-platform filename parsing in loader ([PR#50](https://github.com/phrgab/peaks/pull/50))
- Windows `PermissionError` when loading example CIF structure ([PR#50](https://github.com/phrgab/peaks/pull/50))
- 'Pretty'-formatted normal emission values copied from the display panel that can't be parsed ([PR#53](https://github.com/phrgab/peaks/pull/53))
- `plot_fit` doesn't render multiple fits and sliders during the docs build ([PR#53](https://github.com/phrgab/peaks/pull/53))

### Changed

- Pydantic v1 methods to their v2 equivalents ([PR#48](https://github.com/phrgab/peaks/pull/48))
- Diamond I05 data loader, following the analyser upgrade in 2026 ([PR#53](https://github.com/phrgab/peaks/pull/53))

## [0.5.1] - 2026-05-29

:::{important}
This version introduced a bug affecting the loading of SES-acquired zip files. Please install a later version if you work with this type of data.
:::

### Added

- Cross-references to the API documentation throughout the user guide ([PR#26](https://github.com/phrgab/peaks/pull/26))
- List of accessors that `peaks` adds on top of `xarray` on the explanations page ([PR#26](https://github.com/phrgab/peaks/pull/26))
- conda-forge package publishing
- Demo video showing how to use the ROI panel in the 4D GUI ([PR#30](https://github.com/phrgab/peaks/pull/30))
- D101, D102, D103 and D104 to `ruff` linting rules to enforce docstrings on public classes, methods, functions and packages ([PR#30](https://github.com/phrgab/peaks/pull/30))
- NumPy convention to `ruff` linting ([PR#30](https://github.com/phrgab/peaks/pull/30))
- Support for opening multiple `disp` panels simultaneously in Jupyter environments when `%gui qt6` is enabled ([PR#34](https://github.com/phrgab/peaks/pull/34))
- Helper methods to correct dead or hot pixels in dispersions ([PR#36](https://github.com/phrgab/peaks/pull/36))
- `_repr_html_` methods in the `Metadata` and `MetadataItem` classes, providing proper colour-coded HTML display of `da.metadata` and `da.metadata.<key>` in IPython-compatible environments ([PR#35](https://github.com/phrgab/peaks/pull/35))
- More unit tests for colour-encoded metadata display `core/metadata`: `metadata_method` ([PR#35](https://github.com/phrgab/peaks/pull/35))

### Fixed

- Errors when loading NetCDF files and processing Zarr files saved by `peaks` with `metadata=False` ([PR#20](https://github.com/phrgab/peaks/pull/20))
- Error when calling `plot_tutorial_example_figure` locally ([PR#28](https://github.com/phrgab/peaks/pull/28))
- Empty inline interactive `iplot` when using the `groupby` argument in static CI docs builds ([PR#30](https://github.com/phrgab/peaks/pull/30))
- Colour-coded metadata rendering in docs ([PR#35](https://github.com/phrgab/peaks/pull/35))
- Bug where analysis history was not updated when iterating accessor methods over a `DataTree` ([PR#39](https://github.com/phrgab/peaks/pull/39))

### Changed

- Minor documentation improvements and plenty of cosmetic auto-fixes required by numpy-style docstring format ([PR#23](https://github.com/phrgab/peaks/pull/23), [PR#30](https://github.com/phrgab/peaks/pull/30), [PR#33](https://github.com/phrgab/peaks/pull/33))
- Installation instructions now cover `uv`, conda-forge and the traditional Python venv approach ([PR#28](https://github.com/phrgab/peaks/pull/28))
- `plot_tutorial_example_figure` now renders videos ([PR#30](https://github.com/phrgab/peaks/pull/30))
- Figure aspect ratios in output cells are preserved on docs build ([PR#30](https://github.com/phrgab/peaks/pull/30))
- Installation instructions and contributing guide because the `dev` branch has been deprecated ([PR#40](https://github.com/phrgab/peaks/pull/40))
- `display_metadata` now takes an optional `palette` argument and the default `mode` is now `"HTML"` (was `"ANSI"`) ([PR#35](https://github.com/phrgab/peaks/pull/35))

### Removed

- `peaks/core/GUI/disp_panels/degrid_panel.py` - redundant legacy file ([PR#30](https://github.com/phrgab/peaks/pull/30))
- The `dev` branch ([Discussion#27](https://github.com/phrgab/peaks/discussions/27), [PR#40](https://github.com/phrgab/peaks/pull/40))

:::{note}
The `dev` branch was removed as of `v0.5.1`. Development now happens on `main`.
:::

## [0.5.0] - 2026-04-23

### Added

- Initial unit test suite covering core processing pipeline ([PR#10](https://github.com/phrgab/peaks/pull/10)):
  - `core/fileIO`: `data_loading`, `loc_registry`
  - `core/fitting`: `fit`, `fit_functions`, `models`
  - `core/process`: `data_select`, `differentiate`, `tools`
  - `core/utils`: `interpolation`, `misc`
- Unit tests added to CI workflow ([PR#10](https://github.com/phrgab/peaks/pull/10))
- `BaseOpticsDataLoader` class providing metadata handling for optical components ([PR#11](https://github.com/phrgab/peaks/pull/11))
- `precooling_stage` and `heater_power` entries in `BaseTemperatureDataLoader` ([PR#11](https://github.com/phrgab/peaks/pull/11))
- `heater_power` now in the metadata for data collected at I05, Diamond ([PR#11](https://github.com/phrgab/peaks/pull/11))
- `AGENTS.md` ([PR#13](https://github.com/phrgab/peaks/pull/13))
- Loader added for data collected at SGM4, ASTRID2 in Aarhus ([PR#3](https://github.com/phrgab/peaks/pull/3), [PR#16](https://github.com/phrgab/peaks/pull/16))

### Fixed

- `.smooth` and `.curvature` now preserve pint units during analysis ([PR#10](https://github.com/phrgab/peaks/pull/10))
- Non-deterministic azimuthal origin across python sessions in `radial_cuts` ([PR#10](https://github.com/phrgab/peaks/pull/10))
- Stale docstrings ([PR#10](https://github.com/phrgab/peaks/pull/10))
- OpticsMetadataModel serialisation issue for additional optics axes when saving data ([PR#12](https://github.com/phrgab/peaks/pull/12))

### Changed

- `I05NanoARPESLoader` optics metadata now uses `BaseOpticsDataLoader` instead of the bespoke `I05NanoFocussingMetadataModel` ([PR#11](https://github.com/phrgab/peaks/pull/11))
- Optics metadata stored under `_optics` attribute (previously `_focussing`) ([PR#11](https://github.com/phrgab/peaks/pull/11))
- Zenodo URL to version 3 ([PR#12](https://github.com/phrgab/peaks/pull/12))

:::{attention}
**For Diamond I05-nano data:** As a result, any files containing `I05NanoFocussingMetadataModel` previously saved by `peaks` using `.save` will not reload properly; please re-save from the original `.nxs` files.
:::

### Removed

- `core/utils/consts.py` deprecated as replaced by `scipy.constants` ([PR#10](https://github.com/phrgab/peaks/pull/10))
- `I05NanoFocussingMetadataModel` as replaced by `OpticsMetadataModel` ([PR#11](https://github.com/phrgab/peaks/pull/11))

## [0.4.9] - 2026-03-06

### Fixed

- `numpy`-related errors when converting an `ndim > 0` array to a scalar ([PR#4](https://github.com/phrgab/peaks/pull/4))
- `xarray` warnings about `.argmax()` calls ([PR#4](https://github.com/phrgab/peaks/pull/4))
- Boundary case in `plot_nanofocus` when all focus estimates are identical and `.std()` is 0 ([PR#4](https://github.com/phrgab/peaks/pull/4))

### Added

- Local mirror path support for structure example data, falling back to remote download if unavailable ([PR#5](https://github.com/phrgab/peaks/pull/5))
- CI for lint checks and installation tests ([PR#6](https://github.com/phrgab/peaks/pull/6), [PR#7](https://github.com/phrgab/peaks/pull/7))

### Removed

- The slant correction section from the docs, along with the associated function for downloading the related example data ([PR#5](https://github.com/phrgab/peaks/pull/5))
- `numba` version constraint ([PR#5](https://github.com/phrgab/peaks/pull/5))

## [0.4.8] - 2026-02-04

### Added

- Option in Zenodo downloader to pull example data from local path
- Installation note for Intel macOS

### Changed

- Removed Python version upper bound; now supports Python 3.11-3.14

## [0.4.7] - 2026-01-28

### Fixed

- Subtle crosshair deviations from real value on motor axes in 3D and 4D GUI
- Trimming non-monotonic data at end of travel in Diamond I05-nano ana_polar maps

## [0.4.6] - 2026-01-22

### Fixed

- Zenodo download failures
- Excluded yanked `uncertainties` version 3.2.4
- `lmfit` intersphinx URL for doc building

## [0.4.5] - 2026-01-19

### Fixed

- Bug when summing two FSMs
- Pinned `numba` version to >=0.60.0 to resolve dependency conflicts
- Downloading errors due to rate limiting

## [0.4.4] - 2025-10-15

### Fixed

- Pinned python version to <3.14 due to dependency requirements

## [0.4.3] - 2025-10-14

### Fixed

- k-conversion failure for merged hv scans

## [0.4.2] - 2025-08-08

### Changed

- Reference sign conventions of spatial co-ordinates

### Added

- Spatial sign conventions in various loaders
- Explanations of the sign convention for `deflector_parallel` in doc

## [0.4.1] - 2025-08-08

### Changed

- Minor docs update

## [0.4.0] - 2025-08-06

### Added

- Handling of automatic download of example data for the tutorials
- More robust file path handling in loading, including adding glob character support
- Automated running of tutorial tests and linting in CI, and lint pre-commit hook
- Setting up for publishing on PyPI

### Fixed

- Significant bug fixes in k-conversion

### Changed

- Updated the project documentation and packaging

### Removed

- Some place-holders for modules not yet implemented

## [0.3.1]

### Added

- Utilities for getting and plotting energies of XPS core levels
- General periodic table display

## [0.3.0]

**_Includes several breaking changes:_** significant update to the package, with a number of changes to the underlying data structures and new features.

### Added

- Support for xarray.DataTree structure
- New methods for time-resolved spectra in `time_resolved` module, and new data loader for Artemis@CLF
- Implemented use of `pint` and `pint_xarray` for unit handling; some methods now require this
- Methods for plotting Brillouin zones

### Fixed

- Significant changes to extraction of angles in data processing, including to fix various sign problems in angle conventions (NB sign conventions not yet checked for all loaders)
- Made GUIs a bit more robust against missing metadata

### Changed

- Complete change of base structure for data loaders, moving to class-based structure (not all loaders yet ported)
- New metadata format, including modified methods for setting and getting metadata entries, and metadata models based on `pydantic`
- Moved to a new structure for accessors, lazy importing and other under the hood refactoring
- Changed the structure of the `peaks` options, refactored into `options.py`
- Bumped minimum required version of `xarray` to 2024.10.0 for `DataTree` support
- Updated tutorials to reflect changes in the package
- Other small bug fixes throughout (NB still some bugs remaining, in particular for k-conversion)

### Removed

- Old data loaders moved to `to_depreciate` folder and these are no longer functional. Note, not all of these loaders have been ported to the new class-based structure yet.
- General tidy up of some outdated code
- Removed some modules not yet implemented to improve focus

## [0.2.0]

### Added

- Initial version of core functionality, for alpha testing within the group.
