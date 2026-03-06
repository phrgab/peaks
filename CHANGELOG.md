# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
