# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.5]

### Fixed
- Bug when summing two FSMs
- Pinned numba version to >=0.60.0 to resolve dependency conflicts
- Downloading errors due to rate limiting

## [0.4.4]

### Fixed
- Pinned python version to <3.14 due to dependency requirements


## [0.4.3]

### Fixed
- k-conversion failure for merged hv scans

## [0.4.2]

### Changed
- Reference sign conventions of spatial co-ordinates

### Added
- Spatial sign conventions in various loaders
- Explanations of the sign convention for `deflector_parallel` in doc

## [0.4.1]

### Changed
- Minor docs update

## [0.4.0]

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
***Includes several breaking changes:*** significant update to the package, with a number of changes to the underlying data structures and new features. 

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
