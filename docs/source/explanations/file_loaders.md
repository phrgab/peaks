(file_loaders)=

# File loaders

:::{attention}
If you develop a new data loader, please consider [contributing](#contributing) this to the {py:mod}`peaks` codebase. Please try and minimise the number of new dependencies required.
:::

A core set of file loaders are included within {py:mod}`peaks.core.fileIO.loaders`.
These take raw data files recorded in the spectrometer or facility-specific format, and return structured {py:mod}`xarray` objects with metadata parsed into the standard {py:mod}`peaks` [metadata conventions](#metadata-conventions).

To make a new loader, subclass the most relevant base loader.
Manufacturer-specific base classes are defined for most of the core ARPES spectrometer manufacturers ({py:mod}`~peaks.core.fileIO.base_arpes_data_classes`).
This can make new system-specific data loaders rather compact, largely defining the mapping from {py:mod}`peaks` to local [axis names and co-ordinate system](coordinate-conventions).
For example, for data collected in the SES data format at the A-branch of the [Bloch beamline](http://blochdocs.maxiv.lu.se) (as of 2025), we can subclass the {py:class}`~peaks.core.fileIO.base_arpes_data_classes.base_ses_class.SESDataLoader`, meaning the entire loader can be defined as:

```python
class BlochArpesLoader(SESDataLoader):
    _loc_name = "MAXIV_Bloch_A"  # Unique identifier of the loader
    _loc_description = "A branch (ARPES) of Bloch beamline at Max-IV"  # One-line description
    _loc_url = "https://www.maxiv.lu.se/beamlines-accelerators/beamlines/bloch/"  # Link to the facility
    _analyser_slit_angle = 0 * ureg("deg")  # Angle of the entrance slit, crucial for determining k-space conversion functions

    # Manipulator name conventions, mapping `peaks` to local names
    _manipulator_name_conventions = {
        "polar": "P",
        "tilt": "T",
        "azi": "A",
        "x1": "X",
        "x2": "Y",
        "x3": "Z",
    }

    # Manipulator sign conventions (assumed +1 unless given as -1)
    _manipulator_sign_conventions = {
        "polar": -1,
        "tilt": -1,
    }

    # Sign conventions for the electron analyser angles (including deflectors)
    _analyser_sign_conventions = {
        "deflector_perp": -1,
    }

    # Additional mapping to give some specific metadata units
_SES_metadata_units = {
        f"manipulator_{dim}": ("mm" if dim in ["x1", "x2", "x3"] else "deg")
        for dim in _manipulator_name_conventions.keys()
    }
```

For data saved using a custom format, or for additional features not yet implemented in the existing classes, start from a lower-level base loaders:

- {py:class}`~peaks.core.fileIO.base_data_classes.base_data_class.BaseDataLoader`: the minimum class that can be used;
- {py:class}`~peaks.core.fileIO.base_arpes_data_classes.base_arpes_data_class.BaseARPESDataLoader`: the default class for ARPES data, which defines analyser, manipulator, sample temperature, and photon source metadata;
- {py:class}`~peaks.core.fileIO.base_data_classes.base_hdf5_class.BaseHDF5DataLoader` or {py:class}`~peaks.core.fileIO.base_data_classes.base_ibw_class.BaseIBWDataLoader`: helper mixins for specific data formats.

By defaul, the loader should implement `_load_data()` and `_load_metadata()` as class methods, letting {py:func}`~peaks.core.fileIO.base_data_classes.base_data_class.BaseDataLoader._load()` handle data loading, parsing, and metadata extraction.
`_load_data()` should allow for lazy loading of data where possible.
`_load_metadata()` should avoid a full data load if possible, to allow for quick parsing of metadata only for use, e.g., in logbook systems.
Specific metadata parsers are also required to map the loaded metadata values to validated metadata models: {py:mod}`~peaks.core.metadata.base_metadata_models`.
For examples, see the current loaders in {py:mod}`peaks.core.fileIO.loaders`.

## Metadata conventions

A core feature of {py:mod}`peaks` is that it provides rich metadata, some of which is specifically required in subsequent analysis (e.g. k-conversion).
To abstract facility- and instrument-specific details away, we define a common set of metadata names and fields which are defined via {py:class}`pydantic.BaseModel` models in {py:mod}`~peaks.core.metadata.base_metadata_models` and by our
[co-ordinate conventions](coordinate-conventions).
These can be expanded if required, e.g. to handle custom metadata specific to a given facility or to add additional axes to an optical system, but in general we strongly encourage using a standardised metadata convention wherever possible to ensure interoperability and ease of use across different facilities.
For some fields (e.g. manipulators, optics), reference names should be defined in the loader, mapping from the {py:mod}`peaks` [naming convention](coordinate-conventions)) to the local axis name to aid the experimenter during data acquisition.

To standardise loading of common metadata, additional mixins exist which data loaders can inherit from.
For example, {py:class}`~peaks.core.fileIO.base_arpes_data_classes.base_arpes_data_class.BaseARPESDataLoader` already inherits from {py:class}`~peaks.core.fileIO.base_data_classes.base_photon_source_classes.BasePhotonSourceDataLoader`, {py:class}`~peaks.core.fileIO.base_data_classes.base_temperature_class.BaseTemperatureDataLoader`, and {py:class}`~peaks.core.fileIO.base_data_classes.base_manipulator_class.BaseManipulatorDataLoader` to orchestrate loading of this common metadata for all ARPES data, including defining the metadata parsing logic.
Nano-ARPES loaders additionally inherit from the {py:class}`~peaks.core.fileIO.base_data_classes.base_optics_class.BaseOpticsDataLoader` mixin to define the focussing optics.

## Registering a new data loader

Location identifiers for currently supported file loaders can be accessed as:

```python
import peaks as pks

# Show current loaders
pks.locs()
```

To be used within {py:mod}`peaks`, a new loader must be registered using the {py:func}`register_loader <peaks.core.fileIO.loc_registry.register_loader>` accessor:

```python
from peaks.core.fileIO.loc_registry import register_loader

@register_loader
class NewLoader(...):
    ...
```

It will now become visible in the {py:func}`pks.locs() <peaks.locs>` listing. To use this loader, call {py:func}`pks.load <peaks.core.fileIO.data_loading.load>` passing `loc=<_loc_name>` as an argument. To be able to use automatic data source determination for calling this loader, the {py:class}`peaks.core.fileIO.loc_registry.IdentifyLoc` class must be updated.

## Testing

As well as testing the core file loading functionality, please test that the local axis names show correctly in the display GUIs, and ensure that normal emissions parse as you would expect from the physical angles. If this is incorrect, it likely means that there is a [sign convention](#coordinate-conventions) that is not correctly respected.
