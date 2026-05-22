(file_loaders)=

# File loaders

:::{attention}
If you develop a new data loader, please consider [contributing](#contributing) this to the `peaks` codebase. Please try and minimise the number of new dependencies required.
:::

A core set of file loaders are included within {py:mod}`peaks.core.fileIO.loaders`.
These take the raw data files in the spectrometer or facility-specific format, and return an {py:class}`xarray.DataArray` with the data loaded as the labelled array, and relevant metadata loaded following a standard set of [metadata conventions](#metadata-conventions).
To make a new loader, the general approach is to subclass the most relevant base loader(s), implementing only the required custom logic.

Manufacturer-specific base classes are defined for most of the core ARPES spectrometer manufacturers ({py:mod}`peaks.core.fileIO.base_arpes_data_classes`).
This can make new system-specific data loaders rather compact, largely defining the mapping from `peaks` to local [axis names and co-ordinate system](coordinate-conventions).
For example, for data collected in the SES data format at the A-branch of the [Bloch beamline](http://blochdocs.maxiv.lu.se) (as of 2025), we can subclass the {py:class}`peaks.core.fileIO.base_arpes_data_classes.base_ses_class.SESDataLoader`, meaning the entire loader can be defined as:

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

For data saved using a custom format, or for additional features not yet implemented in the `peaks` loader classes, a more extensive loader function should be written.
This must subclass one of the base loader classes: typically {py:class}`~peaks.core.fileIO.base_arpes_data_classes.base_arpes_data_class.BaseARPESDataLoader` for ARPES data but the base {py:class}`~peaks.core.fileIO.base_data_classes.base_data_class.BaseDataLoader` can be used for other data types.

The loader must define a `_load_data` classmethod which loads the spectral data, handling also logic for optional lazy loading where possible. It should also define a `_load_metadata` classmethod which orchestrates metadata loading from the file.
Where possible, metadata loading should be performed without loading the full data, to allow for quick parsing of metadata only for use, e.g., in logbook systems.
Specific metadata parsers are also required to map the loaded metadata values to the `peaks` [metadata model](#metadata-conventions).
For examples, see the current loaders in {py:mod}`peaks.core.fileIO.loaders`.

## Metadata conventions

A core feature of `peaks` is that it provides rich metadata, some of which is specifically required in subsequent analysis (e.g. k-conversion).
To abstract facility- and instrument-specific details away, we define a common set of metadata names which are defined as [`pydantic`](https://pydantic.dev/docs/validation/latest/get-started/) models in {py:mod}`~peaks.peaks.core.metadata.base_metadata_models` and by our
[co-ordinate conventions](coordinate-conventions).
These can be expanded if required, e.g. to handle custom metadata specific to a given facility or to add additional axes to a manipulator or optical system, but in general we strongly encourage using a standardised metadata convention wherever possible.
For some fields (e.g. manipulators, optics), reference names should be defined in the loader, mapping from the `peaks` naming convention to the local axis name to aid the experimenter during data acquisition.

To standardise loading of common metadata, additional base classes exist which data loaders can inherit from.
For example, {py:class}`~peaks.core.fileIO.base_arpes_data_classes.base_arpes_data_class.BaseARPESDataLoader` already inherits from {py:class}`~peaks.core.fileIO.base_data_classes.base_photon_source_classes.BasePhotonSourceDataLoader`, {py:class}`~peaks.core.fileIO.base_data_classes.base_temperature_class.BaseTemperatureDataLoader`, and {py:class}`~peaks.core.fileIO.base_data_classes.base_manipulator_class.BaseManipulatorDataLoader` to orchestrate loading of this common metadata for all ARPES data, including defining the metadata parsing logic.
Nano-ARPES loaders additionally inherits from the {py:class}`~peaks.core.fileIO.base_data_classes.base_optics_class.BaseOpticsDataLoader`.

## Registering a new data loader

Currently supported file loaders can be accessed as:

```python
import peaks as pks

# Show current loaders
pks.locs()
```

To be used within `peaks`, a new loader must be registered using the {py:func}`register_loader <peaks.core.fileIO.loc_registry.register_loader>` accessor:

```python
from peaks.core.fileIO.loc_registry import register_loader

@register_loader
class NewLoader(...):
    ...
```

It will now become visible in the {py:func}`pks.locs() <peaks.locs>` listing. To use this loader, call {py:func}`pks.load <peaks.core.fileIO.data_loading.load>` passing `loc=<_loc_name>` as an argument. To be able to use automatic data source determination for calling this loader, the {py:class}`peaks.core.fileIO.loc_registry.IdentifyLoc` class must be updated.

## Testing

As well as testing the core file loading functionality, please test that the local axis names show correctly in the display GUIs, and ensure that normal emissions parse as you would expect from the physical angles. If this is incorrect, it likely means that there is a [sign convention](#coordinate-conventions) that is not correctly respected.
