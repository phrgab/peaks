(file_loaders)=
# File loaders
:::{attention}
If you develop a new data loader, please consider [contributing](#contributing) this to the `peaks` codebase. Please try and minimise the number of new dependencies required. 
:::

A core set of file loaders are included within `peaks.core.fileIO.loaders`. To make a new loader, the general approach is to subclass the most relevant base loader(s), implementing only the required custom logic. For data saved using the default format of one of the main ARPES spectrometer manufacturers, this can make the entire loader rather compact, largely defining the mapping from `peaks` to local [axis names and co-ordinate system](coordinate-conventions). 

For example, for data collected in the SES data format at the A-branch of the [Bloch beamline](http://blochdocs.maxiv.lu.se), we can subclass the `peaks.core.fileIO.base_arpes_data_classes_base_ses_class:SESDataLoader`, meaning the entire loader can be defined as: 

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

For data saved using a custom format, or for additional features not yet implemented in the `peaks` loader classes, a full loader function can be written, but this should still subclass e.g. the `BaseARPESDataLoader` class for ARPES data, or should subclass (a combination of) the `BasePhotonSourceDataLoader`, `BaseTemperatureDataLoader`, `BaseManipulatorDataLoader` for other data types. For more examples, see the current loaders in `peaks.core.fileIO.loaders`.

## Registering a new data loader
Currently supported file loaders can be accessed as:
```python
import peaks as pks

# Show current loaders
pks.locs()
```

To be used within `peaks`, a new loader must be registered using the `register_loader` accessor:
```python
from peaks.core.fileIO.loc_registry import register_loader

@register_loader
class NewLoader(...):
    ...
```

It will now become visible in the `pks.locs()` listing. To use this loader, call `pks.load()` passing `loc=<_loc_name>` as an argument. To be able to use automatic data source determination for calling this loader, the `peaks.core.fileIO.loc_registry:IdentifyLoc` class must be updated.

## Testing
As well as testing the core file loading functionality, please test that the local axis names show correctly in the display GUIs, and ensure that normal emissions parse as you would expect from the physical angles. If this is incorrect, it likely means that there is a [sign convention](#coordinate-conventions) that is not correctly respected.