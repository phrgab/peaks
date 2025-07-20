(coordinate-conventions)=
# Coordinate conventions
`peaks` aims to ensure a consistent handling of angle sign conventions in ARPES $k$-conversions, while still facilitating its easy use for on-the-fly analysis during data acquisition, where it is helpful to keep the same sign conventions as for the host system. 

The primary `peaks` coordinate convention is shown [below](#coord_conventions_figure).
```{figure} ../figs/angle_conventions.jpeg
:scale: 50 %
:alt: angle conventions schematic
:name: coord_conventions_figure

`peaks` sample manipulator and electron analyser coordinate conventions.
```
We assume a sample manipulator with a primary rotary axis, which we term `polar`. At the sample stage, a secondary `tilt` and tertiary `azi` axis provide the sample angular positioning control. Similarly, the spectrometer is defined by polar, tilt, and azimuth angles which have the same sign convention as the manipulator ones. For a fixed analyser geometry, the `ana_polar` and `ana_tilt` angles are typically zero, but these allow, e.g., inclusion of a moving analyser. `ana_azi`$=0$ defines that the entrance slit of a hemispherical analyser is aligned along the polar axis. The manipulator also has three spatial dimensions, `x1`, `x2`, `x3` as shown in the [figure](#coord_conventions_figure).

:::{important}
When data are loaded, the local axis names are converted to `peaks` conventions to allow uniform handling in function calls. However, **signs are not changed to match our [conventions](#coordinate-conventions)**. This choice is made so that that reading off a value of the dataset at, e.g. a given angle or spatial position, still corresponds to that of the physical manipualtor/analyser. This significantly simplifies interpretation when using `peaks` for on-the-fly data analysis, reducing load on the experimenter.
:::

While sign conversions are not applied upon data loading, [file loaders](#file_loaders) define dictionaries mapping `peaks` name and sign conventions to those of the physical system. This allows GUIs to be able to provide axis names following the local nomenclature in addition to the `peaks` convention, again as an aid to the experimenter. It also allows having a unified approach for  co-ordinate systems to aid the experimenter, enables a uniquely defined definition of how axes should stack together (e.g. `polar` and `ana_polar`), and allows a consistent co-ordinate system to be used in [$k$-space conversions](#k-space-conversion).

(k-space-conversion)=
## ARPES k-space conversion
ARPES $k$-space conversion follows the approach and definitions from Ishida and Shin, Rev. Sci. Instrum. 89, 043903 (2018):
:::{seealso}
[*Functions to map photoelectron distributions in a variety of setups in angle-resolved photoemission spectroscopy*](https://aip.scitation.org/doi/10.1063/1.5007226) \
Y. Ishida and S. Shin \
Rev. Sci. Instrum. 89, 043903 (2018).
:::
The conversion is exact (no small angle approximations are made). The following analyser types (with or without deflectors) are currently supported:
- Type I (in the nomenclature of Ishidia and Shin): analyser slit aligned with main rotary axis of manipulator (defined in the file loader with `_analyser_slit_angle = 0 * ureg('degrees')`), often called 'vertical' slit for a vertically-oriented sample manipualator;
- Type II: analyser slit perpendicular to main rotary axis of manipulator (`_analyser_slit_angle = 90 * ureg('degrees')`), often called a 'horizontal' slit. 

The `peaks` [angular conventions](#coordinate-conventions) are consistent with those of Ishida and Shin, but: 
- additionally implement analyser rotations; and 
- enforce a fixed set of axis names irrespective of the analyser type, again as an aide to the experimenter who may use multiple different types. 

Any sign changes required from the unconverted data are handled during the $k$-conversion processing. 

:::{danger}
Historically, there have been some bugs in these sign conventions, and this been more carefully checked for some file loaders than others. If things are working correctly, the normal emission (reference angles) determined e.g. using the display panel GUIs, should correspond to the physical normal emissions on the manipulator. Setting these as the reference angles should then lead to $k$-converted data with the normal emission located as expected, and with the signs of the $k$ axes matching the conventions of Ishida and Shin. If you find bugs with any of these aspects, please raise an issue at the `peaks` [GitHub repository](https://github.com/phrgab/peaks).
:::
