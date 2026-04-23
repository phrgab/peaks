# peaks AGENTS.md

`peaks` is a Python package for loading, processing, and analysing
Angle-Resolved Photoemission Spectroscopy (ARPES) data. Data are represented
as `xarray.DataArray` objects and most tools are available both as standalone
functions and as accessor methods (e.g. `data.smooth(...)`).

## Conventions

- **Accessor notation**: all tools listed here are registered as methods on
  `xr.DataArray` (and where applicable `xr.Dataset` / `xr.DataTree`). Call
  them as `data.<tool>(...)`.
- **Axis names**: `eV` (energy), `theta_par` / `k_par` / `kx` / `ky` / `kz`
  (angle or momentum), `x1` / `x2` (spatial), `hv` (photon energy).
- **Units**: axes carry `pint` units. Smoothing/broadening FWHMs should be
  given in axis units (or as `pint.Quantity`).
- **Analysis history**: every tool appends a record to `data.history` for
  provenance tracking.

## Tools

### load

- description: Load one or more ARPES data files into an `xr.DataArray` (single file) or `xr.DataTree` (multiple files). Supports `.ibw`, `.nxs`, `.zip`, `.krx`, `.xy`, `.nc`, `.zarr`, and other beamline formats. Metadata is parsed automatically.
- function: peaks.core.fileIO.data_loading:load
- parameters:
  - fpath: str or list (required) — file path(s) to load
  - lazy: bool (optional, default=None) — load lazily with dask; auto-determined by file size if None
  - loc: str (optional, default=None) — beamline/location identifier; auto-detected if None
  - metadata: bool (optional, default=True) — parse metadata into DataArray attributes
  - parallel: bool (optional, default=False) — load multiple files in parallel (h5/nxs only)
  - quiet: bool (optional, default=False) — suppress warnings
- returns: xr.DataArray, xr.Dataset, or xr.DataTree
- when_to_use: Start of every analysis workflow
- family: io
- strategy: file_load
- cost: low

### save

- description: Save a DataArray or Dataset as a NetCDF file, or a DataTree as a Zarr store. Peaks metadata is serialised into the file attributes where possible.
- function: peaks.core.fileIO.data_saving:save
- parameters:
  - data: xr.DataArray, xr.Dataset, or xr.DataTree (required) — data to save
  - fpath: str (required) — output file path (extension optional; .nc for DataArray/Dataset, .zarr for DataTree)
- returns: None
- when_to_use: Saving processed data for later reuse or sharing
- family: io
- strategy: file_save
- cost: low

### bin_data

- description: Bin (coarsen) data along one or more dimensions by integer factors. A thin wrapper around `xr.DataArray.coarsen` that also updates analysis history.
- function: xr.DataArray accessor .bin_data()
- parameters:
  - binning: int (optional, default=None) — apply the same bin factor to all dimensions; takes priority over per-dimension kwargs
  - boundary: str (optional, default='trim') — boundary handling: 'trim', 'exact', or 'pad'
  - \*\*binning_kwargs: int — per-dimension bin factors, e.g. eV=3, theta_par=2
- returns: xr.DataArray with reduced dimensions
- when_to_use: Low SNR data, large datasets needing size reduction, quick-look analysis
- addresses: noise, large_dataset
- effects: reduces_noise, reduces_resolution
- family: denoising
- strategy: binning
- cost: low

### bin_spectra

- description: Shortcut to bin_data that automatically identifies and bins the spectral dimensions (eV and theta_par/k_par) by a uniform factor, leaving spatial dimensions unchanged.
- function: xr.DataArray accessor .bin_spectra()
- parameters:
  - binning: int (optional, default=2) — bin factor for spectral dimensions
  - boundary: str (optional, default='trim') — boundary handling
- returns: xr.DataArray with binned spectral dimensions
- when_to_use: Quick noise reduction when the spectral dimensions are known but spatial ones should be preserved
- addresses: noise
- effects: reduces_noise, reduces_resolution
- family: denoising
- strategy: binning
- cost: low

### smooth

- description: Gaussian smoothing along one or more dimensions. FWHM values are specified in axis units (or as `pint.Quantity`); dimensions not listed are left unchanged.
- function: xr.DataArray accessor .smooth()
- parameters:
  - \*\*smoothing_kwargs: float or pint.Quantity — FWHM per axis, e.g. eV=0.05, theta_par=0.3
- returns: xr.DataArray with smoothed intensity
- when_to_use: Noisy data where binning would lose too many points; always apply before derivative/curvature analysis
- addresses: noise
- effects: reduces_noise, reduces_resolution
- family: denoising
- strategy: gaussian
- cost: low

### norm

- description: Normalise data intensity. Can normalise to unity (max), by the mean, by an integrated distribution curve along a given dimension, or by the integrated intensity of an arbitrary slice/ROI.
- function: xr.DataArray accessor .norm()
- parameters:
  - dim: str (optional, default=None) — 'all' to normalise by global mean; axis name (e.g. 'eV') to normalise by an integrated DC along that direction; None to normalise to unity
  - \*\*kwargs: slice — coordinate slices defining a ROI to normalise by, e.g. eV=slice(-0.5, -0.3)
- returns: xr.DataArray with normalised intensity
- when_to_use: Before comparing spectra across different scans or photon energies; correcting for varying photon flux
- addresses: intensity_variation
- effects: normalises_intensity
- family: normalization
- strategy: region_based
- cost: low

### bgs

- description: Background subtraction. Supports subtraction of a constant, mean, integrated distribution curve along a given dimension, a region-defined background, or a Shirley background (for XPS).
- function: xr.DataArray accessor .bgs()
- parameters:
  - subtraction: float, pint.Quantity, or str (optional, default=None) — value to subtract; 'all' for mean; axis name for integrated DC; 'Shirley' for iterative Shirley background
  - num_avg: int (optional, default=1) — Shirley: number of points to average at data endpoints
  - offset_start: float (optional, default=0) — Shirley: offset subtracted from start value
  - offset_end: float (optional, default=0) — Shirley: offset subtracted from end value
  - max_iterations: int (optional, default=10) — Shirley: maximum convergence iterations
  - \*\*kwargs: slice — coordinate slices defining the background region
- returns: xr.DataArray with background subtracted
- when_to_use: Removing a slowly varying or step-like background before fitting or derivative analysis
- addresses: background
- effects: removes_background
- family: background_subtraction
- strategy: region_based, shirley
- cost: low

### k_convert

- description: Convert angular ARPES data to momentum (k-space) using the free-electron final-state model. Handles 2D dispersions, 3D Fermi maps, and hν scans (producing kz). Reads analyser geometry and photon energy from metadata.
- function: xr.DataArray accessor .k_convert()
- parameters:
  - eV: slice (optional) — binding energy range for the output, e.g. slice(-1.0, 0.1, 0.002)
  - eV_slice: tuple (optional) — (energy, width) to return a single MDC-like energy slice after conversion
  - kx: slice (optional) — kx range for the output
  - ky: slice (optional) — ky range for the output
  - kz: slice (optional) — kz range for hν scans
  - return_kz_scan_in_hv: bool (optional, default=False) — return hν-scan data as (hv, eV, k_par) not (kz, eV, k_par)
  - quiet: bool (optional, default=False) — suppress progress bar and warnings
- returns: xr.DataArray in k-space (k_par/kx/ky replaces theta_par; kz replaces hv for hν scans)
- when_to_use: Band mapping, Fermi surface mapping, kz dispersion; any analysis requiring momentum coordinates
- addresses: angular_coordinates
- effects: converts_to_momentum
- family: coordinate_transform
- strategy: free_electron_final_state
- cost: medium

### estimate_EF

- description: Quickly estimate the Fermi level position from the peak of the first derivative of the angle-integrated spectrum. For hν scans, fits a polynomial correction. This is an approximate method suitable for seeding fit routines or GUIs, not for publication-quality determination.
- function: xr.DataArray accessor .estimate_EF()
- returns: float (single scan) or numpy.ndarray (hν scan) — estimated Fermi level in eV
- when_to_use: Quick energy calibration check; seed value for fit_gold or EF correction; when Fermi level position is unknown
- addresses: energy_calibration
- effects: identifies_fermi_level
- family: calibration
- strategy: derivative_peak
- cost: low

### fit_gold

- description: Fit a gold reference spectrum to a LinearDosFermiModel (Fermi-Dirac × linear DOS + background, convolved with a Gaussian) to extract EF, temperature, and instrumental energy resolution. For 2D data (gold angle map), also fits and returns a polynomial EF correction as a function of angle.
- function: peaks.core.fitting.fit:fit_gold
- parameters:
  - data: xr.DataArray (required) — 1D or 2D gold reference spectrum
  - EF_correction_type: str (optional, default='poly4') — type of EF correction for 2D data: 'poly4', 'poly3', 'quadratic', 'linear', or 'average'
  - \*\*kwargs — initial parameter overrides (e.g. bg_slope=0)
- returns: xr.Dataset with fit parameters (EF, T, sigma_conv, DOS coefficients, background coefficients), uncertainties, lmfit ModelResult objects, and EF_correction attribute
- when_to_use: Determining instrumental energy resolution; obtaining a precise EF correction before applying to sample data
- addresses: energy_calibration, resolution
- effects: measures_fermi_level, measures_resolution
- family: calibration
- strategy: fermi_dirac_fit
- cost: medium

### fit

- description: Fit an `lmfit.Model` to a DataArray, broadcasting over non-independent dimensions. Sequential fitting (updating initial parameters from the previous slice) is supported for 2D data, which greatly improves convergence along a slowly varying axis.
- function: xr.DataArray accessor .fit()
- parameters:
  - model: lmfit.Model (required) — model to fit
  - params: lmfit.Parameters (required) — initial parameters
  - independent_var: str (optional, default=None) — dimension name for the independent variable; required for 2D+ data
  - sequential: bool (optional, default=True) — update initial params from previous slice when fitting 2D data
  - reverse_sequential_fit_order: bool (optional, default=False) — reverse the order of sequential fitting
- returns: xr.Dataset with best-fit parameter values, uncertainties, and lmfit ModelResult objects
- when_to_use: Peak fitting (MDCs, EDCs), lineshape analysis, extracting band positions and widths
- addresses: peak_extraction
- effects: extracts_parameters
- family: fitting
- strategy: lmfit
- cost: medium

### MDC

- description: Extract one or more Momentum Distribution Curves (MDC) from a dispersion at specified energy value(s), with optional energy integration window.
- function: xr.DataArray accessor .MDC()
- parameters:
  - E: float, list, tuple, or numpy.ndarray (optional, default=0) — energy value(s); tuple format (start, end, step) for a range
  - dE: float (optional, default=0) — total integration window in eV (integrates ±dE/2)
- returns: xr.DataArray — 1D (or stacked 2D) MDC(s)
- when_to_use: Band dispersion extraction, linewidth analysis, Fermi surface extraction from a 3D cube
- addresses: dimensionality_reduction
- effects: extracts_momentum_profile
- family: slicing
- strategy: energy_cut
- cost: low

### EDC

- description: Extract one or more Energy Distribution Curves (EDC) from a dispersion at specified momentum/angle value(s), with optional integration window.
- function: xr.DataArray accessor .EDC()
- parameters:
  - k: float, list, tuple, or numpy.ndarray (optional, default=0) — k or theta_par value(s); tuple format (start, end, step) for a range
  - dk: float (optional, default=0) — total integration window in axis units (integrates ±dk/2)
- returns: xr.DataArray — 1D (or stacked 2D) EDC(s)
- when_to_use: Gap measurement, self-energy analysis, lineshape fitting at fixed momentum
- addresses: dimensionality_reduction
- effects: extracts_energy_profile
- family: slicing
- strategy: momentum_cut
- cost: low

### DC

- description: General distribution curve extraction along any coordinate, at one or more values with an optional integration window. MDC and EDC are convenience wrappers around this function.
- function: xr.DataArray accessor .DC()
- parameters:
  - coord: str (optional, default='eV') — coordinate to cut along
  - val: float, list, tuple, or numpy.ndarray (optional, default=0) — value(s) to extract at; tuple (start, end, step) for a range
  - dval: float (optional, default=0) — total integration window in coordinate units
- returns: xr.DataArray — extracted DC(s)
- when_to_use: Extracting cuts along non-standard axes (e.g. hv, x1, x2, ana_polar)
- addresses: dimensionality_reduction
- effects: extracts_profile
- family: slicing
- strategy: coordinate_cut
- cost: low

### DOS

- description: Integrate over all non-energy dimensions to return the best approximation to the density of states available from the data.
- function: xr.DataArray accessor .DOS()
- returns: xr.DataArray — 1D energy spectrum (DOS)
- when_to_use: Overview of spectral weight; seed for estimate_EF; before Fermi-Dirac fitting
- addresses: dimensionality_reduction
- effects: extracts_dos
- family: slicing
- strategy: full_integration
- cost: low

### tot

- description: Integrate spatial map data over all non-spatial (energy + angle/k) dimensions to produce a real-space intensity image, or over all spatial dimensions to produce a dispersion.
- function: xr.DataArray accessor .tot()
- parameters:
  - spatial_int: bool (optional, default=False) — if True, integrate over spatial (x1, x2) dimensions instead
- returns: xr.DataArray — spatially resolved or spatially integrated map
- when_to_use: Spatial maps (nanoARPES): obtain total intensity image or spatially averaged dispersion
- addresses: dimensionality_reduction
- effects: extracts_spatial_map, extracts_dispersion
- family: slicing
- strategy: full_integration
- cost: low

### deriv

- description: Differentiate data along one or more specified dimensions, in sequence. General-purpose wrapper around `xr.DataArray.differentiate` that preserves metadata and analysis history.
- function: xr.DataArray accessor .deriv()
- parameters:
  - dims: str or list of str (required) — dimension(s) to differentiate along, in order (repeat to obtain higher-order derivatives)
- returns: xr.DataArray with differentiated values
- when_to_use: Custom derivative orders or mixed derivatives not covered by the shortcut functions
- addresses: feature_visibility
- effects: enhances_features, destroys_intensity
- family: visualization
- strategy: finite_difference
- cost: low

### d2E

- description: Shortcut for double differentiation along the energy (eV) axis. Enhances band features and suppresses slowly varying background.
- function: xr.DataArray accessor .d2E()
- returns: xr.DataArray with d^2I/dE^2 values
- when_to_use: Band structure visualisation. Apply AFTER smoothing and LAST in a processing chain — destroys quantitative intensity.
- addresses: feature_visibility
- effects: enhances_features, destroys_intensity
- family: visualization
- strategy: second_derivative
- cost: low

### d2k

- description: Shortcut for double differentiation along the momentum (or angle) dimension. Works whether data is in angle or k-space.
- function: xr.DataArray accessor .d2k()
- returns: xr.DataArray with d^2I/dk^2 values
- when_to_use: Enhancing features along the momentum direction; complement to d2E
- addresses: feature_visibility
- effects: enhances_features, destroys_intensity
- family: visualization
- strategy: second_derivative
- cost: low

### dEdk

- description: Shortcut for sequential differentiation: first along eV, then along the momentum/angle dimension.
- function: xr.DataArray accessor .dEdk()
- returns: xr.DataArray with d^2I/dEdk values
- when_to_use: Mixed derivative enhancement; highlights features that vary in both energy and momentum
- addresses: feature_visibility
- effects: enhances_features, destroys_intensity
- family: visualization
- strategy: mixed_derivative
- cost: low

### dkdE

- description: Shortcut for sequential differentiation: first along the momentum/angle dimension, then along eV.
- function: xr.DataArray accessor .dkdE()
- returns: xr.DataArray with d^2I/dkdE values
- when_to_use: Mixed derivative enhancement; order-reversed complement to dEdk
- addresses: feature_visibility
- effects: enhances_features, destroys_intensity
- family: visualization
- strategy: mixed_derivative
- cost: low

### curvature

- description: 2D curvature analysis following Zhang et al., Rev. Sci. Instrum. 82, 043712 (2011). Better preserves feature sharpness than the second derivative. Free parameters must be provided for both axes of the 2D data; set a parameter to 0 for 1D curvature along the other axis.
- function: xr.DataArray accessor .curvature()
- parameters:
  - \*\*parameter_kwargs: float — free parameters per axis in format axis=value, e.g. eV=1, theta_par=10; both axes required
- returns: xr.DataArray with curvature values
- when_to_use: Publication-quality band visualisation; preferred alternative to d2E for preserving lineshape; apply after smoothing
- addresses: feature_visibility
- effects: enhances_features, destroys_intensity
- family: visualization
- strategy: curvature
- cost: medium

### min_gradient

- description: Minimum gradient analysis following He et al., Rev. Sci. Instrum. 88, 073903 (2017). Computes the ratio of the data to the Gaussian-filtered gradient magnitude; enhances band features while being less sensitive to noise than derivatives.
- function: xr.DataArray accessor .min_gradient()
- parameters:
  - \*\*smoothing_kwargs: float — Gaussian filter FWHM per axis in axis units, e.g. eV=0.05, theta_par=0.2; axes not listed default to 1 pixel
- returns: xr.DataArray with minimum gradient values
- when_to_use: Band visualisation in noisy data; robust alternative to curvature when derivative methods fail
- addresses: feature_visibility, noise
- effects: enhances_features
- family: visualization
- strategy: minimum_gradient
- cost: medium

### sym

- description: Mirror-symmetrise data about a specified coordinate value along one axis. The result is the sum of the original and its mirror image. Can also return just the flipped copy.
- function: xr.DataArray accessor .sym()
- parameters:
  - flipped: bool (optional, default=False) — return only the mirrored data rather than original + mirror
  - fillna: bool (optional, default=True) — fill NaNs in non-overlapping regions with 0
  - \*\*sym_kwarg: float or pint.Quantity — axis and centre value, e.g. theta_par=1.4 or eV=0; only one axis supported; defaults to eV=0
- returns: xr.DataArray — symmetrised (or flipped) data
- when_to_use: Enforcing time-reversal symmetry in dispersions; symmetrising about normal emission; preparing data for Fermi surface analysis
- addresses: asymmetry
- effects: enforces_symmetry
- family: symmetry
- strategy: mirror
- cost: low

### sym_nfold

- description: Apply n-fold rotational symmetry to 2D or 3D data about a specified centre. Useful for enforcing crystal point-group symmetry in Fermi surface maps.
- function: xr.DataArray accessor .sym_nfold()
- parameters:
  - nfold: int (required) — rotational order (e.g. 4 for C4, 6 for C6)
  - expand: bool (optional, default=True) — expand the coordinate grid to show all symmetrised data
  - fillna: bool (optional, default=True) — fill NaNs from rotation with 0 before summing
  - \*\*centre_kwargs: float — centre of rotation per axis, e.g. kx=0.1, ky=0.0; defaults to (0, 0)
- returns: xr.DataArray — n-fold symmetrised data
- when_to_use: Fermi surface maps with known crystal symmetry; increasing effective statistics by exploiting symmetry
- addresses: asymmetry, statistics
- effects: enforces_symmetry, improves_statistics
- family: symmetry
- strategy: rotational
- cost: medium

### estimate_sym_point

- description: Estimate the symmetry centre (e.g. normal emission, Γ point) of data by phase cross-correlation of the data with its mirror image. Returns a dict of {axis: centre_value}.
- function: xr.DataArray accessor .estimate_sym_point()
- parameters:
  - dims: str, list, or tuple (optional, default=None) — dimensions to estimate the centre along; defaults to all dimensions
  - upsample_factor: int (optional, default=100) — subpixel precision factor for phase cross-correlation
- returns: dict mapping dimension name to estimated centre coordinate value
- when_to_use: Before symmetrisation; finding normal emission in a dispersion; locating the Γ point in a Fermi surface map
- addresses: alignment
- effects: identifies_centre
- family: alignment
- strategy: phase_cross_correlation
- cost: low

### rotate

- description: Rotate 2D (or 3D) data by a specified angle about a given centre of rotation, using bilinear interpolation.
- function: xr.DataArray accessor .rotate()
- parameters:
  - rotation: float (required) — rotation angle in degrees (anticlockwise)
  - \*\*centre_kwargs: float — centre of rotation per axis, e.g. kx=0, ky=0; defaults to (0, 0)
- returns: xr.DataArray — rotated data on the original coordinate grid
- when_to_use: Correcting angular misalignment in Fermi surface maps; aligning crystal axes with k-space axes
- addresses: misalignment
- effects: corrects_rotation
- family: alignment
- strategy: rotation
- cost: medium

### degrid

- description: Remove a periodic mesh grid artefact from 2D data by identifying and zeroing high-intensity spikes in the FFT outside a protected central low-frequency region.
- function: xr.DataArray accessor .degrid()
- parameters:
  - width: float (optional, default=0.1) — fraction of total width to protect as the central low-frequency region
  - height: float (optional, default=0.1) — fraction of total height to protect as the central low-frequency region
  - cutoff: float (optional, default=4) — intensity threshold as a multiple of the FFT mean for spike removal
- returns: xr.DataArray — data with grid artefact removed
- when_to_use: data from instruments with a physical mesh; before spatial or spectral analysis
- addresses: grid_artefact
- effects: removes_artefact
- family: artefact_removal
- strategy: fft_filtering
- cost: low

### drop_nan_borders

- description: Trim all-NaN rows/columns from the edges of each dimension of a DataArray or Dataset.
- function: xr.DataArray accessor .drop_nan_borders()
- returns: xr.DataArray or xr.Dataset — trimmed data
- when_to_use: After k-conversion or rotation, which introduce NaN borders; before fitting or display
- addresses: nan_borders
- effects: trims_data
- family: cleaning
- strategy: border_trim
- cost: low

### drop_zero_borders

- description: Trim all-zero rows/columns from the edges of each dimension of a DataArray or Dataset.
- function: xr.DataArray accessor .drop_zero_borders()
- returns: xr.DataArray or xr.Dataset — trimmed data
- when_to_use: After operations that pad with zeros; before fitting or display
- addresses: zero_borders
- effects: trims_data
- family: cleaning
- strategy: border_trim
- cost: low

---

**Interactive tools** — the following tools render interactive widgets and require a Jupyter notebook or equivalent interactive environment.

---

### disp

- description: Interactive GUI viewer for 2D, 3D, or 4D DataArrays, or a list of 2D DataArrays. Displays linked panels (image, MDC, EDC slices) with keyboard-navigable sliders. Also accepts an `xr.DataTree` of 2D scans. Open the in-app Help menu for keyboard shortcuts.
- function: xr.DataArray accessor .disp()
- parameters:
  - primary_dim: str, tuple, or list (optional, default=None) — dimension(s) to treat as the primary axis; for 4D data pass two dims; defaults to conventional peaks ARPES layout
  - exclude_from_centering: str, tuple, list, or None (optional, default='eV') — dimension(s) excluded from auto-centering in the viewer
- returns: None (renders an interactive Panel widget in the notebook)
- when_to_use: Exploratory data inspection for any dimensionality; navigating 3D or 4D cubes interactively; comparing a stack of 2D dispersions side-by-side
- family: display
- strategy: interactive_gui
- cost: low

### iplot

- description: Interactive Bokeh/HoloViews plot via hvplot. Thin wrapper around `data.hvplot(...)` that automatically strips pint units before plotting. Auto-selects plot type by dimensionality (line for 1D, image for 2D, etc.); all hvplot keyword arguments are passed through.
- function: xr.DataArray accessor .iplot()
- parameters:
  - \*args, \*\*kwargs — passed directly to `hvplot`; common options include `x`, `y`, `z`, `cmap`, `clim`, `width`, `height`, `colorbar`, `title`, `xlabel`, `ylabel`; see hvplot documentation for the full list
- returns: holoviews.core.element.Element — an interactive Bokeh plot displayed inline in the notebook
- when_to_use: Quick interactive inspection of any DataArray; panning/zooming into dispersions or Fermi surfaces; preferred over `.plot()` when interactivity is needed
- family: display
- strategy: interactive_hvplot
- cost: low

### plot_DCs

- description: Plot a stack of 1D distribution curves (MDCs, EDCs, or any 1D profiles) as coloured lines, with optional colormap, per-curve vertical offset, and normalisation. Accepts a list of 1D DataArrays, a 2D DataArray (stack), or an `xr.DataTree` of DCs.
- function: xr.DataArray accessor .plot_DCs()
- parameters:
  - titles: list (optional, default=None) — legend labels, one per curve; omit for no legend
  - cmap: str (optional, default='coolwarm') — matplotlib colormap for line colours
  - color: str (optional, default=None) — single colour for all lines; overrides cmap
  - offset: float (optional, default=0) — vertical offset between curves as a fraction of the maximum peak height
  - norm: bool (optional, default=False) — normalise each curve to its maximum
  - stack_dim: str (optional, default='auto') — dimension to iterate over; 'auto' picks the smallest dimension
  - figsize: tuple (optional, default=(6,6)) — figure size
  - linewidth: float (optional, default=1) — line width
  - ax: matplotlib.axes.Axes (optional, default=None) — existing axes to plot on
- returns: None (renders a matplotlib figure)
- when_to_use: Comparing sets of EDCs or MDCs (e.g. temperature series, momentum series); visualising fit residuals; waterfall plots
- family: display
- strategy: line_plot
- cost: low

### plot_3d_stack

- description: Render a 3D stacked surface plot of a data cube, with one dimension stacked along the z-axis and the 2D slices shown as coloured surfaces. Useful for visualising dispersions at multiple energies or a kz map as stacked Fermi surfaces.
- function: xr.DataArray accessor .plot_3d_stack()
- parameters:
  - downsample: int (optional, default=2) — spatial downsampling factor for performance
  - stack_dim: str (optional, default='eV') — dimension to stack along the z-axis
  - vmax: float (optional, default=None) — colour scale maximum; defaults to data maximum
  - cmap: str (optional, default='Purples') — matplotlib colormap
  - figsize: tuple (optional, default=(8,12)) — figure size
  - aspect: tuple (optional, default=None) — 3D box aspect ratio (x, y, z) in data units, e.g. (1, 1, 200)
  - elev: float (optional, default=10.0) — elevation viewing angle in degrees
  - azim: float (optional, default=-60.0) — azimuthal viewing angle in degrees
- returns: None (renders a matplotlib 3D figure)
- when_to_use: Publication figures showing band structure evolution with energy or kz; visual overview of 3D datasets
- family: display
- strategy: 3d_surface_stack
- cost: low

### plot_fit_test

- description: Overlay a model evaluated at a given set of parameters on top of a 1D DataArray, optionally showing individual model components. Useful for checking initial parameter guesses before running a full fit.
- function: xr.DataArray accessor .plot_fit_test()
- parameters:
  - model: lmfit.Model (required) — model to evaluate
  - params: lmfit.Parameters (required) — parameters to evaluate the model with
  - show_components: bool (optional, default=True) — plot individual model components as dashed lines
- returns: None (renders a matplotlib figure)
- when_to_use: Checking and tuning initial parameter guesses before calling .fit(); debugging a model that is not converging
- family: display
- strategy: fit_preview
- cost: low

### plot_nanofocus

- description: Determine and plot the focal position of a nanoARPES focussing scan (i05 nano branch, Diamond Light Source) by finding the scan position at which a spatial feature is sharpest.
- function: xr.DataArray accessor .plot_nanofocus()
- parameters:
  - focus: str (optional, default='defocus') — dimension name representing the focussing direction in the data
- returns: None (renders a matplotlib figure with the estimated focal position)
- when_to_use: nanoARPES experiments at i05 (Diamond); determining the optimal focus position from a defocus scan before acquiring the main dataset
- family: display
- strategy: focus_analysis
- cost: low

### plot_grid

- description: Plot a list (or `xr.DataTree`) of 2D DataArrays arranged on a grid of subplots. Supports shared axes, per-plot titles, and per-plot or global colorbar limits.
- function: peaks.core.display.plotting:plot_grid
- parameters:
  - data: list or xr.DataTree (required) — 2D DataArrays to plot
  - ncols: int (optional, default=3) — number of columns; ignored if nrows is given
  - nrows: int (optional, default=None) — number of rows; overrides ncols
  - titles: list (optional, default=None) — subplot title strings, one per panel
  - sharex: bool (optional, default=False) — share x-axis across panels
  - sharey: bool (optional, default=False) — share y-axis across panels
  - figsize: tuple (optional, default=None) — figure size
  - vmin: float or list (optional, default=None) — colorbar minimum; list for per-panel control
  - vmax: float or list (optional, default=None) — colorbar maximum; list for per-panel control
  - cmap: str or list (optional, default=None) — colormap; list for per-panel control
- returns: None (renders a matplotlib figure)
- when_to_use: Comparing multiple dispersions or Fermi surface maps side by side; temperature/doping series overviews
- family: display
- strategy: grid_plot
- cost: low

### plot_ROI

- description: Overlay a polygon region of interest (ROI) on an existing matplotlib plot. The ROI is specified as a dict of coordinate lists matching the two axes of the plot.
- function: peaks.core.display.plotting:plot_ROI
- parameters:
  - ROI: dict (required) — polygon vertices as {'dim1': [x0,x1,...], 'dim2': [y0,y1,...]}, e.g. {'theta_par': [-8, -5, -3], 'eV': [95.4, 95.4, 95.8]}
  - color: str (optional, default='black') — line colour
  - x: str (optional, default=None) — which ROI key to place on the x-axis
  - y: str (optional, default=None) — which ROI key to place on the y-axis
  - label: str (optional, default=None) — legend label; adds a legend if provided
  - loc: str or int (optional, default='best') — legend location
  - ax: matplotlib.axes.Axes (optional, default=None) — axes to draw on
- returns: None (adds the ROI polygon to the current or specified axes)
- when_to_use: Annotating background or signal regions on a dispersion before normalisation or background subtraction; marking integration windows
- family: display
- strategy: roi_overlay
- cost: low

### plot_kpar_cut

- description: Plot the k_∥ trajectory of an analyser slit cut (kx vs ky) for a given set of goniometer angles and photon energy. Useful for planning or documenting Fermi surface mapping sequences.
- function: peaks.core.display.plotting:plot_kpar_cut
- parameters:
  - hv: float (optional, default=21.2) — photon energy in eV
  - Eb: float (optional, default=0) — binding energy in eV (positive = below EF)
  - theta_par_range: tuple (optional, default=(-15,15)) — (start, stop) range of slit angles in degrees
  - polar: float (optional, default=0) — polar angle in degrees
  - tilt: float (optional, default=0) — tilt angle in degrees
  - defl_perp: float (optional, default=0) — perpendicular deflector angle in degrees
  - ana_type: str (optional, default='I') — analyser geometry: 'I', 'II', 'Ip', or 'IIp'
  - ax: matplotlib.axes.Axes (optional, default=None) — axes to draw on
  - label_cut: bool (optional, default=True) — annotate the cut with the angle values
  - flip: bool (optional, default=False) — swap kx and ky axes
- returns: None (adds the cut trajectory to the current or specified axes)
- when_to_use: Designing a Fermi surface mapping sequence; checking k-space coverage before an experiment; annotating Fermi surface maps with measurement geometry
- family: display
- strategy: k_geometry_plot
- cost: low

### plot_kz_cut

- description: Plot the kz trajectory of an analyser slit cut (kx vs kz) as a function of slit angle for a given photon energy and inner potential. Useful for planning photon energy scans.
- function: peaks.core.display.plotting:plot_kz_cut
- parameters:
  - hv: float (optional, default=21.2) — photon energy in eV
  - Eb: float (optional, default=0) — binding energy in eV
  - polar_or_tilt: float (optional, default=0) — offset angle along the slit direction in degrees
  - theta_par_range: tuple (optional, default=(-15,15)) — (start, stop) slit angle range in degrees
  - V0: float (optional, default=12) — inner potential in eV
  - ax: matplotlib.axes.Axes (optional, default=None) — axes to draw on
  - label_cut: bool (optional, default=True) — annotate the cut with the photon energy
- returns: None (adds the kz cut trajectory to the current or specified axes)
- when_to_use: Planning a photon energy scan; verifying kz periodicity against lattice parameters; annotating kz-dispersion plots
- family: display
- strategy: k_geometry_plot
- cost: low

### radial_cuts

- description: Extract radial cuts of a 2D Fermi surface map (or 3D cube) as a function of azimuthal angle about a central point. Returns a 3D DataArray with dimensions (azimuthal_angle, radial_distance, eV).
- function: xr.DataArray accessor .radial_cuts()
- parameters:
  - num_azi: int (optional, default=361) — number of evenly spaced azimuthal angles between 0° and 360°
  - num_points: int (optional, default=200) — number of radial points per cut
  - radius: float (optional, default=2) — maximum radial distance in data axis units
  - \*\*centre_kwargs: float — centre of the radial fan, e.g. kx=0.1, ky=0.0
- returns: xr.DataArray — radial cuts stacked along an azimuthal_angle dimension
- when_to_use: Warping square Fermi surfaces to circular geometry; azimuthal analysis of Fermi surface maps
- addresses: coordinate_transform
- effects: extracts_radial_profile
- family: slicing
- strategy: radial_cut
- cost: medium

### extract_cut

- description: Extract an arbitrary straight-line cut through 2D data between two specified points in coordinate space.
- function: xr.DataArray accessor .extract_cut()
- parameters:
  - start_point: dict (required) — {'axis1_key': coord1_value, 'axis2_key': coord2_value} of the cut start in data coordinate values, e.g. {'kx': 0.5, 'ky': -0.3}
  - end_point: dict (required) — {'axis1_key': coord1_value, 'axis2_key': coord2_value} of the cut end in data coordinate values, e.g. {'kx': 0.2, 'ky': 0.3}
  - num_points: int (optional, default=None) — number of points along the cut; defaults to the data resolution
- returns: xr.DataArray — 1D cut with a path-length coordinate
- when_to_use: Extracting off-axis cuts in Fermi surface maps; kz–kx or kz–ky cuts through 3D data
- addresses: dimensionality_reduction
- effects: extracts_profile
- family: slicing
- strategy: arbitrary_cut
- cost: low

### mask_data

- description: Mask a region of interest (ROI) defined by a polygon path, returning the masked data and optionally its integrated intensity within the ROI.
- function: xr.DataArray accessor .mask_data()
- parameters:
  - ROI: dict (required) — vertices of the polygonal ROI in coordinate space, e.g. {'theta_par': [-8, -5.5, -3.1, -5.6], 'eV': [95.45, 95.45, 95.77, 95.77]}
  - return_integrated: bool (optional, default=True) — if True, take the mean over the ROI dimensions; if False, return the masked data
- returns: xr.DataArray (masked data or masked-then-averaged data if return_integrated is True)
- when_to_use: Selecting specific Brillouin zone regions for analysis; excluding spurious regions from fitting
- addresses: region_selection
- effects: selects_region
- family: slicing
- strategy: polygon_mask
- cost: low

---

## Quick-fit tools

`quick_fit` is a class-based accessor (`data.quick_fit.<method>(...)`) providing one-liner convenience wrappers around `.fit()` with automatic parameter guessing. All methods return the same `xr.Dataset` as `.fit()` (parameter values, uncertainties, `lmfit.ModelResult`). For 2D+ data, `independent_var` must be specified and the fit broadcasts over the remaining dimensions.

### quick_fit.linear

- description: Quick fit of a linear model (`slope * x + intercept`) to a DataArray, with automatic parameter guessing. Wraps `.fit()` with a `LinearModel`.
- function: xr.DataArray accessor .quick_fit.linear()
- parameters:
  - independent_var: str (optional, default=None) — dimension name for the independent variable; required for 2D+ data
  - \*\*kwargs — override initial parameter values, e.g. slope=1.0
- returns: xr.Dataset with best-fit parameters, uncertainties, and lmfit ModelResult
- when_to_use: Fitting a linear background or trend; quick sanity check on a 1D profile; fitting extracted EF positions vs angle or hv
- family: fitting
- strategy: linear
- cost: low

### quick_fit.poly

- description: Quick fit of a polynomial model of a given degree to a DataArray, with automatic parameter guessing. Wraps `.fit()` with a `PolynomialModel`.
- function: xr.DataArray accessor .quick_fit.poly()
- parameters:
  - degree: int (optional, default=3) — degree of the polynomial
  - independent_var: str (optional, default=None) — dimension name for the independent variable; required for 2D+ data
  - \*\*kwargs — override initial parameter values
- returns: xr.Dataset with best-fit parameters, uncertainties, and lmfit ModelResult
- when_to_use: Fitting a smooth curved background; fitting extracted EF vs angle or hv with a polynomial correction; any smoothly varying 1D trend
- family: fitting
- strategy: polynomial
- cost: low

### quick_fit.gaussian

- description: Quick fit of a Gaussian peak on a linear background (`GaussianModel + LinearModel`) to a DataArray, with automatic parameter guessing. Wraps `.fit()`.
- function: xr.DataArray accessor .quick_fit.gaussian()
- parameters:
  - independent_var: str (optional, default=None) — dimension name for the independent variable; required for 2D+ data
  - \*\*kwargs — override initial parameter values, e.g. center=0.0, sigma=0.05
- returns: xr.Dataset with best-fit parameters (center, sigma, amplitude, slope, intercept), uncertainties, and lmfit ModelResult
- when_to_use: Fitting a single MDC or EDC peak; extracting band positions and linewidths; quick peak characterisation before building a more complex multi-peak model
- family: fitting
- strategy: gaussian
- cost: low

### quick_fit.lorentzian

- description: Quick fit of a Lorentzian peak on a linear background (`LorentzianModel + LinearModel`) to a DataArray, with automatic parameter guessing. Wraps `.fit()`.
- function: xr.DataArray accessor .quick_fit.lorentzian()
- parameters:
  - independent_var: str (optional, default=None) — dimension name for the independent variable; required for 2D+ data
  - \*\*kwargs — override initial parameter values, e.g. center=0.0, sigma=0.05
- returns: xr.Dataset with best-fit parameters (center, sigma, amplitude, slope, intercept), uncertainties, and lmfit ModelResult
- when_to_use: Fitting a single MDC peak with a Lorentzian lineshape (expected for quasiparticle peaks); extracting self-energy from linewidths; preferred over Gaussian for intrinsic linewidth analysis
- family: fitting
- strategy: lorentzian
- cost: low

---

## Brillouin zone tools

Functions in `peaks.bz` for computing and plotting Brillouin zones and high-symmetry points. Require `ase` (Atomic Simulation Environment) and accept either an `ase.Atoms` object (crystal structure) or an `ase.lattice.BravaisLattice` object. **`ase` is an optional dependency** — install with `pip install ase`.

### plot_bz

- description: Plot the bulk or surface Brillouin zone of a crystal structure. Wrapper around `ase.cell.BravaisLattice.plot_bz` with added support for surface BZ selection, rotation, scaling, and repeat tiling.
- function: peaks.bz.plotting:plot_bz
- parameters:
  - structure_or_lattice: ase.Atoms or ase.lattice.BravaisLattice (required) — crystal structure or Bravais lattice
  - surface: tuple of int (optional, default=None) — Miller indices (h, k, l) of the surface; if given, plots the surface BZ instead of the bulk BZ
  - path: str (optional, default='') — high-symmetry k-path to overlay, e.g. 'GZR,MX'; use None for the default DFT path; '' for no path
  - special_points: dict (optional, default=None) — custom high-symmetry points as {'label': [kx, ky, kz]}; None uses standard points; {} hides all points
  - vectors: bool (optional, default=False) — show reciprocal lattice vectors
  - azim: float (optional, default=36) — azimuthal viewing angle for 3D plot in degrees
  - elev: float (optional, default=30) — elevation viewing angle for 3D plot in degrees
  - scale: float (optional, default=1) — scale factor applied to BZ coordinates (output is in Å⁻¹)
  - rotate: float or list of float (optional, default=0) — rotation in degrees: scalar rotates about z; 3-vector is a rotation vector
  - repeat: tuple of int (optional, default=(1,1,1)) — number of BZ repetitions along each reciprocal lattice direction
  - ax: matplotlib.axes.Axes or Axes3D (optional, default=None) — existing axes to plot on
  - show: bool (optional, default=False) — call plt.show() after plotting
  - show_axes: bool (optional, default=None) — show/hide axes; None keeps current state
- returns: None (renders a matplotlib 3D figure)
- when_to_use: Visualising the BZ before an experiment; annotating Fermi surface maps with BZ boundaries; planning k-path for photon energy scans
- family: structure
- strategy: bz_plot
- cost: low

### plot_bz_section

- description: Plot a 2D cross-section through the bulk Brillouin zone at an arbitrary plane (defined by origin and normal vector). Useful for overlaying BZ boundaries on measured Fermi surface maps.
- function: peaks.bz.plotting:plot_bz_section
- parameters:
  - structure_or_lattice: ase.Atoms or ase.lattice.BravaisLattice (required) — crystal structure or Bravais lattice
  - plane_origin: list or np.ndarray (optional, default=[0,0,0]) — a point on the cutting plane in Å⁻¹
  - plane_normal: list or np.ndarray (optional, default=[0,0,1]) — normal vector of the cutting plane
  - repeat: int (optional, default=1) — number of BZ repetitions to draw in each direction
  - ax: matplotlib.axes.Axes (optional, default=None) — existing axes to plot on (recommended: plot on top of a Fermi surface map)
  - show: bool (optional, default=False) — call plt.show() after plotting
  - show_axes: bool (optional, default=None) — show/hide axes
- returns: None (adds BZ boundary lines to the current or specified axes)
- when_to_use: Overlaying BZ boundaries on a measured Fermi surface map or kz section; annotating publication figures; identifying zone-folding features
- family: structure
- strategy: bz_section
- cost: low

### sym_points

- description: Return a pandas DataFrame of the high-symmetry k-points for the given crystal structure or lattice, in Å^-1. Optionally compute the corresponding angle-along-slit for a given photon energy, useful for locating high-symmetry points in raw angle-space data.
- function: peaks.bz.utils:sym_points
- parameters:
  - structure_or_lattice: ase.Atoms or ase.lattice.BravaisLattice (required) — crystal structure or Bravais lattice
  - surface: tuple, list, or np.ndarray (optional, default=None) — Miller indices (h, k, l) for a surface BZ; None gives bulk points
  - hv: float (optional, default=None) — photon energy in eV; if given, adds a column with the angle-along-slit for each k-point (assumes 4.5 eV work function)
- returns: pandas.DataFrame with columns k_x, k_y, k_z, |k|, and optionally angle-along-slit
- when_to_use: Identifying the positions of Γ, X, M, K etc. in Å^-1 for k-conversion validation; finding the expected slit angle for a high-symmetry point at a given photon energy before the measurement; annotating Fermi surface maps
- family: structure
- strategy: symmetry_points
- cost: low

---

## Machine learning tools

Functions in `peaks.ML` for unsupervised analysis of nanoARPES / µ-ARPES spatial mapping data. All accessor methods operate on 4D spatial-mapping DataArrays with dimensions `(x1, x2, eV, theta_par/k_par)`. Require `scikit-learn`. **`scikit-learn` is an optional dependency** — install with `pip install scikit-learn`.

### ML_pre_proc

- description: Convert a 4D spatial-mapping DataArray into a flat `pandas.DataFrame` where each row is a spatial position and each column is a spectral feature (full dispersion, MDC, or EDC). This is the required input format for all other ML tools.
- function: xr.DataArray accessor .ML_pre_proc()
- parameters:
  - extract: str (optional, default='dispersion') — what to use as features per spatial position: 'dispersion' (full 2D spectrum flattened), 'MDC', or 'EDC'
  - E: float (optional, default=0) — energy value for MDC extraction
  - dE: float (optional, default=0) — MDC integration window (total width, integrates ±dE/2)
  - k: float (optional, default=0) — momentum/angle value for EDC extraction
  - dk: float (optional, default=0) — EDC integration window (total width, integrates ±dk/2)
  - scale: bool (optional, default=False) — apply StandardScaler (zero-mean, unit-variance) to features
  - norm: bool (optional, default=False) — normalise the spectrum at each spatial position before feature extraction
- returns: pandas.DataFrame — shape (n_spatial_positions, n_features)
- when_to_use: Always call first before `perform_k_means`, `clusters`, `clusters_explore`, or `PCA_explore`; also useful for custom scikit-learn pipelines on spatial mapping data
- family: ml
- strategy: feature_extraction
- cost: medium

### SM_PCA

- description: Denoise a 4D spatial-mapping DataArray using PCA: flatten the data, reduce to N principal components, then reconstruct. Noise is suppressed because it is distributed across many low-variance components that are discarded. Lower `PCs` gives stronger denoising but may oversimplify the data.
- function: xr.DataArray accessor .SM_PCA()
- parameters:
  - PCs: int (optional, default=10) — number of principal components to retain for reconstruction
- returns: xr.DataArray — reconstructed spatial map with the same dimensions and coordinates as the input
- when_to_use: Noisy nanoARPES spatial maps; before clustering or visualisation to improve SNR; use `PCA_explore` first to choose an appropriate number of PCs
- addresses: noise, large_dataset
- effects: reduces_noise
- family: ml
- strategy: pca_denoise
- cost: medium

### PCA_explore

- description: Scan over a range of numbers of principal components and plot the cumulative explained variance fraction vs number of PCs, with a threshold line. Helps determine the minimum number of PCs needed to capture a given fraction of the variance in the spatial map before denoising or clustering.
- function: xr.DataArray accessor .PCA_explore()
- parameters:
  - PCs_range: range (optional, default=range(1,6)) — range of PC counts to test
  - threshold: float (optional, default=0.95) — target explained variance fraction; a vertical line is drawn at the minimum PC count that reaches this threshold
  - extract: str (optional, default='dispersion') — feature type: 'dispersion', 'MDC', or 'EDC'
  - E: float (optional, default=0) — energy for MDC extraction
  - dE: float (optional, default=0) — MDC integration window
  - k: float (optional, default=0) — momentum for EDC extraction
  - dk: float (optional, default=0) — EDC integration window
  - scale: bool (optional, default=False) — apply StandardScaler before PCA
  - norm: bool (optional, default=False) — normalise each spectrum before feature extraction
- returns: None (renders a matplotlib figure)
- when_to_use: Before `SM_PCA` to choose the number of PCs; before `clusters` or `clusters_explore` to decide whether to use PCA pre-processing and how many components to retain
- family: ml
- strategy: pca_explore
- cost: medium

### clusters_explore

- description: Run k-means clustering over a range of cluster counts on a spatial map and plot both the elbow curve (inertia vs k) and a grid of spatial cluster-label maps. Automatically suggests the optimal k using the elbow heuristic. Optionally applies PCA dimensionality reduction before clustering.
- function: xr.DataArray accessor .clusters_explore()
- parameters:
  - cluster_range: range (optional, default=range(1,7)) — range of cluster counts to test
  - n_init: int or str (optional, default='auto') — number of k-means restarts per k; 'auto' uses scikit-learn's default
  - use_PCA: bool (optional, default=True) — apply PCA before clustering
  - PCs: int (optional, default=3) — number of principal components for PCA pre-processing
  - extract: str (optional, default='dispersion') — feature type: 'dispersion', 'MDC', or 'EDC'
  - E: float (optional, default=0) — energy for MDC extraction
  - dE: float (optional, default=0) — MDC integration window
  - k: float (optional, default=0) — momentum for EDC extraction
  - dk: float (optional, default=0) — EDC integration window
  - scale: bool (optional, default=False) — apply StandardScaler before clustering
  - norm: bool (optional, default=False) — normalise each spectrum before feature extraction
- returns: None (renders matplotlib figures)
- when_to_use: First step of a clustering workflow to identify the optimal number of clusters; before calling `clusters` with a chosen k
- family: ml
- strategy: kmeans_explore
- cost: high

### clusters

- description: Perform k-means clustering on a spatial map for a specified number of clusters, plot the spatial cluster-label map alongside the integrated spectral weight map, and plot the average dispersion for each cluster. Optionally applies PCA before clustering and plots the PCA-reconstructed cluster centres.
- function: xr.DataArray accessor .clusters()
- parameters:
  - num_clusters: int (optional, default=3) — number of clusters
  - n_init: int or str (optional, default='auto') — number of k-means restarts
  - use_PCA: bool (optional, default=True) — apply PCA before clustering
  - PCs: int (optional, default=3) — number of principal components for PCA pre-processing
  - extract: str (optional, default='dispersion') — feature type: 'dispersion', 'MDC', or 'EDC'
  - E: float (optional, default=0) — energy for MDC extraction
  - dE: float (optional, default=0) — MDC integration window
  - k: float (optional, default=0) — momentum for EDC extraction
  - dk: float (optional, default=0) — EDC integration window
  - scale: bool (optional, default=False) — apply StandardScaler before clustering
  - norm: bool (optional, default=False) — normalise each spectrum before feature extraction
  - robust: bool (optional, default=False) — use robust colour scaling in dispersion plots
  - vmin: float (optional, default=None) — colour scale minimum for dispersion plots
  - vmax: float (optional, default=None) — colour scale maximum for dispersion plots
- returns: tuple of (xr.DataArray, list) — (spatial cluster-label map with dims (x1, x2), list of average dispersions per cluster)
- when_to_use: Identifying spatially distinct electronic phases or terminations in nanoARPES; separating contributions from different domains; use after `clusters_explore` to confirm the cluster count
- family: ml
- strategy: kmeans
- cost: high

---

## Time-resolved ARPES tools

Functions in `peaks.time_resolved` for pump-probe / TR-ARPES data. All tools are accessed via the `.tr` accessor on `xr.DataArray` or `xr.DataTree`. The time axis must have dimension `"t"` with units attached as `pint` quantities (or plain floats assumed to be in the axis units).

### tr.mean

- description: Integrate the data over the entire time axis to produce a time-averaged ARPES spectrum.
- function: xr.DataArray accessor .tr.mean()
- returns: xr.DataArray with the `t` dimension removed
- when_to_use: Quick overview of the time-averaged spectrum; checking data quality before time-resolved analysis
- family: time_resolved
- strategy: average
- cost: low

### tr.static

- description: Calculate the equilibrium (static) spectrum by averaging all time points recorded before a specified time zero `t_static`. Defaults to averaging everything before −250 fs.
- function: xr.DataArray accessor .tr.static()
- parameters:
  - t_static: pint.Quantity or float (optional, default=-250 fs) — upper limit of the static time window; if a float is given, the axis units are assumed
- returns: xr.DataArray — static spectrum with `t` dimension removed
- when_to_use: Obtaining the equilibrium reference spectrum for difference calculations; when `t` axis contains pre-t0 data
- family: time_resolved
- strategy: static
- cost: low

### tr.diff

- description: Compute the pump-induced difference spectrum: I(t_select) − I_static, where I_static is the average over all pre-t0 time points.
- function: xr.DataArray accessor .tr.diff()
- parameters:
  - t_select: pint.Quantity, float, slice, or None (optional, default=None) — time point or window to extract the excited spectrum from; if a slice, the mean over that window is used; if None, the difference of the full data cube is returned
  - t_static: pint.Quantity or float (optional, default=-250 fs) — static reference cutoff (same as `tr.static`)
- returns: xr.DataArray — difference spectrum
- when_to_use: Visualising pump-induced spectral changes; extracting dynamics at a specific time delay
- family: time_resolved
- strategy: difference
- cost: low

### tr.set_t0

- description: Recalibrate the time axis by setting a new time zero in-place. Updates the `t` coordinate and the `pump.t0_position` metadata attribute. Use `tr.assign_t0` for a non-mutating version.
- function: xr.DataArray accessor .tr.set_t0()
- parameters:
  - t0: pint.Quantity or float (required) — new time zero expressed in the current time axis units
  - delay_line_roundtrips: int (optional, default=2) — number of round trips in the delay line (used to convert time to stage position)
- returns: None (modifies DataArray in place)
- when_to_use: After identifying t0 from a cross-correlation or other measurement
- family: time_resolved
- strategy: t0_calibration
- cost: low

### tr.assign_t0

- description: Non-mutating version of `tr.set_t0` — returns a new DataArray with the recalibrated time axis.
- function: xr.DataArray accessor .tr.assign_t0()
- parameters:
  - t0: pint.Quantity or float (required) — new time zero
  - delay_line_roundtrips: int (optional, default=2) — delay line round trips
- returns: xr.DataArray with updated `t` coordinate
- when_to_use: When you want to preserve the original data; chaining operations
- family: time_resolved
- strategy: t0_calibration
- cost: low

### tr.set_t0_like

- description: Set the time zero of the current dataset to match the calibrated t0 from a reference dataset. Reads `pump.t0_position` metadata from the reference and applies it in-place.
- function: xr.DataArray accessor .tr.set_t0_like()
- parameters:
  - da_ref: xr.DataArray (required) — reference dataset with a calibrated t0
- returns: None (modifies DataArray in place)
- when_to_use: When multiple scans share the same t0 reference; propagating a calibration from one scan to others
- family: time_resolved
- strategy: t0_calibration
- cost: low

### tr.assign_t0_like

- description: Non-mutating version of `tr.set_t0_like` — returns a new DataArray with the t0 copied from a reference dataset.
- function: xr.DataArray accessor .tr.assign_t0_like()
- parameters:
  - da_ref: xr.DataArray (required) — reference dataset with a calibrated t0
- returns: xr.DataArray with updated `t` coordinate
- when_to_use: When you want to preserve the original data while applying a reference t0
- family: time_resolved
- strategy: t0_calibration
- cost: low

---

## XPS tools

`peaks.xps.CoreLevels` is a standalone class providing core level binding energy and cross-section databases, along with lookup and visualisation tools. It does not operate as an xarray accessor — call class methods directly.

### CoreLevels.by_element

- description: Return a formatted DataFrame of core level binding energies (and optionally kinetic energies at a given photon energy) for one or more elements.
- function: peaks.xps.core_levels:CoreLevels.by_element
- parameters:
  - elements: str or list of str (required) — element symbol(s), e.g. `'Fe'` or `['Fe', 'O']`
  - hv: float (optional) — photon energy in eV; if provided, kinetic energies are also shown
  - max_order: int (optional, default=1) — highest harmonic order to include in the kinetic energy table
- returns: pandas.DataFrame — rows are elements/energies, columns are orbital labels
- when_to_use: Identifying which core levels are accessible at a given photon energy; planning XPS experiments
- family: xps
- strategy: database_lookup
- cost: low

### CoreLevels.by_energy

- description: Search the core level database for all core levels whose binding (or kinetic) energy falls within a tolerance of a specified value. Useful for identifying unknown peaks.
- function: peaks.xps.core_levels:CoreLevels.by_energy
- parameters:
  - energy: float (required) — energy in eV to search around
  - hv: float (optional) — if provided, search is performed on kinetic energy from this photon energy rather than binding energy
  - max_order: int (optional, default=1) — highest harmonic order to include
  - tol: float (optional, default=3) — search window ± tolerance in eV
- returns: pandas.DataFrame — filtered table of matching core levels and elements
- when_to_use: Assigning unknown peaks in an XPS spectrum; checking for contamination or unexpected elements
- family: xps
- strategy: database_lookup
- cost: low

### CoreLevels.plot

- description: Plot vertical marker lines at core level binding or kinetic energies for specified elements on a matplotlib axis. Solid lines for 1st-order, dashed for higher harmonics.
- function: peaks.xps.core_levels:CoreLevels.plot
- parameters:
  - elements: str or list of str (required) — element symbol(s)
  - eV: slice (optional) — energy range to display
  - hv: float (optional) — photon energy; if provided, markers are placed at kinetic energies
  - max_order: int (optional, default=1) — highest harmonic order to show
  - ax: matplotlib.axes.Axes (optional) — axes to plot on; creates a new figure if not provided
  - show_binding_as_negative: bool (optional, default=True) — display binding energies as negative (E−E_F convention)
  - \*\*kwargs — passed to `matplotlib.pyplot.axvline`
- returns: None (renders on the provided or new axes)
- when_to_use: Overlaying reference lines on a measured XPS spectrum; publication figures
- family: xps
- strategy: visualization
- cost: low

### CoreLevels.ptab

- description: Launch an interactive Bokeh periodic table in a Jupyter notebook showing core level binding (or kinetic) energies on hover. Requires Bokeh and a Jupyter environment.
- function: peaks.xps.core_levels:CoreLevels.ptab
- parameters:
  - hv: float (optional) — photon energy in eV; if not provided, binding energies are shown; if provided, kinetic energies are shown
- returns: None (renders an interactive Bokeh figure in the notebook)
- when_to_use: Exploring which elements and core levels are accessible at a given photon energy; experiment planning
- family: xps
- strategy: interactive_visualization
- cost: low

### CoreLevels.BE

- description: Class attribute — nested dictionary of all core level binding energies by element and orbital label. Data from Bearden & Burr (1967), Cardona & Ley (1978), and Fuggle & Mårtensson (1980).
- function: peaks.xps.core_levels:CoreLevels.BE
- returns: dict — `{element: {orbital: binding_energy_in_eV}}`
- when_to_use: Direct programmatic access to the binding energy database
- family: xps
- strategy: database_lookup
- cost: low

### CoreLevels.xc

- description: Class attribute — `xr.DataTree` of photoionisation cross sections as a function of photon energy for all available core levels, organised as `element/orbital` nodes. Data from Elettra Synchrotron.
- function: peaks.xps.core_levels:CoreLevels.xc
- returns: xr.DataTree — nodes are `element/orbital`, each leaf contains a DataArray `data` over the `eV` dimension
- when_to_use: Estimating relative peak intensities at a given photon energy; optimising photon energy for an experiment
- family: xps
- strategy: database_lookup
- cost: low

---

## Metadata accessor

The `.metadata` accessor is available on `xr.DataArray`, `xr.Dataset`, and `xr.DataTree`. It provides structured access to instrument and scan metadata (beamline, manipulator, calibration, etc.) stored in `attrs`, and exposes methods for common metadata editing tasks.

### metadata (display)

- description: Display all structured metadata stored on a DataArray as a colour-formatted nested dictionary. Call `repr(da.metadata)` or `print(da.metadata)` to see it. Navigate with attribute access, e.g. `da.metadata.scan.photon_energy`.
- function: xr.DataArray accessor .metadata
- returns: formatted string (ANSI for terminal, HTML for Jupyter)
- when_to_use: Inspecting what instrument and beamline parameters are attached to a loaded scan

### metadata.set_normal_emission

- description: Set the normal-emission reference angles for the current scan from keyword arguments matching the manipulator axes (e.g. `theta_par`, `polar`). Modifies the `manipulator` metadata section in-place. Required before k-conversion if the sample is not already at normal emission.
- function: xr.DataArray accessor .metadata.set_normal_emission()
- parameters:
  - \*\*kwargs — axis names and their normal-emission values (e.g. `theta_par=12.0, polar=5.0`) or a `norm_values` dict
- returns: None (modifies metadata in place)
- when_to_use: Before `k_convert` when the sample was tilted during the scan; correcting for sample misalignment
- family: calibration
- strategy: normal_emission

### metadata.set_normal_emission_like

- description: Copy the normal-emission reference angles from another DataArray and apply them to the current scan.
- function: xr.DataArray accessor .metadata.set_normal_emission_like()
- parameters:
  - da: xr.DataArray (required) — source DataArray whose normal emission angles to copy
- returns: None (modifies metadata in place)
- when_to_use: When multiple scans share the same sample orientation and one has already been calibrated
- family: calibration
- strategy: normal_emission

### metadata.set_EF_correction

- description: Store a Fermi level correction in the DataArray's calibration metadata. Accepts a constant shift (float/int), a polynomial coefficient dictionary `{'c0': ..., 'c1': ...}` as returned by `fit_gold`, or the `xr.Dataset` output of `fit_gold` directly.
- function: xr.DataArray accessor .metadata.set_EF_correction()
- parameters:
  - EF_correction: float, int, dict, or xr.Dataset (required) — Fermi level correction to store
- returns: None (modifies metadata in place)
- when_to_use: After running `fit_gold`; storing the correction before applying it with `k_convert` or during energy axis calibration

### metadata.set_EF_correction_like

- description: Copy the Fermi level correction from another DataArray and store it in the current scan's calibration metadata.
- function: xr.DataArray accessor .metadata.set_EF_correction_like()
- parameters:
  - da_to_set_like: xr.DataArray (required) — source DataArray with an existing EF correction
- returns: None (modifies metadata in place)
- when_to_use: Propagating a gold fit to a series of sample scans recorded under the same conditions

### metadata.get_EF_correction

- description: Retrieve the stored Fermi level correction from the DataArray's calibration metadata.
- function: xr.DataArray accessor .metadata.get_EF_correction()
- returns: float, int, or dict — the stored EF correction
- when_to_use: Checking or programmatically using the stored correction value

---

## DataTree management tools

Tools for organising multiple ARPES scans in an `xr.DataTree` hierarchy. Accessed via the `.` accessor directly on a DataTree (registered as direct DataTree accessors).

### dt.view

- description: Print a human-readable tree diagram of all branches and leaf nodes in the DataTree, showing scan names and structure.
- function: xr.DataTree accessor .view()
- returns: None (prints to stdout)
- when_to_use: Inspecting the structure of a loaded multi-scan DataTree; finding scan paths before data access
- cost: low

### dt.add

- description: Add data into an existing DataTree — either from a file path/identifier (same arguments as `peaks.load`) or from an existing DataArray, Dataset, or DataTree.
- function: xr.DataTree accessor .add()
- parameters:
  - data_source: str, int, xr.DataArray, xr.Dataset, or xr.DataTree (required) — data to add; strings and ints are passed to `peaks.load`
  - name: str (optional) — name for the new leaf/branch; auto-generated if not provided
  - add_at_root: bool (optional, default=False) — add directly under the root node rather than creating a new branch
  - \*\*kwargs — additional keyword arguments forwarded to `peaks.load` when `data_source` is a file identifier
- returns: xr.DataTree (modified in place, also returned)
- when_to_use: Building up a multi-scan DataTree interactively; adding a new measurement to an existing collection
- cost: low

### dt.add_scan_group

- description: Add a new empty group (branch) node to the DataTree, useful for organising scans into labelled categories.
- function: xr.DataTree accessor .add_scan_group()
- parameters:
  - name: str (optional) — name of the new group; defaults to `scan_group_N` where N makes the name unique
- returns: xr.DataTree (modified in place, also returned)
- when_to_use: Creating organisational structure before adding scans; grouping by sample, temperature, or photon energy
- cost: low

### dt.get_DataArray

- description: Extract the single `xr.DataArray` stored at a DataTree leaf node (the `data` variable of the leaf Dataset).
- function: xr.DataTree accessor .get_DataArray()
- returns: xr.DataArray
- when_to_use: Retrieving the data at a specific node for individual analysis; transitioning from tree-level to array-level operations
- cost: low

### dt.sum_data

- description: Sum all DataArrays in the DataTree along a new axis, returning a single DataArray. Useful for co-adding equivalent scans.
- function: xr.DataTree accessor .sum_data()
- returns: xr.DataArray
- when_to_use: Co-adding repeated measurements for improved statistics
- cost: medium

### dt.subtract_data

- description: Subtract one DataArray in the DataTree from another, returning the difference.
- function: xr.DataTree accessor .subtract_data()
- returns: xr.DataArray
- when_to_use: Computing background-subtracted or reference-subtracted spectra when data is stored in a DataTree
- cost: low

### dt.merge_data

- description: Concatenate or merge DataArrays across the DataTree into a single DataArray, combining scans recorded at different conditions (e.g. photon energies, temperatures) into a higher-dimensional array.
- function: xr.DataTree accessor .merge_data()
- returns: xr.DataArray
- when_to_use: Building kz maps from a series of hv scans; combining temperature-dependent measurements
- cost: medium
