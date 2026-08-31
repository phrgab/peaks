"""Functions to load data from the Bloch beamline at MAX IV Laboratory."""

from dataclasses import dataclass

import dask.array
import h5py
import numpy as np
import pint
import pint_xarray
import xarray as xr

from peaks.core.fileIO.base_arpes_data_classes.base_arpes_data_class import (
    BaseARPESDataLoader,
)
from peaks.core.fileIO.base_arpes_data_classes.base_ses_class import SESDataLoader
from peaks.core.fileIO.base_data_classes.base_hdf5_class import BaseHDF5DataLoader
from peaks.core.fileIO.loc_registry import register_loader
from peaks.core.options import opts
from peaks.core.utils.misc import analysis_warning

ureg = pint_xarray.unit_registry


@register_loader
class BlochArpesLoader(SESDataLoader):
    """SES-based loader for the Bloch ARPES branch at MAX IV."""

    _loc_name = "MAXIV_Bloch_A"
    _loc_description = "A branch (ARPES) of Bloch beamline at Max-IV"
    _loc_url = "https://www.maxiv.lu.se/beamlines-accelerators/beamlines/bloch/"
    _analyser_slit_angle = 0 * ureg("deg")

    _manipulator_name_conventions = {
        "polar": "P",
        "tilt": "T",
        "azi": "A",
        "x1": "X",
        "x2": "Y",
        "x3": "Z",
    }
    _manipulator_sign_conventions = {
        "polar": -1,
        "tilt": -1,
    }
    _analyser_sign_conventions = {
        "deflector_perp": -1,
    }
    _SES_metadata_units = {
        f"manipulator_{dim}": ("mm" if dim in ["x1", "x2", "x3"] else "deg")
        for dim in _manipulator_name_conventions.keys()
    }


# -------------------- Alpha version nexus data loader --------------------
def _read_scalar(group, name, default=None):
    """Read a scalar dataset from an open h5py group/file, decoding bytes."""
    if name not in group:
        return default
    value = group[name][()]
    if isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(-1)[0]
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode(errors="replace")
    return value


def _nxs_unit(dataset, default=""):
    """Read the units attribute of a dataset in a pint-friendly form."""
    if dataset is None:
        return default
    unit = dataset.attrs.get("units", default) or default
    if isinstance(unit, (bytes, np.bytes_)):
        unit = unit.decode(errors="replace")
    unit = str(unit).strip()
    return "" if unit.lower() in {"", "1", "a.u.", "au", "arb", "arb."} else unit


def _nxs_vector(group, name):
    """Read a per-frame dataset as a flat float vector, or None.

    Per-frame readbacks are stored as (N,) or (N, 1), so the read is always
    flattened before use.
    """
    if name not in group:
        return None
    try:
        return np.asarray(group[name][()], dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None


def _get_core_axes(nxdata, shape):
    """Energy (x) vs angle (y)."""

    def one_axis(name, length):
        ds = nxdata.get(name)
        values = (
            np.asarray(ds[()], dtype=np.float64).reshape(-1) if ds is not None else None
        )
        if values is None or values.size != length:
            values = np.arange(length, dtype=np.float64)  # fall back to pixel index
        unit = _nxs_unit(ds)
        if unit == "eV":
            dim = "eV"
        elif unit.lower() == "deg":
            dim = "theta_par"
        else:
            dim = f"{name}_scale"
        return dim, unit, values

    return one_axis("y", shape[1]), one_axis("x", shape[2])


def _manipulator_device_to_dim(device, fallback):
    """Best-effort mapping from Tango device/alias names to peaks manipulator axes.

    Covers both the app aliases (manip_x) and the raw Tango device names
    (B110A-EA01/DIA/MPB-01-X), which end in the axis letter.
    """
    compact = str(device or "").strip().lower()
    if not compact:
        return fallback
    for old in (" ", "-", "/", ".", ":"):
        compact = compact.replace(old, "_")
    if compact in {"x", "x1", "sample_x", "manip_x"} or compact.endswith("_x"):
        return "x1"
    if compact in {"y", "x2", "sample_y", "manip_y"} or compact.endswith("_y"):
        return "x2"
    if compact in {"z", "x3", "sample_z", "manip_z"} or compact.endswith("_z"):
        return "x3"
    if "polar" in compact or compact.endswith("_p"):
        return "polar"
    if "tilt" in compact or compact.endswith("_t"):
        return "tilt"
    if "azi" in compact or compact.endswith("_a"):
        return "azi"
    return fallback


def _manipulator_dim_unit(dim, dataset=None):
    """Unit for a manipulator scan axis, from the file if possible."""
    unit = _nxs_unit(dataset) if dataset is not None else ""
    if unit:
        return unit
    if dim in {"x1", "x2", "x3"}:
        return "mm"
    if dim in {"polar", "tilt", "azi"}:
        return "deg"
    return ""


def _scan_axis(nxdata, nframes):
    """Choose the frame dimension for a stack, if a useful scan axis exists.

    A manipulator scan is deliberately *not* offered here: its per-frame
    position repeats once per raster line, and a repeated dimension coordinate
    breaks ``_apply_conventions`` (which reindexes any descending axis). Spatial
    maps are folded by :func:`_map_geometry` instead; only a scan that failed to
    fold reaches the manipulator candidate below.
    """
    candidates = []
    if "PhotonEnergy" in nxdata:
        candidates.append(("PhotonEnergy", "hv", "eV"))
    if "DeflectorAxisValue" in nxdata:
        scan_axis = str(_read_scalar(nxdata, "DeflectorScanAxis", "")).upper()
        dim = "deflector_perp" if scan_axis.startswith("Y") else "deflector_parallel"
        candidates.append(("DeflectorAxisValue", dim, "deg"))
    if "ManipulatorFastAxis" in nxdata:
        dim = _manipulator_device_to_dim(
            _read_scalar(nxdata, "FastAxisDevice", ""), "manipulator_fast_axis"
        )
        candidates.append(
            (
                "ManipulatorFastAxis",
                dim,
                _manipulator_dim_unit(dim, nxdata.get("ManipulatorFastAxis")),
            )
        )
    if "ScanIndex" in nxdata:  # Fallback for point series
        candidates.append(("ScanIndex", "scan_no", ""))

    for dataset_name, dim_name, unit in candidates:
        values = _nxs_vector(nxdata, dataset_name)
        if values is None or values.size != nframes:
            continue
        # A dimension coordinate must be usable as an index; duplicates make
        # xarray's reindex raise
        if np.unique(values).size != values.size:
            continue
        return dim_name, _nxs_unit(nxdata.get(dataset_name), unit) or unit, values
    return None


# For spatial map foling; from Jacek Osiecki at Bloch, Max IV
@dataclass
class _MapGeometry:
    """A spatial map folded back onto the raster it was acquired on.

    The console flattens every acquisition to a frame stack; for a spatial map
    that discards the geometry entirely.
    """

    index: np.ndarray
    fast_dim: str
    slow_dim: str
    fast_values: np.ndarray
    slow_values: np.ndarray
    fast_unit: str
    slow_unit: str


def _collapse_positions(index, values, axis):
    """Reduce per-frame positions onto one raster axis by averaging the other.

    Fast-fly positions are continuous encoder readbacks, so no two lines land on
    exactly the same coordinate; the mean across lines is the best estimate of
    the shared axis. Unmeasured cells are interpolated rather than left NaN so
    the coordinate stays usable as an index.
    """
    if values is None:
        return None
    gathered = np.where(index >= 0, values[np.where(index >= 0, index, 0)], np.nan)
    with np.errstate(invalid="ignore"):
        collapsed = np.nanmean(gathered, axis=axis)
    if not np.any(np.isfinite(collapsed)):
        return None
    missing = ~np.isfinite(collapsed)
    if np.any(missing):
        positions = np.arange(collapsed.size)
        collapsed[missing] = np.interp(
            positions[missing], positions[~missing], collapsed[~missing]
        )
    return collapsed


def _map_axis_dims(nxdata):
    """Peaks dimension names and units for a map's fast and slow axes."""
    fast_dim = _manipulator_device_to_dim(
        _read_scalar(nxdata, "FastAxisDevice", ""), "x1"
    )
    slow_dim = _manipulator_device_to_dim(
        _read_scalar(nxdata, "SlowAxisDevice", ""), "x2"
    )
    if fast_dim == slow_dim:  # Unrecognised device names must not collide
        fast_dim, slow_dim = "x1", "x2"
    return (
        fast_dim,
        slow_dim,
        _manipulator_dim_unit(fast_dim, nxdata.get("ManipulatorFastAxis")),
        _manipulator_dim_unit(slow_dim, nxdata.get("ManipulatorSlowAxis")),
    )


def _stepped_map_geometry(nxdata, nframes):
    """Fold a stepped manipulator scan using its recorded row/column indices."""
    rows = _nxs_vector(nxdata, "GridRow")
    cols = _nxs_vector(nxdata, "GridColumn")
    if rows is None or cols is None or rows.size != nframes or cols.size != nframes:
        return None
    if not (np.all(np.isfinite(rows)) and np.all(np.isfinite(cols))):
        return None
    nrows, ncols = int(np.nanmax(rows)) + 1, int(np.nanmax(cols)) + 1
    if nrows < 2 or ncols < 2 or nrows * ncols < nframes:
        return None

    index = np.full((nrows, ncols), -1, dtype=np.int64)
    index[rows.astype(int), cols.astype(int)] = np.arange(nframes)

    fast_dim, slow_dim, fast_unit, slow_unit = _map_axis_dims(nxdata)
    fast = _collapse_positions(index, _nxs_vector(nxdata, "ManipulatorFastAxis"), axis=0)
    slow = _collapse_positions(index, _nxs_vector(nxdata, "ManipulatorSlowAxis"), axis=1)
    return _MapGeometry(
        index,
        fast_dim,
        slow_dim,
        fast if fast is not None else np.arange(ncols, dtype=np.float64),
        slow if slow is not None else np.arange(nrows, dtype=np.float64),
        fast_unit,
        slow_unit,
    )


def _looks_like_snake(index, fast):
    """Detect a snake raster the writer did not flag.

    Alternate lines sweeping in opposite directions make a snake regardless of
    what the recipe recorded, and folding one as if it were unidirectional
    mirrors every other line of the map.
    """
    directions = []
    for row in index[:4]:
        members = row[row >= 0]
        if members.size < 2:
            continue
        span = fast[members[-1]] - fast[members[0]]
        if np.isfinite(span) and span != 0:
            directions.append(np.sign(span))
    return len(directions) >= 2 and bool(np.any(np.diff(directions) != 0))


def _fastfly_map_geometry(nxdata, nframes):
    """Fold a fast-fly manipulator scan using its per-frame line index.

    Fast-fly scans move continuously along the fast axis and stamp each frame
    with the line it belongs to; there is no GridRow/GridColumn. Frames in
    acquisition order within a line are that line's columns.
    """
    lines = _nxs_vector(nxdata, "LineIndex")
    fast = _nxs_vector(nxdata, "ManipulatorFastAxis")
    if lines is None or fast is None or lines.size != nframes or fast.size != nframes:
        return None
    if not np.all(np.isfinite(lines)):
        return None

    line_ids = lines.astype(np.int64)
    unique_lines = np.unique(line_ids)
    nrows = int(unique_lines.size)
    if nrows < 2:
        return None
    ncols = int(max(np.count_nonzero(line_ids == line) for line in unique_lines))
    if ncols < 2:
        return None

    index = np.full((nrows, ncols), -1, dtype=np.int64)
    for row, line in enumerate(unique_lines):
        members = np.flatnonzero(line_ids == line)
        index[row, : members.size] = members

    snake = bool(_read_scalar(nxdata, "snakeScanMode", False)) or bool(
        _read_scalar(nxdata, "SnakeScanMode", False)
    )
    if snake or _looks_like_snake(index, fast):
        for row in range(1, nrows, 2):
            members = index[row][index[row] >= 0]
            index[row, : members.size] = members[::-1]

    fast_dim, slow_dim, fast_unit, slow_unit = _map_axis_dims(nxdata)
    fast_values = _collapse_positions(index, fast, axis=0)
    slow_values = _collapse_positions(
        index, _nxs_vector(nxdata, "ManipulatorSlowAxis"), axis=1
    )
    return _MapGeometry(
        index,
        fast_dim,
        slow_dim,
        fast_values if fast_values is not None else np.arange(ncols, dtype=np.float64),
        slow_values if slow_values is not None else np.arange(nrows, dtype=np.float64),
        fast_unit,
        slow_unit,
    )


def _map_geometry(nxdata, nframes):
    """Return the raster geometry of a manipulator scan, or None if not one."""
    if nframes < 4:
        return None
    for builder in (_stepped_map_geometry, _fastfly_map_geometry):
        geometry = builder(nxdata, nframes)
        if geometry is not None:
            return geometry
    return None


def _fold_frames(frames, index):
    """Fold ``[frame, y, x]`` into ``[slow, fast, y, x]``.

    Cells with no frame -- a partial line from an interrupted scan -- become NaN
    rather than zero. Zero is a measurement claim: it says the detector looked
    and saw nothing. NaN says nobody looked.
    """
    folded = np.full(tuple(index.shape) + frames.shape[1:], np.nan, dtype=np.float32)
    measured = index >= 0
    folded[measured] = frames[index[measured]]
    return folded


def _valid_frame_mask(nxdata, nframes):
    """Boolean per-frame validity mask, or None if every frame is valid.

    A partial scan keeps placeholder frames so the file stays rectangular, and
    flags them with ``frame_valid = 0``. They hold zeros; leaving them turns
    "never measured" into "measured zero counts", which biases any integration
    over the scan axis without ever looking wrong.
    """
    valid = _nxs_vector(nxdata, "frame_valid")
    if valid is None or valid.size != nframes:
        return None
    mask = valid >= 0.5
    return None if mask.all() else mask


def _mask_invalid_frames(spectrum, valid):
    """NaN-out invalid frames; works on numpy and dask arrays alike."""
    if valid is None:
        return spectrum
    mask = valid.reshape((-1,) + (1,) * (spectrum.ndim - 1))
    return np.where(mask, spectrum, np.float32(np.nan))


@register_loader
class BlochNexusLoader(BaseHDF5DataLoader, BaseARPESDataLoader):
    """Loader for MAX IV Bloch A-branch (Scienta) PEAK NeXus files."""

    _loc_name = "MAXIV_Bloch_A_New"
    _loc_description = "A branch (ARPES) of Bloch beamline at Max-IV in (Scienta) PEAK written NeXus format"
    _loc_url = "https://www.maxiv.lu.se/beamlines-accelerators/beamlines/bloch/"
    _analyser_slit_angle = 0 * ureg("deg")

    _manipulator_name_conventions = {
        "polar": "polar",
        "tilt": "tilt",
        "azi": "azimuth",
        "x1": "x",
        "x2": "y",
        "x3": "z",
    }
    # [SHU] CHEEEEEEECK!!
    _manipulator_sign_conventions = {
        "polar": -1,
        "tilt": -1,
    }
    _analyser_sign_conventions = {
        "deflector_perp": -1,
    }
    _analyser_name_conventions = {
        "deflector_parallel": "LensThetaX",
        "deflector_perp": "LensThetaY",
    }

    _hdf5_metadata_key_mappings = {
        "timestamp": "entry/start_time",
        "scan_command": None,
        "analyser_model": "FIXED_VALUE:Scienta DA30-L",
        "analyser_eV_type": lambda f: BlochNexusLoader._energy_mode(f),
        "analyser_acquisition_mode": "entry/detector_configuration/measurement_type",
        "analyser_lens_mode": [
            "entry/recipe/step/region/lens_mode",
            "entry/recipe/common/lens_mode",
            "entry/instrument/analyser/analyser_settings/value/lens_mode",
            "entry/instrument/analyzer/analyser_settings/value/lens_mode",
        ],
        "analyser_PE": [
            "entry/recipe/step/region/pass_energy_eV",
            "entry/recipe/common/pass_energy",
            "entry/instrument/analyser/pass_energy/value",
            "entry/instrument/analyzer/pass_energy/value",
            "entry/instrument/analyser/analyser_settings/value/pass_energy",
            "entry/instrument/analyzer/analyser_settings/value/pass_energy",
        ],
        "analyser_eV": lambda f: BlochNexusLoader._get_analyser_eV(f),
        "analyser_step_size": "entry/detector_configuration/received/axes/x/step_mean",
        "analyser_dwell": [
            "entry/recipe/step/region/dwell_time_s",
            "entry/recipe/common/dwell_time_s",
        ],
        "analyser_sweeps": "entry/acquisition_result/completed_units",
        "analyser_deflector_parallel": [
            "entry/recipe/common/lens_theta_x_deg",
            "entry/recipe/step/region/lens_theta_x_deg",
        ],
        "analyser_deflector_perp": [
            "entry/data/DeflectorY",
            "entry/recipe/common/lens_theta_y_deg",
        ],
        "photon_hv": [
            "entry/data/PhotonEnergyAxis",  # hv scan: full scanned range
            "entry/instrument/analyser/photon_energy_readback/value/value",
            "entry/instrument/analyzer/photon_energy_readback/value/value",
        ],
        "temperature_sample": [
            "entry/instrument/analyser/temperature_readbacks/value/LAKESHORE_01_temperature_A/value",
            "entry/instrument/analyzer/temperature_readbacks/value/LAKESHORE_01_temperature_A/value",
        ],
        "temperature_cryostat": [
            "entry/instrument/analyser/temperature_readbacks/value/LAKESHORE_01_temperature_B/value",
            "entry/instrument/analyzer/temperature_readbacks/value/LAKESHORE_01_temperature_B/value",
        ],
        "temperature_setpoint": [
            "entry/instrument/analyser/temperature_readbacks/value/LAKESHORE_01_setpoint_1/value",
            "entry/instrument/analyzer/temperature_readbacks/value/LAKESHORE_01_setpoint_1/value",
        ],
        "manipulator_x1": [
            "entry/instrument/analyser/manipulator_live_positions/value/x/value",
            "entry/instrument/analyzer/manipulator_live_positions/value/x/value",
        ],
        "manipulator_x2": [
            "entry/instrument/analyser/manipulator_live_positions/value/y/value",
            "entry/instrument/analyzer/manipulator_live_positions/value/y/value",
        ],
        "manipulator_x3": [
            "entry/instrument/analyser/manipulator_live_positions/value/z/value",
            "entry/instrument/analyzer/manipulator_live_positions/value/z/value",
        ],
        "manipulator_polar": [
            "entry/instrument/analyser/manipulator_live_positions/value/polar/value",
            "entry/instrument/analyzer/manipulator_live_positions/value/polar/value",
        ],
        "manipulator_tilt": [
            "entry/instrument/analyser/manipulator_live_positions/value/tilt/value",
            "entry/instrument/analyzer/manipulator_live_positions/value/tilt/value",
        ],
        "manipulator_azi": [
            "entry/instrument/analyser/manipulator_live_positions/value/azimuth/value",
            "entry/instrument/analyzer/manipulator_live_positions/value/azimuth/value",
        ],
    }

    _hdf5_metadata_fixed_units = {
        **{
            p: "eV"
            for p in [
                "entry/recipe/step/region/pass_energy_eV",
                "entry/recipe/common/pass_energy",
                "entry/instrument/analyser/pass_energy/value",
                "entry/instrument/analyzer/pass_energy/value",
                "entry/instrument/analyser/analyser_settings/value/pass_energy",
                "entry/instrument/analyzer/analyser_settings/value/pass_energy",
                "entry/detector_configuration/received/axes/x/step_mean",
                "entry/recipe/step/region/center_eV",
                "entry/recipe/energy/center_eV",
            ]
        },
        **{
            p: "s"
            for p in [
                "entry/recipe/step/region/dwell_time_s",
                "entry/recipe/common/dwell_time_s",
            ]
        },
        **{
            p: "deg"
            for p in [
                "entry/recipe/common/lens_theta_x_deg",
                "entry/recipe/step/region/lens_theta_x_deg",
                "entry/recipe/common/lens_theta_y_deg",
            ]
        },
    }

    @staticmethod
    def _sibling_unit(f, key):
        """For units that are a sibling dataset rather than an HDF5 attribute."""
        parent = key.rsplit("/", 1)[0]
        unit = _read_scalar(f, f"{parent}/unit", "")
        return str(unit).strip()

    @classmethod
    def _extract_hdf5_value(cls, f, key, return_extreme_values=True):
        """Handle quirks of the Bloch NeXus metadata format when reading a value."""
        if callable(key):
            key = key(f)
        try:
            value = super()._extract_hdf5_value(f, key, return_extreme_values)
        except (TypeError, ValueError, AttributeError):
            return None
        if isinstance(value, str) and value.strip().lower() in ("", "null"):
            return None
        if (
            isinstance(key, str)
            and not key.startswith("FIXED_VALUE:")
            and isinstance(value, (int, float, np.number, np.ndarray))
            and not isinstance(value, pint.Quantity)
        ):
            unit = cls._sibling_unit(f, key)
            if unit:
                return value * ureg(unit)
        return value

    @staticmethod
    def _is_binding_axis(f):
        mode = (
            _read_scalar(f, "entry/recipe/step/region/energy_mode")
            or _read_scalar(f, "entry/preview/source_energy_mode")
            or ""
        )
        return "binding" in str(mode).lower()

    @staticmethod
    def _kinetic_axis_reference(f):
        """Values needed to put a binding-energy axis back onto kinetic energy."""

        def first_number(paths):
            for path in paths:
                value = _read_scalar(f, path)
                if isinstance(value, (int, float, np.number)) and np.isfinite(value):
                    return float(value)
            return None

        centre_KE = first_number(
            [
                "entry/instrument/analyser/kinetic_energy/value",
                "entry/instrument/analyzer/kinetic_energy/value",
                "entry/instrument/analyser/analyser_settings/value/kinetic_energy",
                "entry/instrument/analyzer/analyser_settings/value/kinetic_energy",
            ]
        )
        binding = first_number(
            [
                "entry/recipe/step/region/center_eV",
                "entry/recipe/step/binding_center_eV",
            ]
        )
        # the first photon energy
        hv_reference = first_number(["entry/preview/excitation_energy"])
        if hv_reference is None:
            hv_values = _nxs_vector(f, "entry/data/PhotonEnergyAxis")
            if hv_values is None:
                hv_values = _nxs_vector(f, "entry/data/PhotonEnergy")
            if hv_values is not None and hv_values.size:
                hv_reference = float(hv_values[0])
        if centre_KE is None or binding is None or hv_reference is None:
            return None
        return centre_KE, binding, hv_reference

    @classmethod
    def _energy_mode(cls, f):
        """Return the analyser eV type, either 'binding' or 'kinetic'.

        A binding-energy axis is put back onto kinetic energy on load where the
        file records enough to do so.
        """
        if cls._is_binding_axis(f) and cls._kinetic_axis_reference(f) is None:
            return "FIXED_VALUE:Binding"
        return "FIXED_VALUE:Kinetic"

    @classmethod
    def _get_analyser_eV(cls, f):
        """Energy window for swept scans or centre kinetic energy for fixed ones."""
        acquisition_mode = str(
            _read_scalar(f, "entry/detector_configuration/measurement_type", "")
        ).lower()
        if "swept" in acquisition_mode and not cls._is_binding_axis(f):
            if "entry/data/x" in f and _nxs_unit(f["entry/data/x"]) == "eV":
                return "entry/data/x"
        for path in [
            "entry/recipe/step/region/center_eV",
            "entry/recipe/energy/center_eV",
            "entry/instrument/analyser/kinetic_energy/value",
            "entry/instrument/analyser/analyser_settings/value/kinetic_energy",
            "entry/instrument/analyzer/kinetic_energy/value",
            "entry/instrument/analyzer/analyser_settings/value/kinetic_energy",
        ]:
            if path in f:
                return path
        return None

    @classmethod
    def _to_kinetic_energy(cls, f, spectrum, dims, coords, units):
        """Rebuild a kinetic-energy axis for a scan stored against binding energy."""
        if "eV" not in dims or not cls._is_binding_axis(f):
            return spectrum, coords, units

        reference = cls._kinetic_axis_reference(f)
        if reference is None:
            analysis_warning(
                "Data energy axis has been loaded as binding energy.",
                "danger",
                "Loading info",
            )
            return spectrum, coords, units

        kinetic_centre, binding_centre, hv_reference = reference
        # Binding energy ascends as kinetic energy descends, so the axis and the
        # data are both reversed to keep the standard low-to-high ordering
        binding = np.asarray(coords["eV"], dtype=np.float64)
        coords["eV"] = (kinetic_centre + binding_centre - binding)[::-1]
        spectrum = np.flip(spectrum, axis=dims.index("eV"))

        if "hv" in coords:
            coords["KE_delta"] = (
                "hv",
                np.asarray(coords["hv"], dtype=np.float64) - hv_reference,
            )
            units["KE_delta"] = "eV"
        return spectrum, coords, units

    @classmethod
    def _load_data(cls, fpath, lazy, **kwargs):
        fpath = str(fpath)
        with h5py.File(fpath, "r") as f:
            if "entry/data" not in f:
                raise ValueError("This file does not contain an /entry/data group.")
            nxdata = f["entry/data"]

            # the signal dataset is called frames
            if "frames" not in nxdata:
                raise ValueError(
                    "This file does not contain the signal dataset /entry/data/frames."
                )
            frames = nxdata["frames"]
            shape = tuple(int(i) for i in frames.shape)
            nframes = shape[0]
            if nframes <= 0:
                raise ValueError("The data file contains no saved frames.")

            signal_unit = _nxs_unit(frames, "counts") or "counts"
            valid = _valid_frame_mask(nxdata, nframes)
            lazy_loading = lazy or (
                lazy is None and int(np.prod(shape)) * 4 > opts.FileIO.lazy_size
            )  # a float32 occupies 4 bytes apparently

            (y_dim, y_unit, y_values), (x_dim, x_unit, x_values) = _get_core_axes(
                nxdata, shape
            )  # these are angles and energies

            geometry = _map_geometry(nxdata, nframes)
            if geometry is None and "ManipulatorFastAxis" in nxdata and nframes >= 4:
                analysis_warning(
                    "Could not reconstruct this spatial map; "
                    "loading as a flat frame stack instead.",
                    "warning",
                    "Spatial map not folded",
                )
            if geometry is not None:
                stack = np.asarray(frames[()], dtype=np.float32)
                stack = _mask_invalid_frames(stack, valid)
                spectrum = _fold_frames(stack, geometry.index)
                if lazy_loading:
                    spectrum = dask.array.from_array(spectrum, chunks="auto")
                dims = [geometry.slow_dim, geometry.fast_dim, y_dim, x_dim]
                coords = {
                    geometry.slow_dim: geometry.slow_values,
                    geometry.fast_dim: geometry.fast_values,
                    y_dim: y_values,
                    x_dim: x_values,
                }
                axis_units = (
                    (geometry.slow_dim, geometry.slow_unit),
                    (geometry.fast_dim, geometry.fast_unit),
                    (y_dim, y_unit),
                    (x_dim, x_unit),
                )
            else:
                # not a spatial map
                scan_choice = _scan_axis(nxdata, nframes)
                if scan_choice is None:
                    frame_dim, frame_unit = "scan_no", ""
                    frame_values = _nxs_vector(nxdata, "frame_index")
                    if frame_values is None or frame_values.size != nframes:
                        frame_values = np.arange(nframes, dtype=np.float64)
                else:
                    frame_dim, frame_unit, frame_values = scan_choice

                spectrum = cls._read_frames(fpath, nxdata, lazy_loading)
                spectrum = _mask_invalid_frames(spectrum, valid)

                dims = [frame_dim, y_dim, x_dim]
                coords = {frame_dim: frame_values, y_dim: y_values, x_dim: x_values}
                axis_units = ((frame_dim, frame_unit), (y_dim, y_unit), (x_dim, x_unit))

                # A single snapshot drops the redundant frame dimension
                if nframes == 1:
                    spectrum = spectrum[0]
                    dims = dims[1:]
                    coords.pop(frame_dim)
                    axis_units = axis_units[1:]

            units = {"spectrum": signal_unit}
            units.update({name: unit for name, unit in axis_units if unit})
            spectrum, coords, units = cls._to_kinetic_energy(
                f, spectrum, dims, coords, units
            )
        return {"spectrum": spectrum, "dims": dims, "coords": coords, "units": units}

    @staticmethod
    def _read_frames(fpath, nxdata, lazy_loading):
        """Read the frame stack."""
        if lazy_loading:
            try:
                ds = xr.open_dataset(
                    fpath,
                    group="entry/data",
                    engine="h5netcdf",
                    phony_dims="sort",
                    chunks="auto",
                )
                return ds["frames"].data.astype(np.float32)
            except Exception:
                analysis_warning(
                    "Lazy loading failed; reading data into memory instead.",
                    "warning",
                    "Lazy load fallback",
                )
        return np.asarray(nxdata["frames"][()], dtype=np.float32)
