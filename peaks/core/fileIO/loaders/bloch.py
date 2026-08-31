"""Functions to load data from the Bloch beamline at MAX IV Laboratory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pint_xarray

from peaks.core.fileIO.base_arpes_data_classes.base_arpes_data_class import (
    BaseARPESDataLoader,
    ureg,
)
from peaks.core.fileIO.base_data_classes.base_hdf5_class import BaseHDF5DataLoader
from peaks.core.fileIO.loc_registry import register_loader

try:  # Optional; only needed when pks.load(..., lazy=True)
    import dask.array as dask_array
except Exception:  # pragma: no cover
    dask_array = None

from peaks.core.fileIO.base_arpes_data_classes.base_ses_class import SESDataLoader

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


_STRING_TYPES = (bytes, np.bytes_)


def _decode(value: Any) -> Any:
    """Decode HDF5 bytes/numpy scalars into ordinary Python values."""
    if isinstance(value, _STRING_TYPES):
        return value.decode(errors="replace")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode(value.item())
        if value.size == 1:
            return _decode(value.reshape(-1)[0])
        return [_decode(v) for v in value.tolist()]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _as_text(value: Any, default: str = "") -> str:
    value = _decode(value)
    if value is None:
        return default
    return str(value)


def _read_scalar(group_or_file: Any, path: str, default: Any = None) -> Any:
    """Read a scalar dataset if it exists; otherwise return default."""
    try:
        node = group_or_file[path]
    except Exception:
        return default
    try:
        return _decode(node[()])
    except Exception:
        return default


def _read_first_scalar(group_or_file: Any, paths: list[str], default: Any = None) -> Any:
    """Read the first non-empty scalar from a list of HDF5 paths."""
    for path in paths:
        value = _read_scalar(group_or_file, path, None)
        if value not in (None, ""):
            return value
    return default


def _read_numeric(group_or_file: Any, paths: list[str], default: Any = None) -> Any:
    """Read the first numeric scalar from a list of HDF5 paths."""
    value = _read_first_scalar(group_or_file, paths, default=None)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except Exception:
        return default


def _read_vector(dataset: Any, fallback_length: int) -> np.ndarray:
    if dataset is None:
        return np.arange(int(fallback_length), dtype=np.float64)
    try:
        values = np.asarray(dataset[()], dtype=np.float64).reshape(-1)
        if values.size == int(fallback_length):
            return values
    except Exception:
        pass
    return np.arange(int(fallback_length), dtype=np.float64)


def _dataset_unit(dataset: Any, default: str = "") -> str:
    if dataset is None:
        return default
    try:
        unit = _as_text(dataset.attrs.get("units", default), default)
    except Exception:
        unit = default
    return _normalise_unit(unit)


def _dataset_label(dataset: Any, fallback: str) -> str:
    if dataset is None:
        return fallback
    try:
        return _as_text(dataset.attrs.get("long_name", fallback), fallback) or fallback
    except Exception:
        return fallback


def _normalise_unit(unit: Any) -> str:
    """Return a pint/PEAKS-friendly unit string."""
    text = _as_text(unit).strip()
    if not text:
        return ""
    aliases = {
        "counts": "count",
        "count": "count",
        "cts": "count",
        "deg": "deg",
        "degree": "deg",
        "degrees": "deg",
        "ev": "eV",
        "electron_volt": "eV",
        "electron volt": "eV",
        "electron volts": "eV",
        "s": "s",
        "sec": "s",
        "second": "s",
        "seconds": "s",
        "ms": "ms",
        "mbar": "mbar",
        "ma": "mA",
        "mm": "mm",
        "um": "um",
        "µm": "um",
        "micron": "um",
        "microns": "um",
        "k": "K",
        "kelvin": "K",
        "": "",
        "1": "",
        "a.u.": "",
        "au": "",
        "arb": "",
        "arb.": "",
    }
    return aliases.get(text, aliases.get(text.lower(), text))


def _clean_dim_name(name: str) -> str:
    """Make a safe xarray dimension/coordinate name."""
    out = []
    for ch in str(name):
        if ch.isalnum() or ch == "_":
            out.append(ch)
        elif ch in " -./:":
            out.append("_")
    result = "".join(out).strip("_")
    return result or "axis"


def _map_detector_axis_name(label: str, unit: str, dataset_name: str, role: str) -> str:
    """Map local detector axis labels into PEAKS coordinate names."""
    text = f"{label} {dataset_name}".lower()
    unit_l = unit.lower()

    if "ev" in unit_l or "energy" in text or "binding" in text or "kinetic" in text:
        return "eV"
    if "deg" in unit_l or "theta" in text or "angle" in text or "deflector" in text:
        return "theta_par"
    if role == "x":
        return "detector_x"
    if role == "y":
        return "detector_y"
    if role == "z":
        return "detector_z"
    return _clean_dim_name(dataset_name)


def _map_manipulator_device_to_peak_dim(device: str, fallback: str) -> str:
    """Best-effort mapping from app/Tango role names to PEAKS manipulator names."""
    text = str(device or "").strip().lower()
    if not text:
        return fallback
    compact = (
        text.replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace(".", "_")
        .replace(":", "_")
    )

    # Prefer explicit manipulator aliases used by the ARPES Console metadata.
    if compact in {
        "x",
        "x1",
        "sample_x",
        "manipulator_x",
        "manip_x",
        "m_x",
    } or compact.endswith("_x"):
        return "x1"
    if compact in {
        "y",
        "x2",
        "sample_y",
        "manipulator_y",
        "manip_y",
        "m_y",
    } or compact.endswith("_y"):
        return "x2"
    if compact in {
        "z",
        "x3",
        "sample_z",
        "manipulator_z",
        "manip_z",
        "m_z",
    } or compact.endswith("_z"):
        return "x3"
    if (
        "polar" in compact
        or compact in {"theta", "th", "sample_theta", "manip_polar"}
        or compact.endswith("_p")
    ):
        return "polar"
    if (
        "tilt" in compact
        or compact in {"chi", "sample_chi", "manip_tilt"}
        or compact.endswith("_t")
    ):
        return "tilt"
    if (
        "azi" in compact
        or "azimuth" in compact
        or compact in {"phi", "sample_phi", "manip_azimuth"}
        or compact.endswith("_a")
    ):
        return "azi"
    return fallback


def _unit_for_manipulator_dim(dim: str, dataset: Any = None) -> str:
    ds_unit = _dataset_unit(dataset, "") if dataset is not None else ""
    if ds_unit:
        return ds_unit
    return (
        "mm"
        if dim in {"x1", "x2", "x3"}
        else "deg"
        if dim in {"polar", "tilt", "azi"}
        else ""
    )


def _scan_axis_name(
    nxdata: h5py.Group, nframes: int
) -> tuple[str, str, np.ndarray] | None:
    """Choose the main frame dimension for stacks, if a useful scan axis exists.

    A manipulator scan is deliberately *not* offered here. Its per-frame position
    repeats once per raster line, and a repeated dimension coordinate breaks
    ``_apply_conventions``, which reindexes any descending dimension. Spatial
    maps are handled by :func:`_map_geometry` instead; only a scan that failed to
    fold reaches the manipulator candidate below.
    """
    candidates: list[tuple[str, str, str]] = []

    if "PhotonEnergy" in nxdata:
        candidates.append(("PhotonEnergy", "hv", "eV"))

    if "DeflectorAxisValue" in nxdata:
        scan_axis = _as_text(_read_scalar(nxdata, "DeflectorScanAxis", ""), "").upper()
        dim = "deflector_perp" if scan_axis.startswith("Y") else "deflector_parallel"
        candidates.append(("DeflectorAxisValue", dim, "deg"))

    if "ManipulatorFastAxis" in nxdata:
        device = _read_scalar(nxdata, "FastAxisDevice", "")
        dim = _map_manipulator_device_to_peak_dim(str(device), "manipulator_fast_axis")
        candidates.append(
            (
                "ManipulatorFastAxis",
                dim,
                _unit_for_manipulator_dim(dim, nxdata.get("ManipulatorFastAxis")),
            )
        )

    # Fallback for point series.
    if "ScanIndex" in nxdata:
        candidates.append(("ScanIndex", "scan_no", ""))

    for dataset_name, dim_name, unit in candidates:
        ds = nxdata.get(dataset_name)
        if ds is None:
            continue
        values = _read_vector(ds, nframes)
        if values.size != nframes:
            continue
        # A dimension coordinate must be usable as an index. Duplicates make
        # xarray's reindex raise, and repeated values are meaningless as an axis
        # even when it does not.
        if np.unique(values).size != values.size:
            continue
        ds_unit = _dataset_unit(ds, unit) or unit
        return dim_name, ds_unit, values

    return None


class _MapGeometry:
    """A manipulator scan folded back into the raster it was acquired on.

    The console flattens every acquisition to a frame stack. For a spatial map
    that discards the geometry entirely: the scan is two-dimensional, but what
    reaches the file is N frames plus per-frame positions. Reading it back as a
    1-D stack gives a non-monotonic position "axis" that is not a coordinate in
    any useful sense, so the grid has to be rebuilt before the data becomes a
    DataArray.

    ``index`` holds the frame number for each raster cell, or -1 where a cell was
    never measured (an interrupted line).
    """

    __slots__ = (
        "index",
        "fast_dim",
        "slow_dim",
        "fast_values",
        "slow_values",
        "fast_unit",
        "slow_unit",
        "kind",
    )

    def __init__(
        self,
        index,
        fast_dim,
        slow_dim,
        fast_values,
        slow_values,
        fast_unit,
        slow_unit,
        kind,
    ):
        self.index = index
        self.fast_dim = fast_dim
        self.slow_dim = slow_dim
        self.fast_values = fast_values
        self.slow_values = slow_values
        self.fast_unit = fast_unit
        self.slow_unit = slow_unit
        self.kind = kind

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(n) for n in self.index.shape)


def _flat_vector(nxdata: h5py.Group, name: str) -> np.ndarray | None:
    """Read a per-frame dataset as a flat float vector, or None."""
    ds = nxdata.get(name)
    if ds is None:
        return None
    try:
        return np.asarray(ds[()], dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None


def _collapse_positions(
    index: np.ndarray, values: np.ndarray | None, axis: int
) -> np.ndarray | None:
    """Reduce per-frame positions onto one raster axis by averaging the other.

    Fast-fly positions are continuous encoder readbacks, so no two lines land on
    exactly the same coordinate; the mean across lines is the best estimate of
    the shared axis. Unmeasured cells are interpolated rather than left NaN, so
    the resulting coordinate stays usable as an index.
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


def _axis_dims_for_map(nxdata: h5py.Group) -> tuple[str, str, str, str]:
    """PEAKS dimension names and units for a map's fast and slow axes."""
    fast_dim = _map_manipulator_device_to_peak_dim(
        str(_read_scalar(nxdata, "FastAxisDevice", "")), "x1"
    )
    slow_dim = _map_manipulator_device_to_peak_dim(
        str(_read_scalar(nxdata, "SlowAxisDevice", "")), "x2"
    )
    if fast_dim == slow_dim:  # Unrecognised device names must not collide.
        fast_dim, slow_dim = "x1", "x2"
    return (
        fast_dim,
        slow_dim,
        _unit_for_manipulator_dim(fast_dim, nxdata.get("ManipulatorFastAxis")),
        _unit_for_manipulator_dim(slow_dim, nxdata.get("ManipulatorSlowAxis")),
    )


def _stepped_map_geometry(nxdata: h5py.Group, nframes: int) -> "_MapGeometry | None":
    """Fold a stepped manipulator scan using its recorded row/column indices."""
    rows = _flat_vector(nxdata, "GridRow")
    cols = _flat_vector(nxdata, "GridColumn")
    if rows is None or cols is None or rows.size != nframes or cols.size != nframes:
        return None
    if not (np.all(np.isfinite(rows)) and np.all(np.isfinite(cols))):
        return None

    nrows = int(np.nanmax(rows)) + 1
    ncols = int(np.nanmax(cols)) + 1
    if nrows < 2 or ncols < 2 or nrows * ncols < nframes:
        return None

    index = np.full((nrows, ncols), -1, dtype=np.int64)
    index[rows.astype(int), cols.astype(int)] = np.arange(nframes)

    fast_dim, slow_dim, fast_unit, slow_unit = _axis_dims_for_map(nxdata)
    fast = _collapse_positions(
        index, _flat_vector(nxdata, "ManipulatorFastAxis"), axis=0
    )
    slow = _collapse_positions(
        index, _flat_vector(nxdata, "ManipulatorSlowAxis"), axis=1
    )
    return _MapGeometry(
        index=index,
        fast_dim=fast_dim,
        slow_dim=slow_dim,
        fast_values=fast if fast is not None else np.arange(ncols, dtype=np.float64),
        slow_values=slow if slow is not None else np.arange(nrows, dtype=np.float64),
        fast_unit=fast_unit,
        slow_unit=slow_unit,
        kind="stepped",
    )


def _looks_like_snake(index: np.ndarray, fast: np.ndarray) -> bool:
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


def _fastfly_map_geometry(nxdata: h5py.Group, nframes: int) -> "_MapGeometry | None":
    """Fold a fast-fly manipulator scan using its per-frame line index.

    Fast-fly scans move continuously along the fast axis and stamp each frame
    with the line it belongs to; there is no GridRow/GridColumn. Frames in
    acquisition order within a line are that line's columns.
    """
    lines = _flat_vector(nxdata, "LineIndex")
    fast = _flat_vector(nxdata, "ManipulatorFastAxis")
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
    if not snake:
        snake = _looks_like_snake(index, fast)
    if snake:
        for row in range(1, nrows, 2):
            members = index[row][index[row] >= 0]
            index[row, : members.size] = members[::-1]

    fast_dim, slow_dim, fast_unit, slow_unit = _axis_dims_for_map(nxdata)
    fast_values = _collapse_positions(index, fast, axis=0)
    slow_values = _collapse_positions(
        index, _flat_vector(nxdata, "ManipulatorSlowAxis"), axis=1
    )
    return _MapGeometry(
        index=index,
        fast_dim=fast_dim,
        slow_dim=slow_dim,
        fast_values=fast_values
        if fast_values is not None
        else np.arange(ncols, dtype=np.float64),
        slow_values=slow_values
        if slow_values is not None
        else np.arange(nrows, dtype=np.float64),
        fast_unit=fast_unit,
        slow_unit=slow_unit,
        kind="fast-fly",
    )


def _map_geometry(nxdata: h5py.Group, nframes: int) -> "_MapGeometry | None":
    """Return the raster geometry of a manipulator scan, or None if it is not one."""
    if nframes < 4:
        return None
    for builder in (_stepped_map_geometry, _fastfly_map_geometry):
        geometry = builder(nxdata, nframes)
        if geometry is not None:
            return geometry
    return None


def _fold_frames(frames: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Fold ``[frame, y, x]`` into ``[slow, fast, y, x]``.

    Cells with no frame -- a partial line from an interrupted scan -- become NaN
    rather than zero. Zero is a measurement claim: it says the detector looked
    and saw nothing. NaN says nobody looked.
    """
    folded = np.full(tuple(index.shape) + frames.shape[1:], np.nan, dtype=np.float32)
    measured = index >= 0
    folded[measured] = frames[index[measured]]
    return folded


def _mask_unmeasured_frames(frames: np.ndarray, nxdata: h5py.Group) -> np.ndarray:
    """Replace ``frame_valid == 0`` placeholder frames with NaN.

    A partial scan keeps placeholder frames so the file stays rectangular, and
    flags them with ``frame_valid = 0``. They hold zeros. Leaving them turns
    "never measured" into "measured zero counts", which biases any integration
    over the scan axis without ever looking wrong.
    """
    valid = _flat_vector(nxdata, "frame_valid")
    if valid is None or valid.size != frames.shape[0]:
        return frames
    invalid = valid < 0.5
    if not np.any(invalid):
        return frames
    frames = frames.astype(np.float32, copy=True)
    frames[invalid] = np.nan
    return frames


def _axis_from_dataset(
    nxdata: h5py.Group,
    name: str,
    length: int,
    *,
    role: str,
) -> tuple[str, str, np.ndarray]:
    ds = nxdata.get(name)
    values = _read_vector(ds, length)
    label = _dataset_label(ds, name)
    unit = _dataset_unit(ds, "")
    dim = _map_detector_axis_name(label, unit, name, role)
    return dim, unit, values


def _find_z_axis(
    nxdata: h5py.Group, length: int, used_names: set[str]
) -> tuple[str, str, np.ndarray]:
    """Find a z/slice axis for [frame, y, x, z] files."""
    preferred = [
        "z",
        "Z",
        "Slice",
        "slice",
        "Energy",
        "Kinetic Energy",
        "Binding Energy",
    ]
    for name in preferred:
        if name in nxdata and name not in used_names:
            return _axis_from_dataset(nxdata, name, length, role="z")

    for name, obj in nxdata.items():
        if name in used_names or name in {"frames", "frame_index"}:
            continue
        if not isinstance(obj, h5py.Dataset):
            continue
        try:
            if obj.shape and int(np.prod(obj.shape)) == int(length):
                return _axis_from_dataset(nxdata, name, length, role="z")
        except Exception:
            continue

    return "detector_z", "", np.arange(int(length), dtype=np.float64)


def _per_frame_aux_coords(
    nxdata: h5py.Group,
    frame_dim: str | tuple[str, ...],
    nframes: int,
    *,
    fold_index: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Read additional per-frame readbacks as auxiliary xarray coordinates.

    ``frame_dim`` is a single dimension name for a stack, or the pair of raster
    dimensions for a folded map. When ``fold_index`` is given, each per-frame
    readback is folded onto the same raster as the data, so a position readback
    or a timestamp stays attached to the cell it belongs to.
    """
    coords: dict[str, Any] = {}
    units: dict[str, str] = {}
    dims = (frame_dim,) if isinstance(frame_dim, str) else tuple(frame_dim)

    skip = {
        "frames",
        "frame_index",
        "x",
        "y",
        "x_display",
        "y_display",
        "DeflectorScanAxis",
        "OrthogonalDeflectorValue",
        "ScanPoints",
        "ActualFrames",
        "RequestedPhotonEnergyPoints",
        "RequestedDeflectorPoints",
        "FastAxisDevice",
        "SlowAxisDevice",
        "SnakeScanMode",
        "snakeScanMode",
        "GridRows",
        "GridColumns",
    }

    rename = {
        "PhotonEnergy": "hv_readback",
        "PhotonEnergySetpoint": "hv_setpoint",
        "LensThetaX": "deflector_parallel",
        "LensThetaY": "deflector_perp",
        "LensThetaXRequested": "deflector_parallel_setpoint",
        "LensThetaYRequested": "deflector_perp_setpoint",
        "DeflectorAxisValue": "deflector_scan_value",
        "ManipulatorFastAxis": "manipulator_fast_axis_value",
        "ManipulatorSlowAxis": "manipulator_slow_axis_value",
        "GridRow": "grid_row",
        "GridColumn": "grid_column",
        "LineIndex": "line_index",
        "FastFlyPassIndex": "fast_fly_pass_index",
        "FastFlyDirection": "fast_fly_direction",
        "TimestampUnix": "timestamp_unix",
        "TimestampMonotonic": "timestamp_monotonic",
        "ScanIndex": "scan_index",
        "PositionSampleFastAxis": "position_sample_fast_axis",
        "PositionSampleSlowAxis": "position_sample_slow_axis",
    }

    for name, obj in nxdata.items():
        if name in skip or not isinstance(obj, h5py.Dataset):
            continue
        try:
            arr = np.asarray(obj[()]).reshape(-1)
        except Exception:
            continue
        if arr.size != nframes:
            continue
        coord_name = rename.get(name, _clean_dim_name(name))
        if coord_name in dims:
            continue
        if fold_index is None:
            coords[coord_name] = (dims[0], arr)
        else:
            # Object dtypes (e.g. FrameTimestampSource) have no NaN to pad an
            # unmeasured cell with, so they are left out of a folded map rather
            # than filled with something misleading.
            if arr.dtype.kind not in "fiub":
                continue
            folded = np.full(fold_index.shape, np.nan, dtype=np.float64)
            measured = fold_index >= 0
            folded[measured] = arr[fold_index[measured]].astype(np.float64)
            coords[coord_name] = (dims, folded)
        unit = _dataset_unit(obj, "")
        if unit:
            units[coord_name] = unit

    return coords, units


def _hdf5_group_to_dict(group: Any) -> dict[str, Any]:
    """Convert a metadata group to nested Python dictionaries."""
    result: dict[str, Any] = {}
    if group is None:
        return result
    try:
        items = list(group.items())
    except Exception:
        return result
    for key, node in items:
        if isinstance(node, h5py.Group):
            result[str(key)] = _hdf5_group_to_dict(node)
        elif isinstance(node, h5py.Dataset):
            try:
                result[str(key)] = _decode(node[()])
            except Exception:
                pass
    return result


def _quantity(value: Any, unit: str) -> Any:
    if value in (None, ""):
        return None
    try:
        return float(value) * ureg(unit)
    except Exception:
        return None


def _per_frame_median(f: h5py.File, name: str) -> float | None:
    """Median of a per-frame bookkeeping dataset, ignoring NaN placeholders."""
    node = f.get(f"entry/data/{name}")
    if node is None:
        return None
    try:
        values = np.asarray(node[()], dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.median(values))


def _acquisition_timing(f: h5py.File) -> tuple[float | None, float]:
    """Return ``(dwell_seconds, sweeps)`` for ONE STORED FRAME.

    ``BaseARPESDataLoader._load`` converts counts to counts/s by dividing the
    data by ``analyser_dwell * analyser_sweeps``, so that product has to be the
    integration time of a single stored frame -- which is exactly what the
    console records as ``frame_acquisition_time``, and what its own
    ``count_rate_formula`` attribute documents as ``frames /
    frame_acquisition_time``.

    Sweeps is therefore the number of acquisitions *summed into a frame*
    (``frame_contributing_frames``), not the number of points in the scan.
    Reading it from ``acquisition_result/completed_units``, as this loader did
    before, happens to be right for a single-point snapshot -- where the two
    coincide -- and is wrong by the number of scan points for everything else:
    a 61-point deflector map came out 61 times too small, silently.

    Dwell is derived from the measured acquisition time rather than the
    requested dwell, so the product reproduces the real integration time even
    when PEAK returns a partial final acquisition.
    """
    contributing = _per_frame_median(f, "frame_contributing_frames")
    sweeps = float(contributing) if contributing and contributing >= 1 else 1.0

    acquisition = _per_frame_median(f, "frame_acquisition_time")
    if acquisition is not None and acquisition > 0:
        return acquisition / sweeps, sweeps

    dwell = _per_frame_median(f, "frame_dwell_time")
    if dwell is not None and dwell > 0:
        return dwell, sweeps

    nominal = _read_numeric(
        f,
        [
            "entry/recipe/step/region/dwell_time_s",
            "entry/recipe/common/dwell_time_s",
            "entry/instrument/analyser/dwell_time/value",
            "entry/instrument/analyzer/dwell_time/value",
        ],
    )
    return nominal, sweeps


def _axis_step_quantity_from_file(
    f: h5py.File, path: str = "entry/data/x", unit: str = "eV"
) -> Any:
    """Derive analyzer energy step size from a numeric detector axis."""
    try:
        if path not in f:
            return None
        values = np.asarray(f[path][()], dtype=np.float64).reshape(-1)
        if values.size < 2:
            return None
        diffs = np.diff(values)
        diffs = diffs[np.isfinite(diffs)]
        diffs = np.abs(diffs[diffs != 0])
        if diffs.size == 0:
            return None
        return float(np.nanmedian(diffs)) * ureg(unit)
    except Exception:
        return None


def _metadata_snapshot_quantity(
    f: h5py.File,
    base_paths: list[str],
    default_unit: str,
) -> Any:
    """Read a metadata snapshot stored either as value/unit or value/value/unit."""
    for base in base_paths:
        if base not in f:
            continue
        value = _read_numeric(f, [f"{base}/value", f"{base}/value/value", base], None)
        if value is None:
            continue
        unit = (
            _normalise_unit(
                _read_first_scalar(
                    f,
                    [
                        f"{base}/unit",
                        f"{base}/value/unit",
                        f"{base}/units",
                        f"{base}/value/units",
                    ],
                    default_unit,
                )
            )
            or default_unit
        )
        return _quantity(value, unit)
    return None


#: Role tags the console attaches to each discovered thermometer, ordered by how
#: well each answers "how cold was the sample". Selecting on the role rather than
#: the alias is what makes this survive a change of cryostat wiring: alias names
#: are site configuration, roles are the contract.
_TEMPERATURE_ROLES = (
    "sample_temperature",
    "sample_coldfinger_temperature",
    "cryostat_temperature",
    "shield_temperature",
)

#: Where the console writes its thermometer snapshot. The analyser group is
#: written under both spellings, and the beamline paths are kept for files from
#: earlier writer revisions.
_TEMPERATURE_ROOTS = (
    "entry/instrument/analyser/temperature_readbacks/value",
    "entry/instrument/analyzer/temperature_readbacks/value",
    "entry/instrument/analyser/temperature_readbacks",
    "entry/instrument/analyzer/temperature_readbacks",
    "entry/metadata/beamline/temperature_readbacks/value",
    "entry/beamline/metadata/temperature_readbacks/value",
    "entry/metadata/beamline/temperature_readbacks",
    "entry/beamline/metadata/temperature_readbacks",
)


def _temperature_readbacks_by_role(f: h5py.File) -> dict[str, tuple[float, str]]:
    """Collect every valid thermometer reading, keyed by its role tag."""
    found: dict[str, tuple[float, str]] = {}
    for root_path in _TEMPERATURE_ROOTS:
        root = f.get(root_path)
        if not isinstance(root, h5py.Group):
            continue
        try:
            items = list(root.items())
        except Exception:
            continue
        for alias, node in items:
            if not isinstance(node, h5py.Group):
                continue
            quality = _as_text(_read_scalar(node, "quality", "ATTR_VALID"), "ATTR_VALID")
            if quality not in ("ATTR_VALID", ""):
                continue
            value = _read_numeric(node, ["value", "value/value"], None)
            if value is None or not np.isfinite(value):
                continue
            unit = _as_text(
                _read_first_scalar(
                    node, ["unit", "value/unit", "units", "value/units"], "K"
                ),
                "K",
            )
            # Tango aliases often report Celsius. Convert to kelvin directly:
            # pint's offset-unit multiplication is awkward inside a metadata model.
            if unit.strip().lower().lstrip("°") in {
                "c",
                "degc",
                "celsius",
                "degree_celsius",
            }:
                value = float(value) + 273.15
            found.setdefault(
                str(_as_text(_read_scalar(node, "role", ""), "")),
                (float(value), str(alias)),
            )
        if found:
            break
    return found


def _first_temperature_quantity_from_file(f: h5py.File) -> Any:
    """Find the sample temperature recorded with this acquisition.

    The console records every thermometer it discovered, each tagged with a
    ``role``. Earlier versions of this loader searched only the beamline metadata
    paths, which this format does not use, so every file silently loaded with a
    NaN temperature even though the value was present.
    """
    direct_paths = [
        "entry/sample/temperature",
        "entry/sample/temperature/value",
        "entry/sample/metadata/temperature",
        "entry/sample/metadata/sample_temperature",
        "entry/instrument/analyser/sample_temperature/value",
        "entry/instrument/analyser/sample_temperature/value/value",
        "entry/instrument/analyzer/sample_temperature/value",
        "entry/instrument/analyzer/sample_temperature/value/value",
        "entry/instrument/sample/temperature/value",
        "entry/metadata/sample/temperature",
        "entry/metadata/sample/sample_temperature",
    ]
    value = _read_numeric(f, direct_paths, None)
    if value is not None:
        return _quantity(value, "K")

    by_role = _temperature_readbacks_by_role(f)
    for role in _TEMPERATURE_ROLES:
        if role in by_role:
            return _quantity(by_role[role][0], "K")

    # The ARPES Console format may legitimately omit sample temperature.
    # PEAKS still expects a temperature_sample metadata field for ARPES objects;
    # return a NaN quantity instead of None so loading does not warn or fail,
    # while making it clear that no physical temperature was recorded.
    return np.nan * ureg("K")


def _manipulator_metadata_from_file(f: h5py.File) -> dict[str, Any]:
    """Extract manipulator live positions from the app's nested metadata snapshots."""
    metadata: dict[str, Any] = {}
    base_candidates = [
        "entry/instrument/analyser/manipulator_live_positions/value",
        "entry/instrument/analyzer/manipulator_live_positions/value",
        "entry/metadata/manipulator/manipulator_live_positions/value",
    ]
    axis_map = {
        "x": "x1",
        "y": "x2",
        "z": "x3",
        "polar": "polar",
        "tilt": "tilt",
        "azimuth": "azi",
        "azi": "azi",
    }
    for base in base_candidates:
        if base not in f:
            continue
        group = f[base]
        for local_axis, peaks_axis in axis_map.items():
            value = _read_numeric(group, [f"{local_axis}/value"], None)
            if value is None:
                continue
            unit = _normalise_unit(_read_first_scalar(group, [f"{local_axis}/unit"], ""))
            if not unit:
                unit = "mm" if peaks_axis in {"x1", "x2", "x3"} else "deg"
            metadata[f"manipulator_{peaks_axis}"] = _quantity(value, unit)
        break
    return metadata


def _lazy_spectrum_from_hdf5(
    cls: type,
    fpath: str,
    shape: tuple[int, ...],
    chunks: tuple[int, ...] | None,
) -> Any:
    """Return a dask array backed by HDF5, keeping the file handle alive.

    A dask array holds only a reference to the ``h5py.Dataset``, so the file has
    to stay open for as long as any chunk might still be computed. The handles
    are cached per path.

    Reusing a live handle rather than opening a second one matters: the previous
    version replaced the cached entry on every load of the same path, dropping
    the old ``h5py.File`` while dask graphs from the earlier DataArray still
    referenced datasets inside it. Once that handle was garbage collected, those
    arrays raised on compute.
    """
    if dask_array is None:
        with h5py.File(fpath, "r") as f:
            return np.asarray(f["entry/data/frames"][()], dtype=np.float32)

    handles = getattr(cls, "_lazy_h5_handles", None)
    if handles is None:
        handles = {}
        cls._lazy_h5_handles = handles

    handle = handles.get(fpath)
    if handle is None or not handle.id.valid:
        handle = h5py.File(fpath, "r")
        handles[fpath] = handle

    dataset = handle["entry/data/frames"]
    if chunks is None:
        chunks = dataset.chunks
    if chunks is None:
        chunks = (1,) + tuple(shape[1:])
    return dask_array.from_array(dataset, chunks=chunks)


def close_lazy_handles(fpath: str | None = None) -> int:
    """Close HDF5 handles held open for lazily loaded DataArrays.

    Any dask-backed DataArray still referencing a closed file will fail on
    compute, so call this only when those are finished with. Returns the number
    of handles closed.
    """
    handles = getattr(BlochNexusLoader, "_lazy_h5_handles", None) or {}
    targets = [str(fpath)] if fpath is not None else list(handles)
    closed = 0
    for key in targets:
        handle = handles.pop(key, None)
        if handle is None:
            continue
        try:
            handle.close()
            closed += 1
        except Exception:
            pass
    return closed


@register_loader
class BlochNexusLoader(BaseHDF5DataLoader, BaseARPESDataLoader):
    """Loader for MAX IV Bloch A-branch PEAK / ARPES Console NeXus files."""

    _loc_name = "MAXIV_Bloch_A_NeXus"
    _loc_description = "A branch (ARPES) of Bloch beamline at Max-IV — PEAK / ARPES Console NeXus format"
    _loc_url = "https://www.maxiv.lu.se/beamlines-accelerators/beamlines/bloch/"

    _manipulator_axes = ["polar", "tilt", "azi", "x1", "x2", "x3"]
    _manipulator_name_conventions = {
        "polar": "polar",
        "tilt": "tilt",
        "azi": "azimuth",
        "x1": "x",
        "x2": "y",
        "x3": "z",
    }
    _manipulator_sign_conventions: dict[str, int] = {}
    _analyser_sign_conventions: dict[str, int] = {}
    _analyser_name_conventions = {
        "deflector_parallel": "LensThetaX",
        "deflector_perp": "LensThetaY",
    }
    _analyser_slit_angle = 0 * ureg("deg")

    _desired_dim_order = [
        "scan_no",
        "hv",
        "temperature_sample",
        "temperature_cryostat",
        "x3",
        "x2",
        "x1",
        "polar",
        "tilt",
        "azi",
        "y_scale",
        "deflector_perp",
        "eV",
        "deflector_parallel",
        "theta_par",
        "detector_z",
        "detector_x",
        "detector_y",
    ]

    @staticmethod
    def is_compatible_file(fpath: str | Path) -> bool:
        """Return True when a file looks like an ARPES Console NeXus file."""
        try:
            with h5py.File(str(fpath), "r") as f:
                if "entry/data/frames" not in f:
                    return False
                definition = _as_text(_read_scalar(f, "entry/definition", ""), "")
                instrument = _as_text(_read_scalar(f, "entry/instrument/name", ""), "")
                signal = _as_text(f["entry/data"].attrs.get("signal", ""), "")
                return (
                    definition == "NXarpes"
                    and instrument == "ARPES"
                    and signal == "frames"
                )
        except Exception:
            return False

    @classmethod
    def _load_data(
        cls, fpath: str, lazy: bool | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Load /entry/data/frames as a PEAKS-compatible DataArray dictionary.

        Keyword arguments
        -----------------
        keep_frame_dim
            Keep a singleton frame dimension for a single-frame file.
        prefer_display_axes
            Use the console's preview axes rather than the raw detector axes.
        as_map
            Fold a manipulator scan back into its two-dimensional raster
            (default). Set ``False`` to get the flat acquisition-order stack,
            which is what you want when inspecting a scan that went wrong.
        mask_invalid
            Replace ``frame_valid == 0`` placeholder frames with NaN (default).
        """
        keep_frame_dim = bool(kwargs.pop("keep_frame_dim", False))
        prefer_display_axes = bool(kwargs.pop("prefer_display_axes", False))
        as_map = bool(kwargs.pop("as_map", True))
        mask_invalid = bool(kwargs.pop("mask_invalid", True))
        if kwargs:
            raise ValueError(
                f"Unexpected keyword arguments for {cls.__name__}: {sorted(kwargs)}"
            )

        fpath = str(fpath)
        with h5py.File(fpath, "r") as f:
            if "entry/data/frames" not in f:
                raise ValueError("This file does not contain /entry/data/frames.")
            nxdata = f["entry/data"]
            frames = nxdata["frames"]
            shape = tuple(int(i) for i in frames.shape)
            if len(shape) not in (3, 4):
                raise ValueError(
                    f"Unsupported ARPES Console frames rank {len(shape)}; expected 3 or 4."
                )
            if shape[0] <= 0:
                raise ValueError("The ARPES Console file contains no saved frames.")

            signal_unit = _dataset_unit(frames, "count") or "count"
            nframes = shape[0]

            # A manipulator scan is a raster, not a stack. Folding it is what
            # keeps its positions usable as coordinates, so it is decided before
            # any 1-D scan axis is considered. Rank-4 detector cubes are already
            # four-dimensional and are never folded further.
            geometry = None
            if as_map and len(shape) == 3:
                geometry = _map_geometry(nxdata, nframes)
            if geometry is not None:
                return cls._load_map(
                    fpath,
                    f,
                    nxdata,
                    frames,
                    shape,
                    geometry,
                    signal_unit=signal_unit,
                    prefer_display_axes=prefer_display_axes,
                    mask_invalid=mask_invalid,
                    lazy=bool(lazy),
                )

            scan_choice = _scan_axis_name(nxdata, nframes)
            if scan_choice is None:
                frame_dim = "scan_no"
                frame_unit = ""
                frame_values = (
                    _read_vector(nxdata.get("frame_index"), nframes)
                    if "frame_index" in nxdata
                    else np.arange(nframes, dtype=np.float64)
                )
            else:
                frame_dim, frame_unit, frame_values = scan_choice

            if prefer_display_axes:
                default_y = _as_text(
                    nxdata.attrs.get("arpes_display_y_axis", "y_display"), "y_display"
                )
                default_x = _as_text(
                    nxdata.attrs.get("arpes_display_x_axis", "x_display"), "x_display"
                )
            else:
                default_y = _as_text(nxdata.attrs.get("arpes_default_y_axis", "y"), "y")
                default_x = _as_text(nxdata.attrs.get("arpes_default_x_axis", "x"), "x")

            if default_y not in nxdata:
                default_y = "y" if "y" in nxdata else "y_display"
            if default_x not in nxdata:
                default_x = "x" if "x" in nxdata else "x_display"

            y_dim, y_unit, y_values = _axis_from_dataset(
                nxdata, default_y, shape[1], role="y"
            )
            x_dim, x_unit, x_values = _axis_from_dataset(
                nxdata, default_x, shape[2], role="x"
            )

            dims = [frame_dim, y_dim, x_dim]
            coords: dict[str, Any] = {
                frame_dim: frame_values,
                y_dim: y_values,
                x_dim: x_values,
            }
            units: dict[str, str] = {"spectrum": signal_unit}
            for name, unit in (
                (frame_dim, frame_unit),
                (y_dim, y_unit),
                (x_dim, x_unit),
            ):
                if unit:
                    units[name] = unit

            used_axis_dataset_names = {default_y, default_x, "frame_index", "frames"}
            if len(shape) == 4:
                z_dim, z_unit, z_values = _find_z_axis(
                    nxdata, shape[3], used_axis_dataset_names
                )
                if z_dim in set(dims):
                    z_dim = "detector_z"
                dims.append(z_dim)
                coords[z_dim] = z_values
                if z_unit:
                    units[z_dim] = z_unit

            aux_coords, aux_units = _per_frame_aux_coords(nxdata, frame_dim, nframes)
            for name, value in aux_coords.items():
                if name not in coords and name not in dims:
                    coords[name] = value
            units.update({k: v for k, v in aux_units.items() if v})

            if lazy:
                spectrum = _lazy_spectrum_from_hdf5(cls, fpath, shape, frames.chunks)
            else:
                spectrum = np.asarray(frames[()], dtype=np.float32)
                if mask_invalid:
                    spectrum = _mask_unmeasured_frames(spectrum, nxdata)

            if nframes == 1 and not keep_frame_dim:
                spectrum = spectrum[0]
                dims = dims[1:]
                coords = {
                    k: v
                    for k, v in coords.items()
                    if k != frame_dim
                    and not (isinstance(v, tuple) and v[0] == frame_dim)
                }
                units.pop(frame_dim, None)

        return {
            "spectrum": spectrum,
            "dims": dims,
            "coords": coords,
            "units": units,
        }

    @classmethod
    def _load_map(
        cls,
        fpath: str,
        f: h5py.File,
        nxdata: h5py.Group,
        frames: h5py.Dataset,
        shape: tuple[int, ...],
        geometry: "_MapGeometry",
        *,
        signal_unit: str,
        prefer_display_axes: bool,
        mask_invalid: bool,
        lazy: bool,
    ) -> dict[str, Any]:
        """Build the DataArray dictionary for a folded manipulator raster.

        The result is four-dimensional, ``[slow, fast, y, x]``, which
        ``_desired_dim_order`` then arranges as ``x2, x1, eV, theta_par``.
        Folding needs the whole stack in memory, so a map is always read eagerly
        even when ``lazy=True`` was requested; a lazy read of a raster would have
        to reorder chunks anyway.
        """
        if prefer_display_axes:
            default_y = _as_text(
                nxdata.attrs.get("arpes_display_y_axis", "y_display"), "y_display"
            )
            default_x = _as_text(
                nxdata.attrs.get("arpes_display_x_axis", "x_display"), "x_display"
            )
        else:
            default_y = _as_text(nxdata.attrs.get("arpes_default_y_axis", "y"), "y")
            default_x = _as_text(nxdata.attrs.get("arpes_default_x_axis", "x"), "x")
        if default_y not in nxdata:
            default_y = "y" if "y" in nxdata else "y_display"
        if default_x not in nxdata:
            default_x = "x" if "x" in nxdata else "x_display"

        y_dim, y_unit, y_values = _axis_from_dataset(
            nxdata, default_y, shape[1], role="y"
        )
        x_dim, x_unit, x_values = _axis_from_dataset(
            nxdata, default_x, shape[2], role="x"
        )

        stack = np.asarray(frames[()], dtype=np.float32)
        if mask_invalid:
            stack = _mask_unmeasured_frames(stack, nxdata)
        spectrum = _fold_frames(stack, geometry.index)
        if lazy and dask_array is not None:
            spectrum = dask_array.from_array(spectrum, chunks=(1, 1) + tuple(shape[1:]))

        dims = [geometry.slow_dim, geometry.fast_dim, y_dim, x_dim]
        coords: dict[str, Any] = {
            geometry.slow_dim: geometry.slow_values,
            geometry.fast_dim: geometry.fast_values,
            y_dim: y_values,
            x_dim: x_values,
        }
        units: dict[str, str] = {"spectrum": signal_unit}
        for name, unit in (
            (geometry.slow_dim, geometry.slow_unit),
            (geometry.fast_dim, geometry.fast_unit),
            (y_dim, y_unit),
            (x_dim, x_unit),
        ):
            if unit:
                units[name] = unit

        aux_coords, aux_units = _per_frame_aux_coords(
            nxdata,
            (geometry.slow_dim, geometry.fast_dim),
            shape[0],
            fold_index=geometry.index,
        )
        for name, value in aux_coords.items():
            if name not in coords and name not in dims:
                coords[name] = value
        units.update({k: v for k, v in aux_units.items() if v})

        return {
            "spectrum": spectrum,
            "dims": dims,
            "coords": coords,
            "units": units,
        }

    @classmethod
    def _load_metadata(cls, fpath: str) -> dict[str, Any]:
        """Load PEAKS ARPES metadata plus raw ARPES Console provenance."""
        metadata: dict[str, Any] = {}
        with h5py.File(str(fpath), "r") as f:
            metadata["timestamp"] = _read_first_scalar(
                f,
                ["entry/start_time", "entry/end_time"],
            )

            recipe_type = _read_first_scalar(
                f, ["entry/recipe/type", "entry/recipe/kind"], ""
            )
            acquisition_mode = _read_first_scalar(
                f, ["entry/recipe/acquisition_mode"], ""
            )
            title = _read_first_scalar(f, ["entry/title"], "ARPES acquisition")
            if recipe_type or acquisition_mode:
                metadata["scan_command"] = " / ".join(
                    str(v) for v in (recipe_type, acquisition_mode) if v
                )
            else:
                metadata["scan_command"] = title

            energy_mode = _read_first_scalar(
                f,
                [
                    "entry/recipe/step/region/energy_mode",
                    "entry/recipe/common/energy_mode",
                    "entry/preview/source_energy_mode",
                ],
                "KINETIC",
            )
            metadata["analyser_eV_type"] = (
                "binding" if "binding" in str(energy_mode).lower() else "kinetic"
            )
            metadata["analyser_acquisition_mode"] = acquisition_mode or None
            metadata["analyser_model"] = "PEAK / ARPES Console"
            metadata["analyser_lens_mode"] = _read_first_scalar(
                f,
                [
                    "entry/recipe/step/region/lens_mode",
                    "entry/recipe/common/lens_mode",
                    "entry/instrument/analyser/analyser_settings/value/lens_mode",
                    "entry/instrument/analyzer/analyser_settings/value/lens_mode",
                ],
            )

            metadata["analyser_PE"] = _quantity(
                _read_numeric(
                    f,
                    [
                        "entry/recipe/step/region/pass_energy_eV",
                        "entry/recipe/common/pass_energy",
                        "entry/instrument/analyser/pass_energy/value",
                        "entry/instrument/analyser/analyser_settings/value/pass_energy",
                        "entry/instrument/analyzer/pass_energy/value",
                        "entry/instrument/analyzer/analyser_settings/value/pass_energy",
                    ],
                ),
                "eV",
            )
            # The nominal (requested) dwell. The measured one that reproduces the
            # count rate is set further down, together with analyser_sweeps.
            nominal_dwell = _read_numeric(
                f,
                [
                    "entry/recipe/step/region/dwell_time_s",
                    "entry/recipe/common/dwell_time_s",
                    "entry/instrument/analyser/dwell_time/value",
                    "entry/instrument/analyser/analyser_settings/value/dwell_time",
                    "entry/instrument/analyzer/dwell_time/value",
                    "entry/instrument/analyzer/analyser_settings/value/dwell_time",
                ],
            )
            metadata["analyser_dwell"] = _quantity(nominal_dwell, "s")
            if nominal_dwell is not None:
                metadata["arpes_console_nominal_dwell_s"] = float(nominal_dwell)

            # Swept regions define min/max/step; fixed regions define center_eV.
            center_eV = _read_numeric(
                f,
                [
                    "entry/recipe/energy/center_eV",
                    "entry/instrument/analyser/kinetic_energy/value",
                    "entry/instrument/analyser/analyser_settings/value/kinetic_energy",
                    "entry/instrument/analyzer/kinetic_energy/value",
                    "entry/instrument/analyzer/analyser_settings/value/kinetic_energy",
                ],
            )
            metadata["analyser_eV"] = _quantity(center_eV, "eV")
            step_meV = _read_numeric(f, ["entry/recipe/energy/step_meV"], None)
            if step_meV is not None:
                metadata["analyser_step_size"] = _quantity(
                    float(step_meV) / 1000.0, "eV"
                )
            else:
                # Analyzer-region files do not always keep step_meV in the recipe,
                # but the saved energy coordinate is exact enough to derive it.
                metadata["analyser_step_size"] = _axis_step_quantity_from_file(
                    f, "entry/data/x", "eV"
                )
            # BaseARPESDataLoader._load converts counts to counts/s by dividing
            # the data by ``dwell * sweeps``, so these two must multiply to the
            # integration time of ONE STORED FRAME. See _acquisition_timing:
            # sweeps is the number of acquisitions summed into a frame, never
            # the number of scan points.
            dwell_s, sweeps = _acquisition_timing(f)
            if dwell_s is not None:
                metadata["analyser_dwell"] = _quantity(dwell_s, "s")
            metadata["analyser_sweeps"] = sweeps

            metadata["analyser_deflector_parallel"] = _quantity(
                _read_numeric(
                    f,
                    [
                        "entry/recipe/step/region/lens_theta_x_deg",
                        "entry/recipe/common/lens_theta_x_deg",
                        "entry/instrument/analyser/analyser_settings/value/lens_theta_x",
                        "entry/instrument/analyzer/analyser_settings/value/lens_theta_x",
                    ],
                ),
                "deg",
            )
            metadata["analyser_deflector_perp"] = _quantity(
                _read_numeric(
                    f,
                    [
                        "entry/recipe/step/region/lens_theta_y_deg",
                        "entry/recipe/common/lens_theta_y_deg",
                        "entry/instrument/analyser/analyser_settings/value/lens_theta_y",
                        "entry/instrument/analyzer/analyser_settings/value/lens_theta_y",
                    ],
                ),
                "deg",
            )

            # Analyzer orientation defaults: DA30 slit angle is handled by _analyser_slit_angle.
            metadata["analyser_polar"] = 0 * ureg("deg")
            metadata["analyser_tilt"] = 0 * ureg("deg")
            metadata["analyser_azi"] = cls._analyser_slit_angle

            # The live beamline readback is preferred over the region's
            # configured excitation_energy_eV, and deliberately so: the
            # configured value is what the operator typed and is never
            # re-validated against the beamline, so it can be left stale from an
            # earlier setup. One of the reference manipulator scans records
            # 60 eV against a 50 eV readback, and only 50 eV is consistent with
            # the kinetic-energy window that file actually measured. Do not
            # reorder these without checking a file where they disagree.
            photon_hv = _metadata_snapshot_quantity(
                f,
                [
                    "entry/metadata/beamline/photon_energy_readback",
                    "entry/beamline/metadata/photon_energy_readback",
                    "entry/instrument/analyser/photon_energy_readback",
                    "entry/instrument/analyzer/photon_energy_readback",
                ],
                "eV",
            )
            configured_hv = _read_numeric(
                f,
                [
                    "entry/preview/excitation_energy",
                    "entry/recipe/step/region/excitation_energy_eV",
                    "entry/recipe/preview_excitation_energy_eV",
                ],
            )
            if photon_hv is None and configured_hv is not None:
                photon_hv = _quantity(configured_hv, "eV")
            if photon_hv is None and "entry/data/PhotonEnergy" in f:
                try:
                    vals = np.asarray(
                        f["entry/data/PhotonEnergy"][()], dtype=np.float64
                    ).reshape(-1)
                    vals = vals[np.isfinite(vals)]
                    if vals.size:
                        photon_hv = float(np.nanmean(vals)) * ureg("eV")
                except Exception:
                    pass
            # PEAKS metadata model expects photon_hv; keep hv as a convenience alias.
            metadata["photon_hv"] = photon_hv
            metadata["hv"] = photon_hv
            # Keep the configured value alongside, so a disagreement stays
            # inspectable rather than being silently discarded.
            if configured_hv is not None:
                metadata["arpes_console_configured_hv_eV"] = float(configured_hv)

            metadata["temperature_sample"] = _first_temperature_quantity_from_file(f)
            metadata.update(_manipulator_metadata_from_file(f))

            # Raw provenance is useful with return_as_dict=True.
            entry = f.get("entry")
            if entry is not None:
                metadata["arpes_console_recipe"] = _hdf5_group_to_dict(
                    entry.get("recipe")
                )
                metadata["arpes_console_preview"] = _hdf5_group_to_dict(
                    entry.get("preview")
                )
                metadata["arpes_console_file_info"] = _hdf5_group_to_dict(
                    entry.get("file_info")
                )
                metadata["arpes_console_acquisition_result"] = _hdf5_group_to_dict(
                    entry.get("acquisition_result")
                )

        return metadata
