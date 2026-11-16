import re
from datetime import datetime, timezone

import dask.array
import h5py
import numpy as np
import pint
import pint_xarray
import xarray as xr
from dateutil.parser import parse

from peaks.core.accessors.accessor_methods import register_accessor
from peaks.core.fileIO.base_arpes_data_classes.base_arpes_data_class import (
    BaseARPESDataLoader,
)
from peaks.core.fileIO.base_data_classes.base_hdf5_class import BaseHDF5DataLoader
from peaks.core.fileIO.base_data_classes.base_optics_class import BaseOpticsDataLoader
from peaks.core.fileIO.loc_registry import register_loader
from peaks.core.metadata.base_metadata_models import (
    MomentumMicroscopeAnalyserMetadataModel,
    MomentumMicroscopeApertureModel,
    MomentumMicroscopeAperturesModel,
    MomentumMicroscopeMetadataModel,
)
from peaks.core.options import opts
from peaks.core.utils.misc import analysis_warning

ureg = pint_xarray.unit_registry


class DiamondNXSLoader(BaseHDF5DataLoader):
    """Helper class for parsing the NeXus file format used at Diamond Light Source."""

    _data_group_key_resolution_order = []
    _core_data_key_resolution_order = []

    @property
    def data_key_resolution_order(self):
        """The priority order of keys to specify the main data group in the file."""
        return self._data_group_key_resolution_order

    @staticmethod
    def get_root_key(f):
        """Get the root key of the file, which is assumed to be the only key in the file at root level.

        Parameters
        ----------
        f : h5py.File
            The h5py file object (should be open).

        Returns
        -------
        root_key : str
            The root key of the file.
        """
        if len(f.keys()) != 1:
            raise ValueError(
                "The file has multiple root keys, which is not expected for the standard Diamond NeXus file."
            )
        return next(iter(f.keys()))

    @classmethod
    def get_data_group_addr(cls, f, root_key=None):
        """Get the main data group from the file, with priority as per _data_key_resolution_order.

        Parameters
        ----------
        f : h5py.File
            The h5py file object (should be open).
        root_key : str, optional
            The root key of the file. If not provided, it is automatically determined as the only key at root level.

        Returns
        -------
        data_group_addr : str
            The full path of the main data group in the file.
        """
        if root_key is None:
            root_key = cls.get_root_key(f)

        # Get all keys within this root level
        scan_keys = f[root_key].keys()

        # Check for the main data group, with priority as per data_key_resolution_order
        data_group_key = None
        for key in cls._data_group_key_resolution_order:
            if key in scan_keys:
                data_group_key = key
                break
        else:
            # If no key from data_key_resolution_order is found, fall back to the first key in the group
            data_group_key = next(iter(scan_keys))

        data_group_addr = f"{root_key}/{data_group_key}"
        return data_group_addr

    @staticmethod
    def _parse_nxs_primary_attr(attr):
        """Parse the attribute value for a signal or primary key to a standard format."""
        if isinstance(attr, bytes):
            attr = attr.decode()

        if isinstance(attr, np.ndarray):
            if attr.size == 1:
                attr = attr.item()

        if isinstance(attr, str):
            if "," in attr:
                attr = [int(i) for i in attr.split(",")]
            else:
                attr = int(attr)
        elif isinstance(attr, list):
            attr = [int(i.decode()) if isinstance(i, bytes) else int(i) for i in attr]
            if len(attr) == 1:
                attr = attr[0]

        return attr

    @staticmethod
    def _parse_nxs_axis_attr(attr):
        """Parse the attribute value for an axis key to a standard format."""
        if isinstance(attr, (list, np.ndarray)):
            attr = attr[0]
        if isinstance(attr, bytes):
            attr = attr.decode()
        return attr.decode() if isinstance(attr, bytes) else attr

    @classmethod
    def get_core_data_key(cls, f, data_address):
        """Find the core data in an HDF5/nxs group. First attempts to resolve data by key located in
        `_core_data_key_resolution_order` class variable. If that fails, fall back to trying to automatically parse
        by looking for a signal attribute flag.

        Parameters
        ----------
        f : h5py.File
            The h5py file object (should be open).

        data_address : str
            The key of the core data in the `data_address` group of the HDF5 file
        """
        # Parse all data entries with a signal key
        data_group = f[data_address]

        # Check for key in the resolution order
        for key in cls._core_data_key_resolution_order:
            if key in data_group.keys():
                return key

        # Otherwise fall back to looking for a signal attribute
        signal_data_mappings = {
            key: data_group[key].attrs.get("signal")
            for key in data_group.keys()
            if "signal" in data_group[key].attrs
        }

        core_data_key = None
        if len(signal_data_mappings) == 1:
            core_data_key = next(iter(signal_data_mappings.keys()))
        elif len(signal_data_mappings) > 1:
            # Take the primary data based on the resolution order of the @signal attr
            core_data_key = min(
                signal_data_mappings,
                key=lambda k: cls._parse_nxs_primary_attr(signal_data_mappings[k]),
            )

        if core_data_key:
            return core_data_key

        # If no signal attribute found, raise an error
        raise ValueError(
            f"None of the expected data keys in {cls._core_data_key_resolution_order} found in the data, "
            f"and no signal attribute found to mark primary data in the group."
        )

    @staticmethod
    def _parse_start_time(f):
        """Return the tz-aware acquisition start time."""
        if "entry1/start_time" not in f:
            return None
        _raw_start_time = f["entry1/start_time"][()]
        _raw_start_time = (
            _raw_start_time.decode()
            if isinstance(_raw_start_time, bytes)
            else str(_raw_start_time)
        )
        start_time = parse(_raw_start_time)
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        return start_time

    @classmethod
    def _get_start_time(cls, fpath):
        """Parse the acquisition start time from the file at `fpath`."""
        with h5py.File(fpath, "r") as f:
            return cls._parse_start_time(f)

    @staticmethod
    def _trim_sweep_artefacts(da, dim):
        """Trim points at either end of a fly-scan axis where the motor was still
        accelerating or decelerating, leaving non-monotonic/repeated coordinate values."""
        val = np.asarray(da[dim].values, dtype=float)
        if val.size < 3:
            return da
        steps = np.diff(val)
        median_step = np.median(steps)
        if not np.isfinite(median_step) or median_step == 0:
            return da
        threshold = 0.1 * np.abs(median_step)  # for motor noise
        direction = np.sign(median_step)
        moving = np.flatnonzero(
            (np.abs(steps) > threshold) & (np.sign(steps) == direction)
        )
        if not moving.size:
            return da
        first_good = int(moving[0])
        last_good = int(moving[-1]) + 1
        n_trim = val.size - (last_good - first_good + 1)
        if n_trim:
            da = da.isel({dim: slice(first_good, last_good + 1)})
            analysis_warning(
                f"Non-monotonic values detected in `{dim}` at the start and/or end of "
                f"travel. Trimmed {n_trim} data point(s).",
                "warning",
                "Removed non-monotonic data",
            )
        return da


@register_loader
class I05ARPESLoader(DiamondNXSLoader, BaseARPESDataLoader):
    """Loader for Diamond I05 HR-ARPES NeXus files."""

    _loc_name = "Diamond_I05_ARPES"
    _loc_description = "HR-ARPES branch line of I05 beamline at Diamond Light Source"
    _loc_url = "https://www.diamond.ac.uk/Instruments/Structures-and-Surfaces/I05.html"
    _analyser_slit_angle = 0 * ureg.deg

    _manipulator_name_conventions = {
        # HR-branch
        "polar": "sapolar",
        "tilt": "satilt",
        "azi": "saazimuth",
        "x1": "sax",
        "x2": "saz",
        "x3": "say",
    }

    _manipulator_sign_conventions = {
        # HR-branch
        "tilt": -1,
        "azi": -1,
        "x1": -1,
    }

    _analyser_name_conventions = {
        "deflector_perp": "deflector_x",  # These two don't exist for R4000 but fine for now
        "deflector_parallel": "deflector_y",  # Should handle them properly when we do the refactor. This was called detector_y at least before July 2026
        "eV": ["energies", "kinetic_energy_center"],
        "theta_par": "angles",
        "hv": ["energy", "value"],
    }

    _hdf5_metadata_key_mappings = {
        "scan_command": "entry1/scan_command",
        "manipulator_polar": "entry1/instrument/manipulator/sapolar",
        "manipulator_tilt": "entry1/instrument/manipulator/satilt",
        "manipulator_azi": "entry1/instrument/manipulator/saazimuth",
        "manipulator_x1": "entry1/instrument/manipulator/sax",
        "manipulator_x2": [
            "entry1/instrument/saz_fly/value",  # fly-scan path must come first
            "entry1/instrument/manipulator/saz",
        ],
        "manipulator_x3": "entry1/instrument/manipulator/say",
        "manipulator_salong": "entry1/instrument/manipulator/salong",
        "analyser_model": lambda f: (
            None
            if (ac_date := DiamondNXSLoader._parse_start_time(f)) is None
            else "FIXED_VALUE:MBS A1"
            if ac_date > I05ARPESLoader._hr_analyser_upgrade_date
            else "FIXED_VALUE:Scienta R4000"
        ),
        "analyser_slit_width": lambda f: (  # If the slit size is 0, assume it's not set and return None, otherwise return the value.
            None
            if (
                v := BaseHDF5DataLoader._extract_hdf5_value(
                    f, "entry1/instrument/analyser/entrance_slit_size"
                )
            )
            == 0
            else v
        ),
        "analyser_slit_width_identifier": [
            lambda f: (
                None  # If the slit size is 0 which is physcially impossible, assume it's not set and return None
                if BaseHDF5DataLoader._extract_hdf5_value(
                    f, "entry1/instrument/analyser/entrance_slit_size"
                )
                == 0
                else "entry1/instrument/analyser/entrance_slit_setting"  # otherwise return it
            ),
            lambda f: (
                None
                if BaseHDF5DataLoader._extract_hdf5_value(
                    f, "entry1/instrument/analyser_total/entrance_slit_size"
                )
                == 0
                else "entry1/instrument/analyser_total/entrance_slit_setting"
            ),
        ],
        "analyser_eV": "entry1/instrument/analyser/energies",
        "analyser_step_size": lambda f: (
            np.ptp(f["entry1/instrument/analyser/energies"])
            / (max(f["entry1/instrument/analyser/energies"].shape) - 1)
            if "entry1/instrument/analyser/energies" in f
            else None
        ),
        "analyser_PE": "entry1/instrument/analyser/pass_energy",
        "analyser_sweeps": [
            "entry1/instrument/analyser/number_of_iterations",
            "entry1/instrument/analyser/number_of_cycles",
            "entry1/instrument/analyser/number_of_frames",
        ],
        "analyser_dwell": "entry1/instrument/analyser/time_for_frames",
        "analyser_lens_mode": "entry1/instrument/analyser/lens_mode",
        "analyser_acquisition_mode": "entry1/instrument/analyser/acquisition_mode",
        "analyser_eV_type": "FIXED_VALUE:kinetic",
        "analyser_deflector_parallel": lambda f: (
            "entry1/instrument/analyser/deflector_y"
            if "entry1/instrument/analyser/deflector_y" in f
            else "entry1/instrument/analyser/detector_y"
            if (ac_date := DiamondNXSLoader._parse_start_time(f))
            and ac_date > I05ARPESLoader._hr_analyser_upgrade_date
            else None
        ),
        "analyser_deflector_perp": lambda f: (
            (
                "entry1/instrument/deflector_x/deflector_x"
                if "entry1/instrument/deflector_x/deflector_x" in f
                else "entry1/instrument/analyser/deflector_x"
            )
            if (ac_date := DiamondNXSLoader._parse_start_time(f))
            and ac_date > I05ARPESLoader._hr_analyser_upgrade_date
            else None
        ),
        "temperature_sample": "entry1/sample/temperature",
        "temperature_cryostat": "entry1/sample/cryostat_temperature",
        "temperature_shield": "entry1/sample/shield_temperature",
        "temperature_setpoint": "entry1/sample/temperature_demand",
        "temperature_heater_power": "entry1/sample/heater_percent",
        "photon_hv": "entry1/instrument/monochromator/energy",
        "photon_polarisation": [
            "entry1/instrument/insertion_device/beam/final_polarisation_label",
            "entry1/instrument/insertion_device/beam/final_polarisation_label\n\t\t\t\t\t\t\t\t",
        ],
        "photon_exit_slit": "entry1/instrument/monochromator/exit_slit_size",
        "timestamp": "entry1/start_time",
    }

    _hdf5_metadata_fixed_units = {
        "entry1/sample/heater_percent": "%",
        "entry1/instrument/analyser/deflector_x": "deg",  # Had to pin this to deg to get the k conv working
        "entry1/instrument/deflector_x/deflector_x": "deg",  # Note, the unit should be mm for a 2D transmission-mode scan but fine for now for metadata display
        "entry1/instrument/analyser/detector_y": "deg",  # Need to revisit this during the refactor
        "entry1/instrument/analyser/deflector_y": "deg",
    }

    _data_group_key_resolution_order = ["analyser"]
    _core_data_key_resolution_order = ["analyser"]

    _hr_analyser_upgrade_date = datetime(2021, 7, 1, tzinfo=timezone.utc)  # R4000 --> A1

    @classmethod
    def _load_data(cls, fpath, lazy, **kwargs):
        """Load the data from a Diamond I05 ARPES NeXus file. Try to automatically determine the core scan structure."""
        with h5py.File(fpath, "r") as f:
            # Parse the data group structure
            data_group_addr = cls.get_data_group_addr(f)
            # Find the signal (i.e. core) data from the signal key
            core_data_key = cls.get_core_data_key(f, data_group_addr)

            # Sort remaining data entries based on these axis and primary keys
            dim_names = set(f[data_group_addr].keys()) - {core_data_key}
            dim_name_to_axis_mapping = {
                dim: cls._parse_nxs_axis_attr(
                    f[f"{data_group_addr}/{dim}"].attrs.get("axis")
                )
                for dim in dim_names
            }

            # Parse the possible dimensions for each axis, excluding current which is always a `metadata` entry
            axis_to_dim_name_mapping = {
                axis: [
                    dim
                    for dim, axis2 in dim_name_to_axis_mapping.items()
                    if (axis == axis2 and dim.lower() != "current")
                ]
                for axis in set(dim_name_to_axis_mapping.values())
            }
            # Sort these based on their primary attribute
            # Get rid of attributes that don't have an axis link
            axis_to_dim_name_mapping.pop(None, None)
            axis_to_dim_name_mapping_sorted = {}
            for axis, dims in axis_to_dim_name_mapping.items():
                axis_to_dim_name_mapping_sorted[axis] = sorted(
                    dims,
                    key=lambda k: int(
                        cls._parse_nxs_primary_attr(
                            f[f"{data_group_addr}/{k}"].attrs.get("primary", 99)
                        )
                    ),
                )

                # Handle edge cases ############################################
                # ana_polar scan via dummy motor (I05 Nano)
                if axis_to_dim_name_mapping_sorted[axis] == [
                    "dummy_motor",
                    "analyser_polar_angle",
                ]:
                    axis_to_dim_name_mapping_sorted[axis] = [
                        "analyser_polar_angle",
                        "dummy_motor",
                    ]
                # Secondary spatial axis in a polar map
                if "sapolar" in axis_to_dim_name_mapping_sorted[axis]:
                    other_dim = list(
                        set(axis_to_dim_name_mapping_sorted[axis]) - set(["sapolar"])
                    )
                    if len(other_dim) == 1 and other_dim[0] in ["sax", "say", "salong"]:
                        axis_to_dim_name_mapping_sorted[axis] = [
                            "sapolar",
                            other_dim[0],
                        ]
                # Fly scan - both motors carry primary=1 with axis '1,2'; fly should be axis 2
                dims_here = axis_to_dim_name_mapping_sorted[axis]
                if len(dims_here) == 2:
                    fly = [
                        d
                        for d in dims_here
                        if "_fly"
                        in str(f[f"{data_group_addr}/{d}"].attrs.get("target", b""))
                    ]
                    if len(fly) == 1:
                        step = next(d for d in dims_here if d != fly[0])
                        axis_to_dim_name_mapping_sorted[axis] = [step, fly[0]]
                # hv scan
                if "energy" in axis_to_dim_name_mapping_sorted[axis]:
                    other_dim = list(
                        set(axis_to_dim_name_mapping_sorted[axis]) - set(["energy"])
                    )
                    axis_to_dim_name_mapping_sorted[axis] = ["energy"]
                    axis_to_dim_name_mapping_sorted[axis].extend(other_dim)

            # Define the primary dims and coords
            data_ndim = f[f"{data_group_addr}/{core_data_key}"].ndim
            dim_mapping = {}
            coords = {}
            for axis, dims in axis_to_dim_name_mapping_sorted.items():
                if "," in axis:
                    axes = axis.split(",")
                    n_axes = len(axes)
                    for i in range(n_axes):
                        dim_mapping[axes[i]] = dims[i]
                        indexer = [0] * n_axes
                        indexer[i] = slice(None)
                        coords[dims[i]] = f[f"{data_group_addr}/{dims[i]}"][
                            tuple(indexer)
                        ]
                elif len(dims) > 0:
                    dim_mapping[axis] = dims[0]
                    coords[dims[0]] = f[f"{data_group_addr}/{dims[0]}"][()].squeeze()
            dims = [dim_mapping.get(str(i + 1), "dummy") for i in range(data_ndim)]
            units = {}
            for dim in dims:
                try:
                    units[dim] = f[f"{data_group_addr}/{dim}"].attrs.get("units")
                except KeyError:
                    pass
            units["spectrum"] = f[f"{data_group_addr}/{core_data_key}"].attrs.get(
                "units", "counts"
            )
            units = {
                key: (cls._parse_nxs_axis_attr(value)) for key, value in units.items()
            }

            # July 2026: Both hv scans and fast fly-scan spatial maps
            # now expose a dataset called 'value'. Renaming.
            fly_motor = None
            if "value" in dims:
                target = cls._parse_nxs_axis_attr(
                    f[f"{data_group_addr}/value"].attrs.get("target", b"")
                )
                parts = target.strip("/").split("/")
                motor = parts[-2] if len(parts) >= 2 and "_fly" in parts[-2] else ""

                known_motors = set(cls._manipulator_name_conventions.values())
                known_motors.discard(None)  # No tilt on nano

                true_name = motor.removesuffix("_fly")
                if (
                    true_name in known_motors
                ):  # should be a spatial map rather than a hv scan
                    fly_motor = true_name

                    dims = [true_name if d == "value" else d for d in dims]
                    if "value" in coords:
                        coords[true_name] = coords.pop("value")
                    if "value" in units:
                        units[true_name] = units.pop("value")
                        if units.get(true_name) is None:
                            root_data_group = data_group_addr.split("/")[0]
                            true_motor_addr = (
                                f"{root_data_group}/instrument/manipulator/{true_name}"
                            )
                            if (
                                true_motor_addr in f
                                and "units" in f[true_motor_addr].attrs
                            ):
                                units[true_name] = cls._parse_nxs_axis_attr(
                                    f[true_motor_addr].attrs["units"]
                                )
                elif motor:
                    analysis_warning(
                        f"This fly scan cannot be recognised due to an unknown axis: `{motor}`.",
                        "warning",
                        "Unrecognised fly scan",
                    )

        # Load the core data in a way that supports lazy loading
        ds = xr.open_dataset(
            fpath,
            group=data_group_addr,
            engine="h5netcdf",
            phony_dims="sort",
            chunks="auto",
        )
        # Map local dimension names keeping I05 convention for now
        ds = ds.rename(
            {orig: new for orig, new in zip(ds[core_data_key].dims, dims, strict=True)}
        )

        # Get the core data
        da = ds[core_data_key]
        da.attrs = {}

        # Apply the coordinates
        coords_to_apply = {dim: coords.get(dim) for dim in dims if dim != "dummy"}
        # Handle the special case of an hv scan, where the kinetic energy is 2D
        for dim, coord in coords_to_apply.copy().items():
            if coord.ndim == 2 and not np.array_equal(
                coords_to_apply["energies"][0], coords_to_apply["energies"][1]
            ):
                # Should be a 2D array with shape (value, KE) where value is the changing hv dim
                hv_coord_label = {"value", "energy"}.intersection(
                    set(coords_to_apply.keys())
                ).pop()

                if coords_to_apply.get(hv_coord_label).shape[0] == coord.shape[0]:
                    coords_to_apply["KE_delta"] = (
                        hv_coord_label,
                        coord[:, 0] - coord[0, 0],
                    )
                    coords_to_apply[dim] = coord[0]
                else:
                    analysis_warning(
                        f"Struggling to parse dimensions of {dim}. If this is a hv scan, may need to "
                        f"manually add a 'KE_delta' dimension.",
                        "warning",
                        "Unexpected data shape",
                    )
            elif coord.ndim == 2 and np.array_equal(
                coords_to_apply["energies"][0], coords_to_apply["energies"][1]
            ):
                # Deal with deflector maps with the energy axis somehow to be 2D
                coords_to_apply[dim] = coord[0]
        da = da.assign_coords(coords_to_apply)

        # Attach the I0 monitor as a coordinate, for normalisation with `norm_by_I0`
        if "current" in dim_name_to_axis_mapping:
            I0 = ds["current"]
            # `current` is the storage ring current (links to /entry1/instrument/source/current, in mA)
            units["I0"] = cls._parse_nxs_axis_attr(I0.attrs.get("units", "mA"))
            da = da.assign_coords(I0=(I0.dims, I0.data))

        # Rename to peaks conventions using _manipulator_name_conventions and _analyser_name_conventions
        dim_names_to_update = {}
        names_to_check = cls._manipulator_name_conventions.copy()
        names_to_check.update(cls._analyser_name_conventions.copy())
        for peaks_name, i05_names in names_to_check.items():
            if isinstance(i05_names, str):
                if i05_names in dims:
                    dim_names_to_update[i05_names] = peaks_name
                    units[peaks_name] = units.pop(i05_names)
            elif isinstance(i05_names, list):
                for i05_name in i05_names:
                    if i05_name in dims:
                        dim_names_to_update[i05_name] = peaks_name
                        units[peaks_name] = units.pop(i05_name)
                        break
        da = da.rename(dim_names_to_update)

        # EDGE CASE - Handle problem with ana_polar data sometimes having non-monotonic/repeated points at end of range
        # Trim back to the last point where the axis was genuinely moving.
        if "ana_polar" in da.dims:
            da = cls._trim_sweep_artefacts(da, "ana_polar")

        # Edge case for manipulator translation motors in fly spatial mapping
        fly_dim = dim_names_to_update.get(fly_motor, fly_motor)
        if fly_dim in da.dims:
            da = cls._trim_sweep_artefacts(da, fly_dim)

        # Load array into memory if lazy loading not required
        if not (lazy or (lazy is None and da.size > opts.FileIO.lazy_size)):
            da = da.compute()

        # Add units where available
        da.name = "spectrum"
        da = da.pint.quantify(units)
        return da.squeeze()


@register_loader
class I05NanoARPESLoader(I05ARPESLoader, BaseOpticsDataLoader):
    """Loader for Diamond I05 nano-ARPES NeXus files."""

    _loc_name = "Diamond_I05_Nano-ARPES"
    _loc_description = "Nano-ARPES branch line of I05 beamline at Diamond Light Source"
    _loc_url = "https://www.diamond.ac.uk/Instruments/Structures-and-Surfaces/I05.html"
    _analyser_slit_angle = 0 * ureg.deg

    _manipulator_axes = ["polar", "tilt", "azi", "x1", "x2", "x3", "defocus"]
    _manipulator_name_conventions = {
        "polar": "smpolar",
        "tilt": None,
        "azi": "smazimuth",
        "x1": "smx",
        "x2": "smy",
        "x3": "smz",
        "defocus": "smdefocus",
    }
    _manipulator_sign_conventions = {
        # Nano-branch
        "azi": -1,
        "x1": -1,
        "x3": -1,
    }

    _analyser_sign_conventions = {  # This will be overridden if the analyser upgrade is detected see class I05NanoNewARPESLoader
        "theta_par": -1,
        "deflector_parallel": -1,  # consistent with theta_par
    }

    _desired_dim_order = [
        "scan_no",
        "hv",
        "temperature_sample",
        "temperature_cryostat",
        "x3",
        "x2",
        "x1",
        "ana_polar",
        "polar",
        "tilt",
        "azi",
        "y_scale",
        "deflector_perp",
        "eV",
        "deflector_parallel",
        "theta_par",
    ]

    _analyser_name_conventions = {
        "deflector_perp": "ThetaY",  # This one and the one below will be overridden if the analyser upgrade is detected
        "deflector_parallel": "ThetaX",  # see class I05NanoNewARPESLoader
        "eV": ["energies", "kinetic_energy_center"],
        "theta_par": "angles",
        "hv": ["energy", "value"],
        "ana_polar": "analyser_polar_angle",
    }

    _optics_name_conventions = {
        "x1": "ZPx",
        "x2": "ZPy",
        "x3": "ZPz",
    }
    _optics_additional_name_conventions = {
        "OSA_x1": "OSAx",
        "OSA_x2": "OSAy",
        "OSA_x3": "OSAz",
    }

    _hdf5_metadata_key_mappings = {
        "scan_command": "entry1/scan_command",
        "manipulator_polar": "entry1/instrument/manipulator/smpolar",
        "manipulator_tilt": "entry1/instrument/manipulator/smtilt",
        "manipulator_azi": "entry1/instrument/manipulator/smazimuth",
        "manipulator_x1": "entry1/instrument/manipulator/smx",
        "manipulator_x2": [
            "entry1/instrument/smy_fly/value",  # fly-scan path must come first
            "entry1/instrument/manipulator/smy",
        ],
        "manipulator_x3": "entry1/instrument/manipulator/smz",
        "manipulator_defocus": "entry1/instrument/manipulator/smdefocus",
        "analyser_model": "FIXED_VALUE:Scienta DA30",  # Overridden in I05NanoNewARPESLoader to handle analyser upgrade
        "analyser_slit_width": [  # If the slit size is 0, assume it's not set and return None, otherwise return the value.
            lambda f: (
                None
                if (
                    v := BaseHDF5DataLoader._extract_hdf5_value(
                        f, "entry1/instrument/analyser/entrance_slit_size"
                    )
                )
                == 0
                else v
            ),
            lambda f: (
                None
                if (
                    v := BaseHDF5DataLoader._extract_hdf5_value(
                        f, "entry1/instrument/analyser_total/entrance_slit_size"
                    )
                )
                == 0
                else v
            ),
        ],
        "analyser_slit_width_identifier": [
            lambda f: (
                None  # If the slit size is 0 which is physcially impossible, assume it's not set and return None
                if BaseHDF5DataLoader._extract_hdf5_value(
                    f, "entry1/instrument/analyser/entrance_slit_size"
                )
                == 0
                else "entry1/instrument/analyser/entrance_slit_setting"  # otherwise return it
            ),
            lambda f: (
                None
                if BaseHDF5DataLoader._extract_hdf5_value(
                    f, "entry1/instrument/analyser_total/entrance_slit_size"
                )
                == 0
                else "entry1/instrument/analyser_total/entrance_slit_setting"
            ),
        ],
        "analyser_eV": [
            "entry1/instrument/analyser/energies",
            "entry1/instrument/analyser_total/energies",
            "entry1/instrument/analyser/kinetic_energy_center",
            "entry1/instrument/analyser_total/kinetic_energy_center",
        ],
        "analyser_step_size": [
            lambda f: (
                np.ptp(f["entry1/instrument/analyser/energies"])
                / (max(f["entry1/instrument/analyser/energies"].shape) - 1)
                if "entry1/instrument/analyser/energies" in f
                else None
            ),
            lambda f: (
                np.ptp(f["entry1/instrument/analyser_total/energies"])
                / (max(f["entry1/instrument/analyser_total/energies"].shape) - 1)
                if "entry1/instrument/analyser_total/energies" in f
                else None
            ),
            "entry1/instrument/analyser/kinetic_energy_step",
            "entry1/instrument/analyser_total/kinetic_energy_step",
        ],
        "analyser_PE": [
            "entry1/instrument/analyser/pass_energy",
            "entry1/instrument/analyser_total/pass_energy",
        ],
        "analyser_sweeps": [
            "entry1/instrument/analyser/number_of_iterations",
            "entry1/instrument/analyser/number_of_cycles",
            "entry1/instrument/analyser/number_of_frames",
            "entry1/instrument/analyser_total/number_of_iterations",
            "entry1/instrument/analyser_total/number_of_cycles",
            "entry1/instrument/analyser_total/number_of_frames",
        ],
        "analyser_dwell": [
            "entry1/instrument/analyser/time_for_frames",
            "entry1/instrument/analyser_total/time_for_frames",
        ],
        "analyser_lens_mode": [
            "entry1/instrument/analyser/lens_mode",
            lambda f: (
                f"FIXED_VALUE:{BaseHDF5DataLoader._extract_hdf5_value(f, 'entry1/instrument/analyser_total/lens_mode')}"
                f"_analyser_total"
                if "entry1/instrument/analyser_total/lens_mode" in f
                else None
            ),
        ],
        "analyser_acquisition_mode": [
            "entry1/instrument/analyser/acquisition_mode",
            "entry1/instrument/analyser_total/acquisition_mode",
        ],
        "analyser_eV_type": "FIXED_VALUE:kinetic",
        "analyser_deflector_parallel": None,  # Overridden in I05NanoNewARPESLoader to handle analyser upgrade
        "analyser_deflector_perp": None,  # Overridden in I05NanoNewARPESLoader to handle analyser upgrade
        "analyser_polar": [
            "entry1/instrument/analyser/analyser_polar_angle",
            "entry1/instrument/analyser_total/analyser_polar_angle",
        ],
        "temperature_sample": "entry1/sample/temperature",
        "temperature_cryostat": "entry1/sample/cryostat_temperature",
        "temperature_shield": "entry1/sample/shield_temperature",
        "temperature_setpoint": "entry1/sample/temperature_demand",
        "temperature_heater_power": "entry1/sample/heater_percent",
        "photon_hv": "entry1/instrument/monochromator/energy",
        "photon_polarisation": [
            "entry1/instrument/insertion_device/beam/final_polarisation_label",
            "entry1/instrument/insertion_device/beam/final_polarisation_label\n\t\t\t\t\t\t\t\t",
        ],
        "photon_exit_slit": "entry1/instrument/monochromator/exit_slit_size",
        "timestamp": "entry1/start_time",
        "optics_OSA_x1": "entry1/instrument/order_sorting_aperture/osax",
        "optics_OSA_x2": "entry1/instrument/order_sorting_aperture/osay",
        "optics_OSA_x3": "entry1/instrument/order_sorting_aperture/osaz",
        "optics_x1": "entry1/instrument/zone_plate/zpx",
        "optics_x2": "entry1/instrument/zone_plate/zpy",
        "optics_x3": "entry1/instrument/zone_plate/zpz",
    }

    _hdf5_metadata_fixed_units = {
        "entry1/sample/heater_percent": "%",
    }

    _data_group_key_resolution_order = ["analyser", "analyser_total"]
    _core_data_key_resolution_order = ["analyser", "analyser_total"]

    _metadata_parsers = [
        "_parse_analyser_metadata",
        "_parse_manipulator_metadata",
        "_parse_photon_metadata",
        "_parse_temperature_metadata",
        "_parse_optics_metadata",
    ]  # List of metadata parsers to apply

    # Handle nano analyser upgrade
    _nano_analyser_upgrade_date = datetime(
        2026, 4, 1, tzinfo=timezone.utc
    )  # DA30 --> A1

    @classmethod
    def _load(cls, fpath, lazy, metadata, quiet, **kwargs):
        """Load the data from Diamond I05 Nano ARPES."""
        if fpath.split(".")[-1] == "zip":
            # Load the data using the SES loader for .zip files
            return cls.load(fpath, loc="SES", lazy=lazy, metadata=metadata)

        # Choose analyser sign and name conventions for deflector based on acquisition date i.e. the analyser used at the time
        # If data are acquired after the analyser upgrade, assign the new analyser sign and name conventions
        if cls is I05NanoARPESLoader:
            start_time = cls._get_start_time(fpath)
            if start_time is not None and start_time > cls._nano_analyser_upgrade_date:
                return cls.load(
                    fpath,
                    loc="Diamond_I05_Nano-ARPES_New",
                    lazy=lazy,
                    metadata=metadata,
                    quiet=quiet,
                    **kwargs,
                )

        # Load the data
        data = super()._load(fpath, lazy, metadata, quiet, **kwargs)

        # Perform post processing on load if required
        if kwargs.pop("slant_correct", False):  # Perform the slant correction
            data = cls.slant_correct(data, kwargs.get("slant_factor", None))

        return data

    @register_accessor(xr.DataArray)
    @staticmethod
    def slant_correct(data, slant_factor=None):
        """Function to remove a slant that was present in data obtained using the Scienta DA30 (9ES210) analyser at the nano
        branch of the I05 beamline at Diamond Light Source in 2021/22.

        Parameters
        ----------
        data : xarray.DataArray
            The data to be corrected.

        slant_factor : float, optional
            The slant factor correction (degrees/eV) to use. Defaults to 8/PE (where PE is the pass energy used).

        Returns
        -------
        corrected_data : xarray.DataArray
            The corrected data.

        Examples
        --------
        Example usage is as follows::

            from peaks import pks

            # Load the data, incorporating slant correction with a custom factor
            disp = pks.load('i05-1-13579.ibw', slant_correct=True, slant_factor=7.9)

            # Load data and apply slant correction afterwards
            disp = pks.load('i05-1-13579.ibw')
            disp_sc = disp.slant_correct()
        """
        # Set the slant factor to the default correction if it has not been supplied
        if slant_factor is None:
            slant_factor = 8 * ureg.deg / data.metadata.analyser.scan.PE.to("eV")
        elif not isinstance(slant_factor, pint.Quantity):
            slant_factor = slant_factor * ureg.deg / ureg.eV
        else:
            slant_factor = slant_factor.to("deg/eV")

        # Define the new angle mapping
        theta_par_values = data.theta_par.pint.to("deg") - (
            slant_factor.magnitude
            * (data.eV.pint.to("eV") - data.eV.pint.to("eV").median())
        )

        # Perform the interpolation onto the corrected grid
        corrected_data = (
            data.pint.dequantify()
            .interp({"theta_par": theta_par_values, "eV": data.eV})
            .pint.quantify()
        )

        # Update the analysis history
        corrected_data.history.add(
            "Slant correction for Diamond nano-ARPES data applied: {factor}".format(
                factor=slant_factor
            )
        )

        return corrected_data


@register_loader
class I05NanoNewARPESLoader(I05NanoARPESLoader):
    """Nano branch of I05 after the 2026 analyser upgrade."""

    _loc_name = "Diamond_I05_Nano-ARPES_New"
    _analyser_sign_conventions = {
        "theta_par": 1,
        "deflector_parallel": 1,  # consistent with theta_par
        "deflector_perp": -1,
    }
    _analyser_name_conventions = {
        **I05NanoARPESLoader._analyser_name_conventions,
        "deflector_perp": "deflector_x",
        "deflector_parallel": "deflector_y",
    }

    _hdf5_metadata_key_mappings = {
        **I05NanoARPESLoader._hdf5_metadata_key_mappings,
        "analyser_model": "FIXED_VALUE:MBS A1",
        "analyser_deflector_parallel": "entry1/instrument/analyser/deflector_y",
        "analyser_deflector_perp": "entry1/instrument/analyser/deflector_x",
    }

    _hdf5_metadata_fixed_units = {
        **I05NanoARPESLoader._hdf5_metadata_fixed_units,
        "entry1/instrument/analyser/deflector_x": "deg",  # Note, the unit should be mm for a 2D transmission-mode scan but fine for now for metadata display
        "entry1/instrument/analyser/deflector_y": "deg",  # Need to revisit this during the refactor
    }


def _read_szip_chunk(fpath, dset_path, block_info=None):
    """Read one szip-compressed HDF5 chunk and decode it with `imagecodecs`, for h5py builds without szip."""
    import imagecodecs

    with h5py.File(fpath, "r") as f:
        dset = f[dset_path]
        _, _, (mask, ppb, bpp, pps), _ = dset.id.get_create_plist().get_filter(0)
        start = tuple(start for start, _ in block_info[None]["array-location"])
        # Frames never written (e.g. an aborted scan) read as the fill value, matching h5py
        if dset.id.get_chunk_info_by_coord(start).byte_offset is None:
            return np.full(block_info[None]["chunk-shape"], dset.fillvalue, dset.dtype)
        filter_mask, raw = dset.id.read_direct_chunk(start)
        chunk_shape, dtype = dset.chunks, dset.dtype
    # A set filter mask bit means HDF5 stored this chunk uncompressed
    if not filter_mask:
        raw = imagecodecs.szip_decode(raw, mask, ppb, bpp, pps, header=True)
    chunk = np.frombuffer(raw, dtype).reshape(chunk_shape)
    # Edge chunks are stored padded to the full chunk shape
    return chunk[tuple(slice(0, n) for n in block_info[None]["chunk-shape"])]


def _i09_detector_addr(f):
    """Address of the instrument group flagged with scan_role=detector (e.g. `dld` or `dld_total`), or None."""
    return next(
        (
            f"entry/instrument/{key}"
            for key, group in f["entry/instrument"].items()
            if DiamondNXSLoader._parse_nxs_axis_attr(group.attrs.get("scan_role", ""))
            == "detector"
        ),
        None,
    )


def _i09_energy_addr(f):
    """Address of the recorded energy: the kinetic energy set point (`Kenergy`) or readback (`kenergy`), else the
    binding energy (`Benergy`, from a predefined work function), e.g. for hv scans at constant binding energy."""
    return next(
        (
            f"entry/instrument/{name}/value"
            for name in ["Kenergy", "kenergy", "Benergy"]
            if f"entry/instrument/{name}/value" in f
        ),
        None,
    )


def _i09_accumulation_time(f):
    """Total accumulation time per point, from the detector argument in the scan command (e.g. `dld 20`).

    The detector's own `count_time` is the exposure per frame, not the accumulation time per point.
    """
    det_addr = _i09_detector_addr(f)
    if not det_addr:
        return None
    scan_command = BaseHDF5DataLoader._extract_hdf5_value(
        f, "entry/diamond_scan/scan_command"
    )
    m = re.search(rf"\b{det_addr.rsplit('/', 1)[1]}\s+([\d.eE+-]+)", str(scan_command))
    return float(m.group(1)) * ureg.s if m else None


@register_loader
class I09MomentumMicroscopeLoader(DiamondNXSLoader, BaseARPESDataLoader):
    """Loader for Diamond I09-2 (J-branch) momentum microscope NeXus files.

    The delay-line detector (DLD) image is kept in raw detector pixels (`detector_y`, `detector_x`); no
    momentum calibration is stored in the file. The analyser slit angle is not meaningful for this detector.

    Flat-field correction is typically required after loading data. Load the raw flat-field image, specifying
    this loc, then apply it, e.g.
    ``data.flat_field_correction(pks.load('Flatfield.tif', loc='Diamond_I09_Momentum_Microscope'))``.
    """

    _loc_name = "Diamond_I09_Momentum_Microscope"
    _loc_description = "Momentum microscope on the J-branch (I09-2) of the I09 beamline at Diamond Light Source"
    _loc_url = "https://www.diamond.ac.uk/Instruments/Structures-and-Surfaces/I09.html"

    # Kenergy is the kinetic energy (= sample bias); jenergy is the J-branch photon energy, in keV
    _analyser_name_conventions = {"eV": "Kenergy", "hv": "jenergy"}
    # Other energy scannables also loaded as eV: kenergy (kinetic readback), Benergy (binding energy, positive)
    _eV_scannables = ["kenergy", "Benergy"]
    # Detector axes ordered like (deflector_perp, eV, theta_par), so disp shows constant energy maps top-left
    _desired_dim_order = [
        *BaseARPESDataLoader._desired_dim_order[
            : BaseARPESDataLoader._desired_dim_order.index("eV")
        ],
        "detector_y",
        "eV",
        "detector_x",
    ]
    # No compound rotation/translation axes yet: record the physical motors. sx1-3 move the sample towards
    # the microscope (combined: focus, polar, tilt); sz1-2 across its view (combined: z, azi). es3x (whole
    # end-station) is not a sample motion, so omitted
    _i09_motors = ["sx1", "sx2", "sx3", "sz1", "sz2"]
    _manipulator_axes = ["polar", "tilt", "azi", "x1", "x2", "x3", *_i09_motors]
    _manipulator_name_conventions = {motor: motor for motor in _i09_motors}
    _scannable_units = {
        "Kenergy": "eV",
        "kenergy": "eV",
        "Benergy": "eV",
        "jenergy": "keV",
    }
    # Drain currents (Femto amplifier channels, e.g. sd11iamp7) are read as the amplifier output voltage (no units
    # in the file). The current is value / gain [V/A], but the gain is not recorded, so they are kept in V
    # panda_scannable: PandA mean of the sample drain amplifier output over the counting window (matches sd11iamp7)
    # assumes PandA is always wired to the sample drain amplifier
    _drain_current_pattern = r"s[a-z]\d+iamp\d+|panda_scannable"
    _drain_current_units = "V"
    _I0_monitor = "sm6iamp27"  # SM6 mirror drain current
    # Only warned for the scan types they apply to, see _load_metadata
    _analyser_include_in_metadata_warn = ["eV", "step_size", "dwell"]
    # Sample temperature not yet recorded at I09-2 (placeholder string in the file)
    _temperature_exclude_from_metadata_warn = BaseARPESDataLoader._temperature_attributes
    _hdf5_metadata_fixed_units = {
        f"entry/instrument/{name}/value": "eV"
        for name in ["Kenergy", "kenergy", "Benergy"]
    }

    _hdf5_metadata_key_mappings = {
        "scan_command": "entry/diamond_scan/scan_command",
        **{
            f"manipulator_{motor}": f"entry/instrument/{motor}/value"
            for motor in _i09_motors
        },
        # Schmitt et al., arXiv:2406.00771
        "analyser_model": "FIXED_VALUE:SPECS Phoibos 225 hybrid (hemisphere + ToF) momentum microscope",
        # Round hemisphere apertures, e.g. "D0.5" = 0.5 mm diameter
        "analyser_slit_width": lambda f: (
            float(m.group(1)) * ureg.mm
            if (
                m := re.fullmatch(
                    r"D([\d.]+)",
                    str(
                        BaseHDF5DataLoader._extract_hdf5_value(
                            f, "entry/instrument/analyser_slit/value"
                        )
                    ),
                )
            )
            else None
        ),
        "analyser_slit_width_identifier": "entry/instrument/analyser_slit/value",
        # Kinetic energy, set by sweeping the (positive) sample bias, or the binding energy if only that is recorded
        # Benergy is positive below EF, so flip to the peaks E-EF convention
        "analyser_eV": lambda f: (
            -I09MomentumMicroscopeLoader._extract_hdf5_value(f, addr)
            if (addr := _i09_energy_addr(f)) and "Benergy" in addr
            else addr
        ),
        # Only from the set point: the readbacks jitter even at fixed energy
        "analyser_step_size": lambda f: (
            np.ptp(f["entry/instrument/Kenergy/value"])
            / (f["entry/instrument/Kenergy/value"].size - 1)
            if "entry/instrument/Kenergy/value" in f
            and f["entry/instrument/Kenergy/value"].size > 1
            else None
        ),
        "analyser_dwell": lambda f: _i09_accumulation_time(f),
        "analyser_eV_type": lambda f: (
            "FIXED_VALUE:binding"
            if (addr := _i09_energy_addr(f)) and "Benergy" in addr
            else "FIXED_VALUE:kinetic"
        ),
        "analyser_fourier_aperture": "entry/instrument/microscope/fourier_select",
        "analyser_fourier_aperture_x": "entry/instrument/microscope/x_fourier",
        "analyser_fourier_aperture_y": "entry/instrument/microscope/y_fourier",
        "analyser_gaussian_aperture": "entry/instrument/microscope/gaussian_select",
        "analyser_gaussian_aperture_x": "entry/instrument/microscope/x_gaussian",
        "analyser_gaussian_aperture_y": "entry/instrument/microscope/y_gaussian",
        # Written as a placeholder string when no temperature scannable is configured
        "temperature_sample": lambda f: (
            "entry/sample/temperature"
            if "entry/sample/temperature" in f
            and f["entry/sample/temperature"].dtype.kind in "fiu"
            else None
        ),
        # jenergy holds the per-point values for an hv scan; pgmenergy is only recorded once per scan
        "photon_hv": lambda f: (
            (
                BaseHDF5DataLoader._extract_hdf5_value(
                    f, "entry/instrument/jenergy/value"
                )
                * ureg.keV
            ).to("eV")
            if "entry/instrument/jenergy/value" in f
            else "entry/instrument/pgm/pgmenergy"
        ),
        "photon_polarisation": "entry/instrument/polarisation/value",
        # I09-2 exit slit; the vertical gap sets the energy resolution
        "photon_exit_slit": "entry/instrument/ss7/ss7ygap",
        "timestamp": "entry/diamond_scan/start_time",
    }

    @classmethod
    def _parse_analyser_metadata(cls, metadata_dict):
        """Parse the standard analyser metadata, adding the microscope apertures."""
        parsed_metadata, warn_list = super()._parse_analyser_metadata(metadata_dict)
        arpes_metadata = parsed_metadata["_analyser"]
        apertures = MomentumMicroscopeAperturesModel(
            **{
                plane: MomentumMicroscopeApertureModel(
                    selected=metadata_dict.get(f"analyser_{plane}_aperture"),
                    x=metadata_dict.get(f"analyser_{plane}_aperture_x"),
                    y=metadata_dict.get(f"analyser_{plane}_aperture_y"),
                )
                for plane in ["fourier", "gaussian"]
            }
        )
        parsed_metadata["_analyser"] = MomentumMicroscopeMetadataModel(
            **{
                **dict(arpes_metadata),
                "analyser": MomentumMicroscopeAnalyserMetadataModel(
                    **dict(arpes_metadata.analyser), apertures=apertures
                ),
            }
        )
        relevant = metadata_dict.get("_i09_analyser_warn", [])
        return parsed_metadata, [w for w in warn_list if w in relevant]

    @staticmethod
    def _is_live_view_image(fpath):
        """Whether the file is an image saved from the live view (a .tif, or a snapshot .nxs without
        `entry/instrument`) rather than a GDA scan. These carry no metadata or beamline identifier."""
        if not h5py.is_hdf5(fpath):
            return True
        with h5py.File(fpath, "r") as f:
            return "entry/instrument" not in f

    @classmethod
    def _load_metadata(cls, fpath):
        """Load the metadata from a GDA scan file; live-view images have none."""
        if cls._is_live_view_image(fpath):
            return {}
        metadata = super()._load_metadata(fpath)
        # Analyser fields apply only to detector scans (not drain-current or mirror scans), step size only when
        # sweeping the kinetic energy
        with h5py.File(fpath, "r") as f:
            nxdata = f["entry"][cls._parse_nxs_axis_attr(f["entry"].attrs["default"])]
            axes = {
                cls._parse_nxs_axis_attr(a) for a in np.atleast_1d(nxdata.attrs["axes"])
            }
            relevant = []
            if _i09_detector_addr(f):
                relevant = ["eV", "dwell"]
                if axes & {"Kenergy", "kenergy"}:
                    relevant.append("step_size")
        metadata["_i09_analyser_warn"] = [f"analyser_{key}" for key in relevant]
        return metadata

    @classmethod
    def _load_live_view_image(cls, fpath):
        """Load a live-view image (e.g. a PEEM image or flat-field reference) as raw detector pixels.

        Both the .tif and snapshot .nxs images share the (`detector_y`, `detector_x`) orientation of `dld/data`.
        """
        if h5py.is_hdf5(fpath):
            with h5py.File(fpath, "r") as f:
                nxdata = f["entry/data"]
                image = nxdata[cls._parse_nxs_axis_attr(nxdata.attrs["signal"])][()]
        else:
            import imagecodecs

            image = imagecodecs.imread(fpath)
        da = xr.DataArray(
            image,
            dims=["detector_y", "detector_x"],
            coords={
                "detector_y": np.arange(image.shape[0]),
                "detector_x": np.arange(image.shape[1]),
            },
            name="spectrum",
        )
        return da.pint.quantify({"spectrum": "counts"})

    @classmethod
    def _load_data(cls, fpath, lazy, **kwargs):
        """Load the data from a Diamond I09-2 momentum microscope NeXus file, or a live-view image."""
        if cls._is_live_view_image(fpath):
            return cls._load_live_view_image(fpath)
        with h5py.File(fpath, "r") as f:
            nxdata = f["entry"][cls._parse_nxs_axis_attr(f["entry"].attrs["default"])]
            det_addr = _i09_detector_addr(f)
            if det_addr:
                # Full detector image if saved, otherwise the integrated ROI (`mcp_roi_total` for GDA < 9.42)
                signal_addr = next(
                    f"{det_addr}/{key}"
                    for key in ["data", "fixed_roi_total", "mcp_roi_total"]
                    if key in f[det_addr]
                )
            else:
                # Scans without a detector (e.g. drain-current XAS, mirror scans): use the default NXdata signal
                signal_addr = (
                    f"{nxdata.name}/{cls._parse_nxs_axis_attr(nxdata.attrs['signal'])}"
                )
            dset = f[signal_addr]

            # Scan axes from the default NXdata group: the position in `axes` is the dimension index
            scan_dims = {}
            coords = {}
            units = {}
            for index, axis in enumerate(np.atleast_1d(nxdata.attrs["axes"])):
                axis = axis.decode() if isinstance(axis, bytes) else str(axis)
                if axis == "." or axis not in nxdata:
                    continue
                # Axis values are written on the full scan grid, so take them along their own dimension
                values = nxdata[axis][()]
                indexer = [0] * values.ndim
                indexer[index] = slice(None)
                scan_dims[index] = axis
                coords[axis] = values[tuple(indexer)]
                units[axis] = nxdata[axis].attrs.get(
                    "units", cls._scannable_units.get(axis)
                )

            scan_rank = int(f["entry/diamond_scan/scan_rank"][()])
            dims = [scan_dims.get(i, f"dummy_{i}") for i in range(scan_rank)]
            detector_dims = ["detector_y", "detector_x"][2 - (dset.ndim - scan_rank) :]
            for dim, size in zip(detector_dims, dset.shape[scan_rank:], strict=True):
                coords[dim] = np.arange(size)
            dims += detector_dims

            # I0 monitor recorded per scan point, unless it is itself the signal (e.g. mirror scans)
            I0_addr = next(
                (
                    addr
                    for addr in [
                        f"{nxdata.name}/{cls._I0_monitor}",
                        f"entry/instrument/{cls._I0_monitor}/value",
                    ]
                    if addr in f and f[addr].ndim > 0
                ),
                None,
            )
            if signal_addr.rsplit("/", 1)[1] == cls._I0_monitor:
                I0_addr = None
            I0 = f[I0_addr][()] if I0_addr else None
            # Drain currents recorded per scan point (e.g. sd11iamp7, the sample drain current)
            scan_streams = {
                key: nxdata[key][()]
                for key in nxdata
                if re.fullmatch(cls._drain_current_pattern, key)
                and key
                not in [
                    cls._I0_monitor,
                    cls._parse_nxs_axis_attr(nxdata.attrs.get("signal")),
                ]
                and nxdata[key].shape == dset.shape[:scan_rank]
            }

            if not det_addr:
                data = dset[()]
            # The DLD image is szip-compressed, but some h5py wheels (e.g. macOS) are built without szip
            elif dset.compression == "szip" and not h5py.h5z.filter_avail(
                h5py.h5z.FILTER_SZIP
            ):
                data = dask.array.map_blocks(
                    _read_szip_chunk,
                    fpath,
                    signal_addr,
                    chunks=dask.array.core.normalize_chunks(dset.chunks, dset.shape),
                    dtype=dset.dtype,
                    meta=np.empty((0,) * dset.ndim, dtype=dset.dtype),
                )
            else:
                data = None

        if data is not None:
            da = xr.DataArray(data, dims=dims)
        else:
            # Load the core data in a way that supports lazy loading (the image is an external link to the dld file)
            group, signal_key = signal_addr.rsplit("/", 1)
            ds = xr.open_dataset(
                fpath,
                group=group,
                engine="h5netcdf",
                phony_dims="sort",
                chunks="auto",
            )
            da = ds[signal_key]
            da = da.rename(dict(zip(da.dims, dims, strict=True)))
            da.attrs = {}
        da = da.assign_coords({dim: coords[dim] for dim in dims if dim in coords})
        if I0 is not None:
            da = da.assign_coords(I0=(dims[:scan_rank], I0))
            units["I0"] = cls._drain_current_units
        da = da.assign_coords(
            {key: (dims[:scan_rank], values) for key, values in scan_streams.items()}
        )
        units.update(dict.fromkeys(scan_streams, cls._drain_current_units))

        # Rename to peaks conventions, with the binding energy in E-EF
        if "Benergy" in da.dims:
            da = da.assign_coords(Benergy=-da.Benergy)
        rename = {
            local: peaks_name
            for peaks_name, local in [
                *cls._analyser_name_conventions.items(),
                *(("eV", local) for local in cls._eV_scannables),
            ]
            if local in da.dims
        }
        da = da.rename(rename)
        units = {rename.get(dim, dim): unit for dim, unit in units.items()}
        units = {
            key: cls._parse_nxs_axis_attr(value)
            for key, value in units.items()
            if value is not None
        }
        if det_addr:
            units["spectrum"] = "counts"
        elif re.fullmatch(cls._drain_current_pattern, signal_addr.rsplit("/", 1)[1]):
            units["spectrum"] = cls._drain_current_units

        # Load array into memory if lazy loading not required
        if not (lazy or (lazy is None and da.size > opts.FileIO.lazy_size)):
            da = da.compute()

        da.name = "spectrum"
        da = da.pint.quantify(units)
        if "hv" in da.dims:
            da = da.pint.to(hv="eV")
        return da.squeeze()
