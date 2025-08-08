from datetime import datetime, timezone
from typing import Optional, Union

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
from peaks.core.fileIO.loc_registry import register_loader
from peaks.core.metadata.base_metadata_models import BaseMetadataModel, Quantity
from peaks.core.options import opts
from peaks.core.utils.misc import analysis_warning

ureg = pint_xarray.unit_registry


class DiamondNXSLoader(BaseHDF5DataLoader):
    """Helper class for parsing the NeXus file format used at Diamond Light Source."""

    _data_group_key_resolution_order = []
    _core_data_key_resolution_order = []

    @property
    def data_key_resolution_order(self):
        """The priority order of keys to specify the main data group in the file"""
        return self._data_group_key_resolution_order

    @staticmethod
    def get_root_key(f):
        """Get the root key of the file, which is assumed to be the only key in the file at root level

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
        """Get the main data group from the file, with priority as per _data_key_resolution_order

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
        """Parse the attribute value for a signal or primary key to a standard format"""
        if isinstance(attr, bytes):
            attr = attr.decode()

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
        """Parse the attribute value for an axis key to a standard format"""
        if isinstance(attr, (list, np.ndarray)):
            attr = attr[0]
        if isinstance(attr, bytes):
            attr = attr.decode()
        return attr.decode() if isinstance(attr, bytes) else attr

    @classmethod
    def get_core_data_key(cls, f, data_address):
        """Find the core data in an HDF5/nxs group. First attempts to resolve data by key located in
        `_core_data_key_resolution_order` class variable. If that fails, fall back to trying to automatically parse
        by looking for a signal attribute flag

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


@register_loader
class I05ARPESLoader(DiamondNXSLoader, BaseARPESDataLoader):
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
        "deflector_perp": "deflector_x",
        "deflector_parallel": "deflector_y",
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
        "manipulator_x2": "entry1/instrument/manipulator/say",
        "manipulator_x3": "entry1/instrument/manipulator/saz",
        "manipulator_salong": "entry1/instrument/manipulator/salong",
        "analyser_model": lambda f: (
            (
                "FIXED_VALUE:MBS A1"
                if parse(
                    (
                        (f["entry1/start_time"][()]).decode()
                        if isinstance(f["entry1/start_time"][()], bytes)
                        else f["entry1/start_time"][()]
                    )
                )
                > datetime(2021, 7, 1, tzinfo=timezone.utc)
                else "FIXED_VALUE:Scienta R4000"
            )
            if "entry1/start_time" in f
            else None
        ),
        "analyser_slit_width": "entry1/instrument/analyser/entrance_slit_size",
        "analyser_slit_width_identifier": "entry1/instrument/analyser_total/entrance_slit_setting",
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
        "analyser_deflector_parallel": "entry1/instrument/deflector_y/deflector_y",
        "analyser_deflector_perp": "entry1/instrument/deflector_x/deflector_x",
        "temperature_sample": "entry1/sample/temperature",
        "temperature_cryostat": "entry1/sample/cryostat_temperature",
        "temperature_shield": "entry1/sample/shield_temperature",
        "temperature_setpoint": "entry1/sample/temperature_demand",
        "photon_hv": "entry1/instrument/monochromator/energy",
        "photon_polarisation": [
            "entry1/instrument/insertion_device/beam/final_polarisation_label",
            "entry1/instrument/insertion_device/beam/final_polarisation_label\n\t\t\t\t\t\t\t\t",
        ],
        "photon_exit_slit": "entry1/instrument/monochromator/exit_slit_size",
        "timestamp": "entry1/start_time",
    }

    _data_group_key_resolution_order = ["analyser"]
    _core_data_key_resolution_order = ["analyser"]

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

        # Normalise data by I0 if required
        if kwargs.get("norm_by_I0", False):
            if "current" in dim_name_to_axis_mapping:
                beam_current = ds["current"]
                beam_current_coords_to_apply = {
                    k: v for k, v in coords_to_apply.items() if k in beam_current.dims
                }
                beam_current = beam_current.assign_coords(beam_current_coords_to_apply)
                bc_units = beam_current.attrs.get("units", "mA")
                beam_current.attrs = {}
                da = da / beam_current
                units["spectrum"] = units["spectrum"] + f"/{bc_units}"
            else:
                analysis_warning(
                    "Could not determine the beam current to normalise data by",
                    "warning",
                    "Missing I0 data",
                )

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

        # Load array into memory if lazy loading not required
        if not (lazy or (lazy is None and da.size > opts.FileIO.lazy_size)):
            da = da.compute()

        # Add units where available
        da.name = "spectrum"
        da = da.pint.quantify(units)
        return da.squeeze()


class I05NanoFocussingMetadataModel(BaseMetadataModel):
    """Model to store metadata for Diamond Light Source Nano-ARPES focussing optics metadata."""

    OSAx: Optional[Union[str, Quantity]] = None
    OSAy: Optional[Union[str, Quantity]] = None
    OSAz: Optional[Union[str, Quantity]] = None
    ZPx: Optional[Union[str, Quantity]] = None
    ZPy: Optional[Union[str, Quantity]] = None
    ZPz: Optional[Union[str, Quantity]] = None


@register_loader
class I05NanoARPESLoader(I05ARPESLoader):
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

    _analyser_sign_conventions = {
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
        "deflector_perp": "ThetaY",
        "deflector_parallel": "ThetaX",
        "eV": ["energies", "kinetic_energy_center"],
        "theta_par": "angles",
        "hv": ["energy", "value"],
        "ana_polar": "analyser_polar_angle",
    }

    _hdf5_metadata_key_mappings = {
        "scan_command": "entry1/scan_command",
        "manipulator_polar": "entry1/instrument/manipulator/smpolar",
        "manipulator_tilt": "entry1/instrument/manipulator/smtilt",
        "manipulator_azi": "entry1/instrument/manipulator/smazimuth",
        "manipulator_x1": "entry1/instrument/manipulator/smx",
        "manipulator_x2": "entry1/instrument/manipulator/smy",
        "manipulator_x3": "entry1/instrument/manipulator/smz",
        "manipulator_defocus": "entry1/instrument/manipulator/smdefocus",
        "analyser_model": "FIXED_VALUE:Scienta DA30",
        "analyser_slit_width": [
            "entry1/instrument/analyser/entrance_slit_size",
            "entry1/instrument/analyser_total/entrance_slit_size",
        ],
        "analyser_slit_width_identifier": [
            "entry1/instrument/analyser/entrance_slit_setting",
            "entry1/instrument/analyser_total/entrance_slit_setting",
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
        "analyser_deflector_parallel": None,
        "analyser_deflector_perp": None,
        "analyser_polar": [
            "entry1/instrument/analyser/analyser_polar_angle",
            "entry1/instrument/analyser_total/analyser_polar_angle",
        ],
        "temperature_sample": "entry1/sample/temperature",
        "temperature_cryostat": "entry1/sample/cryostat_temperature",
        "temperature_shield": "entry1/sample/shield_temperature",
        "temperature_setpoint": "entry1/sample/temperature_demand",
        "photon_hv": "entry1/instrument/monochromator/energy",
        "photon_polarisation": [
            "entry1/instrument/insertion_device/beam/final_polarisation_label",
            "entry1/instrument/insertion_device/beam/final_polarisation_label\n\t\t\t\t\t\t\t\t",
        ],
        "photon_exit_slit": "entry1/instrument/monochromator/exit_slit_size",
        "timestamp": "entry1/start_time",
        "focussing_OSAx": "entry1/instrument/order_sorting_aperture/osax",
        "focussing_OSAy": "entry1/instrument/order_sorting_aperture/osay",
        "focussing_OSAz": "entry1/instrument/order_sorting_aperture/osaz",
        "focussing_ZPx": "entry1/instrument/zone_plate/zpx",
        "focussing_ZPy": "entry1/instrument/zone_plate/zpy",
        "focussing_ZPz": "entry1/instrument/zone_plate/zpz",
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

    @classmethod
    def _load(cls, fpath, lazy, metadata, quiet, **kwargs):
        """Load the data from Diamond I05 Nano ARPES."""
        if fpath.split(".")[-1] == "zip":
            # Load the data using the SES loader for .zip files
            return cls.load(fpath, loc="SES", lazy=lazy, metadata=metadata)

        # Load the data
        data = super()._load(fpath, lazy, metadata, quiet, **kwargs)

        # Perform post processing on load if required
        if kwargs.pop("slant_correct", False):  # Perform the slant correction
            data = cls.slant_correct(data, kwargs.get("slant_factor", None))

        return data

    @classmethod
    def _parse_optics_metadata(cls, metadata_dict):
        """Parse the optics metadata from the metadata dictionary"""

        optics_metadata = I05NanoFocussingMetadataModel(
            OSAx=metadata_dict.get("focussing_OSAx"),
            OSAy=metadata_dict.get("focussing_OSAy"),
            OSAz=metadata_dict.get("focussing_OSAz"),
            ZPx=metadata_dict.get("focussing_ZPx"),
            ZPy=metadata_dict.get("focussing_ZPy"),
            ZPz=metadata_dict.get("focussing_ZPz"),
        )
        return {"_focussing": optics_metadata}, None

    @register_accessor(xr.DataArray)
    @staticmethod
    def slant_correct(data, slant_factor=None):
        """Function to remove a slant that was present in data obtained using the Scienta DA30 (9ES210) analyser at the nano
        branch of the I05 beamline at Diamond Light Source in 2021/22.

        Parameters
        ------------
        data : xarray.DataArray
            The data to be corrected.

        slant_factor : float, optional
            The slant factor correction (degrees/eV) to use. Defaults to 8/PE (where PE is the pass energy used).

        Returns
        ------------
        corrected_data : xarray.DataArray
            The corrected data.

        Examples
        ------------
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
