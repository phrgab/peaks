from datetime import datetime

import dask
import dask.array as da
import h5py
import numpy as np
import pint_xarray  # noqa: F401
import xarray as xr

from peaks.core.fileIO.base_arpes_data_classes.base_arpes_data_class import (
    BaseARPESDataLoader,
    ureg,
)
from peaks.core.fileIO.loc_registry import register_loader


@register_loader
class BaseFeSuMaDataLoader(BaseARPESDataLoader):
    # Define class variables
    _loc_name = "FeSuMa"

    _desired_dim_order = ["detector_x", "eV", "detector_y"]

    @classmethod
    def _load_data(cls, fpath, lazy):
        class FeSuMaImageCube:
            def __init__(self, hdf5_file):
                self.hdf5_file = hdf5_file

                with h5py.File(hdf5_file, "r") as f:
                    # Find all groups that start with '0'
                    self.scans_list = [key for key in f.keys() if key.startswith("0")]

                    # Get the shape and dtype from the first image
                    image0 = f[f"{self.scans_list[0]}/analysisImage"]
                    self.image_shape = image0.shape
                    self.dtype = image0.dtype
                    start = image0.attrs.get("DimOffsetX")
                    delta = image0.attrs.get("DimDeltaX")
                    self.detector_x = np.arange(
                        start, start + delta * image0.shape[0], delta
                    )
                    self.detector_x_unit = image0.attrs.get("ScaleUnitsX")
                    if isinstance(self.detector_x_unit, bytes):
                        self.detector_x_unit = self.detector_x_unit.decode()
                    start = image0.attrs.get("DimOffsetY")
                    delta = image0.attrs.get("DimDeltaY")
                    self.detector_y = np.arange(
                        start, start + delta * image0.shape[1], delta
                    )
                    self.detector_y_unit = image0.attrs.get("ScaleUnitsY")
                    if isinstance(self.detector_y_unit, bytes):
                        self.detector_y_unit = self.detector_y_unit.decode()

                    # Parse Acquisition metadata as integers
                    self.steps = np.array(
                        [
                            int(f[f"{scan}/AcquisitionCurrentStep"][0])
                            for scan in self.scans_list
                        ]
                    )
                    self.sweeps = np.array(
                        [
                            int(f[f"{scan}/AcquisitionCurrentSweep"][0])
                            for scan in self.scans_list
                        ]
                    )

                    # Get acquisition co-ordinate
                    self.acquisition_coord = f["AcquisitionCoordinateNice"][0]
                    if isinstance(self.acquisition_coord, bytes):
                        self.acquisition_coord = self.acquisition_coord.decode()

                    # Get the KE scaling
                    try:
                        # If a KE scan, should list the relevant KE values in the common metadata
                        # Don't trust AcquisitionEkinStop - this sometimes includes an extra step
                        self.eV = np.linspace(
                            float(f["AcquisitionEkinStart"][0]),
                            float(f["AcquisitionEkinStart"][0])
                            + (f["AcquisitionEkinStep"][0] * (len(set(self.steps)) - 1)),
                            len(set(self.steps)),
                        )
                    except KeyError:
                        # Get this from the first scan analyser voltages
                        self.eV = -f[f"{self.scans_list[0]}/AnalyzerUserSetVoltages"][3]

                    # Get delay positions if required
                    if self.acquisition_coord == "Delay Stage":
                        self.delay_pos = np.linspace(
                            float(f["AcquisitionDelayStart"][0]),
                            float(f["AcquisitionDelayStart"][0])
                            + (
                                f["AcquisitionDelayStep"][0] * (len(set(self.steps)) - 1)
                            ),
                            len(set(self.steps)),
                        )

            def __getitem__(self, idx):
                # Lazily load the slice when accessed
                scan_path = f"{self.scans_list[idx]}/analysisImage"
                with h5py.File(self.hdf5_file, "r") as f:
                    data = f[scan_path][:]
                return data

            def __len__(self):
                return len(self.scans_list)

            def shape(self):
                return (len(self.scans_list), *self.image_shape)

        # Instantiate the lazy loader
        lazy_cube = FeSuMaImageCube(fpath)

        # Create a list of Dask arrays from delayed objects
        lazy_slices = [
            da.from_delayed(
                dask.delayed(lazy_cube[i]),
                shape=lazy_cube.image_shape,
                dtype=lazy_cube.dtype,
            )
            for i in range(len(lazy_cube))
        ]

        # Stack the Dask arrays along a new dimension to form the 3D cube
        lazy_data = da.stack(lazy_slices, axis=0)

        # Create the xarray.DataArray with Acquisition metadata as dimensions
        data_array = xr.DataArray(
            lazy_data,
            dims=[
                "steps",
                "detector_x",
                "detector_y",
            ],  # Specifying dimensions for 3D data
            coords={
                "detector_x": lazy_cube.detector_x,
                "detector_y": lazy_cube.detector_y,
                "steps": lazy_cube.steps,  # Coordinates for the "steps" dimension
                "sweep": (
                    "steps",
                    lazy_cube.sweeps,
                ),  # Associating "sweep" with "steps"
            },
            name="spectrum",
        )

        if not lazy:
            data_array = data_array.compute()

        data_array = data_array.groupby("steps").mean()

        if lazy_cube.acquisition_coord == "Kinetic Energy":
            data_array = data_array.assign_coords({"eV": ("steps", lazy_cube.eV)})
            data_array = data_array.swap_dims({"steps": "eV"}).drop_vars("steps")
            scan_dim = "eV"
            scan_units = "eV"
        elif lazy_cube.acquisition_coord == "Delay Stage":
            data_array = data_array.assign_coords(
                {"delay_pos": ("steps", lazy_cube.delay_pos)}
            )
            data_array = data_array.swap_dims({"steps": "delay_pos"}).drop_vars("steps")
            scan_dim = "delay_pos"
            scan_units = "mm"

        return {
            "spectrum": data_array.data,
            "dims": data_array.dims,
            "coords": {dim: coord.data for dim, coord in data_array.coords.items()},
            "units": {
                scan_dim: scan_units,
                "detector_x": lazy_cube.detector_x_unit,
                "detector_y": lazy_cube.detector_y_unit,
                "spectrum": "counts",
            },
        }

    @classmethod
    def _load_metadata(cls, fpath):
        with h5py.File(fpath, "r") as f:
            image0 = f[
                f"{next(key for key in f.keys() if key.startswith('0'))}/analysisImage"
            ]
            dwell_unit = image0.attrs.get("ExposureTime_UNIT")

            # Parse KE
            # Get the KE scaling
            try:
                # If a KE scan, should list the relevant KE values in the common metadata
                # Don't trust AcquisitionEkinStop - this sometimes includes an extra step
                eV = [
                    float(f["AcquisitionEkinStart"][0]),
                    float(f["AcquisitionEkinStop"][0]),
                ] * ureg("eV")
                eV_step = f["AcquisitionEkinStep"][0] * ureg("eV")
            except KeyError:
                # Get this from the first scan analyser voltages
                eV = -f[
                    f"{next(key for key in f.keys() if key.startswith('0'))}/AnalyzerUserSetVoltages"
                ][3] * ureg("eV")
                eV_step = None

            if isinstance(dwell_unit, bytes):
                dwell_unit = dwell_unit.decode()
            metadata = {
                "analyser_sweeps": int(f["AcquisitionNumberOfSweeps"][0]),
                "analyser_dwell": image0.attrs.get("ExposureTime")[()]
                * image0.attrs.get("AccumulationCount")[()]
                * ureg(dwell_unit),
                "analyser_eV": eV,
                "analyser_step_size": eV_step,
                "analyser_eV_type": "kinetic energy cutoff",
                "timestamp": datetime.strptime(
                    f["AcquisitionStartTimeNice"][0].decode(), "%Y-%m-%dT%H:%M:%S.%fZ"
                ).strftime("%Y-%m-%d %H:%M:%S"),
            }

        return metadata
