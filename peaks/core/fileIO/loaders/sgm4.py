import math
from typing import Optional, Union

import h5py
import numpy as np
import pint_xarray
import xarray as xr

from peaks.core.fileIO.base_arpes_data_classes.base_arpes_data_class import (
    BaseARPESDataLoader,
)
from peaks.core.fileIO.base_data_classes.base_hdf5_class import BaseHDF5DataLoader
from peaks.core.fileIO.base_data_classes.base_optics_class import BaseOpticsDataLoader
from peaks.core.fileIO.loc_registry import register_loader
from peaks.core.metadata.base_metadata_models import (
    BaseMetadataModel,
    Quantity,
)

ureg = pint_xarray.unit_registry


class SGM4KeithleyMetadataModel(BaseMetadataModel):
    """Model to store metadata for SGM4 Nano-ARPES focussing optics metadata."""

    KA_I: Optional[Union[str, Quantity]] = None
    KA_V: Optional[Union[str, Quantity]] = None
    KA_R: Optional[Union[str, Quantity]] = None
    KB_I: Optional[Union[str, Quantity]] = None
    KB_V: Optional[Union[str, Quantity]] = None
    KB_R: Optional[Union[str, Quantity]] = None


@register_loader
class SGM4NanoARPESLoader(BaseHDF5DataLoader, BaseARPESDataLoader, BaseOpticsDataLoader):
    _loc_name = "ASTRID2_SGM4"
    _loc_description = (
        "SGM4 endstation at the ASTRID2 synchrotron at Aarhus University, Denmark."
    )
    _loc_url = "https://isa.au.dk/facilities/astrid2/beamlines/AU-sgm4/AU-sgm4.asp"
    _analyser_slit_angle = 0 * ureg.deg
    _analyser_WF = 4.416 * ureg.electron_volt
    _anapolaroffset = -14 * ureg.deg
    _smpolaroffset = +45 * ureg.deg

    _unitdict = {
        "x1": ureg.micrometer,
        "x2": ureg.micrometer,
        "x3": ureg.micrometer,
        "polar": ureg.deg,
        "tilt": ureg.deg,
        "azi": ureg.deg,
        "defocus": ureg.micrometer,
        "CapZ": ureg.micrometer,
        "CapY": ureg.micrometer,
        "CapX": ureg.micrometer,
        "SamX": ureg.micrometer,
        "SamY": ureg.micrometer,
        "SamZ": ureg.micrometer,
        "CapPolar": ureg.deg,
        "eV": ureg.electron_volt,
        "theta_par": ureg.deg,
        "deflector_perp": ureg.deg,
        "deflector_parallel": ureg.deg,
        "scan_no": ureg.count,
        "hv": ureg.electron_volt,
    }

    _manipulator_axes = ["polar", "tilt", "azi", "x1", "x2", "x3", "defocus"]
    _manipulator_name_conventions = {
        "polar": "SamPolar",
        "tilt": "SamTilt",
        "azi": "SamAzi",
        "x1": "FSamX",
        "x2": "FSamY",
        "x3": "FSamZ",
        "defocus": "CapZ",
    }

    _manipulator_sign_conventions = {
        "polar": +1,
        "tilt": -1,
    }

    _analyser_name_conventions = {
        "deflector_perp": "ShiftX",
        "deflector_parallel": "ShiftY",
        "theta_par": "ordinate_range",
        "hv": "hv",
    }

    _optics_name_conventions = {
        "x1": "CapX",
        "x2": "CapY",
        "x3": "CapZ",
    }

    _analyser_sign_conventions = {
        "polar": -1,
    }

    _metadata_parsers = [
        "_parse_analyser_metadata",
        "_parse_manipulator_metadata",
        "_parse_photon_metadata",
        "_parse_temperature_metadata",
        "_parse_optics_metadata",
        "_parse_sgm4Keithley_metadata",
    ]

    @classmethod
    def _parse_sgm4Keithley_metadata(cls, metadata_dict):
        """Metadata model specific to the SGM4 endstation. It refers to the external Keithley power supplies."""
        keithley_metadata = SGM4KeithleyMetadataModel(
            KA_I=metadata_dict.get("keithleyA_Current"),
            KA_V=metadata_dict.get("keithleyA_Voltage"),
            KA_R=metadata_dict.get("keithleyA_Resistance"),
            KB_I=metadata_dict.get("keithleyB_Current"),
            KB_V=metadata_dict.get("keithleyB_Voltage"),
            KB_R=metadata_dict.get("keithleyB_Resistance"),
        )
        return {"_keithley": keithley_metadata}, None

    @classmethod
    def _load_data(cls, fpath, metadata=False, Lazy=False, **kwargs):
        with h5py.File(fpath, "r", swmr=True) as h5file:
            scandetails = h5file["/Entry/Data/ScanDetails/"]
            # Determine whether data is a knife-edge, or ordinary scan.
            if "FastAxis_start" in list(scandetails.keys()):
                # Ensure the naming conventions are correct.
                cls._manipulator_name_conventions = {
                    "polar": "SamPolar",
                    "tilt": "SamTilt",
                    "azi": "SamAzi",
                    "x1": "FSamX",
                    "x2": "FSamY",
                    "x3": "FSamZ",
                    "defocus": "CapZ",
                }
                # Load Energy and k-axis:
                NFast = len(scandetails["FastAxis_names"])
                Fastlen = scandetails["FastAxis_length"][()]
                FastAxis = []
                for iaxis in range(NFast):
                    axistemp = np.array(
                        [
                            scandetails["FastAxis_start"][()][iaxis]
                            + scandetails["FastAxis_step"][()][iaxis] * i
                            for i in range(Fastlen[iaxis])
                        ]
                    )
                    FastAxis.append(axistemp)

                # Load remaining axis:
                NSlow = len(scandetails["SlowAxis_names"])
                Slowlen = scandetails["SlowAxis_length"][()]
                SlowAxis = []
                for iaxis in range(NSlow):
                    axistemp = np.array(
                        [
                            scandetails["SlowAxis_start"][()][iaxis]
                            + scandetails["SlowAxis_step"][()][iaxis] * i
                            for i in range(Slowlen[iaxis])
                        ]
                    )
                    SlowAxis.append(axistemp)
                # Load dataset:
                dset = h5file["Entry/Data/TransformedData"][()]
                TargLen = math.prod(Slowlen)
                Len = len(dset)
                # Reshape:
                if Len == TargLen:
                    dset = dset.reshape(tuple(np.append(Slowlen[::-1], Fastlen)))
                elif Len < TargLen:
                    Warning(
                        "You are loading an only partially complete dataset. The missing entries are represented by nan."
                    )
                    padlen = TargLen - Len
                    padarr = [int(padlen)]
                    padarr.extend(list(dset.shape)[1::])
                    padarr = np.empty(tuple(padarr))
                    padarr[:] = np.nan
                    dset = np.append(dset, padarr).reshape(
                        tuple(np.append(Slowlen[::-1], Fastlen))
                    )
                else:
                    raise ValueError(
                        "Dataset appears to have more entries than expected. This should not happen."
                    )
                # Convert to xarray:
                dimnames = [i.decode() for i in scandetails["SlowAxis_names"][()][::-1]]
                dimnames.extend([i.decode() for i in scandetails["FastAxis_names"][()]])
                axis = SlowAxis[::-1]
                axis.extend(FastAxis)
                coords = {}
                for kn, a in zip(dimnames, axis, strict=True):
                    coords[kn] = a
                dsetxr = xr.DataArray(dset, dims=dimnames, coords=coords)
                # Adjust naming conventions if it is an hv scan
                if "LEG" in dimnames:
                    cls._analyser_name_conventions["hv"] = "LEG"
                elif "MEG" in dimnames:
                    cls._analyser_name_conventions["hv"] = "MEG"
                elif "HEG" in dimnames:
                    cls._analyser_name_conventions["hv"] = "HEG"
                # Renaming Loop dimension:
                if "Loop" in dimnames:
                    dsetxr = dsetxr.rename({"Loop": "scan_no"})
                dsetxr = dsetxr.rename(
                    {"Kinetic Energy": "eV", "OrdinateRange": "theta_par"}
                )
                dsetxr = dsetxr.rename(
                    {
                        v: k
                        for k, v in cls._manipulator_name_conventions.items()
                        if v in dsetxr.dims
                    }
                )
                dsetxr = dsetxr.rename(
                    {
                        v: k
                        for k, v in cls._analyser_name_conventions.items()
                        if v in dsetxr.dims
                    }
                )

                # Add units for coordinate axis.
                dsetxr = dsetxr.pint.quantify({dsetxr.name: ureg.count / ureg.s})
                for coord in dsetxr.coords:
                    dsetxr.coords[coord].attrs["units"] = cls._unitdict[coord]

                # If an hv scan, add the KE_delta coord:
                if "hv" in dsetxr.dims:
                    coords_to_apply = {
                        dim: dsetxr.coords.get(dim)
                        for dim in dsetxr.dims
                        if dim != "dummy"
                    }
                    coord = coords_to_apply["hv"]
                    coords_to_apply["KE_delta"] = coord[:] - coord[0]
                    dsetxr = dsetxr.assign_coords(coords_to_apply)

                # Correcting work function.
                dsetxr.coords["eV"] = dsetxr.coords["eV"] - cls._analyser_WF.magnitude

                # Adding up scan_no/Loop dimension if specified.
                if "SumLoop" in kwargs.keys():
                    if kwargs["SumLoop"]:
                        dsetxr = dsetxr.sum("scan_no")

            else:
                # Change name conventions temporarily.
                cls._manipulator_name_conventions = {
                    "polar": "SamPolar",
                    "tilt": "SamTilt",
                    "azi": "SamAzi",
                    "x1": "SamX",
                    "x2": "SamY",
                    "x3": "SamZ",
                    "defocus": "CapZ",
                }
                # Load remaining axis:
                NSlow = len(scandetails["SlowAxis_names"])
                Slowlen = scandetails["SlowAxis_length"][()]
                SlowAxis = []
                for iaxis in range(NSlow):
                    axistemp = np.array(
                        [
                            scandetails["SlowAxis_start"][()][iaxis]
                            + scandetails["SlowAxis_step"][()][iaxis] * i
                            for i in range(Slowlen[iaxis])
                        ]
                    )
                    SlowAxis.append(axistemp)
                # Load dataset:
                dset = h5file["/Entry/Process/SumData"][()]
                TargLen = math.prod(Slowlen)
                Len = len(dset)
                # Reshape:
                if Len == TargLen:
                    dset = dset.reshape(tuple(Slowlen[::-1]))
                elif Len < TargLen:
                    Warning(
                        "You are loading an only partially complete dataset. The missing entries are represented by nan."
                    )
                    padlen = TargLen - Len
                    padarr = np.array([np.nan] * padlen)
                    dset = np.append(dset, padarr).reshape(tuple(Slowlen[::-1]))
                else:
                    raise ValueError(
                        "Dataset appears to have more entries than expected. This should not happen."
                    )
                # Convert to xarray:
                dimnames = [i.decode() for i in scandetails["SlowAxis_names"][()][::-1]]
                axis = SlowAxis[::-1]
                coords = {}
                for kn, a in zip(dimnames, axis, strict=True):
                    coords[kn] = a
                dsetxr = xr.DataArray(dset, dims=dimnames, coords=coords)
                dsetxr = dsetxr.rename(
                    {
                        v: k
                        for k, v in cls._manipulator_name_conventions.items()
                        if v in dsetxr.dims
                    }
                )
                dsetxr = dsetxr.rename(
                    {
                        v: k
                        for k, v in cls._analyser_name_conventions.items()
                        if v in dsetxr.dims
                    }
                )
                dsetxr = dsetxr.pint.quantify({dsetxr.name: ureg.count / ureg.s})
                for coord in dsetxr.coords:
                    dsetxr.coords[coord].attrs["units"] = cls._unitdict[coord]

            return dsetxr.squeeze()

    @classmethod
    def _load_metadata(cls, fpath):
        with h5py.File(fpath, "r") as f:
            # Necessary to distinguish knifeedge and ordinary scans.
            scandetails = f["/Entry/Data/ScanDetails/"]
            knifeedgescanflag = "FastAxis_start" not in list(scandetails.keys())
            if knifeedgescanflag:
                # During knife edge measurements, the main manipulator
                # axis are the "Real" SamX, SamY and SamZ axis, rather than the
                # "Fake" FSamX, FSamY and FSamZ used in ordinary measurements.
                # The naming conventions are adjusted accordingly here.
                cls._manipulator_name_conventions = {
                    "polar": "SamPolar",
                    "tilt": "SamTilt",
                    "azi": "SamAzi",
                    "x1": "SamX",
                    "x2": "SamY",
                    "x3": "SamZ",
                    "defocus": "CapZ",
                }
            else:
                cls._manipulator_name_conventions = {
                    "polar": "SamPolar",
                    "tilt": "SamTilt",
                    "azi": "SamAzi",
                    "x1": "FSamX",
                    "x2": "FSamY",
                    "x3": "FSamZ",
                    "defocus": "CapZ",
                }

            # Decode info entry:
            if "/Entry/Instrument/Detector/Info" in f:
                infostring = f["/Entry/Instrument/Detector/Info"][()].decode()
                infolist = infostring.split("\r\n")
                Ekinstart = float(infolist[0].replace("Kinetic Start Energy: ", ""))
                Ekinstop = float(infolist[1].replace("Kinetic End Energy: ", ""))
                # If the scan is a sweep scan, save start and stop Ekin, else save the center Ekin.
                if "FAT" in f["Entry/Data/Command"][()].decode().split(" "):
                    Ekin = np.array([Ekinstart, Ekinstop]) * ureg.eV
                else:
                    Ekin = (Ekinstart + Ekinstop) / 2 * ureg.eV
                Sweeps = int(infolist[2].replace("Samples: ", ""))
                Estep = float(infolist[3].replace("Step Width: ", "")) * ureg.eV
                Epass = float(infolist[4].replace("Pass Energy: ", "")) * ureg.eV
                Tdwell = float(infolist[5].replace("Dwell Time: ", "")) * ureg.s
                LensMode = infolist[6].replace("Lens Mode: ", "")
            else:
                Warning("Metadata for Ekin, Sweeps, Dwell ")
                Ekin = None
                Sweeps = None
                Estep = None
                Epass = None
                Tdwell = None
                LensMode = None

            # Decode slow axis entry for deflector scan values:
            if "/Entry/Data/ScanDetails/SlowAxis_names" in f:
                Slownames = f["/Entry/Data/ScanDetails/SlowAxis_names"][()]
                # Account for legacy names
                Slownames = [b"ShiftX" if i == b"SALX" else i for i in Slownames]
                Slownames = [b"ShiftY" if i == b"SALY" else i for i in Slownames]
                if (b"ShiftX" in Slownames) or (b"ShiftY" in Slownames):
                    SlowStart = f["/Entry/Data/ScanDetails/SlowAxis_start"][()]
                    SlowStep = f["/Entry/Data/ScanDetails/SlowAxis_step"][()]
                    SlowLength = f["/Entry/Data/ScanDetails/SlowAxis_length"][()]
                    if b"ShiftX" in Slownames:
                        isx = np.where(np.array(Slownames) == b"ShiftX")[0].item()
                        ShiftX = np.array(
                            [
                                SlowStart[isx].item(),
                                SlowStart[isx].item()
                                + (SlowLength[isx].item() - 1) * SlowStep[isx].item(),
                            ]
                        )
                    else:
                        ShiftX = 0
                    if b"ShiftY" in Slownames:
                        isy = np.where(np.array(Slownames) == b"ShiftY")[0].item()
                        ShiftY = np.array(
                            [
                                SlowStart[isy].item(),
                                SlowStart[isy].item()
                                + (SlowLength[isy].item() - 1) * SlowStep[isy].item(),
                            ]
                        )
                    else:
                        ShiftY = 0
                else:
                    ShiftX = 0
                    ShiftY = 0
            else:
                ShiftX = None
                ShiftY = None

            # Get time stamp:
            if "Entry/Data/Timestamp" in f:
                timestamp = f["Entry/Data/Timestamp"][()][0].decode()

            # Get Analyser angles:
            if "/Entry/Instrument/Detector/AnaAzi" in f:
                AnaAzi = np.mean(f["/Entry/Instrument/Detector/AnaAzi"][()]) * ureg.deg
            else:
                AnaAzi = None
            if "/Entry/Instrument/Detector/AnaPolar" in f:
                AnaPolar = (
                    np.mean(f["/Entry/Instrument/Detector/AnaPolar"][()])
                ) * ureg.deg - cls._anapolaroffset
            else:
                AnaPolar = None

            # Get scan command
            if "Entry/Data/Command" in f:
                ScanCommand = f["Entry/Data/Command"][()].decode()
            else:
                ScanCommand = None

            # Get SamPolar/Tilt/Azi:
            if "Entry/Instrument/Positioner/SamPolar" in f:
                SamPolar = (
                    np.mean(f["Entry/Instrument/Positioner/SamPolar"][()]) / 1000
                ) * ureg.deg - cls._smpolaroffset
            else:
                SamPolar = None
            if "Entry/Instrument/Positioner/SamTilt" in f:
                SamTilt = (
                    np.mean(f["Entry/Instrument/Positioner/SamTilt"][()]) * ureg.deg
                )
            else:
                SamTilt = None
            if "Entry/Instrument/Positioner/SamAzi" in f:
                SamAzi = (
                    np.mean(f["Entry/Instrument/Positioner/SamAzi"][()])
                    / 1000
                    * ureg.deg
                )
            else:
                SamAzi = None

            # Get Sam x,y,z:
            if ("Entry/Instrument/Positioner/FSamX" in f) or (
                "Entry/Instrument/Positioner/SamX" in f
            ):
                if knifeedgescanflag:
                    FSamX = (
                        np.array(
                            [
                                min(f["Entry/Instrument/Positioner/SamX"][()]),
                                max(f["Entry/Instrument/Positioner/SamX"][()]),
                            ]
                        )
                        * ureg.micrometer
                    )
                else:
                    FSamX = (
                        np.array(
                            [
                                min(f["Entry/Instrument/Positioner/FSamX"][()]),
                                max(f["Entry/Instrument/Positioner/FSamX"][()]),
                            ]
                        )
                        * ureg.micrometer
                    )
            else:
                FSamX = None

            if ("Entry/Instrument/Positioner/FSamY" in f) or (
                "Entry/Instrument/Positioner/SamY" in f
            ):
                if knifeedgescanflag:
                    FSamY = (
                        np.array(
                            [
                                min(f["Entry/Instrument/Positioner/SamY"][()]),
                                max(f["Entry/Instrument/Positioner/SamY"][()]),
                            ]
                        )
                        * ureg.micrometer
                    )
                else:
                    FSamY = (
                        np.array(
                            [
                                min(f["Entry/Instrument/Positioner/FSamY"][()]),
                                max(f["Entry/Instrument/Positioner/FSamY"][()]),
                            ]
                        )
                        * ureg.micrometer
                    )
            else:
                FSamY = None

            if ("Entry/Instrument/Positioner/FSamZ" in f) or (
                "Entry/Instrument/Positioner/SamZ" in f
            ):
                if knifeedgescanflag:
                    FSamZ = (
                        np.array(
                            [
                                min(f["Entry/Instrument/Positioner/SamZ"][()]),
                                max(f["Entry/Instrument/Positioner/SamZ"][()]),
                            ]
                        )
                        * ureg.micrometer
                    )
                else:
                    FSamZ = (
                        np.array(
                            [
                                min(f["Entry/Instrument/Positioner/FSamZ"][()]),
                                max(f["Entry/Instrument/Positioner/FSamZ"][()]),
                            ]
                        )
                        * ureg.micrometer
                    )
            else:
                FSamZ = None

            # Get Manipulator focus:
            if "Entry/Instrument/Positioner/CapZ" in f:
                CapZ = (
                    np.array(
                        [
                            min(f["Entry/Instrument/Positioner/CapZ"][()]),
                            max(f["Entry/Instrument/Positioner/CapZ"][()]),
                        ]
                    )
                    * ureg.micrometer
                )
            else:
                CapZ = None

            # Get light source settings:
            if "/Entry/Instrument/Monochromator/Photon Energy" in f:
                hv = f["/Entry/Instrument/Monochromator/Photon Energy"][()][0] * ureg.eV
            else:
                hv = None
            if "/Entry/Instrument/Monochromator/Exit Slit" in f:
                ExitSlit = (
                    f["/Entry/Instrument/Monochromator/Exit Slit"][()][0]
                    * ureg.millimeter
                )
            else:
                ExitSlit = None

            # Get optics settings:
            if "Entry/Instrument/Positioner/CapX" in f:
                CapX = (
                    np.mean(f["Entry/Instrument/Positioner/CapX"][()]) * ureg.micrometer
                )
            else:
                CapX = None
            if "Entry/Instrument/Positioner/CapY" in f:
                CapY = (
                    np.mean(f["Entry/Instrument/Positioner/CapY"][()]) * ureg.micrometer
                )
            else:
                CapY = None
            if "Entry/Instrument/Positioner/CapZ" in f:
                # Might be a bit confusing, that this entry will also be a single value, for focus scans.
                # In that case though, the scanned range will be stored under manipulator defocus.
                CapZ = (
                    np.mean(f["Entry/Instrument/Positioner/CapZ"][()]) * ureg.micrometer
                )
            else:
                CapZ = None

            # Get temperature settings:
            if "Entry/Instrument/Temperature/Cold Head" in f:
                val = f["Entry/Instrument/Temperature/Cold Head"][()]
                if val.size > 0:
                    ColdHead = np.mean(val)
                else:
                    ColdHead = None
            else:
                ColdHead = None
            if "Entry/Instrument/Temperature/Heater Power" in f:
                val = f["Entry/Instrument/Temperature/Heater Power"][()]
                if val.size > 0:
                    HeaterPower = np.mean(val)
                else:
                    HeaterPower = None
            else:
                HeaterPower = None
            if "Entry/Instrument/Temperature/Sample Stage" in f:
                val = f["Entry/Instrument/Temperature/Sample Stage"][()]
                if val.size > 0:
                    SampleStage = np.mean(val)
                else:
                    SampleStage = None
            else:
                SampleStage = None
            if "Entry/Instrument/Temperature/Stinger" in f:
                val = f["Entry/Instrument/Temperature/Stinger"][()]
                if val.size > 0:
                    Stinger = np.mean(val)
                else:
                    Stinger = None
            else:
                Stinger = None

            # Get Keithley settings:
            if "/Entry/Instrument/Device/KeithleyA_Current" in f:
                val = f["/Entry/Instrument/Device/KeithleyA_Current"]
                if val.size > 0:
                    KA_I = np.mean(val)
                else:
                    KA_I = None
            else:
                KA_I = None

            if "/Entry/Instrument/Device/KeithleyA_Voltage" in f:
                val = f["/Entry/Instrument/Device/KeithleyA_Voltage"]
                if val.size > 0:
                    KA_V = np.mean(val)
                else:
                    KA_V = None
            else:
                KA_V = None

            if "/Entry/Instrument/Device/KeithleyA_Resistance" in f:
                val = f["/Entry/Instrument/Device/KeithleyA_Resistance"]
                if val.size > 0:
                    KA_R = np.mean(val)
                else:
                    KA_R = None
            else:
                KA_R = None

            if "/Entry/Instrument/Device/KeithleyB_Current" in f:
                val = f["/Entry/Instrument/Device/KeithleyB_Current"]
                if val.size > 0:
                    KB_I = np.mean(val)
                else:
                    KB_I = None
            else:
                KB_I = None

            if "/Entry/Instrument/Device/KeithleyB_Voltage" in f:
                val = f["/Entry/Instrument/Device/KeithleyB_Voltage"]
                if val.size > 0:
                    KB_V = np.mean(val)
                else:
                    KB_V = None
            else:
                KB_V = None

            if "/Entry/Instrument/Device/KeithleyB_Resistance" in f:
                val = f["/Entry/Instrument/Device/KeithleyB_Resistance"]
                if val.size > 0:
                    KB_R = np.mean(val)
                else:
                    KB_R = None
            else:
                KB_R = None

            metadata = {
                "timestamp": timestamp,
                "scan_command": ScanCommand,
                "manipulator_polar": SamPolar,
                "manipulator_tilt": SamTilt,
                "manipulator_azi": SamAzi,
                "manipulator_x1": FSamX,
                "manipulator_x2": FSamY,
                "manipulator_x3": FSamZ,
                "manipulator_defocus": CapZ,
                "analyser_model": "Phoibos 150 SAL",
                "analyser_eV": Ekin,
                "analyser_eV_type": "kinetic",
                "analyser_step_size": Estep,
                "analyser_PE": Epass,
                "analyser_dwell": Tdwell,
                "analyser_sweeps": Sweeps,
                "analyser_azi": AnaAzi,
                "analyser_polar": AnaPolar,
                "analyser_tilt": None,
                "analyser_lens_mode": LensMode,
                "analyser_deflector_perp": ShiftX,
                "analyser_deflector_parallel": ShiftY,
                "photon_hv": hv,
                "photon_exit_slit": ExitSlit,
                "photon_polarisation": "LH",
                "optics_x1": CapX,
                "optics_x2": CapY,
                "optics_x3": CapZ,
                "temperature_cryostat": ColdHead,
                "temperature_heater_power": HeaterPower,
                "temperature_sample": SampleStage,
                "temperature_precooling_stage": Stinger,
                "keithleyA_Current": KA_I,
                "keithleyA_Voltage": KA_V,
                "keithleyA_Resistance": KA_R,
                "keithleyB_Current": KB_I,
                "keithleyB_Voltage": KB_V,
                "keithleyB_Resistance": KB_R,
            }
        return metadata
