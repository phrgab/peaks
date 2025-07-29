import os

import natsort
import numpy as np
import pint_xarray
import xarray as xr
from scipy import constants
from tqdm.notebook import tqdm

from peaks.core.fileIO.base_arpes_data_classes.base_arpes_data_class import (
    BaseARPESDataLoader,
)
from peaks.core.fileIO.base_data_classes.base_photon_source_classes import (
    BasePumpProbeClass,
)
from peaks.core.fileIO.loc_registry import register_loader
from peaks.core.utils.misc import analysis_warning

ureg = pint_xarray.unit_registry


@register_loader
class ArtemisPhoibos(BaseARPESDataLoader, BasePumpProbeClass):
    """Data loader for Artemis TR-ARPES system, with the Specs Phoibos analyser."""

    _loc_name = "CLF_Artemis"
    _loc_description = "SPECS Phoibos 100 analyser at CLF Artemis facility"
    _loc_url = "https://www.clf.stfc.ac.uk/Pages/Artemis.aspx"
    _analyser_slit_angle = 90 * ureg.deg

    _manipulator_name_conventions = {
        "polar": "polar",
        "azi": "azi",
        "x1": "x",
        "x2": "y",
        "x3": "z",
    }

    _metadata_parsers = [
        "_parse_analyser_metadata",
        "_parse_manipulator_metadata",
        "_parse_photon_metadata",
        "_parse_temperature_metadata",
        "_parse_pump_photon_metadata",
    ]

    _desired_dim_order = [
        "t",
        "x3",
        "x2",
        "x1",
        "polar",
        "tilt",
        "azi",
        "eV",
        "theta_par",
    ]

    _analyser_include_in_metadata_warn = [
        "eV",
        "PE",
        "sweeps",
        "azi",
    ]
    _manipulator_exclude_from_metadata_warn = ["x1", "x2", "x3", "polar", "tilt", "azi"]
    _temperature_exclude_from_metadata_warn = [
        "cryostat",
        "setpoint",
        "shield",
    ]

    @classmethod
    def _load_data(cls, fpath, lazy, **kwargs):
        """Artemis core data loading

        Parameters
        ----------
        fpath : str
            Path to folder containing time series data. In Artemis structure, should contain Info.tsv,
            Stat Log.tsv and N=xxx folders with the data
        lazy : bool
            Not supported for Artemis data
        kwargs : dict
            Additional keyword arguments for the loader. Defaults used if these are not specified:
            - for image transformation:
                - Ang_offset_px
                - Edge_pos
                - Edge_slope
                - calib2d_fpath
            - for partial data loading:
                - num_scans

        """

        if os.path.splitext(fpath)[1] == ".h5":
            # If the file is an HDF5 file, should be a FeSuMa scan; use the BaseFeSuMaDataLoader
            from peaks.core.fileIO.base_arpes_data_classes.base_fesuma_class import (
                BaseFeSuMaDataLoader,
            )

            data_dict = BaseFeSuMaDataLoader._load_data(fpath, lazy, **kwargs)
            if "delay_pos" in data_dict.get("dims") and "t" not in data_dict.get("dims"):
                coords = data_dict.get("coords")
                units = data_dict.get("units")
                delay_pos_m = (
                    coords.get("delay_pos") * ureg(units.get("delay_pos"))
                ).to("m")
                t = ((delay_pos_m - delay_pos_m[0]) * 2 / ureg.c).to("fs").magnitude
                coords["t"] = t
                coords["delay_pos"] = ("t", coords.pop("delay_pos"))
                new_dims = []
                for dim in data_dict.get("dims"):
                    if dim == "delay_pos":
                        new_dims.append("t")
                    else:
                        new_dims.append(dim)
                data_dict["dims"] = new_dims
                data_dict.get("units")["t"] = "fs"

            return data_dict

        # Parse keyword arguments
        Ang_Offset_px = kwargs.pop("Ang_Offset_px", 0)
        Edge_pos = kwargs.pop("Edge_pos", 10)
        Edge_slope = kwargs.pop("Edge_slope", -1)
        num_scans = kwargs.pop("num_scans", None)
        calib2d_fpath = kwargs.pop(
            "calib2d_fpath",
            os.path.join(
                os.path.dirname(__file__), "calib_files", "Artemis_phoibos100.calib2d"
            ),
        )
        if kwargs:
            raise ValueError(
                f"Unexpected keyword arguments: {kwargs}. "
                f"Valid arguments: Ang_Offset_px, Edge_pos, Edge_slope, num_scans, calib2d_fpath"
            )

        # Read folder structure and get test scan
        scan_folders = cls._get_scan_folders(fpath)
        test_scan = cls._get_test_scan(fpath, scan_folders)
        nx_pixel, ny_pixel = test_scan.shape

        # Load and cache metadata from info file
        metadata_dict = cls._load_metadata(fpath, return_dict_with_raw_keys=True)
        metadata_dict["num_scans"] = num_scans  # Add passed num_scans
        cls._metadata_cache[fpath] = metadata_dict

        # Determine some core parameters / set defaults
        binning = int(1920 / nx_pixel)
        PixelSize = 0.00645
        magnification = 4.41
        WF = 4.215
        KE = float(metadata_dict["KE"])
        PE = float(metadata_dict["PE"])
        rr = (KE - WF) / PE  # retarding ratio

        # Extract calibration data
        calib2d_in = cls._read_calib2d_file(calib2d_fpath)
        calib2d, lens_rr_data = cls._extract_calibration_info(calib2d_in, metadata_dict)

        # Interpolate calibration data
        calib2d = cls._interpolate_calibration_data(
            calib2d, lens_rr_data, rr, metadata_dict
        )

        # Calculate transformations
        eV_xarray, theta_par_xarray, E_correction_xarray, Angular_correction_xarray = (
            cls._calculate_transformations(
                calib2d,
                metadata_dict,
                nx_pixel,
                ny_pixel,
                binning,
                magnification,
                PixelSize,
                Ang_Offset_px,
                Edge_pos,
                Edge_slope,
            )
        )

        # Load and transform data
        return cls._load_and_transform_data(
            fpath,
            scan_folders,
            num_scans,
            nx_pixel,
            ny_pixel,
            eV_xarray,
            theta_par_xarray,
            E_correction_xarray,
            Angular_correction_xarray,
            metadata_dict,
            WF,
        )

    @staticmethod
    def _get_scan_folders(fname):
        """Retrieve list of scan folders from the data directory."""
        file_list = natsort.natsorted(os.listdir(fname))
        scan_folders = [item for item in file_list if "N=" in item]
        return scan_folders

    @staticmethod
    def _get_test_scan(fname, scan_folders):
        """Load an example dispersion to determine image size."""
        test_scan_path = os.path.join(
            fname,
            scan_folders[0],
            natsort.natsorted(os.listdir(os.path.join(fname, scan_folders[0])))[0],
        )
        test_scan = np.loadtxt(test_scan_path)
        return test_scan

    @staticmethod
    def _read_calib2d_file(calib2d_fpath):
        """Read the calib2d file containing calibration data."""
        with open(calib2d_fpath, "r") as f:
            calib2d_in = f.readlines()
        return calib2d_in

    @staticmethod
    def _extract_calibration_info(calib2d_in, meta):
        """Extract calibration data relevant to the current lens mode."""
        calib2d = {}
        lens_rr_data = {
            "lens_rr": [],
            "aInner": [],
            "Da1": {"Da1_1": [], "Da1_2": [], "Da1_3": []},
            "Da3": {"Da3_1": [], "Da3_2": [], "Da3_3": []},
            "Da5": {"Da5_1": [], "Da5_2": [], "Da5_3": []},
            "Da7": {"Da7_1": [], "Da7_2": [], "Da7_3": []},
        }

        i0 = None
        i1 = None

        for ct, line in enumerate(calib2d_in):
            # Extract constants
            if "eRange" in line:
                calib2d["eRange"] = [float(j) for j in line.split()[2:4]]
            if "De1" in line:
                calib2d["De1"] = float(line.split("= ")[1])

            # Find relevant block
            if meta["LensMode"] in line and "#" in line:
                i0 = ct + 1
            elif "# ===" in line and i0 and ct - i0 > 2:
                i1 = ct
                break

        calib_lens = calib2d_in[i0:i1] if i1 else calib2d_in[i0:]

        for line in calib_lens:
            # Static parameters
            if "aUnit" in line:
                calib2d["aUnit"] = line.split('"')[1]
            if "aRange" in line:
                calib2d["aRange"] = [float(j) for j in line.split()[2:4]]
            if "eShift" in line:
                calib2d["eShift"] = [float(j) for j in line.split()[2:5]]

            # Retarding ratio-dependent parameters
            if meta["LensMode"] in line and "@" in line:
                rr_value = float(line.split("@")[1].split("]")[0])
                lens_rr_data["lens_rr"].append(rr_value)
            if "aInner" in line:
                aInner_value = float(line.split("= ")[1].split("#")[0])
                lens_rr_data["aInner"].append(aInner_value)
            for Da in ["Da1", "Da3", "Da5", "Da7"]:
                if Da in line:
                    temp_values = [float(j) for j in line.split()[2:5]]
                    lens_rr_data[Da][f"{Da}_1"].append(temp_values[0])
                    lens_rr_data[Da][f"{Da}_2"].append(temp_values[1])
                    lens_rr_data[Da][f"{Da}_3"].append(temp_values[2])

        return calib2d, lens_rr_data

    @staticmethod
    def _interpolate_calibration_data(calib2d, lens_rr_data, rr, meta):
        """Interpolate calibration data for the current retarding ratio."""
        lens_rr = lens_rr_data["lens_rr"]
        calib2d["aInner"] = np.interp(rr, lens_rr, lens_rr_data["aInner"])
        eV_coords = np.array(calib2d["eShift"]) * float(meta["PE"]) + float(meta["KE"])

        for Da in ["Da1", "Da3", "Da5", "Da7"]:
            Da_values = np.array(
                [
                    np.interp(rr, lens_rr, lens_rr_data[Da][f"{Da}_1"]),
                    np.interp(rr, lens_rr, lens_rr_data[Da][f"{Da}_2"]),
                    np.interp(rr, lens_rr, lens_rr_data[Da][f"{Da}_3"]),
                ]
            )
            calib2d[Da] = xr.DataArray(
                data=Da_values, dims="eV", coords={"eV": eV_coords}
            )

        # Polynomial fits for Da coefficients
        calib2d["Da_value"] = eV_coords
        for Da in ["Da1", "Da3", "Da5", "Da7"]:
            calib2d[f"{Da}_coeff"] = np.polyfit(calib2d["Da_value"], calib2d[Da], 2)
        return calib2d

    @staticmethod
    def _calculate_transformations(
        calib2d,
        meta,
        nx_pixel,
        ny_pixel,
        binning,
        magnification,
        PixelSize,
        Ang_Offset_px,
        Edge_pos,
        Edge_slope,
    ):
        """Calculate energy and angular correction transformations."""
        # Energy range
        Ek_low = float(meta["KE"]) + float(calib2d["eRange"][0]) * float(meta["PE"])
        Ek_high = float(meta["KE"]) + float(calib2d["eRange"][1]) * float(meta["PE"])
        theta_par_low = float(calib2d["aRange"][0])
        theta_par_high = float(calib2d["aRange"][1])

        # Create energy and angle arrays
        eV_xarray = xr.DataArray(
            data=np.linspace(Ek_low, Ek_high, nx_pixel),
            dims="eV",
            coords={"eV": np.linspace(Ek_low, Ek_high, nx_pixel)},
        )
        theta_par_xarray = xr.DataArray(
            data=np.linspace(theta_par_low, theta_par_high, ny_pixel),
            dims="theta_par",
            coords={"theta_par": np.linspace(theta_par_low, theta_par_high, ny_pixel)},
        )

        # Energy correction
        E_correction = np.round(
            (eV_xarray - float(meta["KE"]))
            / float(meta["PE"])
            / float(calib2d["De1"])
            / magnification
            / (PixelSize * binning)
            + nx_pixel / 2
        )

        # Interpolation for energy correction
        eV_i = xr.DataArray(
            data=eV_xarray.data,
            dims="eV_i",
            coords={"eV_i": np.linspace(0, nx_pixel - 1, nx_pixel)},
        )
        E_correction_interp = xr.DataArray(
            data=E_correction,
            dims="eV_i",
            coords={"eV_i": np.linspace(0, nx_pixel - 1, nx_pixel)},
        )
        E_correction_xarray = xr.DataArray(
            data=eV_i.interp(eV_i=E_correction_interp).data,
            dims="eV",
            coords={"eV": eV_xarray.eV},
        )

        # Interpolate Da coefficients
        Da_coeffs = {}
        for Da in ["Da1", "Da3", "Da5", "Da7"]:
            Da_coeffs[Da] = np.polyval(calib2d[f"{Da}_coeff"], eV_xarray)
            Da_coeffs[Da] = xr.DataArray(
                data=Da_coeffs[Da], dims="eV", coords={"eV": eV_xarray.eV}
            )

        # Calculate zInner_data and zInner_Diff_data
        theta_par = theta_par_xarray
        zInner_data = (
            Da_coeffs["Da1"] * theta_par
            + 1e-2 * Da_coeffs["Da3"] * theta_par**3
            + 1e-4 * Da_coeffs["Da5"] * theta_par**5
            + 1e-6 * Da_coeffs["Da7"] * theta_par**7
        )
        zInner_Diff_data = (
            Da_coeffs["Da1"]
            + 3e-2 * Da_coeffs["Da3"] * theta_par**2
            + 5e-4 * Da_coeffs["Da5"] * theta_par**4
            + 7e-6 * Da_coeffs["Da7"] * theta_par**6
        )

        # MCP Position calculations
        aInner = calib2d["aInner"]
        MCP_Position_mm_1 = zInner_data.where(abs(theta_par) <= aInner).fillna(0)
        MCP_Position_mm_2 = (
            (
                np.sign(theta_par)
                * (zInner_data + (abs(theta_par) - aInner) * zInner_Diff_data)
            )
            .where(abs(theta_par) > aInner)
            .fillna(0)
        )
        MCP_Position_mm = MCP_Position_mm_1 + MCP_Position_mm_2

        # Edge correction
        Edge_Coef = (
            np.tan(np.radians(Edge_slope))
            / MCP_Position_mm.interp(eV=float(meta["KE"]), theta_par=Edge_pos).data
            / float(meta["PE"])
            / float(calib2d["De1"])
        )
        w_LinearCorrection = 1 / (1 + (Edge_Coef * (eV_xarray - float(meta["KE"]))))

        Angular_correction = (
            (w_LinearCorrection * MCP_Position_mm)
            / magnification
            / (PixelSize * binning)
            + ny_pixel / 2
            + Ang_Offset_px
        )

        # Interpolation for angular correction
        theta_par_i = xr.DataArray(
            data=theta_par_xarray.data,
            dims="theta_par_i",
            coords={"theta_par_i": np.linspace(0, ny_pixel - 1, ny_pixel)},
        )
        Angular_correction_interp = xr.DataArray(
            data=Angular_correction,
            dims=["eV_i", "theta_par_i"],
            coords={
                "eV_i": np.linspace(0, nx_pixel - 1, nx_pixel),
                "theta_par_i": np.linspace(0, ny_pixel - 1, ny_pixel),
            },
        )
        Angular_correction_xarray = xr.DataArray(
            data=theta_par_i.interp(theta_par_i=Angular_correction_interp).data,
            dims=["eV", "theta_par"],
            coords={"eV": eV_xarray.eV, "theta_par": theta_par_xarray.theta_par},
        )

        return (
            eV_xarray,
            theta_par_xarray,
            E_correction_xarray,
            Angular_correction_xarray,
        )

    @staticmethod
    def _load_and_transform_data(
        fname,
        scan_folders,
        num_scans,
        nx_pixel,
        ny_pixel,
        eV_xarray,
        theta_par_xarray,
        E_correction_xarray,
        Angular_correction_xarray,
        meta,
        WF,
    ):
        """Load raw data and apply transformations."""
        # Determine scan folder
        if num_scans:
            scan_folder = os.path.join(fname, f"N={num_scans}")
        else:
            scan_folder = os.path.join(fname, scan_folders[-1])
            num_scans = int(scan_folders[-1].split("=")[1])

        # Get scan names
        scan_names = [
            i for i in natsort.natsorted(os.listdir(scan_folder)) if ".tsv" in i
        ]
        num_delays = len(scan_names)

        # Initialize data arrays
        delay_pos = np.empty(num_delays)
        data = np.empty((nx_pixel, ny_pixel, num_delays))

        # Load data
        for i, scan_name in tqdm(
            enumerate(scan_names), total=len(scan_names), desc="Loading scans"
        ):
            delay_pos[i] = float(scan_name.split(".tsv")[0])
            data[:, :, i] = np.loadtxt(os.path.join(scan_folder, scan_name))

        # Calculate delay times
        delay_time = (
            (delay_pos - float(meta["Time Zero"])) * 2 / constants.c * 1e12
        )  # Convert to fs

        # Create xarray DataArray
        data_xarray = xr.DataArray(
            data=data,
            dims=["eV", "theta_par", "t"],
            coords={
                "eV": eV_xarray.eV,
                "theta_par": theta_par_xarray.theta_par,
                "t": delay_time,
            },
        )

        # Apply transformations
        spectrum = data_xarray.interp(
            eV=E_correction_xarray, theta_par=Angular_correction_xarray
        )

        return {
            "spectrum": spectrum.data,
            "dims": ["eV", "theta_par", "t"],
            "coords": {
                "eV": eV_xarray.eV.data - WF,
                "theta_par": theta_par_xarray.theta_par.data,
                "t": delay_time,
                "delay_pos": ("t", delay_pos),
            },
            "units": {
                "eV": "eV",
                "theta_par": "deg",
                "t": "fs",
                "delay_pos": "mm",
                "spectrum": "counts",
            },
        }

    @classmethod
    def _load_metadata(cls, fpath, return_dict_with_raw_keys=False):
        if os.path.splitext(fpath)[1] == ".h5":
            # If the file is an HDF5 file, should be a FeSuMa scan; use the BaseFeSuMaDataLoader
            from peaks.core.fileIO.base_arpes_data_classes.base_fesuma_class import (
                BaseFeSuMaDataLoader,
            )

            return BaseFeSuMaDataLoader._load_metadata(fpath)

        # Check if there is a cached version (only cache in this loader with keys in artemis format)
        metadata_dict_artemis_keys = cls._metadata_cache.get(fpath)

        if not metadata_dict_artemis_keys:
            # Load metadata from info file
            with open(os.path.join(fpath, "Info.tsv"), "r") as f:
                meta = f.readlines()
            metadata_dict_artemis_keys = {
                item.split(":", 1)[0]: item.split(":", 1)[1]
                for item in meta[0].split(";")
                if len(item.split(":")) == 2
            }
            for key, value in metadata_dict_artemis_keys.copy().items():
                try:
                    if key != "Delay List":
                        metadata_dict_artemis_keys[key] = float(value)
                except ValueError:
                    pass

            # Load stats file for parsing e.g. sample temperature
            metadata_dict_artemis_keys.update(cls._load_stats(fpath))

        if return_dict_with_raw_keys:
            return metadata_dict_artemis_keys

        # Otherwise parse to peaks format
        return cls._parse_metadata(metadata_dict_artemis_keys, fpath)

    @staticmethod
    def _load_stats(fpath):
        # Load and parse stats file
        stat_log = np.loadtxt(os.path.join(fpath, "Stat Log.tsv"))
        return {
            "measurement_time": stat_log[:, 0],
            "measurement_stats": stat_log[:, 1],
            "sample_temp": stat_log[:, 2],
        }

    @classmethod
    def _get_max_num_scans(cls, meta, fpath):
        if meta.get("num_scans"):
            return int(meta["num_scans"])
        else:
            scan_folders = cls._get_scan_folders(fpath)
            return max(int(folder.split("=")[1]) for folder in scan_folders)

    @classmethod
    def _parse_metadata(cls, metadata_dict_artemis_keys, fpath):
        """Convert Artemis metadata dictionary keys to peaks keys, and add some default metadata."""

        def _parse_delays(input_string):
            if input_string is None:
                return None
            ranges = (
                [input_string] if "LF" not in input_string else input_string.split("#LF")
            )
            delays = []

            for r in ranges:
                if "_" in r:
                    start, stop, step = map(int, r.split("_"))
                    delays.extend(
                        range(start, stop + step, step)
                    )  # +step to include the endpoint if it's on the step

            delays = np.asarray(delays)
            return np.array([delays[0], delays[-1]]) * ureg("fs")

        def _parse_tempterature(input_array):
            if input_array is None:
                return None
            input_array = input_array * ureg("K")
            # Check if temperature varies significantly over the run
            temp_diff = abs(np.ptp(input_array))
            if temp_diff > 5 * ureg("K"):
                analysis_warning(
                    f"Significant temperature variation of {temp_diff:.2f} over the run, from "
                    f"{min(input_array):.2f} to {max(input_array):.2f}. Average value of {np.mean(input_array):.2f} "
                    f"used in metadata. To view the full data, load the stats file: "
                    f"`from peaks.core.fileIO.loaders.clf import ArtemisPhoibos; ArtemisPhoibos.load_stats(fpath)`",
                    "warning",
                    "Temperature variation warning",
                )
            return np.mean(input_array)

        # Define metadata
        return {
            "analyser_model": "Phoibos 100",
            "analyser_slit_width": None,
            "analyser_slit_width_identifier": None,
            "analyser_eV": (
                metadata_dict_artemis_keys.get("KE") * ureg("eV")
                if metadata_dict_artemis_keys.get("KE")
                else None
            ),
            "analyser_eV_type": "kinetic",
            "analyser_step_size": None,
            "analyser_PE": (
                metadata_dict_artemis_keys.get("PE") * ureg("eV")
                if metadata_dict_artemis_keys.get("PE")
                else None
            ),
            "analyser_sweeps": cls._get_max_num_scans(metadata_dict_artemis_keys, fpath),
            "analyser_dwell": None,
            "analyser_lens_mode": metadata_dict_artemis_keys.get("LensMode"),
            "analyser_acquisition_mode": "fixed",
            "analyser_polar": None,
            "analyser_tilt": None,
            "analyser_azi": cls._analyser_slit_angle,
            "timestamp": metadata_dict_artemis_keys.get("Run Start"),
            "manipulator_polar": None,
            "manipulator_tilt": None,
            "manipulator_azi": None,
            "manipulator_x1": None,
            "manipulator_x2": None,
            "manipulator_x3": None,
            "photon_hv": "Excitation Energy",
            "pump_hv": None,
            "pump_power": None,
            "pump_delay": _parse_delays(metadata_dict_artemis_keys.get("Delay List")),
            "pump_polarisation": None,
            "pump_t0_position": float(metadata_dict_artemis_keys.get("Time Zero"))
            * ureg("mm"),
            "temperature_sample": _parse_tempterature(
                metadata_dict_artemis_keys.get("sample_temp")
            ),
        }

    @classmethod
    def load_stats(cls, fpath, metadata=True, quiet=False):
        """Return the signal and temperature stats vs. time from an Artemis measurement run

        Parameters
        ----------
        fpath : str
            Path to the folder containing the Artemis data

        metadata : bool
            Whether to include additional metadata in the returned dataset

        quiet : bool
            Whether to suppress warnings, defaults to False

        Returns
        -------
        xarray.Dataset
            Dataset containing the temperature and signal stats vs. measurement time
        """

        stats = cls._load_stats(fpath)
        stats_da = xr.DataArray(
            data=stats["measurement_stats"],
            dims="measurement_time",
            coords={"measurement_time": stats["measurement_time"]},
            name="stats",
        ).pint.quantify({"stats": "counts", "measurement_time": "min"})
        temp_da = xr.DataArray(
            data=stats["sample_temp"],
            dims="measurement_time",
            coords={"measurement_time": stats["measurement_time"]},
            name="temperature",
        ).pint.quantify({"temperature": "K", "measurement_time": "min"})
        ds = xr.Dataset({"stats": stats_da, "temperature": temp_da})

        # Load the metadata for adding to the ds if desired
        metadata_dict = cls.load_metadata(
            fpath=fpath,
            loc=cls._loc_name,
            quiet=quiet,
            load_metadata_from_file=metadata,
        )

        ds.attrs.update(metadata_dict)

        return ds
