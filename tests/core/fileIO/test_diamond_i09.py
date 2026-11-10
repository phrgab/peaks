import dask.array
import h5py
import numpy as np
import pytest

from peaks.core.fileIO.loaders.diamond import (
    _i09_accumulation_time,
    _i09_energy_addr,
    _read_szip_chunk,
)


@pytest.mark.skipif(
    not h5py.h5z.filter_avail(h5py.h5z.FILTER_SZIP),
    reason="writing the reference file needs an h5py build with szip",
)
def test_read_szip_chunk_matches_h5py(tmp_path):
    fpath = tmp_path / "dld.h5"
    frames = np.arange(2 * 5 * 6, dtype="int32").reshape(2, 5, 6)
    with h5py.File(fpath, "w") as f:
        dset = f.create_dataset(
            "data",
            shape=(4, 5, 6),
            dtype="int32",
            chunks=(1, 5, 6),
            compression="szip",
            fillvalue=-1,
        )
        # Frames 1 and 3 are never written, as in an aborted scan
        dset[0], dset[2] = frames
        expected = dset[()]

    data = dask.array.map_blocks(
        _read_szip_chunk,
        str(fpath),
        "data",
        chunks=dask.array.core.normalize_chunks((1, 5, 6), (4, 5, 6)),
        dtype="int32",
        meta=np.empty((0, 0, 0), dtype="int32"),
    )
    np.testing.assert_array_equal(data.compute(), expected)
    assert (expected[[1, 3]] == -1).all()


@pytest.mark.parametrize(
    "detector, scan_command, expected",
    [
        ("dld", "scan Kenergy 101 106 0.025 dld 20 sm6iamp27", 20),
        ("dld_total", "scan dummy_a 0 10 1 dld_total 0.4 w 1", 0.4),
        ("dld", "scan dummy_a 0 10 1 dld_total 0.4", None),
    ],
)
def test_i09_accumulation_time(tmp_path, detector, scan_command, expected):
    with h5py.File(tmp_path / "scan.nxs", "w") as f:
        f.create_group(f"entry/instrument/{detector}").attrs["scan_role"] = "detector"
        f["entry/diamond_scan/scan_command"] = scan_command
        result = _i09_accumulation_time(f)
    assert (result is None) if expected is None else result.m_as("s") == expected


@pytest.mark.parametrize(
    "recorded, expected",
    [
        (["Benergy", "kenergy", "Kenergy"], "Kenergy"),
        (["Benergy", "kenergy"], "kenergy"),
        (["Benergy"], "Benergy"),
        ([], None),
    ],
)
def test_i09_energy_addr(tmp_path, recorded, expected):
    with h5py.File(tmp_path / "scan.nxs", "w") as f:
        for name in recorded:
            f[f"entry/instrument/{name}/value"] = np.ones(3)
        result = _i09_energy_addr(f)
    assert result == (expected and f"entry/instrument/{expected}/value")


@pytest.mark.parametrize("fname", ["snapshot.nxs", "image.tif"])
def test_i09_live_view_image_needs_loc(tmp_path, fname):
    import imagecodecs

    import peaks as pks

    image = np.arange(4 * 6, dtype="int32").reshape(4, 6)
    fpath = tmp_path / fname
    if fname.endswith(".nxs"):
        with h5py.File(fpath, "w") as f:
            f["entry/data/Live camera stream"] = image
            f["entry/data"].attrs["signal"] = "Live camera stream"
    else:
        imagecodecs.imwrite(fpath, image)

    with pytest.raises(ValueError, match="could not be determined"):
        pks.load(str(fpath))
    da = pks.load(str(fpath), loc="Diamond_I09_Momentum_Microscope")
    assert da.dims == ("detector_y", "detector_x")
    np.testing.assert_array_equal(da.pint.dequantify().values, image)


@pytest.mark.parametrize("monitor_is_signal", [False, True])
def test_i09_I0_coord(tmp_path, monitor_is_signal):
    import peaks as pks

    fpath = tmp_path / "scan.nxs"
    monitor = np.linspace(1, 2, 5)
    with h5py.File(fpath, "w") as f:
        f["entry/diamond_scan/scan_rank"] = 1
        if monitor_is_signal:
            # Mirror scan: the monitor is the signal, with a copy in `instrument`
            f["entry"].attrs["default"] = "sm6iamp27"
            nxdata = f.create_group("entry/sm6iamp27")
            nxdata.attrs.update(signal="sm6iamp27", axes=["sm3fpitch"])
            nxdata["sm3fpitch"] = np.arange(5.0)
            nxdata["sm6iamp27"] = monitor
            f["entry/instrument/sm6iamp27/value"] = monitor
        else:
            f["entry"].attrs["default"] = "dld_total"
            nxdata = f.create_group("entry/dld_total")
            nxdata.attrs.update(signal="fixed_roi_total", axes=["Kenergy"])
            nxdata["Kenergy"] = np.linspace(100, 101, 5)
            nxdata["sm6iamp27"] = monitor
            nxdata["sd11iamp7"] = monitor * 3
            nxdata["panda_scannable"] = monitor * 2
            nxdata["checkbeam"] = np.ones(5)
            detector = f.create_group("entry/instrument/dld_total")
            detector.attrs["scan_role"] = "detector"
            detector["fixed_roi_total"] = np.full(5, 10.0)

    da = pks.load(str(fpath), loc="Diamond_I09_Momentum_Microscope", quiet=True)
    if monitor_is_signal:
        assert "I0" not in da.coords
        assert da.pint.units == pks.ureg.Unit("V")
    else:
        np.testing.assert_array_equal(da.I0.values, monitor)
        np.testing.assert_array_equal(da.sd11iamp7.values, monitor * 3)
        assert da.sd11iamp7.dims == ("eV",)
        assert da.sd11iamp7.pint.units == pks.ureg.Unit("V")
        assert da.panda_scannable.pint.units == pks.ureg.Unit("V")
        assert "checkbeam" not in da.coords
        assert da.I0.pint.units == pks.ureg.Unit("V")
        normed = da.norm_by_I0()
        np.testing.assert_allclose(normed.pint.dequantify().values, 10 / monitor)
        assert normed.pint.units == pks.ureg.Unit("count / V")


@pytest.mark.parametrize(
    "axis, sign, eV_type", [("kenergy", 1, "kinetic"), ("Benergy", -1, "binding")]
)
def test_i09_energy_axis_to_eV(tmp_path, axis, sign, eV_type):
    import peaks as pks

    fpath = tmp_path / "scan.nxs"
    energy = np.linspace(1, 2, 5)
    with h5py.File(fpath, "w") as f:
        f["entry/diamond_scan/scan_rank"] = 1
        f["entry"].attrs["default"] = "dld_total"
        nxdata = f.create_group("entry/dld_total")
        nxdata.attrs.update(signal="fixed_roi_total", axes=[axis])
        nxdata[axis] = energy
        f[f"entry/instrument/{axis}/value"] = energy
        detector = f.create_group("entry/instrument/dld_total")
        detector.attrs["scan_role"] = "detector"
        detector["fixed_roi_total"] = np.full(5, 10.0)

    da = pks.load(str(fpath), loc="Diamond_I09_Momentum_Microscope", quiet=True)
    assert da.dims == ("eV",)
    np.testing.assert_allclose(
        np.sort(da.eV.pint.dequantify().values), np.sort(sign * energy)
    )
    # Metadata holds the energy range
    np.testing.assert_allclose(
        np.sort(da.metadata.analyser.scan.eV.to("eV").magnitude),
        sorted(sign * energy[[0, -1]]),
    )
    assert da.metadata.analyser.scan.eV_type == eV_type


def test_i09_map_dim_order(tmp_path):
    import peaks as pks

    fpath = tmp_path / "scan.nxs"
    with h5py.File(fpath, "w") as f:
        f["entry/diamond_scan/scan_rank"] = 1
        f["entry"].attrs["default"] = "dld"
        nxdata = f.create_group("entry/dld")
        nxdata.attrs.update(signal="data", axes=["Kenergy", ".", "."])
        nxdata["Kenergy"] = np.linspace(100, 101, 3)
        detector = f.create_group("entry/instrument/dld")
        detector.attrs["scan_role"] = "detector"
        detector["data"] = np.ones((3, 4, 5))

    da = pks.load(str(fpath), loc="Diamond_I09_Momentum_Microscope", quiet=True)
    # Constant energy maps (detector_y, detector_x) are the outer dims, as for deflector maps
    assert da.dims == ("detector_y", "eV", "detector_x")


def test_i09_manipulator_motors(tmp_path):
    import peaks as pks

    fpath = tmp_path / "scan.nxs"
    with h5py.File(fpath, "w") as f:
        f["entry/diamond_scan/scan_rank"] = 1
        f["entry"].attrs["default"] = "dld_total"
        nxdata = f.create_group("entry/dld_total")
        nxdata.attrs.update(signal="fixed_roi_total", axes=["Kenergy"])
        nxdata["Kenergy"] = np.linspace(100, 101, 5)
        detector = f.create_group("entry/instrument/dld_total")
        detector.attrs["scan_role"] = "detector"
        detector["fixed_roi_total"] = np.full(5, 10.0)
        for i, motor in enumerate(["sx1", "sx2", "sx3", "sz1", "sz2", "es3x"]):
            f[f"entry/instrument/{motor}/value"] = float(i)
            f[f"entry/instrument/{motor}/value"].attrs["units"] = "mm"

    manipulator = pks.load(
        str(fpath), loc="Diamond_I09_Momentum_Microscope", quiet=True
    ).metadata.manipulator
    assert manipulator.sz2.value == 4 * pks.ureg.mm
    assert manipulator.sz2.local_name == "sz2"
    assert manipulator.polar.value is None
    assert not hasattr(manipulator, "es3x")


@pytest.mark.parametrize(
    "detector, axis, expected",
    [
        (True, "Kenergy", {"analyser_eV", "analyser_dwell", "analyser_step_size"}),
        (True, "dummy_a", {"analyser_eV", "analyser_dwell"}),  # fixed-energy scan
        (False, "sm3fpitch", set()),  # drain-current / mirror scan
    ],
)
def test_i09_analyser_metadata_warnings(tmp_path, monkeypatch, detector, axis, expected):
    import peaks as pks
    from peaks.core.fileIO.loaders.diamond import I09MomentumMicroscopeLoader

    warned = []
    monkeypatch.setattr(
        I09MomentumMicroscopeLoader,
        "_warn_metadata",
        classmethod(lambda cls, metadata, keys: warned.extend(keys)),
    )
    fpath = tmp_path / "scan.nxs"
    with h5py.File(fpath, "w") as f:
        f["entry/diamond_scan/scan_rank"] = 1
        f["entry"].attrs["default"] = "scan"
        nxdata = f.create_group("entry/scan")
        nxdata.attrs.update(signal="signal", axes=[axis])
        nxdata[axis] = np.linspace(100, 101, 5)
        nxdata["signal"] = np.ones(5)
        if detector:
            group = f.create_group("entry/instrument/dld_total")
            group.attrs["scan_role"] = "detector"
            group["fixed_roi_total"] = np.ones(5)
        else:
            f["entry/instrument/sm6iamp27/value"] = np.ones(5)

    pks.load(str(fpath), loc="Diamond_I09_Momentum_Microscope")
    assert {key for key in warned if key.startswith("analyser_")} == expected
    assert "temperature_sample" not in warned
