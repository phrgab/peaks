import dask.array
import h5py
import numpy as np
import pytest

from peaks.core.fileIO.loaders.diamond import _i09_accumulation_time, _read_szip_chunk


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
