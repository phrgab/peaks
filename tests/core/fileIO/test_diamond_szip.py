import dask.array
import h5py
import numpy as np
import pytest

from peaks.core.fileIO.loaders.diamond import _read_szip_chunk


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
