import itertools

import numpy as np
import pytest

from morphocut.core import Pipeline
from morphocut.hdf5 import HDF5Writer
from morphocut.stream import Unpack

h5py = pytest.importorskip("h5py")


def test_HDF5Writer_extend(tmp_path):
    filenames = [tmp_path / "test0.h5", tmp_path / "test1.h5"]
    values = np.arange(10 * 5 * 5).reshape(10, 5, 5)
    with Pipeline() as pipeline:
        h5_fn = Unpack(filenames)
        arr = Unpack(values)
        HDF5Writer(h5_fn, {"arr": arr}, dataset_mode="extend")

    pipeline.run()

    for h5_fn in filenames:
        with h5py.File(h5_fn, "r") as h5f:
            h5_values = h5f["arr"][:]

        np.testing.assert_equal(h5_values, values.reshape(50, 5))


def test_HDF5Writer_append(tmp_path):
    h5_fn = tmp_path / "test.h5"
    values = np.arange(10 * 5 * 5).reshape(10, 5, 5)
    names = [f"{i}" for i in range(10)]

    with Pipeline() as pipeline:
        name, arr = Unpack(zip(names, values)).unpack(2)
        HDF5Writer(h5_fn, {"name": name, "arr": arr}, dataset_mode="append")

    pipeline.run()

    with h5py.File(h5_fn, "r") as h5f:
        h5_values = h5f["arr"][:]
        h5_names = h5f["name"].asstr()[:]

    assert list(h5_names) == names
    np.testing.assert_equal(h5_values, values)
