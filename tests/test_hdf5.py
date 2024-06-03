import itertools

import h5py
import pytest

from morphocut.core import Pipeline
from morphocut.hdf5 import HDF5Writer
from morphocut.stream import Unpack
import itertools


def test_HDF5Writer_extend(tmp_path):
    h5_fn = tmp_path / "test.h5"
    values = [range(10) for _ in range(100)]
    with Pipeline() as pipeline:
        arr = Unpack(values)
        HDF5Writer(h5_fn, {"arr": arr}, dataset_mode="extend")

    pipeline.run()

    with h5py.File(h5_fn, "r") as h5f:
        h5_values = list(h5f["arr"][:])

    values_expected = list(itertools.chain(*values))
    assert h5_values == values_expected
