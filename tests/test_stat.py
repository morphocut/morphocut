import numpy as np

from morphocut import Pipeline
from morphocut.stat import RunningMedian
from morphocut.stream import Unpack


def test_RunningMedian_scalar():
    with Pipeline() as p:
        value = Unpack(range(20))
        running_median = RunningMedian(value, n_init=2)

    p.run()


def test_RunningMedian_numpy():
    with Pipeline() as p:
        value = Unpack(np.arange(20)[:, np.newaxis] * np.ones((10)))
        running_median = RunningMedian(value, n_init=2)

    p.run()
