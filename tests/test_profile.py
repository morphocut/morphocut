import pytest
from timer_cm import Timer

from morphocut import Node, Pipeline
from morphocut.profile import Profile
from morphocut.stream import Unpack
from tests.helpers import Sleep

N = 100
DURATION1 = 0.001
DURATION2 = 0.01


def test_Profile():

    with Pipeline() as pipeline:
        Unpack(range(N))

        # Sleep beforehand to make sure that only the inner Sleep is profiled.
        Sleep(DURATION2)

        with Profile("Sleep") as profile_sleep:
            Sleep(DURATION1)

        # Sleep afterwards to make sure that only the inner Sleep is profiled.
        Sleep(DURATION2)

    objects = list(pipeline.transform_stream())

    assert len(objects) == N

    overhead = profile_sleep._average - DURATION1
    print(f"Overhead {overhead:g}s")

    # Make sure that the overhead is much less than the difference between both durations
    assert overhead < (DURATION2 - DURATION1) / 10
