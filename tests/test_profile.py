import pytest
from timer_cm import Timer

from morphocut import Node, Pipeline
from morphocut.profile import Profile
from morphocut.stream import Unpack
from tests.helpers import Sleep

N = 100
DURATION_INNER = 0.001
DURATION_OUTER = 0.01


@pytest.mark.slow
def test_Profile():

    with Pipeline() as pipeline:
        Unpack(range(N))

        # Sleep beforehand to make sure that only the inner Sleep is profiled.
        Sleep(DURATION_OUTER)

        with Profile("Sleep") as profile_sleep:
            Sleep(DURATION_INNER)

        # Sleep afterwards to make sure that only the inner Sleep is profiled.
        Sleep(DURATION_OUTER)

    objects = list(pipeline.transform_stream())

    assert len(objects) == N

    overhead = profile_sleep._average - DURATION_INNER
    print(f"Overhead {overhead:g}s")

    assert DURATION_INNER <= profile_sleep._average < DURATION_OUTER
