import pytest
from timer_cm import Timer

from morphocut import Node, Pipeline
from morphocut.profile import Profile
from morphocut.stream import Unpack
from tests.helpers import Sleep

N = 1000
DURATION = 0.001


def test_Profile():

    with Pipeline() as pipeline:
        Unpack(range(N))
        with Profile("Sleep") as profile_sleep:
            Sleep()

    objects = list(pipeline.transform_stream())

    assert len(objects) == N

    overhead = profile_sleep._average - DURATION

    print(f"Overhead {overhead:g}s")
