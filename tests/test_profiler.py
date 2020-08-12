import pytest

from morphocut import Pipeline, Call

from morphocut.profiler import Profiler
from morphocut.parallel import ParallelPipeline

import time


def test_Profiler():
    with Pipeline() as pipeline:
        Call(lambda: None)

    with Profiler() as profiler:
        pipeline.run()

    events = profiler.collect()

    assert len(events) == 2


def test_Profiler_parallel():
    with Pipeline() as pipeline:
        with ParallelPipeline():
            Call(lambda: None)

    with Profiler() as profiler:
        pipeline.run()

    events = profiler.collect()

    assert len(events) == 2

    print(events)
