from time import sleep

import pytest
from timer_cm import Timer

from morphocut import Node, Pipeline
from morphocut.parallel import ParallelPipeline
from morphocut.stream import FromIterable


class Sleep(Node):
    def transform(self):
        sleep(0.01)


N_STEPS = 32


def test_ParallelPipeline():

    with Pipeline() as pipeline:
        level1 = FromIterable(range(N_STEPS))
        level2 = FromIterable(range(N_STEPS))
        Sleep()

    with Timer("sequential") as t:
        expected_result = [
            (obj[level1], obj[level2]) for obj in pipeline.transform_stream()
        ]

    elapsed_sequential = t.elapsed

    with Pipeline() as pipeline:
        level1 = FromIterable(range(N_STEPS))
        with ParallelPipeline(4, parent=pipeline) as pp:
            level2 = FromIterable(range(N_STEPS))
            Sleep()

    with Timer("parallel") as t:
        result = [(obj[level1], obj[level2]) for obj in pipeline.transform_stream()]

    elapsed_parallel = t.elapsed

    assert result == expected_result

    assert elapsed_parallel < elapsed_sequential


class SomeException(Exception):
    pass


def test_exception_parent():

    with Pipeline() as pipeline:
        level1 = FromIterable(range(N_STEPS))
        with ParallelPipeline(4, parent=pipeline) as pp:
            level2 = FromIterable(range(N_STEPS))
            Sleep()

    stream = pipeline.transform_stream()

    try:
        with pytest.raises(SomeException):
            for i, obj in enumerate(stream):
                if i == 10:
                    raise SomeException()
    finally:
        stream.close()

    # TODO: Make sure that all processes are stopped


class Raiser(Node):
    def transform(self):
        raise SomeException()


def test_exception_child():

    with Pipeline() as pipeline:
        level1 = FromIterable(range(N_STEPS))
        with ParallelPipeline(4, parent=pipeline) as pp:
            Raiser()

    with pytest.raises(SomeException):
        pipeline.run()

    # TODO: Make sure that all processes are stopped
