"""Test morphocut.parallel."""


import multiprocessing
import os
import signal

import pytest
from timer_cm import Timer

from morphocut import Call, Node, Pipeline
from morphocut.parallel import ParallelPipeline, WorkerDiedException
from morphocut.stream import Unpack
from tests.helpers import Sleep

N_STEPS = 31


def test_parallel_pipeline():
    print("start method:", multiprocessing.get_start_method())
    inp = list(range(N_STEPS))
    with Pipeline() as pipeline:
        x = Unpack(inp)
        with ParallelPipeline(2):
            pass

    result = [obj[x] for obj in pipeline.transform_stream()]

    assert set(result) == set(inp)


def test_speed():

    with Pipeline() as pipeline:
        level1 = Unpack(range(N_STEPS))
        level2 = Unpack(range(N_STEPS))
        Sleep()

    with Timer("sequential") as t:
        expected_result = [
            (obj[level1], obj[level2]) for obj in pipeline.transform_stream()
        ]

    elapsed_sequential = t.elapsed

    with Pipeline() as pipeline:
        level1 = Unpack(range(N_STEPS))
        level2 = Unpack(range(N_STEPS))
        with ParallelPipeline(4):
            Sleep()

    with Timer("parallel") as t:
        result = [(obj[level1], obj[level2]) for obj in pipeline.transform_stream()]

    elapsed_parallel = t.elapsed

    # Make sure the calculated result matches the expected (ignoring order)
    assert sorted(result) == sorted(expected_result)

    assert elapsed_parallel < elapsed_sequential


class SomeException(Exception):
    pass


def test_exception_main_thread():

    with Pipeline() as pipeline:
        level1 = Unpack(range(N_STEPS))
        with ParallelPipeline(4) as pp:
            level2 = Unpack(range(N_STEPS))
            Sleep()

    stream = pipeline.transform_stream()

    try:
        with pytest.raises(SomeException):
            for i, obj in enumerate(stream):
                if i == 10:
                    raise SomeException()
    finally:
        stream.close()


class Raiser(Node):
    def transform(self):
        raise SomeException("foo")


def test_exception_worker():

    with Pipeline() as pipeline:
        level1 = Unpack(range(N_STEPS))
        with ParallelPipeline(4) as pp:
            Sleep()
            Raiser()

    with pytest.raises(SomeException, match="foo"):
        pipeline.run()

    # TODO: Make sure that all processes are stopped


class KeyErrorRaiser(Node):
    def transform(self):
        raise KeyError("foo")


def test_KeyError():
    with Pipeline() as pipeline:
        with ParallelPipeline(4):
            KeyErrorRaiser()

    with pytest.raises(KeyError, match="foo"):
        pipeline.run()


def test_exception_upstream():

    with Pipeline() as pipeline:
        level1 = Unpack(range(N_STEPS))
        Raiser()
        with ParallelPipeline(4) as pp:
            level2 = Unpack(range(N_STEPS))

    with pytest.raises(SomeException, match="foo"):
        pipeline.run()


@pytest.mark.parametrize("num_workers", [1, 2, 3, 4])
def test_num_workers(num_workers):
    with Pipeline() as pipeline:
        level1 = Unpack(range(N_STEPS))
        with ParallelPipeline(num_workers) as pp:
            level2 = Unpack(range(N_STEPS))

    pipeline.run()


def test_worker_die():

    with Pipeline() as pipeline:
        level1 = Unpack(range(N_STEPS))
        with ParallelPipeline(4):
            Call(lambda: os.kill(os.getpid(), signal.SIGKILL))

    with pytest.raises(
        WorkerDiedException,
        match=r".*_Worker-\d+ died unexpectedly. Exit code: -SIGKILL",
    ):
        pipeline.run()
