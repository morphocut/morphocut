from morphocut.parallel import ParallelPipeline
from morphocut import Pipeline, Node
from morphocut.stream import FromIterable
from time import sleep
from timer_cm import Timer


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
