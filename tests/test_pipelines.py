from morphocut.pipelines import MergeNodesPipeline, DataParallelPipeline
from morphocut.core import Call, Pipeline
from morphocut.stream import Unpack
import pytest


def test_DataParallelPipeline():
    with Pipeline() as p1:
        a = Unpack(range(100))
        b = Call(lambda x: x + 1, a)

    expected = [obj[b] for obj in p1.transform_stream()]

    with Pipeline() as p2:
        a = Unpack(range(100))
        with DataParallelPipeline(executor=8):
            b = Call(lambda x: x + 1, a)

    actual = [obj[b] for obj in p2.transform_stream()]

    assert expected == actual

    with Pipeline() as p3:
        a = Unpack(range(100))
        with DataParallelPipeline(executor=8):
            b = Call(lambda x: x / 0, a)

    with pytest.raises(ZeroDivisionError):
        p3.run()
