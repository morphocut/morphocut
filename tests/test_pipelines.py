from morphocut.pipelines import MergeNodesPipeline, DataParallelPipeline, AggregateErrorsPipeline
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


def test_MergeNodesPipeline():
    with Pipeline() as p:
        a = Unpack(range(100))
        with MergeNodesPipeline() as m:
            b = Call(lambda x: x + 1, a)
            c = Call(lambda x: x * 2, a)

    expected_b = [obj[b] for obj in p.transform_stream()]
    expected_c = [obj[c] for obj in p.transform_stream()]

    assert expected_b == list(range(1, 101))
    assert expected_c == list(range(0, 200, 2))


def test_MergeNodesPipeline_on_error():
    error_handler_called = False

    def error_handler(e):
        nonlocal error_handler_called
        assert isinstance(e, ZeroDivisionError)
        error_handler_called = True

    with Pipeline() as p:
        a = Unpack(range(100))
        with MergeNodesPipeline(on_error=error_handler) as m:
            b = Call(lambda x: x + 1, a)
            c = Call(lambda x: x / 0 if x == 50 else x, a)

    p.run()

    assert error_handler_called, "Error handler was not called"



def test_AggregateErrorsPipeline():
    def error_handler(e, x):
        assert isinstance(e, ZeroDivisionError)
        assert x == 50

    with Pipeline() as p:
        a = Unpack(range(100))
        with AggregateErrorsPipeline() as m:
            b = Call(lambda x: x + 1, a)
            c = Call(lambda x: x / 0 if x == 50 else x, a)

    with pytest.raises(ZeroDivisionError):
        p.run()


def test_DataParallelPipeline_executer():
    with Pipeline() as p:
        a = Unpack(range(100))
        with DataParallelPipeline(executor=8):
            b = Call(lambda x: x / 0 if x == 50 else x, a)

    with pytest.raises(ZeroDivisionError):
        p.run()

