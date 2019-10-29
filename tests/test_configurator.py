import pprint

import pytest
from timer_cm import Timer

from morphocut import Node, Output, Pipeline, RawOrVariable, ReturnOutputs
from morphocut.configurator import Configurator
from morphocut.stream import FromIterable

N_ITEMS = 10000


@ReturnOutputs
@Output("result")
class Add(Node):
    """
    A Node.

    Args:
        a: A parameter.
    """

    def __init__(self, a: RawOrVariable[int], b: RawOrVariable[int]):
        super().__init__()
        self.a = a
        self.b = b

    def transform(self, a, b):
        return a + b


def test_Configurator_passthrough():
    with Pipeline() as pipeline:
        conf = Configurator()

        a = FromIterable(range(N_ITEMS))

        b = conf.node("add1", Add, a, a)
        c = conf.node("add2", Add, a, a)

    assert b.node.id == "add1"
    assert c.node.id == "add2"

    with Timer("Configurator_passthrough ({})".format(N_ITEMS)):
        stream = list(pipeline.transform_stream())
    result = [o[b] for o in stream]
    assert result == [2 * x for x in range(N_ITEMS)]


def test_Configurator_set():

    with Pipeline() as pipeline:
        conf = Configurator()

        a = FromIterable(range(N_ITEMS))

        conf.variable("a", a)
        b = conf.node("add1", Add, a)
        c = conf.node("add2", Add, a)

    # add1: Add
    # In:
    #   - a: int = a
    #   - b: int = <missing>
    # add2: Add
    # In:
    #   - a: int = a
    #   - b: int = <missing>

    conf.set("add1.b", a)
    conf.set("add2.b", 1)

    with Timer("Configurator_missing ({})".format(N_ITEMS)):
        stream = list(pipeline.transform_stream())

    result = [o[b] for o in stream]
    assert result == [2 * x for x in range(N_ITEMS)]

    result = [o[c] for o in stream]
    assert result == [x + 1 for x in range(N_ITEMS)]


def test_Configurator_missing():

    with Pipeline() as pipeline:
        conf = Configurator()

        a = FromIterable(range(N_ITEMS))

        conf.variable("a", a)
        b = conf.node("add1", Add, a)

    with pytest.raises(KeyError):
        pipeline.run()
