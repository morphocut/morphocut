from morphocut.graph import Node, Input, Output
import copy
import pytest


@Output("outp")
class Source(Node):
    def transform(self):
        for i in range(10):
            yield i


@Input("inp")
@Output("outp")
class Inner(Node):
    def transform(self, inp):
        return copy.copy(inp)


@Input("inp")
@Output("outp1")
@Output("outp2")
class InnerMultiOut(Node):
    def transform(self, inp):
        return copy.copy(inp)


@Input("inp1")
@Input("inp2")
@Output("outp")
class InnerMultiIn(Node):
    def transform(self, inp1, inp2):
        return copy.copy(inp1)


@Input("inp")
class Sink(Node):
    def transform(self, inp):
        return None


def test_node():
    source = Source()
    # 1A: bind unambigous node port with *args
    inner = Inner()(source)
    # 1B: bind explicit node port with *args
    inner2 = Inner()(inner.outp)
    # 2A: bind unambigous node port with **kargs
    inner3 = Inner()(inp=inner2)
    # 2B: bind explicit node port with **kargs
    inner4 = Inner()(inp=inner3.outp)
    sink = Sink()(inner4)

    # Check the ports
    assert isinstance(inner.inp, Input)
    assert isinstance(inner.outp, Output)

    assert inner.inp._node is inner
    assert inner.outp._node is inner

    # Unexpected keyword argument
    with pytest.raises(TypeError):
        Inner()(foo=source)

    # Ambigous output
    innermulti = InnerMultiOut()(source)
    with pytest.raises(ValueError):
        Inner()(innermulti)

    # Predecessors
    assert inner2.get_predecessors() == {inner}

    innermultiin = InnerMultiIn()(source, inner)
    assert innermultiin.get_predecessors() == {source, inner}
