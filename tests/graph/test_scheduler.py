import copy

from morphocut.graph import Input, Node, Output
from morphocut.graph.scheduler import SimpleScheduler, _discover_ancestors


@Output("outp")
class Source(Node):
    def transform_stream(self, stream):
        for i in range(10):
            obj = self.prepare_output({}, i)

            assert i in obj.values()

            yield obj


@Input("inp")
@Output("outp")
class Clone(Node):
    def transform(self, inp):
        return copy.copy(inp)


@Input("inp")
class Sink(Node):
    def transform_stream(self, stream):
        numbers = []
        for obj in stream:
            values = self.prepare_input(obj)
            numbers.append(values["inp"])

            yield obj

        assert numbers == list(range(10))


def test_simple_scheduler():
    source = Source()
    clone = Clone()(source)
    sink = Sink()(clone)

    assert _discover_ancestors(sink) == {sink, clone, source}

    sched = SimpleScheduler(sink)
    sched.run()
