from morphocut import graph

_pipeline_stack = []


class Pipeline:
    def __init__(self):
        self.nodes = []

    def __enter__(self):
        # Push self to pipeline stack
        _pipeline_stack.append(self)

        return self

    def __exit__(self, *_):
        # Pop self from pipeline stack
        item = _pipeline_stack.pop()

        assert item is self

    def transform_stream(self, stream=None):
        if stream is None:
            stream = [{}]

        for node in self.nodes:
            stream = node.transform_stream(stream)

        return stream

    def run(self):
        for _ in self.transform_stream():
            pass

    def _add_node(self, node):
        self.nodes.append(node)

    def __str__(self):
        return "Pipeline([{}])".format(", ".join(str(n) for n in self.nodes))
