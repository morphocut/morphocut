import collections


class DependencyError(Exception):
    pass


def _discover_ancestors(sink):
    ancestors = set([sink])

    stack = collections.deque([sink])
    while stack:
        n = stack.pop()

        for p in n.get_predecessors():
            if p not in ancestors:
                ancestors.add(p)
                stack.append(p)

    return ancestors


def _calculate_parallel_nodes(sink):
    ancestors = _discover_ancestors(sink)

    done = set()
    parallel_nodes = []

    while ancestors:
        ready = {
            n for n in ancestors
            if all(p in done for p in n.get_predecessors())}

        if not ready:
            raise DependencyError("Dependencies can not be satisfied.")

        parallel_nodes.append(ready)

        for n in ready:
            done.add(n)
            ancestors.remove(n)

    return parallel_nodes


class Pipeline:
    def __init__(self, nodes=[]):
        if nodes is None:
            nodes = []

        self.nodes = nodes

    def append(self, node):
        self.nodes.append(node)

    def transform_stream(self, stream):
        for node in self.nodes:
            stream = node.transform_stream(stream)

        return stream


class SimpleScheduler:
    def __init__(self, sink):
        self.sink = sink

    def to_pipeline(self):
        pipeline = Pipeline()

        parallel_nodes = _calculate_parallel_nodes(self.sink)

        for nodes in parallel_nodes:
            for n in nodes:
                pipeline.append(n)

        return pipeline

    def run(self):
        parallel_nodes = _calculate_parallel_nodes(self.sink)

        stream = []

        for nodes in parallel_nodes:
            for n in nodes:
                stream = n.transform_stream(stream)

        for _ in stream:
            pass
