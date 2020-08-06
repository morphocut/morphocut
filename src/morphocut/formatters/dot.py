import pathlib
from typing import Optional

import pydot

from morphocut.core import Node, Pipeline, Variable
from itertools import count

class DotFormatter:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.counter = count()

    def save(self, path_or_handle=None, rankdir="TB"):
        graph = pydot.Dot(graph_type="digraph", strict=True, rankdir=rankdir)

        for c in self.pipeline.children:
            if isinstance(c, Node):
                self._format_node(graph, c, graph)
            elif isinstance(c, Pipeline):
                self._format_pipeline(graph, c, graph)

        if isinstance(path_or_handle, pathlib.Path):
            path_or_handle = str(path_or_handle)

        close = False
        if isinstance(path_or_handle, str):
            path_or_handle = open(path_or_handle, "w")
            close = True

        try:
            path_or_handle.write(graph.to_string())
        finally:
            if close:
                path_or_handle.close()

    def _format_node(self, graph: pydot.Graph, node: Node, root: pydot.Graph):
        info = node.get_info()

        idx = next(self.counter)

        subgraph = pydot.Subgraph(
            f"{idx}", rank="same"
        )

        subgraph.add_node(pydot.Node(f"step_{idx}", style="invis"))
        if idx > 0:
            root.add_edge(pydot.Edge(f"step_{idx-1}", f"step_{idx}", style="invis"))

        subgraph.add_node(pydot.Node(node.id, label=info["label"], shape="rect"))

        for predecessor in get_predecessors(node):
            if node.id != predecessor.id:
                root.add_edge(pydot.Edge(predecessor.id, node.id))

        graph.add_subgraph(subgraph)

    def _format_pipeline(
        self,
        graph: pydot.Graph,
        pipeline: Pipeline,
        root:  pydot.Graph
    ):
        subgraph = pydot.Subgraph(
            f"cluster_{pipeline.id}", label=pipeline.__class__.__name__
        )

        for c in pipeline.children:
            if isinstance(c, Node):
                self._format_node(subgraph, c, root)
            elif isinstance(c, Pipeline):
                self._format_pipeline(subgraph, c, root)

        graph.add_subgraph(subgraph)


def get_predecessors(obj, visited=None):
    if visited is None:
        visited = set()

    if id(obj) in visited:
            return
    visited.add(id(obj))

    if isinstance(obj, Variable):
        if obj.parent is not None:
            yield obj.parent

    elif isinstance(obj, Node):
        for attr in dir(obj):
            value = getattr(obj, attr)
            yield from get_predecessors(value, visited)

    elif isinstance(obj, Iterable):
        for v in obj:
            yield from get_predecessors(v, visited)

    elif isinstance(obj, Mapping):
        for v in obj.values():
            yield from get_predecessors(v, visited)


from collections.abc import Iterable, Mapping