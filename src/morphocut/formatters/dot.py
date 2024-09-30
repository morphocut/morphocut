import pathlib
from collections.abc import Iterable, Mapping
from itertools import count

import pydot

from morphocut.core import Node, Pipeline, Variable

class DotFormatter:
    """
    A class to format and save a graph representation of a Pipeline using the DOT language.


    Args:
        pipeline (Pipeline): The pipeline to be formatted and visualized.
    """

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.counter = count()

    def save(self, path_or_handle, rankdir="TB"):
        """
        Save the graph representation of the pipeline to a file or file-like object.

        Args:
            path_or_handle (str or file-like): The path to the file or a writable file-like object to save the DOT format.
            rankdir (str, optional): The direction of graph layout. Defaults to "TB" (top to bottom).
        """

        if isinstance(path_or_handle, str):
            path_or_handle = pathlib.Path(path_or_handle)

        close = False
        if hasattr(path_or_handle, "open"):
            path_or_handle = path_or_handle.open("w")
            close = True
        
        graph = pydot.Dot(graph_type="digraph", strict=True, rankdir=rankdir)

        for c in self.pipeline.children:
            if isinstance(c, Node):
                self._format_node(graph, c, graph)
            elif isinstance(c, Pipeline):
                self._format_pipeline(graph, c, graph)

        try:
            path_or_handle.write(graph.to_string())
        finally:
            if close:
                path_or_handle.close()

    def _format_node(self, graph: pydot.Graph, node: Node, root: pydot.Graph):
        idx = next(self.counter)

        subgraph = pydot.Subgraph(f"{idx}", rank="same")

        subgraph.add_node(pydot.Node(f"step_{idx}", style="invis"))
        if idx > 0:
            root.add_edge(pydot.Edge(f"step_{idx-1}", f"step_{idx}", style="invis"))

        subgraph.add_node(pydot.Node(node.id, label=node.get_label(), shape="rect"))

        for predecessor in get_predecessors(node):
            if node.id != predecessor.id:
                root.add_edge(pydot.Edge(predecessor.id, node.id))

        graph.add_subgraph(subgraph)

    def _format_pipeline(
        self, graph: pydot.Graph, pipeline: Pipeline, root: pydot.Graph
    ):
        subgraph = pydot.Subgraph(
            f"cluster_{pipeline.id}", label=pipeline.get_label()
        )

        for c in pipeline.children:
            if isinstance(c, Node):
                self._format_node(subgraph, c, root)
            elif isinstance(c, Pipeline):
                self._format_pipeline(subgraph, c, root)

        graph.add_subgraph(subgraph)


def get_predecessors(obj, _visited=None):
    """
    Recursively yield all predecessor objects (e.g., parent Nodes or Variables) of the given object.

    Args:
        obj (object): The object from which to retrieve predecessors.
        visited (set, optional): A set of visited object IDs to avoid circular references.

    Yields:
        object: Predecessor objects related to the input object.
    """

    if _visited is None:
        _visited = set()

    if id(obj) in _visited:
        return
    _visited.add(id(obj))

    if isinstance(obj, Variable):
        if obj.parent is not None:
            yield obj.parent

    elif isinstance(obj, Node):
        for attr in dir(obj):
            value = getattr(obj, attr)
            yield from get_predecessors(value, _visited)

    elif isinstance(obj, Iterable):
        for v in obj:
            yield from get_predecessors(v, _visited)

    elif isinstance(obj, Mapping):
        for v in obj.values():
            yield from get_predecessors(v, _visited)


