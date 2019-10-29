import inspect
from functools import partial
from typing import Mapping, Tuple

from morphocut.core import (
    Node,
    ReturnOutputs,
    Stream,
    StreamObject,
    Variable,
    raw_node,
    resolve_variable,
)


class IdNode(Node):
    def __init__(self, keys):
        super().__init__()

        self.keys = keys
        self.outputs = [Variable(k, self) for k in self.keys]
        self.values = {}

    def transform_stream(self, stream: Stream) -> Stream:
        for obj in stream:
            values = resolve_variable(obj, self.values)

            for variable, k in zip(self.outputs, self.keys):
                try:
                    obj[variable] = resolve_variable(obj, values[k])
                except KeyError:
                    raise KeyError("Parameter {} is missing".format(k))

            yield obj


class Configurator:
    def __init__(self):
        self.parameters = {}  # type: Mapping[str, Tuple[IdNode, Any]]

    def node(self, id, node_cls, *args, **kwargs):
        node_cls = raw_node(node_cls)
        signature = inspect.signature(node_cls)

        ba = signature.bind_partial(*args, **kwargs)
        missing = signature.parameters.keys() - ba.arguments.keys()

        if missing:
            missing = sorted(missing)

            missing_canonical = ["{}.{}".format(id, k) for k in missing]

            id_node = IdNode(missing_canonical)
            self.parameters.update({k: id_node for k in missing_canonical})

            assert len(id_node.outputs) == len(missing)

            ba.arguments.update({k: v for k, v in zip(missing, id_node.outputs)})

        node = node_cls(*ba.args, **ba.kwargs)
        node.id = id
        return node()

    def variable(self, name, variable):
        """Make a variable known to the Configurator."""

    def set(self, name, value):
        """Set parameter."""
        id_node = self.parameters[name]
        id_node.values[name] = value
