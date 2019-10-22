"""Core components of the MorphoCut processing graph."""

import inspect
import operator
from collections import abc
from functools import wraps
from typing import (
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

_pipeline_stack = []  # type: List[Pipeline] # pylint: disable=invalid-name


def _resolve_variable(obj, variable_or_value):
    if isinstance(variable_or_value, Variable):
        return obj[variable_or_value]

    if isinstance(variable_or_value, tuple):
        return tuple(_resolve_variable(obj, v) for v in variable_or_value)

    if isinstance(variable_or_value, dict):
        return {k: _resolve_variable(obj, v) for k, v in variable_or_value.items()}

    return variable_or_value


T = TypeVar("T")


class Variable(Generic[T]):
    """
    A Variable identifies a value in the stream.

    Variables are (almost) never instanciated manually, they are created when calling a Node.

    Attributes:
        name: The name of the Variable.
        node: The node that created the Variable.
    """

    __slots__ = ["name", "node", "hash"]

    def __init__(self, name: str, node: "Node"):
        self.name = name
        self.node = node
        self.hash = hash((node.id, name))

    def __str__(self):
        return "<Variable {}.{}>".format(self.node, self.name)

    def __getattr__(self, name):
        return LambdaNode(getattr, self, name)

    def __getitem__(self, key):
        return LambdaNode(operator.getitem, self, key)

    def __setitem__(self, key, value):
        return LambdaNode(operator.setitem, self, key, value)


# Types
RawOrVariable = Union[T, Variable[T]]
NodeCallReturnType = Union[None, Variable, Tuple[Variable]]
Stream = Iterable["StreamObject"]


class Node:
    """
    Base class for all nodes.

    A Node applies creates, updates or deletes stream objects.
    """

    def __init__(self):
        self.id = "{:x}".format(id(self))

        # Bind outputs to self
        outputs = getattr(self.__class__, "outputs", [])
        self.outputs = [self.__bind_output(o) for o in outputs]

        # Register with pipeline
        try:
            # pylint: disable=protected-access
            _pipeline_stack[-1]._add_node(self)
        except IndexError:
            raise RuntimeError("Empty pipeline stack") from None

    def __bind_output(self, port: "Output"):
        """Bind self to port and return a variable."""
        variable = port.create_variable(self)

        return variable

    def __call__(self) -> NodeCallReturnType:
        """Return outputs."""

        try:
            outputs = self.__dict__["outputs"]
        except KeyError:
            raise RuntimeError(
                "'{type}' is not initialized properly. Did you forget a super().__init__() in the constructor?".format(
                    type=type(self).__name__
                )
            )

        # Return outputs
        if not outputs:
            return None
        if len(outputs) == 1:
            # If one output, return exactly this
            return outputs[0]
        # Otherwise, return list of outputs
        return outputs

    def prepare_input(self, obj, names):
        """Return a tuple corresponding to the input ports."""

        if isinstance(names, str):
            return _resolve_variable(obj, getattr(self, names))

        return tuple(
            _resolve_variable(obj, v) for v in (getattr(self, n) for n in names)
        )

    def prepare_output(self, obj, *values):
        """Update obj using the values corresponding to the output ports."""

        if not self.outputs:
            if any(values):
                raise ValueError(
                    "No output port specified but transform returned a value."
                )

            return obj

        while True:
            n_values = len(values)
            n_outputs = len(self.outputs)
            if n_values != n_outputs:
                # If values is a nested tuple, unnest and retry
                if n_values == 1 and isinstance(values[0], tuple):
                    values = values[0]
                    continue
                raise ValueError(
                    "Length of values does not match number of output ports: {} vs. {}".format(
                        n_values, n_outputs
                    )
                )
            break

        for variable, r in zip(self.outputs, values):
            obj[variable] = r

        return obj

    def after_stream(self):
        """
        Do something after the stream was processed.

        Called by transform_stream after stream processing is done.
        *Override this in your own subclass.*
        """

    def _get_parameter_names(self):
        """Inspect self.transform to get the parameter names."""
        return [
            p.name
            for p in inspect.signature(
                self.transform  # pylint: disable=no-member
            ).parameters.values()
            if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        ]

    def transform_stream(self, stream: Stream) -> Stream:
        """
        Transform a stream.

        By default, this calls ``self.transform`` with appropriate parameters.
        ``transform`` has to be implemented by a subclass if ``transform_stream`` is not overridden.

        Override if the stream has to be altered in some way, i.e. objects are created, deleted, re-arranged, ...
        """

        names = self._get_parameter_names()

        for obj in stream:
            parameters = self.prepare_input(obj, names)

            result = self.transform(*parameters)  # pylint: disable=no-member

            self.prepare_output(obj, result)

            yield obj

        self.after_stream()

    def __str__(self):
        return "{}()".format(self.__class__.__name__)


class Output:
    """
    Define an Output of a node.

    Args:
        name (str): Name of the output.
        type (type, optional): Type of the output.
        doc (str, optional): Description  of the output.
    """

    def __init__(
        self, name: str, type: Optional[Type] = None, doc: Optional[str] = None
    ):
        self.name = name
        self.type = type
        self.doc = doc
        self.node_cls = None

    def create_variable(self, node: Node):
        """Return a _Variable with a reference to the node."""

        return Variable(self.name, node)

    def __repr__(self):
        return '{}("{}", {})'.format(self.__class__.__name__, self.name, self.node_cls)

    def __call__(self, cls):
        """Add this output to the list of a nodes outputs."""

        if not issubclass(cls, Node):
            raise ValueError(
                "This decorator is meant to be applied to a subclass of Node."
            )

        try:
            outputs = cls.outputs
        except AttributeError:
            outputs = cls.outputs = []

        outputs.insert(0, self)

        self.node_cls = cls

        return cls


def ReturnOutputs(node_cls):
    """Turn Node into a function returning Output variables."""
    if not issubclass(node_cls, Node):
        raise ValueError("This decorator is meant to be applied to a subclass of Node.")

    @wraps(node_cls)
    def wrapper(*args, **kwargs) -> NodeCallReturnType:
        return node_cls(*args, **kwargs)()

    wrapper._node_cls = node_cls
    wrapper.__mro__ = node_cls.__mro__
    return wrapper


@ReturnOutputs
@Output("result")
class LambdaNode(Node):
    """
    Apply a function to the supplied variables.

    For every object in the stream, apply ``clbl`` to the corresponding stream variables.

    Args:
        clbl: A callable.
        *args: Positional arguments to ``clbl``.
        **kwargs: Keyword-arguments to ``clbl``.

    Returns:
        Variable: The result of the function invocation.

    Example:
        .. code-block:: python

            def foo(bar):
                return bar

            baz = ... # baz is a stream variable.
            result = LambdaNode(foo, baz)

    """

    def __init__(self, clbl: Callable, *args, **kwargs):
        super().__init__()
        self.clbl = clbl
        self.args = args
        self.kwargs = kwargs

    def transform(self, clbl, args, kwargs):
        """Apply clbl to the supplied arguments."""
        return clbl(*args, **kwargs)

    def __str__(self):
        return "{}({})".format(self.__class__.__name__, self.clbl.__name__)


class StreamObject(abc.MutableMapping):
    __slots__ = ["data"]

    def __init__(self, data: Dict = None):
        if data is None:
            data = {}
        self.data = data

    def copy(self) -> "StreamObject":
        return StreamObject(self.data.copy())

    def _as_key(self, obj):
        if isinstance(obj, Variable):
            return obj.hash
        return obj

    def __setitem__(self, key, value):
        self.data[self._as_key(key)] = value

    def __delitem__(self, key):
        del self.data[self._as_key(key)]

    def __getitem__(self, key):
        return self.data[self._as_key(key)]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


class Pipeline:
    """
    A Pipeline manages the execution of nodes.

    Nodes defined inside the pipeline context will be added to the pipeline.
    When the pipeline is executed, stream objects are passed
    from one node to the next in the same order.

    Example:
        .. code-block:: python

            with Pipeline() as pipeline:
                ...

            pipeline.run()
    """

    def __init__(self):
        self.nodes = []  # type: List[Node]

    def __enter__(self):
        # Push self to pipeline stack
        _pipeline_stack.append(self)

        return self

    def __exit__(self, *_):
        # Pop self from pipeline stack
        item = _pipeline_stack.pop()

        assert item is self

    def transform_stream(self, stream: Optional[Stream] = None) -> Stream:
        """
        Run the stream through all nodes and return it.

        Args:
            stream: A stream to transform.
                *This argument is solely to be used internally.*

        Returns:
            Stream: An iterable of stream objects.

        """
        if stream is None:
            stream = [StreamObject()]

        for node in self.nodes:  # type: Node
            stream = node.transform_stream(stream)

        return stream

    def run(self):
        """
        Run the complete pipeline.

        This is a convenience method to be used in place of:

        .. code-block:: python

            for _ in pipeline.transform_stream():
                pass
        """
        for _ in self.transform_stream():
            pass

    def _add_node(self, node: Node):
        self.nodes.append(node)

    def __str__(self):
        return "Pipeline([{}])".format(", ".join(str(n) for n in self.nodes))
