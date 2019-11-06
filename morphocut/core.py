"""Core components of the MorphoCut processing graph."""

import inspect
import operator
import typing
import warnings
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
    A Variable identifies a value in a stream object.

    Variables are (almost) never instanciated manually, they are created when calling a Node.

    Attributes:
        name: The name of the Variable.
        node: The node that created the Variable.

    Operations:
        Variables support the following operations.
        Each operation is realized as a new Node in the Pipeline,
        so use them sparingly.
        Operator and method can be used interchangeably (if both present).

        +-----------------------+-----------------------+---------------------------------+
        |       Operation       |      Operator         |             Method              |
        +=======================+=======================+=================================+
        | Addition              | ``a + b``             | ``a.add(b)``                    |
        +-----------------------+-----------------------+---------------------------------+
        | Containment Test      |                       | ``a.contains(b)``, ``b.in_(a)`` |
        +-----------------------+-----------------------+---------------------------------+
        | True Division         | ``a / b``             | ``a.truediv(b)``                |
        +-----------------------+-----------------------+---------------------------------+
        | Integer Division      | ``a // b``            | ``a.floordiv(b)``               |
        +-----------------------+-----------------------+---------------------------------+
        | Bitwise And           | ``a & b``             | ``a.and_(b)``                   |
        +-----------------------+-----------------------+---------------------------------+
        | Bitwise Exclusive Or  | ``a ^ b``             | ``a.xor(b)``                    |
        +-----------------------+-----------------------+---------------------------------+
        | Bitwise Inversion     | ``~ a``               | ``a.invert(b)``                 |
        +-----------------------+-----------------------+---------------------------------+
        | Bitwise Or            | ``a | b``             | ``a.or_(b)``                    |
        +-----------------------+-----------------------+---------------------------------+
        | Exponentiation        | ``a ** b``            | ``a.pow(b)``                    |
        +-----------------------+-----------------------+---------------------------------+
        | Identity              |                       | ``a.is_(b)``                    |
        +-----------------------+-----------------------+---------------------------------+
        | Identity              |                       | ``a.is_not(b)``                 |
        +-----------------------+-----------------------+---------------------------------+
        | Indexed Assignment    | ``obj[k] = v``        |                                 |
        +-----------------------+-----------------------+---------------------------------+
        | Indexed Deletion      | ``del obj[k]``        |                                 |
        +-----------------------+-----------------------+---------------------------------+
        | Indexing              | ``obj[k]``            |                                 |
        +-----------------------+-----------------------+---------------------------------+
        | Left Shift            | ``a << b``            | ``a.lshift(b)``                 |
        +-----------------------+-----------------------+---------------------------------+
        | Modulo                | ``a % b``             | ``a.mod(b)``                    |
        +-----------------------+-----------------------+---------------------------------+
        | Multiplication        | ``a * b``             | ``a.mul(b)``                    |
        +-----------------------+-----------------------+---------------------------------+
        | Matrix Multiplication | ``a @ b``             | ``a.matmul(b)``                 |
        +-----------------------+-----------------------+---------------------------------+
        | Negation (Arithmetic) | ``- a``               | ``a.neg(b)``                    |
        +-----------------------+-----------------------+---------------------------------+
        | Negation (Logical)    |                       | ``a.not_()``                    |
        +-----------------------+-----------------------+---------------------------------+
        | Positive              | ``+ a``               | ``a.pos(b)``                    |
        +-----------------------+-----------------------+---------------------------------+
        | Right Shift           | ``a >> b``            | ``a.rshift(b)``                 |
        +-----------------------+-----------------------+---------------------------------+
        | Slice Assignment      | ``seq[i:j] = values`` |                                 |
        +-----------------------+-----------------------+---------------------------------+
        | Slice Deletion        | ``del seq[i:j]``      |                                 |
        +-----------------------+-----------------------+---------------------------------+
        | Slicing               | ``seq[i:j]``          |                                 |
        +-----------------------+-----------------------+---------------------------------+
        | Subtraction           | ``a - b``             | ``a.sub(b)``                    |
        +-----------------------+-----------------------+---------------------------------+
        | Ordering              | ``a < b``             | ``a.lt(b)``                     |
        +-----------------------+-----------------------+---------------------------------+
        | Ordering              | ``a <= b``            | ``a.leq(b)``                    |
        +-----------------------+-----------------------+---------------------------------+
        | Equality              | ``a == b``            | ``a.eq(b)``                     |
        +-----------------------+-----------------------+---------------------------------+
        | Difference            | ``a != b``            | ``a.ne(b)``                     |
        +-----------------------+-----------------------+---------------------------------+
        | Ordering              | ``a >= b``            | ``a.ge(b)``                     |
        +-----------------------+-----------------------+---------------------------------+
        | Ordering              | ``a > b``             | ``a.gt(b)``                     |
        +-----------------------+-----------------------+---------------------------------+

        ``a``, ``b``, ``i``, ``j`` and ``k`` can be either
        :py:class:`Variable` instances or raw values.
    """

    __slots__ = ["name", "node", "hash"]

    def __init__(self, name: str, node: "Node"):
        self.name = name
        self.node = node
        self.hash = hash((node.id, name))

    def __str__(self):
        return "<Variable {}.{}>".format(self.node, self.name)

    def __repr__(self):
        return self.__str__()

    # Attribute access
    def __getattr__(self, name):
        return LambdaNode(getattr, self, name)

    # Item access
    def __getitem__(self, key):
        return LambdaNode(operator.getitem, self, key)

    def __setitem__(self, key, value):
        return LambdaNode(operator.setitem, self, key, value)

    def __delitem__(self, key):
        return LambdaNode(operator.delitem, self, key)

    # Rich comparison methods
    def __lt__(self, other):
        return LambdaNode(operator.lt, self, other)

    def __le__(self, other):
        return LambdaNode(operator.le, self, other)

    def __eq__(self, other):
        return LambdaNode(operator.eq, self, other)

    def __ne__(self, other):
        return LambdaNode(operator.ne, self, other)

    def __gt__(self, other):
        return LambdaNode(operator.gt, self, other)

    def __ge__(self, other):
        return LambdaNode(operator.ge, self, other)

    # Binary arithmetic operations
    def __add__(self, other):
        return LambdaNode(operator.add, self, other)

    def __sub__(self, other):
        return LambdaNode(operator.sub, self, other)

    def __mul__(self, other):
        return LambdaNode(operator.mul, self, other)

    def __matmul__(self, other):
        return LambdaNode(operator.matmul, self, other)

    def __truediv__(self, other):
        return LambdaNode(operator.truediv, self, other)

    def __floordiv__(self, other):
        return LambdaNode(operator.floordiv, self, other)

    def __mod__(self, other):
        return LambdaNode(operator.mod, self, other)

    def __pow__(self, other):
        return LambdaNode(operator.pow, self, other)

    def __lshift__(self, other):
        return LambdaNode(operator.lshift, self, other)

    def __rshift__(self, other):
        return LambdaNode(operator.rshift, self, other)

    def __and__(self, other):
        return LambdaNode(operator.and_, self, other)

    def __xor__(self, other):
        return LambdaNode(operator.xor, self, other)

    def __or__(self, other):
        return LambdaNode(operator.or_, self, other)

    # Binary arithmetic operations with reflected (swapped) operands
    def __radd__(self, other):
        return LambdaNode(operator.add, other, self)

    def __rsub__(self, other):
        return LambdaNode(operator.sub, other, self)

    def __rmul__(self, other):
        return LambdaNode(operator.mul, other, self)

    def __rmatmul__(self, other):
        return LambdaNode(operator.matmul, other, self)

    def __rtruediv__(self, other):
        return LambdaNode(operator.truediv, other, self)

    def __rfloordiv__(self, other):
        return LambdaNode(operator.floordiv, other, self)

    def __rmod__(self, other):
        return LambdaNode(operator.mod, other, self)

    def __rpow__(self, other):
        return LambdaNode(operator.pow, other, self)

    def __rlshift__(self, other):
        return LambdaNode(operator.lshift, other, self)

    def __rrshift__(self, other):
        return LambdaNode(operator.rshift, other, self)

    def __rand__(self, other):
        return LambdaNode(operator.and_, other, self)

    def __rxor__(self, other):
        return LambdaNode(operator.xor, other, self)

    def __ror__(self, other):
        return LambdaNode(operator.or_, other, self)

    # Unary arithmetic operations
    def __neg__(self):
        return LambdaNode(operator.neg, self)

    def __pos__(self):
        return LambdaNode(operator.pos, self)

    def __abs__(self):
        return LambdaNode(operator.abs, self)

    def __invert__(self):
        return LambdaNode(operator.invert, self)

    # Above operators without underscores
    getattr = __getattr__
    lt = __lt__
    le = __le__
    eq = __eq__
    ne = __ne__
    gt = __gt__
    ge = __ge__
    add = __add__
    sub = __sub__
    mul = __mul__
    matmul = __matmul__
    truediv = __truediv__
    floordiv = __floordiv__
    mod = __mod__
    pow = __pow__
    lshift = __lshift__
    rshift = __rshift__
    and_ = __and__
    xor = __xor__
    or_ = __or__
    neg = __neg__
    pos = __pos__
    abs = __abs__
    invert = __invert__

    # Special operators
    def not_(self):
        """Return the outcome of not obj."""
        return LambdaNode(operator.not_, self)

    def truth(self):
        """Return True if obj is true, and False otherwise."""
        return LambdaNode(operator.truth, self)

    def is_(self, other):
        """Return ``self is other``. Tests object identity."""
        return LambdaNode(operator.is_, self, other)

    def is_not(self, other):
        """Return ``self is not other``. Tests object identity."""
        return LambdaNode(operator.is_not, self, other)

    def in_(self, other):
        """Return the outcome of the test  ``self in other``. Tests containment."""
        return LambdaNode(operator.contains, other, self)

    def contains(self, other):
        """Return the outcome of the test  ``other in self``. Tests containment."""
        return LambdaNode(operator.contains, self, other)


# Types
RawOrVariable = Union[T, Variable[T]]
NodeCallReturnType = Union[None, Variable, Tuple[Variable]]
Stream = Iterable["StreamObject"]


class Node:
    """Base class for all nodes."""

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
            raise RuntimeError(
                "Empty pipeline stack. {} has to be called in a pipeline context.".format(
                    self.__class__.__name__
                )
            ) from None

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
            if any(v is not None for v in values):
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

        Override if the stream has to be altered in some way,
        i.e. objects are created, deleted or re-arranged.
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
    Define an Output of a Node.

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
        args = [self.clbl.__name__]
        args.extend(str(a) for a in self.args)
        args.extend("{}={}".format(k, v) for k, v in self.kwargs.items())
        return "{}({})".format(self.__class__.__name__, ", ".join(args))


class StreamObjectKeyError(KeyError):
    def __str__(self):
        return "{}\nYou probably removed this key from the stream.".format(
            super().__str__()
        )


class StreamObject(abc.MutableMapping):
    __slots__ = ["data"]

    def __init__(self, data: Dict = None):
        if data is None:
            data = {}
        self.data = data

    def copy(self) -> "StreamObject":
        return StreamObject(self.data.copy())

    @staticmethod
    def _as_key(obj):
        if isinstance(obj, Variable):
            return obj.hash
        return obj

    def __setitem__(self, key, value):
        self.data[self._as_key(key)] = value

    def __delitem__(self, key):
        del self.data[self._as_key(key)]

    def __getitem__(self, key):
        try:
            return self.data[self._as_key(key)]
        except KeyError:
            raise StreamObjectKeyError(key) from None

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

    Args:
        parent (Pipeline, optional): A parent pipeline to attach to.
            If None and nested in an existing Pipeline, attach to this one.

    Example:
        .. code-block:: python

            with Pipeline() as pipeline:
                ...

            pipeline.run()
    """

    def __init__(self, parent: Optional["Pipeline"] = None):
        self.nodes = []  # type: List[Node]

        if parent is None:
            try:
                parent = _pipeline_stack[-1]
            except IndexError:
                pass

        if parent is not None:
            parent.add_child(self)

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
