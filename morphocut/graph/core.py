"""Core components of the MorphoCut processing graph."""

import inspect

from morphocut.graph.pipeline import _pipeline_stack


class Node:
    """Represents a node in the computation graph."""
    def __init__(self):
        # Bind outputs to self
        outputs = getattr(self.__class__, "outputs", [])
        self.outputs = [self.__bind_output(o) for o in outputs]

        # Register with pipeline
        try:
            _pipeline_stack[-1]._add_node(self)  #pylint: disable=protected-access
        except IndexError:
            raise RuntimeError("Empty pipeline stack") from None

    def __bind_output(self, port):
        """Bind self to port and return a variable.
        """

        variable = port.create_variable(self)

        return variable

    def __call__(self):
        """Return outputs."""

        try:
            outputs = self.__dict__["outputs"]
        except KeyError:
            raise RuntimeError(
                "'{type}' is not initialized properly. Did you forget a super().__init__() in the constructor?"
                .format(type=type(self).__name__)
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
        """Returns a tuple corresponding to the input ports."""

        return tuple(
            obj[v] if isinstance(v, _Variable) else v
            for v in (getattr(self, n) for n in names)
        )

    def prepare_output(self, obj, *values):
        """Updates obj using the values corresponding to the output ports."""

        if not self.outputs:
            if any(values):
                raise ValueError(
                    "No output port specified but transform returned a value."
                )

            return obj

        if len(values) != len(self.outputs):
            raise ValueError(
                "Length of values does not match number of output ports."
            )

        for outp, r in zip(self.outputs, values):
            obj[outp] = r

        return obj

    def after_stream(self):
        """Called after transform_stream is done.

        Override this in your own implementation.
        """
        pass

    def _get_parameter_names(self):
        """Inspect self.transform to get the parameter names."""
        return [
            p.name
            for p in inspect.signature(self.transform).parameters.values()  # pylint: disable=no-member
            if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        ]

    def transform_stream(self, stream):
        """Apply transform to every object in the stream.
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

    def __getattr__(self, name):
        raise AttributeError(
            "'{type}' has no attribute '{name}'. Did you forget a () after {type}(...)?"
            .format(type=type(self).__name__, name=name)
        )


class Output:
    """Stores meta data about a output of a Node.

    This is used as a decorator.

    Example:
        @Output("bar")
        class Foo(Node):
            ...

    """
    def __init__(self, name, help=None):
        self.name = name
        self.help = help
        self.node_cls = None

    def create_variable(self, node):
        """
        Return a copy of self with a reference to the node.
        """

        return _Variable(self.name, node)

    def __repr__(self):
        return "{}(\"{}\", )".format(
            self.__class__.__name__, self.name, self.node_cls
        )

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

        print("Done.")

        return cls


class _Variable:
    __slots__ = ["name", "node"]

    def __init__(self, name, node):
        self.name = name
        self.node = node

    #TODO: __eq__, __hash__
