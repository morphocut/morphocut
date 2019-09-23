import abc
import collections
import inspect
import itertools

from morphocut.graph.pipeline import _pipeline_stack
from morphocut.graph.port import Output


class Node:
    """Represents a node in the computation graph."""
    def __init__(self):
        # Bind ports to self
        outputs = getattr(self.__class__, "outputs", [])

        self.outputs = []

        for port in outputs:
            self.outputs.append(self.__bind_port(port))

        try:
            _pipeline_stack[-1]._add_node(self)
        except IndexError:
            raise RuntimeError("Empty pipeline stack") from None

    def __bind_port(self, port):
        """Bind self to port and create attribute for self.
        """
        if hasattr(self, port.name):
            raise ValueError("Duplicate port '{}'.".format(port.name))

        port = port.bind_instance(self)

        # Attach port to self
        setattr(self, port.name, port)

        return port

    def __call__(self):
        # Return outputs
        if not self.outputs:
            return None
        if len(self.outputs) == 1:
            # If one output, return exactly this
            return self.outputs[0]
        # Otherwise, return list of outputs
        return self.outputs

    def prepare_input(self, obj, names):
        """Returns a tuple corresponding to the input ports."""

        return tuple(
            v.get_value(obj) if isinstance(v, Output) else v
            for v in (getattr(self, n) for n in names))

    def prepare_output(self, obj, *values):
        """Updates obj using the values corresponding to the output ports."""

        if not self.outputs:
            if any(values):
                raise ValueError(
                    "No output port specified but transform returned a value.")

            return obj

        if len(values) != len(self.outputs):
            raise ValueError(
                "Length of values does not match number of output ports.")

        for outp, r in zip(self.outputs, values):
            obj[outp] = r

        return obj

    def after_stream(self):
        pass

    def _get_parameter_names(self):
        """Inspect self.transform to get the parameter names."""
        return [
            p.name
            for p in inspect.signature(self.transform).parameters.values()
            if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        ]

    def transform_stream(self, stream):
        """Apply transform to every object in the stream.
        """

        names = self._get_parameter_names()

        for obj in stream:
            parameters = self.prepare_input(obj, names)

            try:
                result = self.transform(*parameters)  # pylint: disable=no-member
            except TypeError as exc:
                raise TypeError("{} in {}".format(exc, self)) from None

            self.prepare_output(obj, result)

            yield obj

        self.after_stream()

    def __str__(self):
        return "{}()".format(self.__class__.__name__)
