import abc
import collections
import itertools


class Node:
    """Represents a node in the computation graph."""

    def __init__(self):
        # Bind ports to self
        inputs = getattr(self.__class__, "inputs", [])
        outputs = getattr(self.__class__, "outputs", [])

        self.inputs = []
        self.outputs = []

        for port in inputs:
            self.inputs.append(self.__bind_port(port))

        for port in outputs:
            self.outputs.append(self.__bind_port(port))

    def __bind_port(self, port):
        """Bind self to port and create attribute for self.
        """
        if hasattr(self, port.name):
            raise ValueError("Duplicate port '{}'.".format(port.name))

        port = port.bind_instance(self)

        # Attach port to self
        setattr(self, port.name, port)

        return port

    def __call__(self, *args, **kwargs):
        """Binds output ports of other nodes to input ports of this node."""

        inputs = collections.OrderedDict(
            (inp.name, inp) for inp in self.inputs)

        for inp, outp in zip(inputs.values(), args):
            if isinstance(outp, Node):
                outp = outp.get_output()

            inp.bind(outp)

        for name, outp in kwargs.items():
            try:
                inp = inputs[name]
            except KeyError:
                raise TypeError(
                    "Node got an unexpected keyword argument '{}'.".format(name))

            if isinstance(outp, Node):
                outp = outp.get_output()

            inp.bind(outp)

        # Return self for chaining
        return self

    def transform(self, *args, **kwargs):
        raise NotImplementedError()

    def prepare_input(self, obj):
        """Returns a dictionary corresponding to the input ports."""
        return {inp.name: obj[inp._bind] for inp in self.inputs}

    def prepare_output(self, obj, *values):
        """Updates obj using the values corresponding to the output ports."""

        if len(values) != len(self.outputs):
            raise ValueError(
                "Length of values does not match number of output ports.")

        for outp, r in zip(self.outputs, values):
            obj[outp] = r

        return obj

    def transform_stream(self, stream):
        """Apply transform to every object in the stream.
        """

        for obj in stream:
            parameters = self.prepare_input(obj)

            result = self.transform(**parameters)

            self.prepare_output(obj, result)

            yield obj

    def get_output(self):
        if len(self.outputs) != 1:
            raise ValueError("Node has not exactly one output.")

        return self.outputs[0]

    def get_predecessors(self):
        """Returns the set of predecessors of the current node."""
        return {inp._bind._node for inp in self.inputs}

    def __str__(self):
        return "{}()".format(self.__class__.__name__)
