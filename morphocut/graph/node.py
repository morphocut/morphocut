import abc
import collections
import itertools

from morphocut.graph.pipeline import _pipeline_stack


class Node:
    """Represents a node in the computation graph."""

    def __init__(self):
        # Bind ports to self
        inputs = getattr(self.__class__, "inputs", [])
        outputs = getattr(self.__class__, "outputs", [])

        self.inputs = []
        self.outputs = []
        self._after = None

        for port in inputs:
            self.inputs.append(self.__bind_port(port))

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

    def __call__(self, *args, **kwargs):
        """Binds output ports of other nodes to input ports of this node."""

        inputs = collections.OrderedDict(
            (inp.name, inp) for inp in self.inputs)

        # Treat *args
        for i, inp in enumerate(inputs.values()):
            if inp.multi:
                outs = [o.get_output() if isinstance(o, Node)
                        else o for o in args[i:]]

                inp.bind(outs)

                # There's nothing after a multi
                break
            else:
                try:
                    outp = args[i]
                except IndexError:
                    break

                if isinstance(outp, Node):
                    outp = outp.get_output()
                inp.bind(outp)

        # Treat **kwargs
        for name, outp in kwargs.items():
            try:
                inp = inputs[name]
            except KeyError:
                raise TypeError(
                    "Node got an unexpected keyword argument '{}'.".format(name))

            if isinstance(outp, Node):
                outp = outp.get_output()

            inp.bind(outp)

        # Return value
        if not self.outputs:
            # If no outputs, this is a terminal node. Return self to be able to schedule.
            return self
        if len(self.outputs) == 1:
            # If one output, return exactly this
            return self.outputs[0]
        # Otherwise, return list of outputs
        return self.outputs

    def after(self, other):
        self._after = other

    def transform(self, *args, **kwargs):
        raise NotImplementedError()

    def prepare_input(self, obj):
        """Returns a tuple corresponding to the input ports."""
        return tuple(
            inp.get_value(obj)
            if inp._bind is not None else None
            for inp in self.inputs
        )

    def prepare_input_dict(self, obj):
        """Returns a dictionary corresponding to the input ports."""
        return {
            inp.name: inp.get_value(obj)
            if inp._bind is not None else None
            for inp in self.inputs
        }

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

    def transform_stream(self, stream):
        """Apply transform to every object in the stream.
        """

        for obj in stream:
            parameters = self.prepare_input(obj)

            try:
                result = self.transform(*parameters)
            except TypeError as exc:
                raise TypeError("{} in {}".format(exc, self)) from None

            self.prepare_output(obj, result)

            yield obj

        self.after_stream()

    def get_output(self):
        if len(self.outputs) != 1:
            raise ValueError("Node has not exactly one output.")

        return self.outputs[0]

    def _validate_inputs(self):
        missing_inputs = [
            inp.name for inp in self.inputs
            if inp.required and inp._bind is None
        ]

        if missing_inputs:
            msg = "{}: Inputs {} are not bound".format(
                self, ", ".join(missing_inputs))
            raise ValueError(msg)

    def get_predecessors(self):
        """Returns the set of predecessors of the current node."""
        self._validate_inputs()

        predecessors = {
            inp._bind._node for inp in self.inputs
            if inp._bind is not None
        }
        if self._after is not None:
            predecessors.add(self._after)

        return predecessors

    def __str__(self):
        return "{}()".format(self.__class__.__name__)
