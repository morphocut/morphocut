import abc
import copy
from morphocut.graph import Node


class Port:
    """Stores meta data about a port of a Node."""

    def __init__(self, name, help=None):
        self.name = name
        self.help = help
        self._node = None

    @abc.abstractmethod
    def __call__(self, method):
        raise NotImplementedError()

    def bind_instance(self, node):
        """
        Return a copy of self with a reference to the node.
        """

        port = copy.copy(self)
        port._node = node

        return port

    def __repr__(self):
        return "{}(\"{}\")".format(self.__class__.__name__, self.name)


class Input(Port):
    """Stores meta data about an input port of a Node."""

    def __init__(self, name, required=True, multi=False, help=None):
        super().__init__(name, help)

        self.required = required
        self.multi = multi
        self._bind = None

    def __call__(self, cls):
        if not issubclass(cls, Node):
            raise ValueError(
                "This decorator is meant to be applied to a subclass of Node.")

        try:
            inputs = cls.inputs
        except AttributeError:
            inputs = cls.inputs = []

        inputs.insert(0, self)

        return cls

    def bind(self, output):
        """Bind this input to another output."""
        self._bind = output

    def get_value(self, obj):
        if not self.multi:
            return obj[self._bind]
        else:
            return tuple(obj[b] for b in self._bind)


class Output(Port):
    """Stores meta data about an output port of a Node."""

    def __call__(self, cls):
        if not issubclass(cls, Node):
            raise ValueError(
                "This decorator is meant to be applied to a subclass of Node.")

        try:
            outputs = cls.outputs
        except AttributeError:
            outputs = cls.outputs = []

        outputs.insert(0, self)

        return cls
