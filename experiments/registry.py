class NodeRegistry:
    def __init__(self):
        self.nodes = set()

    def register(self, *nodes: List[Type[Node]]):
        self.nodes.update(nodes)

    def pipeline_factory(self, pipeline_spec):
        """Construct a pipeline according to the spec.
        """
        ...

    @staticmethod
    def _port_to_tuple(port: Port):
        return (
            None,
            inspect.cleandoc(port.help) if port.help else None
        )

    @staticmethod
    def _parse_docstr(obj):
        try:
            return docstring_parser.parse(obj.__doc__)
        except:
            print("Error parsing docstring of {}".format(obj.__name__))
            raise

    @staticmethod
    def _parse_arguments(node_cls: Type[Node]):
        # Use type annotations to determine the type.
        # Use the docstring for each argument.

        annotations = node_cls.__init__.__annotations__

        # Get docstring for each argument
        arg_desc = {
            p.arg_name: p.description
            for p in NodeRegistry._parse_docstr(node_cls).params
        }
        arg_desc.update({
            p.arg_name: p.description
            for p in NodeRegistry._parse_docstr(node_cls.__init__).params
        })

        return {
            k: (annotations[k], arg_desc[k])
            for k in annotations.keys() & arg_desc.keys()
        }

    @classmethod
    def _node_to_dict(cls, node_cls: Type[Node]):
        doc = cls._parse_docstr(node_cls)
        return {
            "name": node_cls.__name__,
            "short_description": doc.short_description,
            "long_description": doc.long_description,
            "inputs": {p.name: cls._port_to_tuple(p) for p in getattr(node_cls, "inputs", [])},
            "outputs": {p.name: cls._port_to_tuple(p) for p in getattr(node_cls, "outputs", [])},
            "options": cls._parse_arguments(node_cls),
        }

    def to_dict(self) -> dict:
        return {
            n.__name__: self._node_to_dict(n) for n in self.nodes
        }
