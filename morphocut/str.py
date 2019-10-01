from morphocut.graph import Node, Output


@Output("string")
class Format(Node):
    def __init__(self, fmt, *args, **kwargs):
        super().__init__()
        self.fmt = fmt
        self.args = args
        self.kwargs = kwargs

    def transform(self, fmt: str, args: tuple, kwargs: dict):
        return fmt.format(*args, **kwargs)