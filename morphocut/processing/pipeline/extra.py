from morphocut.processing.pipeline import NodeBase


class Raw(NodeBase):
    """
    Use objects from a raw iterator.
    """

    def __init__(self, iterable):
        self.iterable = iterable

    def __call__(self, input=None):
        if input is not None:
            raise ValueError("input should be None.")

        for obj in self.iterable:
            yield obj
