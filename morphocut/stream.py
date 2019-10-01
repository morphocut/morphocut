import itertools

from morphocut._optional import import_optional_dependency
from morphocut.graph import Node


class TQDM(Node):
    def __init__(self):
        super().__init__()
        self._tqdm = import_optional_dependency("tqdm")

    def transform_stream(self, stream):
        for obj in self._tqdm.tqdm(stream):
            yield obj


class Slice(Node):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def transform_stream(self, stream):
        for obj in itertools.islice(stream, *self.args):
            yield obj
