"""
Manipulate MorphoCut streams and show diagnostic information.
"""

import itertools

from morphocut._optional import import_optional_dependency
from morphocut.graph import Node


class TQDM(Node):
    """
    Provide a progress indicator via `tqdm`_.

    .. _tqdm: https://github.com/tqdm/tqdm
    """

    def __init__(self, description=None):
        super().__init__()
        self._tqdm = import_optional_dependency("tqdm")
        self.description = description

    def transform_stream(self, stream):
        progress = self._tqdm.tqdm(stream)
        for obj in progress:

            description = self.prepare_input(obj, "description")

            if description:
                progress.set_description(description)

            yield obj


class Slice(Node):

    def __init__(self, *args):
        super().__init__()
        self.args = args

    def transform_stream(self, stream):
        for obj in itertools.islice(stream, *self.args):
            yield obj
