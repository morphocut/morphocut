"""
Manipulate MorphoCut streams and show diagnostic information.
"""

import itertools
import pprint
from queue import Queue
from threading import Thread

from morphocut._optional import import_optional_dependency
from morphocut import Node, Output

__all__ = ["TQDM", "Slice"]


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
        progress.close()


class Slice(Node):

    def __init__(self, *args):
        super().__init__()
        self.args = args

    def transform_stream(self, stream):
        for obj in itertools.islice(stream, *self.args):
            yield obj


class StreamBuffer(Node):
    """
    Buffer the stream.

    This allows proceessing while I/O bound Nodes wait for data.
    """

    _sentinel = object()

    def __init__(self, maxsize):
        super().__init__()
        self.queue = Queue(maxsize)

    def _fill_queue(self, stream):
        try:
            for obj in stream:
                self.queue.put(obj)
        finally:
            self.queue.put(self._sentinel)

    def transform_stream(self, stream):
        """Apply transform to every object in the stream.
        """

        thread = Thread(target=self._fill_queue, args=(stream, ), daemon=True)
        thread.start()

        while True:
            obj = self.queue.get()
            if obj == self._sentinel:
                break
            yield obj

        # Join filler
        thread.join()


class PrintObjects(Node):

    def __init__(self, *args):
        super().__init__()
        self.args = args

    def transform_stream(self, stream):
        for obj in stream:
            print(id(obj))
            for outp in self.args:
                print(outp.name)
                pprint.pprint(obj[outp])
            yield obj


@Output("index")
class Enumerate(Node):

    def transform_stream(self, stream):
        for i, obj in enumerate(stream):
            yield self.prepare_output(obj, i)


@Output("value")
class FromIterable(Node):
    """Insert values from the supplied iterator into the stream."""

    def __init__(self, iterable):
        super().__init__()
        self.iterable = iterable

    def transform_stream(self, stream):
        """Transform a stream."""

        for obj in stream:
            iterable = self.prepare_input(obj, "iterable")

            for value in iterable:
                yield self.prepare_output(obj.copy(), value)
