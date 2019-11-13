"""Manipulate MorphoCut streams and show diagnostic information."""

import itertools
import pprint
from queue import Queue
from threading import Thread
from typing import Iterable, Optional, Tuple

from morphocut._optional import import_optional_dependency
from morphocut.core import (
    Node,
    Output,
    RawOrVariable,
    ReturnOutputs,
    StreamObject,
    Variable,
    closing_if_closable,
)

__all__ = ["TQDM", "Slice"]


@ReturnOutputs
class TQDM(Node):
    """
    Show a dynamically updating progress bar using `tqdm`_.

    .. note::
       The external dependency `tqdm`_ is required to use this Node.

    .. _tqdm: https://github.com/tqdm/tqdm
    
    Args:
        description (str): Description of the progress bar.

    Example:
        .. code-block:: python

            with Pipeline() as pipeline:
                TQDM("Description")

        Output: Description|███████████████████████| [00:00, 2434.24it/s]

    """

    def __init__(
        self, description: Optional[RawOrVariable[str]] = None, monitor_interval=None
    ):
        super().__init__()
        self._tqdm = import_optional_dependency("tqdm")
        self.description = description
        self.monitor_interval = monitor_interval

    def transform_stream(self, stream):
        with closing_if_closable(stream), self._tqdm.tqdm(stream) as progress:
            if self.monitor_interval is not None:
                progress.monitor_interval = self.monitor_interval

            for obj in progress:

                description = self.prepare_input(obj, "description")

                if description:
                    progress.set_description(description)

                yield obj


@ReturnOutputs
class Slice(Node):
    def __init__(self, *args: Optional[int]):
        super().__init__()
        self.args = args

    def transform_stream(self, stream):
        with closing_if_closable(stream):
            for obj in itertools.islice(stream, *self.args):
                yield obj


@ReturnOutputs
class StreamBuffer(Node):
    """
    Buffer the stream.

    This allows proceessing while I/O bound Nodes wait for data.
    """

    _sentinel = object()

    def __init__(self, maxsize: int):
        super().__init__()
        self.queue = Queue(maxsize)

    def _fill_queue(self, stream):
        try:
            with closing_if_closable(stream):
                for obj in stream:
                    self.queue.put(obj)
        finally:
            self.queue.put(self._sentinel)

    def transform_stream(self, stream):
        thread = Thread(target=self._fill_queue, args=(stream,), daemon=True)
        thread.start()

        while True:
            obj = self.queue.get()
            if obj == self._sentinel:
                break
            yield obj

        # Join filler
        thread.join()


@ReturnOutputs
class PrintObjects(Node):
    def __init__(self, *args: Tuple[Variable]):
        super().__init__()
        self.args = args

    def transform_stream(self, stream):
        with closing_if_closable(stream):
            for obj in stream:
                print("Stream object at 0x{:x}".format(id(obj)))
                for outp in self.args:
                    print("{}: ".format(outp.name), end="")
                    pprint.pprint(obj[outp])
                yield obj


@ReturnOutputs
@Output("index")
class Enumerate(Node):
    def transform_stream(self, stream):
        with closing_if_closable(stream):
            for i, obj in enumerate(stream):
                yield self.prepare_output(obj, i)


@ReturnOutputs
@Output("value")
class Unpack(Node):
    """
    Unpack values from an iterable into the stream.

    Args:
        iterable (Iterable or Variable): An iterable to unpack.

    Returns:
       Variable: One value from the iterable.

    Example:
        .. code-block:: python

            with Pipeline() as p:
                a = Unpack([1,2,3])
                # The stream now consists of three objects:
                # {a: 1}, {a: 2}, {a: 3}
                b = Unpack([1,2,3])
                # The stream now consists of nine objects:
                # {a: 1, b: 1}, {a: 1, b: 2}, {a: 1, b: 3},
                # {a: 2, b: 1}, {a: 2, b: 2}, {a: 2, b: 3},
                # {a: 3, b: 1}, {a: 3, b: 2}, {a: 3, b: 3}

    See Also:
        :py:class:`~morphocut.stream.Pack`
    """

    def __init__(self, iterable: RawOrVariable[Iterable]):
        super().__init__()
        self.iterable = iterable

    def transform_stream(self, stream):
        """Transform a stream."""

        with closing_if_closable(stream):
            for obj in stream:
                iterable = self.prepare_input(obj, "iterable")

                for value in iterable:
                    yield self.prepare_output(obj.copy(), value)


@ReturnOutputs
class Pack(Node):
    """
    Pack values of subsequent objects in the stream into one tuple.

    Args:
        size (int or Variable): Number of objects to aggregate.
        *variables (Variable): Variables to pack.

    Returns:
       One or more Variable: One output Variable per input Variable.

    Example:
        .. code-block:: python

            with Pipeline() as p:
                a = Unpack([1,2,3])
                # The stream now consists of three objects:
                # {a: 1}, {a: 2}, {a: 3}
                a123 = Pack(3, a)
                # The stream now consists one object:
                # {a: 1, a123: (1,2,3)}

    See Also:
        :py:class:`~morphocut.stream.Unpack`
    """

    def __init__(self, size, *variables):
        super().__init__()
        self.size = size
        self.variables = variables
        # Mess with self.outputs
        self.outputs = [Variable(v.name, self) for v in self.variables]

    def transform_stream(self, stream):
        while True:
            packed = list(itertools.islice(stream, self.size))

            if not packed:
                break

            packed_values = tuple(tuple(o[v] for o in packed) for v in self.variables)

            yield self.prepare_output(packed[0], *packed_values)


@ReturnOutputs
class Filter(Node):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def transform_stream(self, stream):
        with closing_if_closable(stream):
            for obj in stream:
                if not self.function(obj):
                    continue

                yield obj


class FilterVariables(Node):
    """
    Only keep the specified Variables in the stream.

    This might speed up processing.
    """

    def __init__(self, *variables):
        super().__init__()
        self.keys = {
            StreamObject._as_key(v)  # pylint: disable=protected-access
            for v in variables
        }

    def transform_stream(self, stream):
        with closing_if_closable(stream):
            for obj in stream:
                yield StreamObject({k: v for k, v in obj.items() if k in self.keys})
