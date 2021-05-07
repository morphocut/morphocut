"""Manipulate MorphoCut streams and show diagnostic information."""

import itertools
import pprint
from queue import Queue
from threading import Thread
from typing import Callable, Iterable, Optional, Tuple, Union

import tqdm
from deprecated.sphinx import deprecated

from morphocut.core import (
    Node,
    Output,
    RawOrVariable,
    ReturnOutputs,
    Stream,
    StreamObject,
    Variable,
    closing_if_closable,
)

__all__ = [
    "Enumerate",
    "Filter",
    "FilterVariables",
    "Pack",
    "PrintObjects",
    "Slice",
    "StreamBuffer",
    "Progress",
    "Unpack",
]


@ReturnOutputs
class Progress(Node):
    """
    Show a dynamically updating progress bar using `tqdm`_.

    .. _tqdm: https://github.com/tqdm/tqdm

    Args:
        description (str): Description of the progress bar.

    Example:
        .. code-block:: python

            with Pipeline() as pipeline:
                Progress("Description")

        Output: Description|███████████████████████| [00:00, 2434.24it/s]
    """

    def __init__(
        self, description: Optional[RawOrVariable[str]] = None, monitor_interval=None
    ):
        super().__init__()
        self.description = description
        self.monitor_interval = monitor_interval

    def transform_stream(self, stream: Stream):
        with closing_if_closable(stream), tqdm.tqdm(stream) as progress:
            if self.monitor_interval is not None:
                progress.monitor_interval = self.monitor_interval

            for obj in progress:

                description = self.prepare_input(obj, "description")

                if description:
                    progress.set_description(description)

                yield obj


@deprecated(reason="Deprecated in favor of Progress.", version="0.2.x")
def TQDM(*args, **kwargs):
    return Progress(*args, **kwargs)

TQDM.__doc__ = Progress.__doc__


@ReturnOutputs
class Slice(Node):
    """
    |stream| Slice the :py:obj:`~morphocut.core.Stream`.

    Filter objects in the :py:obj:`~morphocut.core.Stream` based on their index.

    Args:
        start (int, optional): Skip this many objects upfront.
        stop (int, optional): Stop at this index.
        step (int, optional): Skip this many objects in every step.
    """

    def __init__(self, *args: Optional[int]):
        super().__init__()
        self.args = args

    def transform_stream(self, stream: Stream):
        with closing_if_closable(stream):
            for obj in itertools.islice(stream, *self.args):
                yield obj


@ReturnOutputs
class StreamBuffer(Node):
    """
    Buffer the stream.

    Args:
        maxsize (int): Maximum size of the buffer.

    This allows continued processing while I/O bound Nodes wait for data.
    """

    _sentinel = object()

    def __init__(self, maxsize: int):
        super().__init__()
        self.queue = Queue(maxsize)

    def _fill_queue(self, stream: Stream):
        try:
            with closing_if_closable(stream):
                for obj in stream:
                    self.queue.put(obj)
        finally:
            self.queue.put(self._sentinel)

    def transform_stream(self, stream: Stream):
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
    r"""
    Print the contents of :py:class:`~morphocut.core.StreamObject`\ s.

    For debugging purposes only.

    Args:
       *args (Variable): Variables to display.
    """

    def __init__(self, *args: Tuple[Variable]):
        super().__init__()
        self.args = args

    def transform_stream(self, stream: Stream):
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
    """
    Enumerate objects in the :py:obj:`~morphocut.core.Stream`.

    Args:
        start (int, default 0): Start value of the counter.

    Returns:
        Variable[int]: Index (from start).
    """

    def __init__(self, start: int = 0):
        super().__init__()
        self.start = start

    def transform_stream(self, stream: Stream):
        with closing_if_closable(stream):
            for i, obj in enumerate(stream, start=self.start):
                yield self.prepare_output(obj, i)


@ReturnOutputs
@Output("value")
class Unpack(Node):
    """
    |stream| Unpack values from an iterable into the :py:obj:`~morphocut.core.Stream`.

    The result is basically the cross-product of the stream with the iterable.

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

    def transform_stream(self, stream: Stream):
        """Transform a stream."""

        with closing_if_closable(stream):
            for obj in stream:
                iterable = self.prepare_input(obj, "iterable")

                for value in iterable:
                    yield self.prepare_output(obj.copy(), value)


@ReturnOutputs
class Pack(Node):
    """
    |stream| Pack values of subsequent objects in the stream into one tuple.

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

    def transform_stream(self, stream: Stream):
        while True:
            packed = list(itertools.islice(stream, self.size))

            if not packed:
                break

            packed_values = tuple(tuple(o[v] for o in packed) for v in self.variables)

            yield self.prepare_output(packed[0], *packed_values)


class _Predicate:
    def __init__(self, variable: Variable):
        self.variable = variable

    def __call__(self, obj: StreamObject) -> bool:
        return obj[self.variable]


@ReturnOutputs
class Filter(Node):
    """
    |stream| Filter objects in the :py:obj:`~morphocut.core.Stream`.

    After this node, the stream will only contain objects for
    which `function` evaluates to `True`.

    Args:
        predicate (Variable or callable):
            If the predicate is true, the object will stay in the stream.
            If a callable, it will receive a
            :py:class:`~morphocut.core.StreamObject` and must return a bool.

    Example:
        .. code-block:: python

            with Pipeline() as p:
                a = Unpack([1,2,3])
                # The stream now consists of three objects:
                # {a: 1}, {a: 2}, {a: 3}

                # Keep only objects where a>2
                Filter(a>2)
                # The stream now consists of 1 object:
                # {a: 3}

                ## OR:
                # Keep only objects where a>2.
                # Here, `obj` is the current StreamObject.
                # This form might be preferred for performance reasons.
                Filter(lambda obj: obj[a] > 2)

    """

    def __init__(self, predicate: Union[Variable, Callable[[StreamObject], bool]]):
        super().__init__()

        if isinstance(predicate, Variable):
            self.predicate = _Predicate(predicate)
        else:
            self.predicate = predicate

    def transform_stream(self, stream: Stream):
        with closing_if_closable(stream):
            for obj in stream:
                if not self.predicate(obj):
                    continue

                yield obj


@ReturnOutputs
class FilterVariables(Node):
    r"""
    |stream| Only keep the specified Variables in the stream.

    This might reduce memory usage and speed up processing, especially when
    :py:class:`~morphocut.core.StreamObject`\ s have to be sent to other processes.
    """

    def __init__(self, *variables):
        super().__init__()
        self.keys = {
            StreamObject._as_key(v)  # pylint: disable=protected-access
            for v in variables
        }

    def transform_stream(self, stream: Stream):
        with closing_if_closable(stream):
            for obj in stream:
                yield StreamObject({k: v for k, v in obj.items() if k in self.keys})
