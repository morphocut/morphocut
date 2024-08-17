"""Manipulate MorphoCut streams and show diagnostic information."""

import itertools
import pprint
from typing import Callable, Collection, Optional, Union

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
from morphocut.utils import StreamEstimator, buffered_generator

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
        with closing_if_closable(stream), tqdm.tqdm(
            stream, unit_scale=True
        ) as progress:
            if self.monitor_interval is not None:
                progress.monitor_interval = self.monitor_interval

            for n_processed, obj in enumerate(progress):

                description = self.prepare_input(obj, "description")

                if description:
                    progress.set_description(description)

                if obj.n_remaining_hint is not None:
                    progress.total = n_processed + obj.n_remaining_hint

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
            for n_seen, obj in enumerate(itertools.islice(stream, *self.args)):
                if obj.n_remaining_hint is not None:
                    # Slice into the total number of objects (n_seen + remaining)
                    n_total = len(
                        range(*slice(*self.args).indices(obj.n_remaining_hint + n_seen))
                    )
                    # Subtract n_seen to get remaining
                    obj.n_remaining_hint = max(
                        0,
                        n_total - n_seen,
                    )
                yield obj


@ReturnOutputs
class StreamBuffer(Node):
    """
    |stream| Buffer a stream to improve throughput by decoupling data
    production from data consumption.

    This allows for continued processing of data while I/O-bound nodes
    are waiting for data. Buffering the stream enables asynchronous prefetching and
    minimizes the time spent waiting for data to be available.

    Args:
        maxsize (int): The maximum number of items to buffer. If the buffer is full, further
            data production is paused until items are consumed.

    Example:
        .. code-block:: python

            with Pipeline() as pipeline:
                # Produce data (preferably I/O-bound)
                ...
                StreamBuffer(maxsize=100)
                # Process data
                ...
    """

    _sentinel = object()

    def __init__(self, maxsize: int):
        super().__init__()
        self.maxsize = maxsize

    def transform_stream(self, stream: Stream):
        @buffered_generator(self.maxsize)
        def buffered_stream():
            with closing_if_closable(stream):
                yield from stream

        return buffered_stream()


@ReturnOutputs
class PrintObjects(Node):
    r"""
    Print the contents of :py:class:`~morphocut.core.StreamObject`\ s.

    For debugging purposes only.

    Args:
       *args (Variable): Variables to display.
    """

    def __init__(self, *args: Variable):
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
    |stream| Unpack values from a collection into the :py:obj:`~morphocut.core.Stream`.

    The result is basically the cross-product of the stream with the iterable.

    Args:
        collection (Collection or Variable): An iterable to unpack.

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

    def __init__(self, collection: RawOrVariable[Collection]):
        super().__init__()
        self.collection = collection

    def transform_stream(self, stream: Stream):
        """Transform a stream."""

        with closing_if_closable(stream):
            stream_estimator = StreamEstimator()
            for obj in stream:
                collection = tuple(self.prepare_input(obj, "collection"))
                with stream_estimator.consume(
                    obj.n_remaining_hint, est_n_emit=len(collection)
                ) as incoming:
                    for value in collection:
                        yield self.prepare_output(
                            obj.copy(), value, n_remaining_hint=incoming.emit()
                        )


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
        stream_estimator = StreamEstimator()

        while True:
            packed = list(itertools.islice(stream, self.size))

            if not packed:
                break

            with stream_estimator.consume(
                packed[0].n_remaining_hint, n_consumed=len(packed)
            ) as incoming:

                packed_values = tuple(
                    tuple(o[v] for o in packed) for v in self.variables
                )

                yield self.prepare_output(
                    packed[0], *packed_values, n_remaining_hint=incoming.emit()
                )


class _Predicate:
    def __init__(self, variable: Variable):
        self.variable = variable

    def __call__(self, obj: StreamObject) -> bool:
        return obj[self.variable]


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
            stream_estimator = StreamEstimator()
            for obj in stream:
                with stream_estimator.consume(obj.n_remaining_hint) as incoming:

                    if not self.predicate(obj):
                        continue

                    obj.n_remaining_hint = incoming.emit()
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


@ReturnOutputs
@Output("n_remaining_hint")
class RemainingHint(Node):
    """
    Extract n_remaining_hint from an object.
    """

    def transform_stream(self, stream: Stream):
        with closing_if_closable(stream):
            for obj in stream:
                yield self.prepare_output(obj, obj.n_remaining_hint)
