import itertools
from typing import Any, Iterator, Tuple, Union

from morphocut.core import Stream, StreamObject, Variable, resolve_variable


def stream_groupby(
    stream: Stream, by=Union[Variable, Tuple[Variable]]
) -> Iterator[Tuple[Any, Iterator[StreamObject]]]:
    """
    Split a stream into sub-streams by key.

    Every time the value of the `by` changes, a new sub-stream is generated.
    The sub-stream is itself an iterator that shares the underlying stream with stream_groupby.

    Args:
        stream (Stream): A MorphoCut stream.
        by: (Variable or value or tuple thereof): The values to group by.

    Yields:
        `(key, sub_stream)`, where `key` is a value or a tuple and `sub_stream` is the corresponding sub-stream.
    """

    keyfunc = lambda obj: resolve_variable(obj, by)

    return itertools.groupby(stream, keyfunc)
