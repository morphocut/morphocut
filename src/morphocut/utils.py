import itertools
from typing import Any, Iterator, Tuple, Union, overload, TypeVar

from morphocut.core import RawOrVariable, Stream, StreamObject, resolve_variable

T = TypeVar("T")


@overload
def stream_groupby(
    stream: Stream, by: RawOrVariable[T]
) -> Iterator[Tuple[T, Iterator[StreamObject]]]:
    ...


@overload
def stream_groupby(
    stream: Stream, by: Tuple[RawOrVariable[T]]
) -> Iterator[Tuple[Tuple[T], Iterator[StreamObject]]]:
    ...


def stream_groupby(
    stream: Stream, by=Union[RawOrVariable[T], Tuple[RawOrVariable[T]]]
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
