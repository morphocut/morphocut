"""Utilities"""

import itertools
from typing import Any, Iterator, Optional, Tuple, TypeVar, Union, overload

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


class _ConsumedObjectContext:
    def __init__(
        self, estimator: "StreamEstimator", n_consumed: int, est_n_emit: Optional[float]
    ) -> None:
        self.estimator = estimator
        self.n_consumed = n_consumed
        self.est_n_emit = est_n_emit

        self.n_emitted = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        estimator = self.estimator
        # Update number of processed objects
        estimator.n_consumed += self.n_consumed
        estimator.n_emitted += self.n_emitted

        # Update global rate estimate
        estimator.rate = estimator.n_emitted / estimator.n_consumed

        if estimator.n_remaining_in is not None:
            # Decrease number of remaining inputs
            estimator.n_remaining_in -= self.n_consumed

    def emit(self):
        """
        Record the emission of one object and return an estimate of the remaining output length.
        """

        n_remaining_hint = None

        if (
            self.estimator.n_remaining_in is not None
            and self.estimator.rate is not None
        ):
            if self.est_n_emit is not None:
                # We know how many objects we will emit:
                # Use precise calculation.
                n_remaining_hint = round(
                    (self.estimator.n_remaining_in - self.n_consumed)
                    * self.estimator.rate
                    + self.est_n_emit
                    - self.n_emitted
                )
            else:
                # We don't know how many objects we will emit:
                # Use global rate estimate.
                n_remaining_hint = round(
                    self.estimator.n_remaining_in * self.estimator.rate - self.n_emitted
                )

        if n_remaining_hint is not None:
            n_remaining_hint = max(1, n_remaining_hint)

        self.n_emitted += 1

        return n_remaining_hint


class StreamEstimator:
    """
    Record how many objects are consumed and emitted and calculate the rate.

    This should be used in `StreamTransformers` that alter the number of objects in the stream
    to update the estimate the number of remaining objects.

    Example:
        .. code-block:: python

            est = StreamEstimator()

            for obj in stream:
                # We're expecting 10 emitted objects for every consumed object:
                local_estimate = 10
                with est.consume(obj.n_remaining_hint, est_n_emit=local_estimate) as incoming:
                    for _ in range(10):
                        yield self.prepare_output(
                            obj.copy(), value, n_remaining_hint=incoming.emit()
                        )
    """

    def __init__(self) -> None:
        self.n_remaining_in = None
        self.n_consumed = 0
        self.n_emitted = 0
        self.rate: Optional[float] = None

    def consume(
        self,
        n_remaining_hint: Optional[int],
        *,
        est_n_emit: Optional[float] = None,
        n_consumed=1,
    ):
        """Context manager for an incoming object."""

        if n_remaining_hint is not None:
            # Set n_remaining to a new estimate
            self.n_remaining_in = n_remaining_hint

        if self.rate is None and est_n_emit is not None:
            self.rate = est_n_emit / n_consumed

        return _ConsumedObjectContext(self, n_consumed, est_n_emit)
