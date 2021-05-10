from typing import Optional

from morphocut.core import (
    Node,
    Output,
    ReturnOutputs,
    Stream,
    closing_if_closable,
)


class _ObjectContext:
    def __init__(self, estimator: "StreamEstimator", n_consumed) -> None:
        self.estimator = estimator
        self.n_consumed = n_consumed

    def __enter__(self):
        pass

    def __exit__(self, *args):
        estimator = self.estimator
        # Update number of processed objects
        estimator.n_processed += self.n_consumed
        # Update global estimate
        estimator.global_estimate = estimator.n_emitted / estimator.n_processed

        if estimator.n_remaining_in is not None:
            # Decrease number of remaining inputs
            estimator.n_remaining_in -= self.n_consumed


class StreamEstimator:
    """
    Estimate the number of remaining objects in the stream.

    Example:
        .. code-block:: python

            est = StreamEstimator()

            for obj in stream:
                with est.incoming_object(obj.n_remaining_hint):
                    for ...:
                        yield prepare_object(obj.copy(), n_remaining_hint=est.emit())
    """

    def __init__(self) -> None:
        self.n_remaining_in = None
        self.n_processed = 0
        self.n_emitted = 0
        self.n_emitted_local = 0
        self.global_estimate: Optional[float] = None
        self.local_estimate: Optional[int] = None

    def incoming_object(
        self,
        n_remaining_hint: Optional[int],
        *,
        local_estimate: Optional[int] = None,
        n_consumed=1,
    ):
        """Context manager for an incoming object."""

        if n_remaining_hint is not None:
            # Set n_remaining to a new estimate
            self.n_remaining_in = n_remaining_hint

        self.local_estimate = local_estimate
        if self.global_estimate is None:
            self.global_estimate = local_estimate

        # Reset n_emitted_local
        self.n_emitted_local = 0

        return _ObjectContext(self, n_consumed)

    def emit(self):
        """
        Emit one object and returns an estimate of the remaining output length.
        """

        n_remaining_hint = None
        if self.n_remaining_in is not None and self.global_estimate is not None:
            if self.local_estimate is not None:
                n_remaining_hint = max(
                    1,
                    (
                        round(
                            (self.n_remaining_in - 1) * self.global_estimate
                            + self.local_estimate
                        )
                        - self.n_emitted_local
                    ),
                )
            else:
                n_remaining_hint = max(
                    1,
                    round(self.n_remaining_in * self.global_estimate)
                    - self.n_emitted_local,
                )

        self.n_emitted += 1
        self.n_emitted_local += 1

        return n_remaining_hint


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
