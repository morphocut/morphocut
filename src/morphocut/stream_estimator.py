from typing import Optional


class _IncomingObjectContex:
    def __init__(self, estimator: "StreamEstimator", n_consumed) -> None:
        self.estimator = estimator
        self.n_consumed = n_consumed
        self.n_emitted = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        estimator = self.estimator
        # Update number of processed objects
        estimator.n_consumed += self.n_consumed
        estimator.n_emitted += self.n_emitted

        # Update global estimate
        estimator.global_estimate = estimator.n_emitted / estimator.n_consumed

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
            and self.estimator.global_estimate is not None
        ):
            if self.estimator.local_estimate is not None:
                n_remaining_hint = max(
                    1,
                    (
                        round(
                            (self.estimator.n_remaining_in - 1)
                            * self.estimator.global_estimate
                            + self.estimator.local_estimate
                        )
                        - self.n_emitted
                    ),
                )
            else:
                n_remaining_hint = max(
                    1,
                    round(
                        self.estimator.n_remaining_in * self.estimator.global_estimate
                    )
                    - self.n_emitted,
                )

        self.n_emitted += 1

        return n_remaining_hint


class StreamEstimator:
    """
    Estimate the number of remaining objects in the stream.

    Example:
        .. code-block:: python

            est = StreamEstimator()

            for obj in stream:
                # We're expecting 10 emitted objects for every consumed object:
                local_estimate = 10
                with est.consume(obj.n_remaining_hint, local_estimate=local_estimate) as incoming:
                    for _ in range(10):
                        yield self.prepare_output(
                            obj.copy(), value, n_remaining_hint=incoming.emit()
                        )
    """

    def __init__(self) -> None:
        self.n_remaining_in = None
        self.n_consumed = 0
        self.n_emitted = 0
        self.global_estimate: Optional[float] = None
        self.local_estimate: Optional[int] = None

    def consume(
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

        return _IncomingObjectContex(self, n_consumed)
