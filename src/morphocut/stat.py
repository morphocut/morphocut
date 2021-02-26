import collections
import itertools
from typing import Any, Optional

import numpy as np

from morphocut.core import (
    Node,
    Output,
    ReturnOutputs,
    Stream,
    Variable,
    _pipeline_stack,
    closing_if_closable,
)
from morphocut.parallel import ParallelPipeline

from morphocut.filters import ExponentialSmoothingFilter as ExponentialSmoothing


@ReturnOutputs
@Output("agg_value")
class RunningMedian(Node):
    """
    Calculate the running median of a value over a stream of objects.

    Parameters:
        value (Variable of numpy.ndarray): Values to calculate the running mean for.
        n_init (int): Number of objects to initialize the median estimate.

    Returns:
        Variable[numpy.ndarray]: Running median approximation of ``value``.

    This node uses the efficient approximation of the median from:
    Mcfarlane, N. J. B., & Schofield, C. P. (1995).
    Segmentation and tracking of piglets in images.
    In Machine Vision and Applications (Vol. 8).
    """

    def __init__(self, value: Variable[np.ndarray], n_init: int = 10):
        super().__init__()
        self.value = value
        self.n_init = n_init
        self.median: Optional[Any] = None

        assert not any(
            isinstance(p, ParallelPipeline) for p in _pipeline_stack
        ), "RunningMedian can not be used in a ParallelPipeline context"

    def transform_stream(self, stream: Stream) -> Stream:
        """Transform a stream."""

        with closing_if_closable(stream):
            # Initial approximation
            # TODO: This does not work correctly in situations when transform_stream
            # is called repeatedly, e.g. ParallelPipeline.
            if self.median is None:
                yield from self._initialize_median(stream)

            # Process
            for obj in stream:
                value = self.prepare_input(obj, "value")

                # Update according to Mcfarlane & Schofield
                mask = value > self.median

                if np.isscalar(mask):
                    self.median += mask
                else:
                    self.median[mask] += 1

                mask = value < self.median

                if np.isscalar(mask):
                    self.median -= mask
                else:
                    self.median[mask] -= 1

                yield self.prepare_output(obj, self.median)

        self.after_stream()

    def _initialize_median(self, stream):
        objects = []
        values = []
        for obj in itertools.islice(stream, self.n_init):
            value = self.prepare_input(obj, "value")
            objects.append(obj)
            values.append(value)

        self.median = np.median(values, axis=0)

        for obj in objects:
            yield self.prepare_output(obj, self.median)
