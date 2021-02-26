"""
Filters applied to a sliding window of stream objects.
"""

from abc import abstractmethod
from collections import deque

import numpy as np

from morphocut.core import (
    Node,
    Output,
    RawOrVariable,
    ReturnOutputs,
    Stream,
    check_stream,
    closing_if_closable,
)


class _BaseFilter(Node):
    def __init__(self, value: RawOrVariable, size: int = 5, centered=True):
        super().__init__()

        self.value = value

        if size <= 0:
            raise ValueError("size must be positive")

        self.centered = centered

        if centered:
            if not size % 2:
                raise ValueError("size must be odd if centered")

        self.size = size

    @abstractmethod
    def _update(self, value):
        return None

    def transform_stream(self, stream: Stream) -> Stream:
        stream = check_stream(stream)

        obj_queue = deque()
        response_queue = deque()

        with closing_if_closable(stream):
            # Lead-in: Initialize filter
            for _ in range(self.size):
                obj = next(stream)
                obj_queue.append(obj)

                value = self.prepare_input(obj, "value")
                response = self._update(value)

                response_queue.append(response)

            if self.centered:
                for _ in range(self.size // 2):
                    response_queue.popleft()

            # Normal operation
            for obj in stream:
                obj_queue.append(obj)
                value = self.prepare_input(obj, "value")
                response_queue.append(self._update(value))

                obj = obj_queue.popleft()
                response = response_queue.popleft()

                yield self.prepare_output(obj, response)

            # Lead-out: Yield rest of the queue, invalidating old filter responses
            while obj_queue:
                obj = obj_queue.popleft()
                response = response_queue.popleft()
                yield self.prepare_output(obj, response)
                response_queue.append(self._update(None))

            assert not obj_queue


class _UFuncFilter(_BaseFilter):
    ufunc: np.ufunc

    def __init__(self, value: RawOrVariable, size: int = 5, centered=True):
        super().__init__(value, size=size, centered=centered)

        self._acc = None
        self._valid = None
        self._i = 0

    def _update(self, value):
        if value is None:
            if self._valid is None:
                raise ValueError("First value must not be None")
            self._valid[self._i] = False
        else:
            value = np.asarray(value)

            if self._acc is None:
                self._acc = np.zeros((self.size,) + value.shape)
                self._valid = np.zeros((self.size,), dtype=bool)

            self._acc[self._i] = value
            self._valid[self._i] = True

        # Increase i
        self._i = (self._i + 1) % self.size

        if self._valid.any():
            return self.__class__.ufunc(self._acc[self._valid], axis=0)
        return None


@ReturnOutputs
@Output("response")
class MaxFilter(_UFuncFilter):
    ufunc = np.max


@ReturnOutputs
@Output("response")
class MinFilter(_UFuncFilter):
    ufunc = np.min


@ReturnOutputs
@Output("response")
class MedianFilter(_UFuncFilter):
    ufunc = np.median


@ReturnOutputs
@Output("response")
class MeanFilter(_UFuncFilter):
    ufunc = np.mean
