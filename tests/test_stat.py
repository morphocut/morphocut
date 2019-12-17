from queue import Queue
from morphocut import Pipeline

import pytest

from morphocut import Pipeline
from morphocut.stat import (
    ExponentialSmoothing
)

def test_ExponentialSmoothing():
    with Pipeline() as pipeline:
        value = 5
        alpha = 0.8
        result = ExponentialSmoothing(value, alpha)

    stream = pipeline.transform_stream()
    obj = next(stream)
    assert obj[result] == 5
