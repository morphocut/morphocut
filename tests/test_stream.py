from morphocut import Pipeline
from morphocut.stream import Slice

import pytest


def test_Slice():
    # Assert that the stream is sliced
    items = "ABCDEFG"

    with Pipeline() as pipeline:
        result = Slice(2)()

    stream = list(pipeline.transform_stream(items))
    obj = stream

    assert obj[0] == 'A'
    assert obj[1] == 'B'

    # Assert that the stream is sliced from the specified start and end
    with Pipeline() as pipeline:
        result = Slice(2, 4)()

    stream = list(pipeline.transform_stream(items))
    obj = stream

    assert obj[0] == 'C'
    assert obj[1] == 'D'