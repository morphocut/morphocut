from queue import Queue
from morphocut import Pipeline
from morphocut.stream import Slice, StreamBuffer, PrintObjects, TQDM

import pytest

def test_TQDM():
    # Assert that the progress bar works with stream
    items = range(5)

    with Pipeline() as pipeline:
        result = TQDM("Description")

    stream = pipeline.transform_stream(items)
    obj = list(stream)

    assert obj == [0, 1, 2, 3, 4]
    assert result.description == 'Description'

def test_Slice():
    # Assert that the stream is sliced
    items = "ABCDEFG"

    with Pipeline() as pipeline:
        result = Slice(2)

    stream = pipeline.transform_stream(items)
    obj = list(stream)

    assert obj == ['A', 'B']

    # Assert that the stream is sliced from the specified start and end
    with Pipeline() as pipeline:
        result = Slice(2, 4)

    stream = pipeline.transform_stream(items)
    obj = list(stream)

    assert obj == ['C', 'D']

def test_StreamBuffer():
    # Assert that the stream is buffered
    maxsize = 5
    items = "12345"

    with Pipeline() as pipeline:
        result = StreamBuffer(maxsize)

    stream = result.transform_stream(items)
    obj = list(stream)

    assert obj[0] == '1'
    assert obj[1] == '2'
    assert obj[2] == '3'
    assert obj[3] == '4'
    assert obj[4] == '5'

class TestClass:
    def __init__(self, name):
        self.name = name

@pytest.mark.xfail
def test_PrintObjects():
    # Assert that the stream is buffered
    items = "12345"
    arg = TestClass("test")

    with Pipeline() as pipeline:
        result = PrintObjects(arg)

    stream = pipeline.transform_stream(items)

    assert stream.__next__() == '1'
    