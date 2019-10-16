from queue import Queue
from morphocut import Pipeline
from morphocut.stream import Slice, StreamBuffer, PrintObjects

import pytest


def test_Slice():
    # Assert that the stream is sliced
    items = "ABCDEFG"

    with Pipeline() as pipeline:
        result = Slice(2)

    obj = result.transform_stream(items)

    assert obj.__next__() == 'A'
    assert obj.__next__() == 'B'

    # Assert that the stream is sliced from the specified start and end
    with Pipeline() as pipeline:
        result = Slice(2, 4)

    obj = result.transform_stream(items)

    assert obj.__next__() == 'C'
    assert obj.__next__() == 'D'

def test_StreamBuffer():
    # Assert that the stream is buffered
    maxsize = 5
    items = "12345"

    with Pipeline() as pipeline:
        result = StreamBuffer(maxsize)

    obj = result.transform_stream(items)

    assert obj.__next__() == '1'
    assert obj.__next__() == '2'
    assert obj.__next__() == '3'
    assert obj.__next__() == '4'
    assert obj.__next__() == '5'

'''
class TestClass:
    def __init__(self, name):
        self.name = name

def test_PrintObjects():
    # Assert that the stream is buffered
    items = "12345"
    arg = TestClass("test")

    with Pipeline() as pipeline:
        result = PrintObjects(arg)

    stream = result.transform_stream(items)

    assert stream.__next__() == '1'
    '''