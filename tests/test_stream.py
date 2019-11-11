from queue import Queue
from morphocut import Pipeline
from morphocut.stream import Slice, StreamBuffer, PrintObjects, TQDM, FromIterable

import pytest


def test_TQDM():
    # Assert that the progress bar works with stream
    with Pipeline() as pipeline:
        item = FromIterable(range(10))
        result = TQDM("Description")

    stream = pipeline.transform_stream()
    result = [o[item] for o in stream]

    assert result == list(range(10))


def test_Slice():
    # Assert that the stream is sliced
    items = "ABCDEFG"

    with Pipeline() as pipeline:
        result = Slice(2)

    stream = pipeline.transform_stream(items)
    obj = list(stream)

    assert obj == ["A", "B"]

    # Assert that the stream is sliced from the specified start and end
    with Pipeline() as pipeline:
        result = Slice(2, 4)

    stream = pipeline.transform_stream(items)
    obj = list(stream)

    assert obj == ["C", "D"]


def test_StreamBuffer():
    with Pipeline() as pipeline:
        item = FromIterable(range(10))
        result = StreamBuffer(1)

    stream = pipeline.transform_stream()
    result = [o[item] for o in stream]

    assert result == list(range(10))


def test_FromIterable():
    values = list(range(10))

    with Pipeline() as pipeline:
        value = FromIterable(values)

    stream = pipeline.transform_stream()

    result = [o[value] for o in stream]

    assert values == result


def test_PrintObjects(capsys):
    values = list(range(10))

    with Pipeline() as pipeline:
        value = FromIterable(values)
        PrintObjects(value)

    # TODO: Capture output and compare

    # https://docs.pytest.org/en/latest/capture.html#accessing-captured-output-from-a-test-function
    # pipeline.run()
    stream = pipeline.transform_stream()
    result = [o[value] for o in stream]

    captured = capsys.readouterr()
    print(captured.out)
    assert result == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # captured = capsys.readouterr()
    # assert captured.out == '9'
