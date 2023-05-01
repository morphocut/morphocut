from queue import Queue
from typing import List, Optional, Sequence

import pytest
import tqdm

from morphocut import Pipeline
from morphocut.core import StreamObject
from morphocut.stream import (
    Enumerate,
    Filter,
    FilterVariables,
    Pack,
    PrintObjects,
    Progress,
    Slice,
    StreamBuffer,
    Unpack,
)


def test_Progress(monkeypatch: pytest.MonkeyPatch):
    # Monkeypatch tqdm so that we can extract tqdm instance attributes
    tqdm_instance: List[Optional[tqdm.tqdm]] = [None]
    tqdm_cls = tqdm.tqdm

    def mock_tqdm(*args, **kwargs):
        tqdm_instance[0] = tqdm_cls(*args, **kwargs)
        return tqdm_instance[0]

    monkeypatch.setattr(tqdm, "tqdm", mock_tqdm)

    # Assert that the progress bar works with stream
    with Pipeline() as pipeline:
        item = Unpack(range(10))
        result = Progress("Description")

    stream = pipeline.transform_stream()
    result = [o[item] for o in stream]

    assert result == list(range(10))
    assert tqdm_instance[0].total == 10


def test_Slice():
    # Assert that the stream is sliced
    items = "ABCDEFG"

    with Pipeline() as pipeline:
        item = Unpack(items)
        result = Slice(2)

    stream = list(pipeline.transform_stream())
    result = [obj[item] for obj in stream]
    n_remaining = [obj.n_remaining_hint for obj in stream]

    assert result == ["A", "B"]
    assert n_remaining == [2, 1]

    # Assert that the stream is sliced from the specified start and end
    with Pipeline() as pipeline:
        item = Unpack(items)
        result = Slice(2, 4)

    stream = list(pipeline.transform_stream())
    result = [obj[item] for obj in stream]
    n_remaining = [obj.n_remaining_hint for obj in stream]

    assert result == ["C", "D"]
    assert n_remaining == [2, 1]


def test_StreamBuffer():
    with Pipeline() as pipeline:
        item = Unpack(range(10))
        StreamBuffer(1)

    stream = pipeline.transform_stream()
    objects = [o for o in stream]

    assert objects[0].n_remaining_hint == 10
    assert [o[item] for o in objects] == list(range(10))


def test_Unpack():
    values = list(range(10))

    with Pipeline() as pipeline:
        value = Unpack(values)

    stream = pipeline.transform_stream()
    objects = [o for o in stream]

    assert objects[0].n_remaining_hint == 10
    assert values == [o[value] for o in objects]


def test_Pack():
    values = list(range(10))

    with Pipeline() as pipeline:
        value = Unpack(values)
        values_packed = Pack(2, value)

    objects = [o for o in pipeline.transform_stream()]

    # TODO:
    # assert objects[0].n_remaining_hint == 5

    assert [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)] == [
        o[values_packed] for o in objects
    ]


def test_PrintObjects(capsys):
    values = list(range(10))

    with Pipeline() as pipeline:
        value = Unpack(values)
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


def last_n_remaining_hint(objects: Sequence[StreamObject]) -> Optional[int]:
    n_remaining_hint = None
    for obj in objects:
        if obj.n_remaining_hint is not None:
            n_remaining_hint = obj.n_remaining_hint
    return n_remaining_hint


def test_Filter():
    values = list(range(10))

    with Pipeline() as pipeline:
        value = Unpack(values)
        Filter(lambda obj: obj[value] % 2 == 0)

    objects = [o for o in pipeline.transform_stream()]

    assert [v for v in values if v % 2 == 0] == [o[value] for o in objects]

    # n_remaining_hint of last object is 1
    n_remaining = [o.n_remaining_hint for o in objects]
    assert n_remaining[-1] == 1


def test_Filter_highlevel():
    values = list(range(10))

    with Pipeline() as pipeline:
        value = Unpack(values)
        predicate = value % 2 == 0
        Filter(predicate)

    objects = [o for o in pipeline.transform_stream()]

    assert [v for v in values if v % 2 == 0] == [o[value] for o in objects]

    # n_remaining_hint of last object is 1
    n_remaining = [o.n_remaining_hint for o in objects]
    assert n_remaining[-1] == 1


def test_FilterVariables():
    values = list(range(10))

    with Pipeline() as pipeline:
        a = Unpack(values)
        b = Unpack(values)
        FilterVariables(b)

    stream = list(pipeline.transform_stream())

    for o in stream:
        assert a not in o
        assert b in o


def test_Enumerate():
    with Pipeline() as pipeline:
        a = Unpack(range(10))
        i = Enumerate()

    stream = pipeline.transform_stream()

    for obj in stream:
        assert obj[a] == obj[i]
