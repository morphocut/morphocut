from morphocut.batch import BatchPipeline
from morphocut.core import Call, Pipeline
from morphocut.stream import Unpack
import pytest
import itertools

def chunks(it, size):
    it = iter(it)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk:
            break

        yield chunk

@pytest.mark.parametrize(
    "seq_len",
    [
        5, 10, 100, 111
    ],
)
def test_BatchPipeline(seq_len):
    batch_size = 10
    values = list(range(seq_len))
    with Pipeline() as pipeline:
        a = Unpack(values)
        with BatchPipeline(batch_size):
            # Inside BatchPipeline, a is a Sequence
            b = Call(sum, a)

    result = list(pipeline.transform_stream())

    assert [r[a] for r in result] == values

    # Calculate expected values for b
    b_expected = [sum(chunk) for chunk in chunks(values, batch_size)]

    assert [r[b] for r in result[::batch_size]] == b_expected
