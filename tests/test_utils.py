from morphocut.core import Pipeline
from morphocut.stream import Unpack
from morphocut.utils import stream_groupby

def test_stream_groupby():
    with Pipeline() as p:
        a = Unpack([1,1,2,2,3,3,4,4,5,5])

    for key, substream in stream_groupby(p.transform_stream(), a):
        values = [obj[a] for obj in substream]
        assert len(values) == 2
        assert values == [key]*len(values)
        