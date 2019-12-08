from morphocut import Node, Output, ReturnOutputs, Pipeline
from morphocut.encore import Encore
from morphocut.stream import Unpack
from tests.helpers import Const


@ReturnOutputs
@Output("maximum")
class GlobalMax(Node):
    def __init__(self, value):
        super().__init__()
        self.max = float("-inf")
        self.value = value

        assert isinstance(self.parent, Encore)

    def transform(self, value):
        if not self.parent.encore:
            self.max = max(self.max, value)
        return self.max


def test_encore():
    with Pipeline() as pipeline:
        with Encore():
            item = Unpack(range(10))

            global_max = GlobalMax(item)

        item2 = global_max - item

    stream = pipeline.transform_stream()
    result = [o[item] for o in stream]

    assert result == list(range(10))
