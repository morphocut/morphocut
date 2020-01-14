import numpy as np

from morphocut import Node, Output, ReturnOutputs


@ReturnOutputs
@Output("array")
class AsType(Node):
    def __init__(self, array, dtype, **kwargs):
        super().__init__()

        self.array = array
        self.dtype = dtype
        self.kwargs = kwargs

    def transform(self, array):
        return array.astype(self.dtype, **self.kwargs)
