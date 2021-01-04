from typing import Callable

from morphocut import Node, Output, RawOrVariable, ReturnOutputs, closing_if_closable
from morphocut._optional import UnavailableObject

try:
    import torch
    import torch.utils.data as torch_utils_data
except ImportError:
    torch = UnavailableObject("torch")
    torch_utils_data = UnavailableObject("torch.utils.data")


class _Envelope:
    __slots__ = ["data"]

    def __init__(self, data):
        self.data = data


@ReturnOutputs
@Output("output")
class PyTorch(Node):
    def __init__(self, model: Callable, image: RawOrVariable):
        super().__init__()
        self.model = model
        self.image = image

        class _StreamDataset(torch_utils_data.IterableDataset):
            def __init__(self, node, stream):
                self.node = node
                self.stream = stream

            def __iter__(self):
                with closing_if_closable(self.stream) as stream:
                    for obj in stream:
                        yield (self.node.prepare_input(obj, ("image",)), _Envelope(obj))

        self._StreamDataset = _StreamDataset

    def transform_stream(self, stream):
        stream_ds = self._StreamDataset(self, stream)
        dl = torch_utils_data.DataLoader(stream_ds, batch_size=128, num_workers=0)

        with torch.no_grad():
            for batch_image, batch_obj in dl:
                batch_output = self.model(batch_image)

                for output, env_obj in zip(batch_output, batch_obj):
                    print("output", output)
                    yield self.prepare_output(env_obj.data, output)
