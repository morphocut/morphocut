import typing as T

from morphocut import Node, Output, ReturnOutputs
from morphocut._optional import import_optional_dependency


class _Envelope:
    __slots__ = ["data"]

    def __init__(self, data):
        self.data = data


@ReturnOutputs
@Output("output")
class PyTorch(Node):

    def __init__(self, model: T.Callable, image):
        super().__init__()
        self.model = model
        self.image = image

        self._torch = import_optional_dependency(
            "torch",
            "Visit https://pytorch.org/ for instructions.",
            "1.2",
        )
        import torch.utils.data
        self._torch_utils_data = torch.utils.data

        class _StreamDataset(torch.utils.data.IterableDataset):

            def __init__(self, node, stream):
                self.node = node
                self.stream = stream

            def __iter__(self):
                for obj in self.stream:
                    yield (
                        self.node.prepare_input(obj, ("image", )),
                        _Envelope(obj),
                    )

        self._StreamDataset = _StreamDataset

    def transform_stream(self, stream):
        stream_ds = self._StreamDataset(self, stream)
        dl = self._torch_utils_data.DataLoader(
            stream_ds, batch_size=128, num_workers=0
        )

        with self._torch.no_grad():
            for batch_image, batch_obj in dl:
                batch_output = self.model(batch_image)

                for output, env_obj in zip(batch_output, batch_obj):
                    print("output", output)
                    yield self.prepare_output(env_obj.data, output)
