from contextlib import nullcontext
from typing import TYPE_CHECKING
from morphocut.batch import BatchPipeline

import pytest

from morphocut.core import Pipeline
from morphocut.stream import Unpack
from morphocut.torch import PyTorch

if TYPE_CHECKING:
    import torch.nn
else:
    torch = pytest.importorskip("torch")


class MyModule(torch.nn.Module):
    def __init__(self, output_key=None) -> None:
        super().__init__()
        self.output_key = output_key

    def forward(self, input):
        if self.output_key is None:
            return input

        return {self.output_key: input}


@pytest.mark.parametrize(
    "device",
    [None, "cpu"],
)
@pytest.mark.parametrize(
    "queue_size",
    [0, 2],
)
@pytest.mark.parametrize(
    "batch",
    [True, False],
)
@pytest.mark.parametrize(
    "output_key",
    [None, "foo"],
)
def test_PyTorch(device, queue_size, batch, output_key):
    module = MyModule(output_key)

    with Pipeline() as p:
        input = Unpack([torch.tensor([float(i)]) for i in range(100)])

        block = BatchPipeline(2) if batch else nullcontext(p)
        with block:
            result = PyTorch(
                module,
                input,
                is_batch=batch,
                device=device,
                queue_size=queue_size,
                output_key=output_key,
            )

    results = [o[result] for o in p.transform_stream()]
