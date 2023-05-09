from contextlib import ExitStack
from typing import TYPE_CHECKING

import pytest

from morphocut.batch import BatchPipeline
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
    "n_parallel",
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
def test_PyTorch(device, n_parallel, batch, output_key):
    module = MyModule(output_key)

    with Pipeline() as p:
        input = Unpack([torch.tensor([float(i)]) for i in range(100)])

        with ExitStack() as stack:
            if batch:
                stack.enter_context(BatchPipeline(2))
            result = PyTorch(
                module,
                input,
                is_batch=batch,
                device=device,
                n_parallel=n_parallel,
                output_key=output_key,
            )

    results = [o[result] for o in p.transform_stream()]
