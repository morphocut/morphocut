from contextlib import ExitStack, nullcontext
from typing import TYPE_CHECKING

import numpy as np
import pytest

from morphocut.batch import BatchedPipeline
from morphocut.core import Pipeline
from morphocut.stream import Unpack
from morphocut.torch import PyTorch

if TYPE_CHECKING:
    import torch.nn
else:
    torch = pytest.importorskip("torch")


class IdentityModule(torch.nn.Module):
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
    "batch",
    [True, False],
)
@pytest.mark.parametrize(
    "output_key",
    [None, "foo"],
)
@pytest.mark.parametrize(
    "input_dtype",
    [np.uint8, np.float32, np.float64],
)
@pytest.mark.parametrize(
    "input_ndim",
    [0, 1, 2, 3],
)
def test_PyTorch(device, batch, output_key, input_dtype, input_ndim):
    module = IdentityModule(output_key)

    input_data = [
        np.array(i, dtype=input_dtype).reshape((1,) * input_ndim) for i in range(100)
    ]

    with Pipeline() as p:
        input = Unpack(input_data)

        block = BatchedPipeline(2) if batch else nullcontext(p)
        with block:
            result = PyTorch(
                module,
                input,
                is_batch=batch,
                device=device,
                output_key=output_key,
            )

    output_data = [o[result] for o in p.transform_stream()]

    np.testing.assert_equal(output_data, input_data)
