from typing import TYPE_CHECKING, List, Tuple, Union

from morphocut import Node, Output, RawOrVariable, ReturnOutputs, closing_if_closable
from morphocut._optional import UnavailableObject
from morphocut.batch import Batch
from morphocut.utils import buffered_generator

if TYPE_CHECKING:  # pragma: no cover
    import torch
else:
    try:
        import torch
    except ImportError:  # pragma: no cover
        torch = UnavailableObject("torch")


def _stack_pin(tensors: Union[Tuple["torch.Tensor"], List["torch.Tensor"]]):
    n_tensors = len(tensors)
    first = tensors[0]
    size = (n_tensors,) + first.size()
    out = torch.empty(size, dtype=first.dtype, pin_memory=True)
    torch.stack(tensors, out=out)
    return out


@ReturnOutputs
@Output("output")
class PyTorch(Node):
    """
    Apply a PyTorch module to the input.

    Args:
        module (torch.nn.Module): PyTorch module.
        input (input, Variable): Input.
        device (str or torch.device, optional): Device.
        n_parallel (int, optional): Run multiple computations in parallel.
            0 means synchronous computations.
        is_batch (bool, optional): Assume that input is a batch.
        output_key (optional): If the module has multiple outputs, output_key selects one of them.
        pin_memory (bool, optional): Use pinned memory for faster CPU-GPU transfer.
            Only applicable for CUDA devices.

    Example:
        .. code-block:: python

            module = ...
            with Pipeline() as pipeline:
                input = ...
                output = PyTorch(module, input)
    """

    def __init__(
        self,
        module: "torch.nn.Module",
        input: RawOrVariable,
        device=None,
        n_parallel=0,
        is_batch=True,
        output_key=None,
        pin_memory=False,
    ):
        super().__init__()

        print("PyTorch.device: ", device)

        if device is not None:
            device = torch.device(device)

        self.device = device
        module = module.to(device)

        # Enable evaluation mode
        module.eval()

        self.model = module
        self.input = input
        self.n_parallel = n_parallel
        self.is_batch = is_batch
        self.output_key = output_key
        self.pin_memory = pin_memory

    def transform_stream(self, stream):
        @buffered_generator(self.n_parallel)
        def output_gen():
            with torch.no_grad(), closing_if_closable(stream):
                for obj in stream:
                    input = self.prepare_input(obj, "input")

                    # Assemble batch
                    if isinstance(input, Batch):
                        input = (
                            _stack_pin(input)
                            if self.n_parallel and self.pin_memory
                            else torch.stack(input)
                        )
                    elif not self.is_batch:
                        input = torch.as_tensor(input).unsqueeze(0)

                    if self.device is not None:
                        input = input.to(self.device, non_blocking=False)  # type: ignore

                    output = self.model(input)

                    if self.output_key is not None:
                        output = output[self.output_key]

                    yield obj, output

        for obj, output in output_gen():
            output = output.cpu().numpy()

            if not self.is_batch:
                output = output[0]

            yield self.prepare_output(obj, output)
