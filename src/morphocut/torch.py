from contextlib import ExitStack
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

from morphocut import Node, Output, RawOrVariable, ReturnOutputs
from morphocut._optional import UnavailableObject
from morphocut.batch import Batch


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
            Only applicable for CUDA devices. If None, enabled by default for CUDA devices.
        pre_transform (callable, optional): Transformation to apply to the individual input values.
        autocast (bool, optional): Enable automatic mixed precision inference to improve performance.

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
        device: Union[None, str, torch.device] = None,
        is_batch=None,
        output_key=None,
        pin_memory=None,
        pre_transform: Optional[Callable] = None,
        autocast=False,
    ):
        super().__init__()

        print("PyTorch.device: ", device)

        if device is not None:
            device = torch.device(device)

        if pin_memory is None and device is not None:
            pin_memory = device.type == "cuda"

        self.device = device
        module = module.to(device)

        # Enable evaluation mode
        module.eval()

        self.model = module
        self.input = input
        self.is_batch = is_batch
        self.output_key = output_key
        self.pin_memory = pin_memory
        self.pre_transform = pre_transform

        if autocast and device is None:
            raise ValueError("Supply a device when using autocast.")

        self.autocast = autocast

    def transform(self, input):
        with ExitStack() as stack:
            stack.enter_context(torch.no_grad())

            if self.autocast:
                stack.enter_context(torch.autocast(self.device.type))  # type: ignore

            is_batch = (
                isinstance(input, Batch) if self.is_batch is None else self.is_batch
            )

            # Assemble batch
            if is_batch:
                if self.pre_transform is not None:
                    input = [self.pre_transform(inp) for inp in input]
                input = [torch.as_tensor(inp) for inp in input]

                input = _stack_pin(input) if self.pin_memory else torch.stack(input)
                is_batch = True
            else:
                if self.pre_transform is not None:
                    input = self.pre_transform(input)
                input = torch.as_tensor(input)

            if not is_batch:
                # Add batch dimension
                input = input.unsqueeze(0)

            if self.device is not None:
                input = input.to(self.device)  # type: ignore

            output = self.model(input)

            if self.output_key is not None:
                output = output[self.output_key]

            output = output.cpu().numpy()

            if not is_batch:
                # Remove batch dimension
                output = output[0]

            return output
