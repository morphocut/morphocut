from typing import TYPE_CHECKING, Callable, List, Tuple, Union

from morphocut import Node, Output, RawOrVariable, ReturnOutputs, closing_if_closable
from morphocut._optional import UnavailableObject
from morphocut.batch import Batch
import threading
import queue

if TYPE_CHECKING:
    import torch
else:
    try:
        import torch
    except ImportError:
        torch = UnavailableObject("torch")


def stack_ex(
    tensors: Union[Tuple["torch.Tensor"], List["torch.Tensor"]], pin_memory: bool = True
):
    n_tensors = len(tensors)
    first = tensors[0]
    size = (n_tensors,) + first.size()
    out = torch.empty(size, dtype=first.dtype, pin_memory=pin_memory)
    torch.stack(tensors, out=out)
    return out


@ReturnOutputs
@Output("output")
class PyTorch(Node):
    def __init__(
        self,
        model: Callable,
        image: RawOrVariable,
        device=None,
        synchronous=True,
        queue_size=1,
        is_batch=True,
    ):
        super().__init__()

        print("PyTorch.device: ", device)

        if device is not None:
            model = model.to(device)

        # Enable evaluation mode
        model.eval()

        self.model = model
        self.image = image
        self.device = device
        self.synchronous = synchronous
        self.queue_size = queue_size
        self.is_batch = is_batch

    def _transform_stream_sync(self, stream):
        with torch.no_grad(), closing_if_closable(stream):  # type: ignore
            for obj in stream:
                image = self.prepare_input(obj, "image")

                # Assemble batch
                if isinstance(image, Batch):
                    image = torch.stack(image)
                elif not self.is_batch:
                    image = torch.as_tensor(image).unsqueeze(0)

                if self.device is not None:
                    image = image.to(self.device, non_blocking=False)  # type: ignore

                output = self.model(image).cpu().numpy()

                if not self.is_batch:
                    output = output[0]

                yield self.prepare_output(obj, output)

    def _transform_stream_async(self, stream):
        q = queue.Queue(maxsize=self.queue_size)

        def executor():
            with torch.no_grad(), closing_if_closable(stream):  # type: ignore
                for obj in stream:
                    image = self.prepare_input(obj, "image")

                    # Assemble batch
                    if isinstance(image, Batch):
                        image = stack_ex(image, pin_memory=True)
                    elif not self.is_batch:
                        image = torch.as_tensor(image).unsqueeze(0)

                    if self.device is not None:
                        image = image.to(self.device, non_blocking=True)  # type: ignore

                    q.put((obj, self.model(image)))

                q.put(None)

        t = threading.Thread(target=executor)
        t.start()

        while True:
            msg = q.get()
            if msg is None:
                break

            obj, output = msg

            output = output.cpu().numpy()

            if not self.is_batch:
                output = output[0]

            yield self.prepare_output(obj, output)

    def transform_stream(self, stream):
        if self.synchronous:
            return self._transform_stream_sync(stream)
        else:
            return self._transform_stream_async(stream)
