from typing_extensions import Literal
from typing import Any, IO, List, Mapping, Tuple, TypeVar, Union
from morphocut.core import Stream

import numpy as np
import h5py

from morphocut import Node, RawOrVariable, ReturnOutputs, closing_if_closable

T = TypeVar("T")
MaybeTuple = Union[T, Tuple[T]]
MaybeList = Union[T, List[T]]


@ReturnOutputs
class HDF5Writer(Node):
    """
    Write arrays to a HDF5 file.

    Args:
        file_name (str): Location of the output file.
        data (Mapping or Variable): Data to write (name => array).
        file_mode (str, optional): Opening mode of the HDF5 file.
        dataset_mode (str, optional): Dataset behavior.
            "create": Create dataset with arr.
                The dataset is created
            "append": Append arr to a dataset.
                The data must have the same shape as the existing data, without the first dimension.
            "extend": Extend a dataset with arr.
                The data must have the same shape as the existing data, except the first dimension.

    Example:
        .. code-block:: python

            with Pipeline() as pipeline:
                image_fn = ...
                image = ImageReader(image_fn)
                HDF5Writer("path/to/file.h5", {image_fn: image})
            pipeline.transform_stream()

    .. seealso::
        For more information about the metadata fields, see the project import page of EcoTaxa.
    """

    # TODO: Make file_name a stream variable
    # TODO: Implement more `dataset_mode`s

    def __init__(
        self,
        file_name: str,
        data: Mapping[str, RawOrVariable[Any]],
        *,
        file_mode="w",
        dataset_mode: RawOrVariable[Literal["create", "append", "extend"]] = "create",
        compression=None,
        verbose=False,
    ):
        super().__init__()

        assert dataset_mode == "extend"

        self.file_name = file_name
        self.data = data
        self.file_mode = file_mode
        self.dataset_mode = dataset_mode
        self.compression = compression
        self.verbose = verbose

    def _transform_stream_extend(self, stream: Stream):
        with h5py.File(self.file_name, self.file_mode) as h5, closing_if_closable(
            stream
        ):
            datasets: Mapping[str, h5py.Dataset] = {}

            # Offsets per dataset
            offsets: Mapping[str, int] = {}

            for obj in stream:
                data: Mapping[str, Any] = self.prepare_input(obj, "data")  # type: ignore

                for dset, arr in data.items():
                    batch_size = len(arr)

                    if dset not in datasets:
                        if isinstance(arr[0], str):
                            data_shape = tuple()
                            dtype = h5py.string_dtype()
                        else:
                            arr = np.array(arr)
                            data_shape = arr[0].shape
                            dtype = arr.dtype

                        initial_shape = (batch_size * 2,) + data_shape

                        if self.verbose:
                            print(
                                f"Creating dataset {dset}: {initial_shape}, dtype = {dtype}"
                            )

                        datasets[dset] = h5.create_dataset(
                            dset,
                            shape=initial_shape,
                            maxshape=(None,) + data_shape,
                            chunks=(batch_size,) + data_shape,
                            dtype=dtype,
                            compression=self.compression,
                        )
                        offsets[dset] = 0

                    if offsets[dset] + batch_size > datasets[dset].shape[0]:
                        new_length = datasets[dset].shape[0] * 2
                        if self.verbose:
                            print(
                                f"Resizing dataset {dset} to new length: {new_length:,d}"
                            )

                        datasets[dset].resize(new_length, axis=0)

                    datasets[dset][offsets[dset] : offsets[dset] + batch_size] = arr

                    offsets[dset] += batch_size

                yield obj

            for name, dataset in datasets.items():
                if self.verbose:
                    print(
                        f"Truncating dataset {name} to actual length: {offsets[name]:,d}"
                    )

                dataset.resize(offsets[name], axis=0)

    def transform_stream(self, stream):
        if self.dataset_mode == "extend":
            return self._transform_stream_extend(stream)
        else:
            raise ValueError(f"Unknown dataset mode: {self.dataset_mode}")
