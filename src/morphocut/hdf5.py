import logging
from typing import IO, Any, List, Literal, Mapping, Optional, Tuple, TypeVar, Union

import numpy as np

from morphocut import Node, RawOrVariable, ReturnOutputs, closing_if_closable
from morphocut.core import Stream

T = TypeVar("T")
MaybeTuple = Union[T, Tuple[T]]
MaybeList = Union[T, List[T]]

logger = logging.getLogger(__name__)


@ReturnOutputs
class HDF5Writer(Node):
    """
    Write arrays to a HDF5 file.

    Args:
        file_name (str): Location of the output file.
        data (Mapping or Variable): Data to write (name => array).
        file_mode (str, optional): Opening mode of the HDF5 file.
        dataset_mode (str, optional): Dataset behavior.

            "append": Append an array of shape :math:`(*)` to a dataset of shape :math:`(N,*)`.
                The data must have the same shape as the existing data, without the first dimension.
            "extend": Extend a dataset of shape :math:`(N,*)` with an array of shape :math:`(M,*)`.
                The data must have the same shape as the existing data, except in the first dimension.

        compression: Compression strategy. See :py:meth:`h5py.Group.create_dataset`.
        chunk_size: Chunk size. See :py:meth:`h5py.Group.create_dataset`.


    Example:
        .. code-block:: python

            with Pipeline() as pipeline:
                image_fn = ...
                image = ImageReader(image_fn)
                HDF5Writer("path/to/file.h5", {image_fn: image})
            pipeline.transform_stream()

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
        chunk_size: Optional[int] = None,
    ):
        super().__init__()

        self.file_name = file_name
        self.data = data
        self.file_mode = file_mode
        self.dataset_mode = dataset_mode
        self.compression = compression
        self.chunk_size = chunk_size

    def _transform_stream_extend(self, stream: Stream):
        import h5py

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

                        logger.info(
                            f"Creating dataset {dset}: {initial_shape}, dtype = {dtype}"
                        )

                        datasets[dset] = h5.create_dataset(
                            dset,
                            shape=initial_shape,
                            maxshape=(None,) + data_shape,
                            chunks=(self.chunk_size or batch_size,) + data_shape,
                            dtype=dtype,
                            compression=self.compression,
                        )
                        offsets[dset] = 0

                    if offsets[dset] + batch_size > datasets[dset].shape[0]:
                        new_length = datasets[dset].shape[0] * 2
                        logger.info(
                            f"Resizing dataset {dset} to new length: {new_length:,d}"
                        )

                        datasets[dset].resize(new_length, axis=0)

                    datasets[dset][offsets[dset] : offsets[dset] + batch_size] = arr

                    offsets[dset] += batch_size

                yield obj

            for name, dataset in datasets.items():
                logger.info(
                    f"Truncating dataset {name} to actual length: {offsets[name]:,d}"
                )

                dataset.resize(offsets[name], axis=0)

    def _transform_stream_append(self, stream: Stream):
        import h5py

        with h5py.File(self.file_name, self.file_mode) as h5, closing_if_closable(
            stream
        ):
            datasets: Mapping[str, h5py.Dataset] = {}

            # Offsets per dataset
            offsets: Mapping[str, int] = {}

            chunk_size = self.chunk_size or 1024

            for obj in stream:
                data: Mapping[str, Any] = self.prepare_input(obj, "data")  # type: ignore

                for dset, value in data.items():
                    if dset not in datasets:
                        if isinstance(value, str):
                            data_shape = tuple()
                            dtype = h5py.string_dtype()
                        else:
                            value = np.array(value)
                            data_shape = value.shape
                            dtype = value.dtype

                        initial_shape = (chunk_size,) + data_shape

                        logger.info(
                            f"Creating dataset {dset}: {initial_shape}, dtype = {dtype}"
                        )

                        datasets[dset] = h5.create_dataset(
                            dset,
                            shape=initial_shape,
                            maxshape=(None,) + data_shape,
                            chunks=(chunk_size,) + data_shape,
                            dtype=dtype,
                            compression=self.compression,
                        )
                        offsets[dset] = 0

                    if offsets[dset] + 1 > datasets[dset].shape[0]:
                        new_length = datasets[dset].shape[0] * 2
                        logger.info(
                            f"Resizing dataset {dset} to new length: {new_length:,d}"
                        )

                        datasets[dset].resize(new_length, axis=0)

                    datasets[dset][offsets[dset]] = value

                    offsets[dset] += 1

                yield obj

            for name, dataset in datasets.items():
                logger.info(
                    f"Truncating dataset {name} to actual length: {offsets[name]:,d}"
                )

                dataset.resize(offsets[name], axis=0)

    def transform_stream(self, stream):
        if self.dataset_mode == "extend":
            return self._transform_stream_extend(stream)

        if self.dataset_mode == "append":
            return self._transform_stream_append(stream)

        raise ValueError(f"Unknown dataset mode: {self.dataset_mode}")
