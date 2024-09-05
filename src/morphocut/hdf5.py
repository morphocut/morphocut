import logging
from typing import Any, List, Literal, Mapping, Optional, Tuple, TypeVar, Union

import numpy as np
from morphocut import Node, RawOrVariable, ReturnOutputs, closing_if_closable
from morphocut.core import Stream

from .utils import stream_groupby

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

            "create": Create a dataset for each object.
                Each key can only be used once.
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

    def __init__(
        self,
        file_name: RawOrVariable[str],
        data: Union[
            Mapping[str, RawOrVariable[Any]],
            List[Tuple[RawOrVariable[str], RawOrVariable[Any]]],
        ],
        *,
        file_mode="w",
        dataset_mode: RawOrVariable[Literal["create", "append", "extend"]] = "append",
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

    def _transform_stream_create(self, stream: Stream):
        import h5py

        with closing_if_closable(stream):
            for file_name, file_group in stream_groupby(stream, by=self.file_name):
                with h5py.File(file_name, self.file_mode) as h5:
                    for obj in file_group:
                        data: Union[Mapping[str, Any], List[Tuple[str, Any]]] = self.prepare_input(obj, "data")  # type: ignore

                        if isinstance(data, Mapping):
                            data = data.items()  # type: ignore

                        for dset, arr in data:
                            if isinstance(arr[0], str):
                                data_shape = tuple()
                                dtype = h5py.string_dtype()
                            else:
                                arr = np.array(arr, copy=False)
                                data_shape = arr[0].shape
                                dtype = arr.dtype

                            h5.create_dataset(
                                dset,
                                data=arr,
                                dtype=dtype,
                                compression=self.compression,
                            )

                        yield obj

    def _transform_stream_extend(self, stream: Stream):
        import h5py

        with closing_if_closable(stream):
            for file_name, file_group in stream_groupby(stream, by=self.file_name):
                with h5py.File(file_name, self.file_mode) as h5:
                    datasets: Mapping[str, h5py.Dataset] = {}

                    # Offsets per dataset
                    offsets: Mapping[str, int] = {}

                    for obj in file_group:
                        data: Union[Mapping[str, Any], List[Tuple[str, Any]]] = self.prepare_input(obj, "data")  # type: ignore

                        if isinstance(data, Mapping):
                            data = data.items()  # type: ignore

                        for dset, arr in data:
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
                                    chunks=(self.chunk_size or batch_size,)
                                    + data_shape,
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

                            datasets[dset][
                                offsets[dset] : offsets[dset] + batch_size
                            ] = arr

                            offsets[dset] += batch_size

                        yield obj

                    for name, dataset in datasets.items():
                        logger.info(
                            f"Truncating dataset {name} to actual length: {offsets[name]:,d}"
                        )

                        dataset.resize(offsets[name], axis=0)

    def _transform_stream_append(self, stream: Stream):
        import h5py

        with closing_if_closable(stream):
            for file_name, file_group in stream_groupby(stream, by=self.file_name):
                with h5py.File(file_name, self.file_mode) as h5:
                    datasets: Mapping[str, h5py.Dataset] = {}

                    # Offsets per dataset
                    offsets: Mapping[str, int] = {}

                    chunk_size = self.chunk_size or 1024

                    for obj in file_group:
                        data: Mapping[str, Any] = self.prepare_input(obj, "data")  # type: ignore

                        for name, value in data.items():
                            if name not in datasets:
                                if isinstance(value, str):
                                    data_shape = tuple()
                                    dtype = h5py.string_dtype()
                                else:
                                    value = np.array(value)
                                    data_shape = value.shape
                                    dtype = value.dtype

                                initial_shape = (chunk_size,) + data_shape

                                logger.info(
                                    f"Creating dataset {name}: {initial_shape}, dtype = {dtype}"
                                )

                                datasets[name] = h5.create_dataset(
                                    name,
                                    shape=initial_shape,
                                    maxshape=(None,) + data_shape,
                                    chunks=(chunk_size,) + data_shape,
                                    dtype=dtype,
                                    compression=self.compression,
                                )
                                offsets[name] = 0

                            if offsets[name] >= datasets[name].shape[0]:
                                new_length = datasets[name].shape[0] * 2
                                logger.info(
                                    f"Resizing dataset {name} to new length: {new_length:,d}"
                                )

                                datasets[name].resize(new_length, axis=0)

                            datasets[name][offsets[name]] = value

                            offsets[name] += 1

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

        if self.dataset_mode == "create":
            return self._transform_stream_create(stream)

        raise ValueError(f"Unknown dataset mode: {self.dataset_mode}")
