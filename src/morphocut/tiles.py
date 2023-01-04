import itertools
from typing import Iterable, List, Mapping, Optional, Tuple
from morphocut import Pipeline
from morphocut.core import Node, Stream, StreamTransformer, Variable, check_stream
import numpy as np
from morphocut.utils import stream_groupby


def complete_range(start, stop, step):
    i = start
    while True:
        yield i
        i += step
        if i > stop:
            break


def unpad(array, pad_width):
    slices = tuple(
        slice(before, len - after)
        for len, (before, after) in zip(array.shape, pad_width)
    )
    return array[slices]


class TiledPipeline(Pipeline):
    def __init__(
        self,
        tile_shape: Tuple,
        *variables: Variable[np.ndarray],
        tile_stride: Optional[Tuple] = None,
        pad=True,
        pad_kwargs: Optional[Mapping] = None,
    ):
        super().__init__()

        if tile_stride is None:
            tile_stride = tile_shape

        self.tile_shape = tile_shape
        self.tile_stride = tile_stride
        self.variables = variables
        self.pad = pad

        self.pad_kwargs = pad_kwargs or {}

        self._placeholders = [Variable(f"{v.name}_old", self) for v in variables]
        self._slice_padding = Variable("_slice_padding", self)
        self._tiling_id = Variable("_tiling_id", self)

    def transform_stream(self, stream: Optional[Stream] = None) -> Stream:
        """
        Run the stream through all nodes and return it.

        Args:
            stream: A stream to transform.
                *This argument is solely to be used internally.*

        Returns:
            Stream: An iterable of stream objects.
        """

        stream = check_stream(stream)

        stream = self._tile_all(stream)

        for child in self.children:
            child: StreamTransformer

            stream = child.transform_stream(stream)
            assert stream is not None, f"{child!r}.transform_stream returned None"

        stream = self._untile_all(stream)

        return stream

    def _gen_slices_padding_1d(self, arr_shape, tile_shape, tile_stride):
        pad = ((tile_shape - arr_shape) % tile_stride) // 2 if self.pad else 0
        return tuple(
            (
                slice(max(0, off), off + tile_shape),
                (max(0, -off), max(0, off + tile_shape - arr_shape)),
            )
            for off in complete_range(-pad // 2, arr_shape - 1, tile_stride)
        )

    def _gen_slices(self, arr: np.ndarray):
        # Calculate slices for each dimension
        slices = [
            self._gen_slices_padding_1d(arr_shape_, tile_shape_, tile_stride_)
            for arr_shape_, tile_shape_, tile_stride_ in zip(
                arr.shape, self.tile_shape, self.tile_stride
            )
        ]

        return (zip(*slice_padding) for slice_padding in itertools.product(*slices))

    def _tile_all(self, stream: Stream) -> Stream:
        for tiling_id, obj in enumerate(stream):
            for slice, padding in self._gen_slices(obj[self.variables[0]]):
                obj_new = obj.copy()
                obj_new[self._tiling_id] = tiling_id
                for v, p in zip(self.variables, self._placeholders):
                    # Store original
                    obj_new[p] = obj[v]

                    # Store sliced and padded version
                    obj_new[v] = np.pad(obj[v][slice], padding, **self.pad_kwargs)
                obj_new[self._slice_padding] = slice, padding
                yield obj_new

    def _untile_all(self, stream: Stream) -> Stream:
        locals_ = self.locals()

        for tiling_id, group in stream_groupby(stream, self._tiling_id):
            group = list(group)

            obj_new = group[0].copy()

            # Restore previous, untiled arrays
            for v, p in zip(self.variables, self._placeholders):
                obj_new[v] = obj_new[p]
                del obj_new[p]

            base_shape = obj_new[self.variables[0]].shape[: len(self.tile_shape)]

            # Stitch new variables
            for v in locals_:
                arr = None
                for obj in group:
                    tile = obj[v]

                    if tile is None:
                        # This variable is not an array
                        break

                    if arr is None:
                        arr = np.zeros(
                            base_shape + tile.shape[len(self.tile_shape) :],
                            dtype=tile.dtype,
                        )
                    # Slice and unpad
                    slice, padding = obj[self._slice_padding]
                    arr[slice] = unpad(tile, padding)
                obj_new[v] = arr
            yield obj_new
