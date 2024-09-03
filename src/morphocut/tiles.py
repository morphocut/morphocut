import itertools
from typing import Iterator, Literal, Mapping, Optional, Tuple
from morphocut import Pipeline
from morphocut.core import Stream, StreamTransformer, Variable, check_stream
from .exception_utils import exc_add_note
import numpy as np
from morphocut.utils import stream_groupby
import morphocut.stitch


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


BlendStrategy = Literal["overwrite", "linear"]


class TiledPipeline(Pipeline):
    def __init__(
        self,
        tile_shape: Tuple,
        *variables: Variable[np.ndarray],
        tile_stride: Optional[Tuple] = None,
        pad=True,
        pad_kwargs: Optional[Mapping] = None,
        blend_strategy: BlendStrategy = "overwrite",
    ):
        super().__init__()

        if tile_stride is None:
            tile_stride = tile_shape

        self.tile_shape = tile_shape
        self.tile_stride = tile_stride
        self.variables = variables
        self.pad = pad
        self.pad_kwargs = pad_kwargs or {}
        self.blend_strategy = morphocut.stitch.Frame.validate_blend_strategy(
            blend_strategy
        )

        self._placeholders = [Variable(f"{v.name}_old", self) for v in variables]
        self._key_padding = Variable("_slice_padding", self)
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

    def _gen_key_padding_1d(
        self, arr_shape, tile_shape, tile_stride
    ) -> Tuple[Tuple[slice, Tuple[int, int]], ...]:
        pad = ((tile_shape - arr_shape) % tile_stride) // 2 if self.pad else 0
        return tuple(
            (
                slice(max(0, off), min(off + tile_shape, arr_shape)),
                (max(0, -off), max(0, off + tile_shape - arr_shape)),
            )
            for off in complete_range(-pad // 2, arr_shape - 1, tile_stride)
        )

    def _gen_key_padding(
        self, arr: np.ndarray
    ) -> Iterator[Tuple[Tuple[slice, ...], Tuple[int, int]]]:
        # Calculate slices for each dimension
        key_padding = [
            self._gen_key_padding_1d(arr_shape_, tile_shape_, tile_stride_)
            for arr_shape_, tile_shape_, tile_stride_ in zip(
                arr.shape, self.tile_shape, self.tile_stride
            )
        ]

        return (
            zip(*slice_padding) for slice_padding in itertools.product(*key_padding)
        )  # type: ignore

    def _tile_all(self, stream: Stream) -> Stream:
        for tiling_id, obj in enumerate(stream):
            for key, padding in self._gen_key_padding(obj[self.variables[0]]):
                obj_new = obj.copy()
                obj_new[self._tiling_id] = tiling_id
                for v, p in zip(self.variables, self._placeholders):
                    # Store original
                    obj_new[p] = obj[v]

                    n_extra_dim = obj[v].ndim - len(padding)
                    padding = padding + ((0, 0),) * n_extra_dim

                    # Store sliced and padded version
                    try:
                        obj_new[v] = np.pad(obj[v][key], padding, **self.pad_kwargs)
                    except Exception as exc:
                        exc_add_note(
                            exc, f"{obj[v][key].shape}, {padding}, {self.pad_kwargs}"
                        )
                        raise exc

                obj_new[self._key_padding] = key, padding
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

            # Stitch new variables using stitch.Frame
            for v in locals_:
                if not isinstance(obj_new[v], (np.ndarray, morphocut.stitch.Region)):
                    continue

                print(v)

                frame = morphocut.stitch.Frame(blend_strategy=self.blend_strategy)  # type: ignore
                for obj in group:
                    tile = obj[v]

                    if tile is None:
                        # This variable is not an array
                        break

                    # Slice and unpad
                    key, padding = obj[self._key_padding]

                    shape_before = tile.shape
                    tile = unpad(tile, padding)
                    try:
                        frame[key] = tile
                    except Exception as exc:
                        exc_add_note(
                            exc,
                            f"shape_before={shape_before}, shape_after={tile.shape}, padding={padding}, key={key}",
                        )
                        raise exc

                obj_new[v] = frame.asarray()

            yield obj_new
