from itertools import zip_longest
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.lib.mixins

from morphocut.core import (
    Node,
    RawOrVariable,
    ReturnOutputs,
    Stream,
    Variable,
    closing_if_closable,
)
from morphocut.utils import StreamEstimator, stream_groupby


def _wrap_array_property(name):
    return property(lambda self: getattr(self.array, name))


def _wrap_array_method(name):
    def wrapper(self, *args, **kwargs):
        method = getattr(self.array, name)
        result = method(*args, **kwargs)
        return self._convert_result(result)

    return wrapper


class Region(numpy.lib.mixins.NDArrayOperatorsMixin):
    """
    Drop-in replacement for np.ndarray that knows about its location in a larger array ("frame").

    Attributes:
        array (np.ndarray): The actual data.
        key (tuple of slice): Location in the frame.
    """

    def __init__(self, array: np.ndarray, key: Tuple[slice, ...]) -> None:
        self.array = array
        self.key = key

    def _convert_result(self, arr):
        # If result has the same shape as self.array, we can assume that we're still anchored in the frame
        if (
            isinstance(arr, np.ndarray)
            and arr.shape[: len(self.key)] == self.array.shape[: len(self.key)]
        ):
            return Region(arr, key=self.key)
        return arr

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(x.array if isinstance(x, Region) else x for x in inputs)
        out = kwargs.get("out", ())
        if out:
            kwargs["out"] = tuple(x.array if isinstance(x, Region) else x for x in out)
        result = getattr(ufunc, method)(*inputs, **kwargs)

        if type(result) is tuple:
            # multiple return values
            return tuple(self._convert_result(x) for x in result)
        elif method == "at":
            # no return value
            return None
        else:
            # one return value
            return self._convert_result(result)

    def __array__(self, dtype=None):
        return np.asarray(self.array, dtype=dtype)

    def __setitem__(self, key, value):
        self.array[key] = value

    def __getitem__(self, key: Union[slice, Tuple[slice, ...]]):
        if isinstance(key, slice):
            key = (key,)
        elif isinstance(key, tuple):
            if not all(isinstance(sl, slice) for sl in key):
                return self.array[key]
        elif key is ...:
            key = tuple(slice(0, stop) for stop in self.shape[: len(self.key)])
        else:
            # Boolean, None, ...
            return self.array[key]

        key = key if isinstance(key, tuple) else (key,)

        fillvalue = object()

        new_key: List[slice] = []
        for sl0, sl1 in zip_longest(self.key, key, fillvalue=fillvalue):
            if sl0 is fillvalue:
                new_key.append(slice(sl1.start, sl1.stop, sl1.step))
                continue
            if sl1 is fillvalue:
                new_key.append(sl0)
                continue

            assert isinstance(sl0, slice), f"Not a slice: {sl0!r}"
            assert isinstance(sl1, slice), f"Not a slice: {sl1!r}"

            sl0_step = sl0.step or 1
            sl1_step = sl1.step or 1
            start = sl0.start + sl0_step * sl1.start
            stop = sl0.start + sl0_step * sl1.stop if sl1.stop is not None else sl0.stop
            step = sl0_step * sl1_step

            if sl0.stop is not None and stop > sl0.stop:
                stop = sl0.stop

            new_key.append(slice(start, stop, step))

        return Region(self.array[key], tuple(new_key))

    # Accessors for ndarray interoperability
    shape = _wrap_array_property("shape")
    dtype = _wrap_array_property("dtype")
    ndim = _wrap_array_property("ndim")

    astype = _wrap_array_method("astype")

    sum = _wrap_array_method("sum")
    max = _wrap_array_method("max")
    min = _wrap_array_method("min")

    reshape = _wrap_array_method("reshape")


class Frame:
    """
    Virtual frame containing individual regions.

    Args:
        fill_value (scalar, optional): Fill value.
        shape (None or int or tuple thereof, optional):
            Shape of the Frame. Can contain None for dimensions without a fixed size.
            Infered from assigned data if None.
        dtype (data-type, optional):
            The desired data-type for the frame.
            Infered from assigned data if None.
        empty_none (bool, optional):
            Return None when a frame[key] does not contain any regions.
    """

    def __init__(
        self,
        fill_value=0,
        shape: Optional[Tuple[Optional[int], ...]] = None,
        dtype=None,
        empty_none=False,
    ) -> None:
        self.fill_value = fill_value
        self.keylen = None
        self._shape = shape
        self.dtype = dtype
        self.empty_none = empty_none

        self._regions: List[Region] = []

    @property
    def key_shape(self) -> Tuple[int, ...]:
        assert self._shape is not None

        # Calculate maximum stop of each dimension over all regions
        dims_regs = zip(*(r.key for r in self._regions))
        key_shape = tuple(
            max(s.stop for s in dim) if self._shape[i] is None else self._shape[i]
            for i, dim in enumerate(dims_regs)
        )

        return key_shape  # type: ignore

    @property
    def shape(self) -> Tuple[int]:
        assert self._shape is not None
        assert self.keylen is not None

        if None not in self._shape:
            return self._shape  # type: ignore

        return self.key_shape + self._shape[self.keylen :]  # type: ignore

    @property
    def ndim(self):
        return len(self.shape)

    def _validate_key(self, key, getitem=False) -> Tuple[slice, ...]:
        if not isinstance(key, tuple):
            key = (key,)

        for s in key:
            # Only slices
            if not isinstance(s, slice):
                raise IndexError(
                    f"Invalid key: {key}. Only slice and tuple of slice are allowed."
                )

            # slice.step is None or 1
            if s.step is not None and s.step != 1:
                raise IndexError(f"Invalid key: {key}. step has to be 1 or None.")

        # Infere start/stop=None from shape
        def update_slice(sl: slice, sh: Optional[int]):
            start, stop = sl.start, sl.stop

            if start is None:
                start = 0

            if stop is None:
                if sh is None:
                    raise IndexError(
                        f"Invalid key: {key}. Can not infer stop from shape {self._shape}"
                    )

                stop = sh

            if sh is not None:
                if getitem:
                    stop = min(stop, sh)
                else:
                    if stop > sh:
                        raise IndexError(f"Invalid key: {key}. Out of range")

            return slice(start, stop)

        shape = (
            self.shape
            if getitem
            else self._shape if self._shape is not None else (None,) * len(key)
        )
        key = tuple(update_slice(sl, sh) for sl, sh in zip(key, shape))

        return key

    def __setitem__(self, key, value: np.ndarray):
        """Add a region by slices."""
        # Convert and validate key
        key = self._validate_key(key)

        # Set or check self.keylen
        keylen = len(key)
        if self.keylen is None:
            self.keylen = keylen
        else:
            if self.keylen != keylen:
                raise ValueError(
                    f"Mixed key length is not allowed ({self.keylen} / {keylen})"
                )

        # Set or check self._shape
        data_shape = value.shape[keylen:]
        if self._shape is None:
            self._shape = (None,) * keylen + data_shape
        else:
            # Make sure that data shape is consistent across all regions
            if self._shape[keylen:] != data_shape:
                raise ValueError(
                    f"Mixed shape is not allowed ({self._shape} / {value.shape})"
                )

            # Check out-of-bounds of key
            if not all(
                s.stop <= self._shape[i]
                for i, s in enumerate(key)
                if self._shape[i] is not None
            ):
                raise ValueError(f"key {key} is out of bounds for shape {self._shape}")

        # Set or check self.dtype
        if self.dtype is None:
            self.dtype = value.dtype
        else:
            if value.dtype != self.dtype:
                raise ValueError(
                    f"Mixed dtype is not allowed ({self.dtype} / {value.dtype})"
                )

        self._regions.append(Region(value, key))

    def _get_intersecting_regions(self, key: Tuple[slice]) -> List[Region]:
        """Find regions that intersect with the given key."""
        result = []
        for r in self._regions:
            intersection_key = tuple(
                slice(max(ks.start, rs.start), min(ks.stop, rs.stop))
                for ks, rs in zip(key, r.key)
            )
            if all(s.start <= s.stop for s in intersection_key):
                source_key = tuple(
                    slice(ss.start - rs.start, ss.stop - rs.start)
                    for rs, ss in zip(r.key, intersection_key)
                )
                target_key = tuple(
                    slice(ss.start - ks.start, ss.stop - ks.start)
                    for ks, ss in zip(key, intersection_key)
                )
                result.append((r, source_key, target_key))
        return result

    def __getitem__(self, key) -> Optional[Region]:
        """Extract part of the frame as a new region."""

        # [...]
        if key is ...:
            # Calculate maximum length of each dimension
            key = tuple(slice(0, stop) for stop in self.key_shape)

        # [a:b,c:d]
        else:
            # Convert and validate key
            key = self._validate_key(key, getitem=True)

        regions = self._get_intersecting_regions(key)

        if not regions and self.empty_none:
            # Return None instead of constructing an empty array.
            return None

        shape = tuple(s.stop - s.start for s in key) + self.shape[len(key) :]
        array = np.full(shape, self.fill_value, self.dtype)

        # Stitch regions
        for region, source_key, target_key in regions:
            # TODO: Apply better blending (maximum, mininum, weighted, ...)
            array[target_key] = region.array[source_key]

        return Region(array, key)

    def iter_regions(self) -> Iterator[Region]:
        yield from self._regions

    @property
    def n_regions(self):
        return len(self._regions)

    def __array__(self, dtype=None) -> np.ndarray:
        """Convert whole frame to array by stitching regions."""

        whole_frame = self[...]

        if whole_frame is None:
            return np.full(self.shape, self.fill_value, dtype)

        return np.asarray(whole_frame.array, dtype=dtype)

    def _merge_regions(self, max_distance) -> "Frame":
        # Merge regions separated by <= max_distance
        # NB: Merged regions might overlap afterwards
        def _validate_bounds(
            self, arrays: Tuple[Region | np.ndarray, ...]
        ) -> Tuple[slice, ...]:
            # Validate shapes
            for ax in self.axes:
                if not all(arr.shape[ax] == arrays[0].shape[ax] for arr in arrays[1:]):
                    raise ValueError(
                        f"Arrays do not match in dimension {ax}: {[arr.shape[ax] for arr in arrays]}"
                    )

            anchored_arrays = [arr for arr in arrays if isinstance(arr, Region)]

            # Validate bounding key
            for ax in self.axes:
                if not all(
                    arr.key[ax] == arrays[0].key[ax] for arr in anchored_arrays[1:]
                ):
                    raise ValueError(
                        f"Arrays do not match in bounding range {ax}: {[arr.shape[ax] for arr in anchored_arrays]}"
                    )

            return tuple(anchored_arrays[0].key[ax] for ax in self.axes)

        def _range_dist(self, r0: slice, r1: slice) -> int:
            start = max(r0.start, r1.start)
            stop = min(r0.stop, r1.stop)

            return max(0, start - stop)

        def _label_bounds(self, bounds) -> Tuple[int, np.ndarray]:
            if self.max_dist is None:
                # All bounds are part of the same cluster
                return 1, np.zeros(len(bounds))

            # Extracted connected components
            adjacency = scipy.sparse.dok_array((len(bounds),) * 2, dtype=bool)
            for i in range(len(bounds)):
                for j in range(i + 1, len(bounds)):
                    if (
                        np.linalg.norm(
                            [
                                self._range_dist(bi, bj)
                                for bi, bj in zip(bounds[i], bounds[j])
                            ]
                        )
                        <= self.max_dist
                    ):
                        adjacency[i, j] = 1

            (
                n_components,
                labels,
            ) = scipy.sparse.csgraph.connected_components(adjacency, directed=False)

            return n_components, labels

        # Validate and calculate bounds
        bounds = [_validate_bounds(arrays) for arrays in group_arrays]

        # Label connected components
        n_components, labels = _label_bounds(bounds)

        frame = Frame(self.fill_value, self.shape, self.dtype, self.empty_none)

        # TODO: for each label extract bounding box and paste into new frame

        return frame


def _validate_multiparameter(value, n_inputs, name, nested_sequence=False):
    if isinstance(value, Sequence) and (
        not nested_sequence or isinstance(value[0], Sequence)
    ):
        if len(value) != n_inputs:
            raise ValueError(
                f"{name} has a unexpected length: {len(value)} vs. {n_inputs}"
            )
        return value

    return (value,) * n_inputs


@ReturnOutputs
class Stitch(Node):
    """
    Stitch regions onto frames.
    """

    def __init__(
        self,
        *inputs: RawOrVariable[np.ndarray],
        groupby,
        offset,
        fill_value=0,
        shape: Optional[Tuple[Optional[int], ...]] = None,
        dtype=None,
        empty_none=False,
    ) -> None:
        super().__init__()

        self.inputs = inputs
        self.groupby = groupby
        self.offset = offset

        n_inputs = len(inputs)

        self.fill_value = _validate_multiparameter(fill_value, n_inputs, "fill_value")
        self.shape = _validate_multiparameter(
            shape, n_inputs, "shape", nested_sequence=True
        )
        self.dtype = _validate_multiparameter(dtype, n_inputs, "dtype")
        self.empty_none = _validate_multiparameter(empty_none, n_inputs, "empty_none")

        # Variable number of outputs
        self.outputs = [
            Variable(arg.name if isinstance(arg, Variable) else f"stichted{i}", self)
            for i, arg in enumerate(inputs)
        ]

    def transform_stream(self, stream: Stream) -> Stream:
        with closing_if_closable(stream):
            stream_estimator = StreamEstimator()

            for _, group in stream_groupby(stream, self.groupby):
                # self.fill_value, self.shape, self.dtype, self.empty_none can be single or sequence (individual for each input)
                frames = [
                    Frame(fv, s, dt, en)
                    for fv, s, dt, en in zip(
                        self.fill_value, self.shape, self.dtype, self.empty_none
                    )
                ]

                group = list(group)

                with stream_estimator.consume(
                    group[0].n_remaining_hint, est_n_emit=1, n_consumed=len(group)
                ) as incoming_group:

                    for obj in group:
                        inputs, offset = self.prepare_input(obj, ("inputs", "offset"))

                        if isinstance(offset, int):
                            offset = (offset,)

                        key_len = len(offset)

                        # Validate common shape of inputs
                        if not all(
                            inputs[0].shape[:key_len] == inp.shape[:key_len]
                            for inp in inputs[1:]
                        ):
                            raise ValueError(
                                f"Input shapes do not match: {[inp.shape for inp in inputs]}"
                            )

                        key = tuple(
                            slice(o, o + l) for o, l in zip(offset, inputs[0].shape)
                        )

                        for frame, inp in zip(frames, inputs):
                            frame[key] = inp

                    yield self.prepare_output(
                        group[0].copy(),
                        *frames,
                        n_remaining_hint=incoming_group.emit(),
                    )
