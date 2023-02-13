import numpy.lib.mixins


class AnchoredArray(numpy.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, array: np.ndarray, ranges: Tuple[range, ...]) -> None:
        self.array = array
        self.ranges = ranges

    @classmethod
    def from_offsets(cls, array: np.ndarray, offsets: Tuple[int, ...]):
        ranges = tuple(
            range(offs, offs + size) for offs, size in zip(offsets, array.shape)
        )
        return cls(array, ranges)

    @classmethod
    def from_anchor(cls, anchor: np.ndarray, *ranges: range):
        return cls(
            anchor[tuple(slice(r.start, r.stop, r.step) for r in ranges)], ranges
        )

    def __convert_result(self, arr: np.ndarray) -> np.ndarray | "AnchoredArray":
        # If result has the same shape as self.array, we can assume that we still have an anchored array
        if arr.shape[: len(self.ranges)] == self.array.shape[: len(self.ranges)]:
            return AnchoredArray(arr, ranges=self.ranges)
        return arr

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(x.array if isinstance(x, AnchoredArray) else x for x in inputs)
        out = kwargs.get("out", ())
        if out:
            kwargs["out"] = tuple(
                x.array if isinstance(x, AnchoredArray) else x for x in out
            )
        result = getattr(ufunc, method)(*inputs, **kwargs)

        if type(result) is tuple:
            # multiple return values
            return tuple(self.__convert_result(x) for x in result)
        elif method == "at":
            # no return value
            return None
        else:
            # one return value
            return self.__convert_result(result)

    def __array__(self):
        return self.array

    def __getitem__(self, slices: slice | Tuple[slice, ...]):
        slices = slices if isinstance(slices, tuple) else (slices,)

        fillvalue = object()

        new_ranges: List[range] = []
        for sl0, sl1 in zip_longest(self.ranges, slices, fillvalue=fillvalue):
            if sl0 is fillvalue:
                new_ranges.append(range(sl1.start, sl1.stop, sl1.step))
                continue
            if sl1 is fillvalue:
                new_ranges.append(sl0)
                continue

            assert isinstance(sl0, slice)
            assert isinstance(sl1, slice)

            start = sl0.start + sl0.step * sl1.start
            stop = sl0.start + sl0.step * sl1.stop if sl1.stop is not None else sl0.stop
            step = sl0.step * sl1.step

            if sl0.stop is not None and stop > sl0.stop:
                stop = sl0.stop

            new_ranges.append(range(start, stop, step))

            sl0.indices

        return AnchoredArray(self.array[slices], tuple(new_ranges))

    @property
    def shape(self):
        return self.array.shape


@ReturnOutputs
class ClusterStitch(Node):
    def __init__(
        self, *arrays: RawOrVariable[AnchoredArray], groupby, max_dist=None, axes=(0, 1)
    ) -> None:
        super().__init__()

        self.arrays = arrays
        self.groupby = groupby
        self.max_dist = max_dist
        self.axes = axes

        # Variable number of outputs
        self.outputs = [
            Variable(arg.name if isinstance(arg, Variable) else f"stichted{i}", self)
            for i, arg in enumerate(arrays)
        ]

    def _validate_bounds(
        self, arrays: Tuple[AnchoredArray | np.ndarray, ...]
    ) -> Tuple[range, ...]:
        # Validate shapes
        for ax in self.axes:
            if not all(arr.shape[ax] == arrays[0].shape[ax] for arr in arrays[1:]):
                raise ValueError(
                    f"Arrays do not match in dimension {ax}: {[arr.shape[ax] for arr in arrays]}"
                )

        anchored_arrays = [arr for arr in arrays if isinstance(arr, AnchoredArray)]

        # Validate bounding ranges
        for ax in self.axes:
            if not all(
                arr.ranges[ax] == arrays[0].ranges[ax] for arr in anchored_arrays[1:]
            ):
                raise ValueError(
                    f"Arrays do not match in bounding range {ax}: {[arr.shape[ax] for arr in anchored_arrays]}"
                )

        return tuple(anchored_arrays[0].ranges[ax] for ax in self.axes)

    def _range_dist(self, r0: range, r1: range) -> int:
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

    def transform_stream(self, stream: Stream) -> Stream:
        with closing_if_closable(stream):
            stream_estimator = StreamEstimator()

            for _, group in stream_groupby(stream, self.groupby):
                group = list(group)

                with stream_estimator.consume(
                    group[0].n_remaining_hint, est_n_emit=1, n_consumed=len(group)
                ) as incoming_group:

                    # Extract all arrays of the group
                    group_arrays: Tuple[Tuple[AnchoredArray]] = tuple(
                        self.prepare_input(obj, "arrays") for obj in group
                    )
                    group_arrays = tuple(zip(*group_arrays))

                    # Validate and calculate bounds
                    bounds = [self._validate_bounds(arrays) for arrays in group_arrays]

                    # Label connected components
                    n_components, labels = self._label_bounds(bounds)

                    for l in range(n_components):
                        indices = (labels == l).nonzero()[0]
                        label_arrays = [group_arrays[i] for i in indices]
                        label_bounds = [bounds[i] for i in indices]

                        target_shape = [len(b) for b in label_bounds]

                        # Create accumulator arrays
                        # Use float to avoid overflows
                        stitched = [
                            np.zeros(target_shape + inp.shape[2:], dtype=float)
                            for inp in first_input
                        ]

                        for row in data.itertuples():
                            for i, inp in enumerate(row.input):
                                sl = (
                                    slice(row.y, row.y + row.h),
                                    slice(row.x, row.x + row.w),
                                )
                                np.maximum(stitched[i][sl], inp, out=stitched[i][sl])

                        stitched = [
                            st.astype(inp.dtype)
                            for st, inp in zip(stitched, first_input)
                        ]

                        yield self.prepare_output(
                            group[0].copy(),
                            stitched,
                            n_remaining_hint=incoming_group.emit(),
                        )