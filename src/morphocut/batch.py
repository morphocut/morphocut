import itertools
from typing import Optional, Sequence, Union, Tuple

from morphocut.core import (
    Pipeline,
    Stream,
    StreamObject,
    StreamTransformer,
    Variable,
    check_stream,
)
from morphocut.utils import stream_groupby


class Batch(tuple):
    """Special sequence type that is recognized by BatchPipeline."""

    pass


class BatchedPipeline(Pipeline):
    """
    Combine consecutive objects into a batch.

    Example:
        .. code-block:: python

            with Pipeline() as p:
                # a is a scalar
                a = ...
                with BatchedPipeline():
                    # a is a sequence
                    ...

                # a is a scalar again


    """

    def __init__(
        self,
        batch_size,
        *,
        parent: Optional["Pipeline"] = None,
        groupby: Union[None, Variable, Sequence[Variable]] = None,
    ):
        super().__init__(parent=parent)

        if isinstance(groupby, Sequence):
            groupby = tuple(groupby)
        elif isinstance(groupby, Variable):
            groupby = (groupby,)

        # Ensure that all groupby fields are variables
        if groupby is not None and False in (isinstance(k, Variable) for k in groupby):
            raise ValueError("All groupby fields need to be Variables.")

        self.batch_size = batch_size
        self.groupby: Optional[Tuple[Variable]] = groupby  # type: ignore

        self._n_remaining_hint_field = id(object())

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

        stream = self._pack(stream)

        for child in self.children:
            child: StreamTransformer

            stream = child.transform_stream(stream)
            assert stream is not None, f"{child!r}.transform_stream returned None"

        stream = self._unpack(stream)

        return stream

    def _pack(self, stream: Stream) -> Stream:
        for group_key, group in stream_groupby(stream, self.groupby):

            while True:
                batch = tuple(itertools.islice(group, self.batch_size))

                if not batch:
                    break

                # Find first valid n_remaining_hint in batch
                n_remaining_hint_orig = [obj.n_remaining_hint for obj in batch]

                n_remaining_hint_batch = next(
                    (x for x in n_remaining_hint_orig if x is not None),
                    None,
                )

                if n_remaining_hint_batch is not None:
                    n_remaining_hint_batch = max(
                        1, round(n_remaining_hint_batch / self.batch_size)
                    )

                # Transpose
                elem = batch[0]
                obj = StreamObject(
                    {key: Batch([d[key] for d in batch]) for key in elem},
                    n_remaining_hint=n_remaining_hint_batch,
                )

                obj[self._n_remaining_hint_field] = n_remaining_hint_orig

                # Reset groupby fields to scalar value
                if self.groupby is not None:
                    for k, v in zip(self.groupby, group_key):
                        obj[k] = v

                yield obj

    def _unpack(self, stream: Stream) -> Stream:
        locals_hashes = set(v.hash for v in self.locals())

        for batch in stream:
            n_remaining_hint_orig = batch.pop(self._n_remaining_hint_field)

            for i, n_remaining_hint in enumerate(n_remaining_hint_orig):
                obj = {
                    k: batch[k][i]
                    if batch[k] is not None
                    and (isinstance(batch[k], Batch) or (k in locals_hashes))
                    else batch[k]
                    for k in batch
                }
                yield StreamObject(obj, n_remaining_hint=n_remaining_hint)
