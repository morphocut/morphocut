import itertools
from typing import Optional, Sequence, Union

from morphocut.core import Pipeline, Stream, StreamObject, Variable


class Batch(tuple):
    """Special sequence type that is recognized by BatchPipeline."""

    pass


class BatchPipeline(Pipeline):
    """Combine consecutive objects into a batch."""

    def __init__(
        self,
        batch_size,
        *,
        parent: Optional["Pipeline"] = None,
        groupby: Union[None, Variable, Sequence[Variable]] = None
    ):
        super().__init__(parent=parent)

        if isinstance(groupby, Sequence):
            groupby = tuple(groupby)
        elif isinstance(groupby, Variable):
            groupby = (groupby,)

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

        stream = self.check_stream(stream)

        stream = self._pack(stream)

        for child in self.children:  # type: StreamTransformer
            stream = child.transform_stream(stream)

        stream = self._unpack(stream)

        return stream

    def _keyfunc(self, obj):
        if self.groupby is None:
            return None

        return tuple(obj[k] for k in self.groupby)

    def _pack(self, stream: Stream) -> Stream:
        for key, group in itertools.groupby(stream, self._keyfunc):  # type: ignore
            key: tuple

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
                    for k, v in zip(self.groupby, key):
                        obj[k] = v

                yield obj

    def _unpack(self, stream: Stream) -> Stream:
        for batch in stream:
            n_remaining_hint_orig = batch.pop(self._n_remaining_hint_field)

            for i, n_remaining_hint in enumerate(n_remaining_hint_orig):
                obj = {
                    k: batch[k][i] if isinstance(batch[k], Batch) else batch[k]
                    for k in batch
                }
                yield StreamObject(obj, n_remaining_hint=n_remaining_hint)
