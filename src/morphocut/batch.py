from typing import Optional, Sequence, Union
from morphocut.core import Pipeline, Stream, StreamObject, Variable
import itertools

class Batch(tuple):
    """Special sequence type that is recognized by BatchPipeline."""
    pass

class BatchPipeline(Pipeline):
    """Combine consecutive objects into a batch."""
    
    def __init__(
        self, batch_size, *, parent: Optional["Pipeline"] = None, groupby:Union[None,Variable,Sequence[Variable]]=None
    ):
        super().__init__(parent=parent)

        if isinstance(groupby, Sequence):
            groupby = tuple(groupby)
        elif isinstance(groupby, Variable):
            groupby = (groupby,)

        self.batch_size = batch_size
        self.groupby: Optional[Tuple[Variable]] = groupby # type: ignore

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
        for key, group in itertools.groupby(stream, self._keyfunc):
            while True:
                batch = tuple(itertools.islice(group, self.batch_size))
                if not batch:
                    break

                stream_length = None
                for obj in batch:
                    stream_length = obj.stream_length or stream_length

                if stream_length is not None:
                    stream_length //= self.batch_size

                # Transpose
                elem = batch[0]
                obj = StreamObject({key: Batch([d[key] for d in batch]) for key in elem}, stream_length=stream_length)

                # Reset groupby fields to scalar value
                if self.groupby is not None:
                    for k, v in zip(self.groupby, key):
                        obj[k] = v

                yield obj



    def _unpack(self, stream: Stream) -> Stream:
        for batch in stream:
            # Get the batch size from the first Batch
            batch_size = len(next(b for b in batch.values() if isinstance(b, Batch)))

            stream_length = batch.stream_length * batch_size if batch.stream_length is not None else None

            for i in range(batch_size):
                obj = {k: batch[k][i] if isinstance(batch[k], Batch) else batch[k] for k in batch}
                yield StreamObject(obj, stream_length=stream_length)