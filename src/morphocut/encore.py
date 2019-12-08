from typing import Optional

from morphocut.core import Pipeline, Stream, StreamObject, StreamTransformer


class Encore(Pipeline):
    """Repeat the steps in a pipeline."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.encore = False

    def transform_stream(self, stream: Optional[Stream] = None) -> Stream:
        """
        Run the stream through all nodes and return it.

        Args:
            stream: A stream to transform.
                *This argument is solely to be used internally.*

        Returns:
            Stream: An iterable of stream objects.
        """
        if stream is None:
            stream = [StreamObject()]

        # Here, the stream is not automatically closed,
        # as this would happen instantaneously.

        orig_stream = stream

        for child in self.children:  # type: StreamTransformer
            stream = child.transform_stream(stream)

        # Execute
        for _ in stream:
            pass

        # Run a second time, now with encore flag
        self.encore = True
        stream = orig_stream
        for child in self.children:  # type: StreamTransformer
            stream = child.transform_stream(stream)

        return stream
