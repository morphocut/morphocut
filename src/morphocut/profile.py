from typing import Optional

from morphocut.core import Pipeline, Stream, StreamObject, Variable
from time import perf_counter

SI_PREFIXES = ["", "m", "μ", "n"]


class Profile(Pipeline):
    """
    Profile pipeline nodes.

    The average lead time of objects in this pipeline context is measured
    and printed after the end of processing.

    Args:
        name (str): The name of this profiling operation.

    Example:
        .. code-block:: python

            with Pipeline() as pipeline:
                # Un-profiled processing
                ...
                with Profile("some name"):
                    # Profiled processing
                    ...

            pipeline.run()
    """

    def __init__(self, name):
        super().__init__()

        self.name = name
        self._start = Variable("_sentinel", self)
        self._average = 0
        self._n = 0

    def _insert_sentinel(self, stream: Stream) -> Stream:
        for obj in stream:
            obj[self._start] = perf_counter()
            yield obj

    def _format_average(self):
        value = self._average

        for prefix in SI_PREFIXES:
            if value > 1:
                break
            value *= 1000

        return f"{value:.2f}{prefix}s"

    def transform_stream(self, stream: Optional[Stream] = None) -> Stream:
        if stream is None:
            stream = [StreamObject()]

        stream = self._insert_sentinel(stream)

        # Here, the stream is not automatically closed,
        # as this would happen instantaneously.
        for child in self.children:  # type: StreamTransformer
            stream = child.transform_stream(stream)

        for obj in stream:
            diff = perf_counter() - obj.pop(self._start)

            self._n += 1
            self._average = self._average + (diff - self._average) / self._n

            yield obj

        print(
            f"Profile {self.name}: {self._n:,d} objects, avg. {self._format_average()}"
        )
