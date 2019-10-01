from morphocut.graph import Node, Output
from morphocut._optional import import_optional_dependency
import pims


@Output("frame")
class VideoReader(Node):
    """Read frames from video files.

    .. note::
        To use this reader, you need to have `PyAV`_ and `PIMS`_ installed.

        .. _PyAV: https://docs.mikeboers.com/pyav/develop/installation.html
        .. _PIMS: http://soft-matter.github.io/pims/stable

    Args:
        path: Path to a video file.
        **kwargs: Additional keyword parameters for pims.PyAVReaderIndexed

    Outputs:
        frame (pims.Frame): The frame.

        - frame_no: Frame number.
        - metadata: Frame metadata.
    """
    def __init__(self, path):
        super().__init__()

        self.path = path

        import_optional_dependency("av")
        self._pims = import_optional_dependency("pims")

    def transform_stream(self, stream):
        for obj in stream:
            path = self.prepare_input(obj, "path")
            reader = self._pims.PyAVReaderIndexed(path, **self.kwargs)

            for frame in reader:
                yield self.prepare_output(obj.copy(), frame)


@Output("frame")
@Output("series")
class BioformatsReader(Node):
    """Read frames from Bioformats files.

    Bio-Formats is a software tool for reading and writing image data using standardized, open formats.

    .. note::
        To use this reader, you need to have `JPype`_ and `PIMS`_ installed.

        .. _JPype: https://github.com/jpype-project/jpype
        .. _PIMS: http://soft-matter.github.io/pims/stable

    Args:
        path: Path to a Bioformats file.
        **kwargs: Additional keyword parameters for pims.BioformatsReader

    Outputs:
        - frame (pims.Frame): The frame.
            - frame_no: Frame number.
            - metadata: Frame metadata.
        - series (int): The series.
    """
    def __init__(self, path, **kwargs):
        super().__init__()

        self.path = path
        self.kwargs = kwargs

        import_optional_dependency("jpype")
        self._pims = import_optional_dependency("pims")

    def transform_stream(self, stream):
        for obj in stream:
            path = self.prepare_input(obj, "path")

            series = self.kwargs.pop("series", None)

            reader = self._pims.bioformats.BioformatsReader(path, **self.kwargs)

            if series is None:
                series = range(reader.size_series)
            else:
                series = [series]

            for s in series:
                reader.series = s
                for frame in reader:
                    yield self.prepare_output(obj.copy(), frame, s)
