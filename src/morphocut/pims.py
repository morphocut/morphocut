"""
Through PIMS, MorphoCut supports reading Bioformats and Video.

    "`PIMS`_ is a lazy-loading interface to
    sequential data with numpy-like slicing."

.. note::
    `PIMS`_ is required to use the nodes defined in this module.

.. _PIMS: http://soft-matter.github.io/pims/stable
"""
from typing import Optional

from morphocut import Node, Output, RawOrVariable, ReturnOutputs, closing_if_closable
from morphocut._optional import import_optional_dependency


@ReturnOutputs
@Output("frame")
class VideoReader(Node):
    """
    |stream| Read frames from video files.

    .. note::
        `PyAV`_ is required to use this reader.

        .. _PyAV: https://docs.mikeboers.com/pyav/develop/installation.html


    Args:
        path: Path to a video file.
        **kwargs: Additional keyword parameters for :py:class:`pims.PyAVReaderIndexed`.

    Example:
        .. code-block:: python

            frame = VideoReader(path)

            # frame (pims.Frame): The frame.
            #   frame.frame_no (int): Frame number.
            #   frame.metadata (dict): Frame metadata.
    """

    def __init__(self, path: RawOrVariable[str], **kwargs):
        super().__init__()

        self.path = path
        self.kwargs = kwargs

        import_optional_dependency("av")
        self._pims = import_optional_dependency("pims")

    def transform_stream(self, stream):
        with closing_if_closable(stream):
            for obj in stream:
                path = self.prepare_input(obj, "path")
                reader = self._pims.PyAVReaderIndexed(path, **self.kwargs)

                for frame in reader:
                    yield self.prepare_output(obj.copy(), frame)


@ReturnOutputs
@Output("frame")
@Output("series")
class BioformatsReader(Node):
    """
    |stream| Read frames from Bio-Formats files.

    Bio-Formats is a software tool for reading and writing image data using standardized, open formats.
    It is able to read `over 150 file formats`_, including OME-TIFF and Amnis FlowSight (.cif).

    .. _over 150 file formats: https://docs.openmicroscopy.org/bio-formats/latest/supported-formats.html

    .. note::
        `JPype`_ is required to use this reader.

        On first use of `BioformatsReader`, the required java library `loci_tools.jar`
        will be automatically downloaded from openmicroscopy.org.

        .. _JPype: https://github.com/jpype-project/jpype

    Args:
        path (str): Path to a Bioformats file.
        meta (bool, optional): When true, the metadata object is generated. Takes time to build.
        series (int, optional):  Active image series index. Defaults to None, meaning that all series are read.
        **kwargs: Additional keyword parameters for pims.BioformatsReader

    Example:
        .. code-block:: python

            frame, series = BioformatsReader(path)

            # frame (pims.Frame): The frame.
            #   frame.frame_no (int): Frame number.
            #   frame.metadata (dict): Frame metadata.
            # series (int): The series extracted from the file.
    """

    def __init__(
        self,
        path: RawOrVariable[str],
        meta: RawOrVariable[bool],
        series: Optional[RawOrVariable[int]] = None,
        **kwargs,
    ):
        super().__init__()

        self.path = path
        self.meta = meta
        self.series = series
        self.kwargs = kwargs

        import_optional_dependency("jpype")
        self._pims = import_optional_dependency("pims")

    def transform_stream(self, stream):
        with closing_if_closable(stream):
            for obj in stream:
                path, meta, series, kwargs = self.prepare_input(
                    obj, ("path", "meta", "series", "kwargs")
                )

                reader = self._pims.bioformats.BioformatsReader(
                    path, meta=meta, **kwargs
                )

                if series is None:
                    series = range(reader.size_series)
                else:
                    series = [series]

                for s in series:
                    reader.series = s
                    for frame in reader:
                        yield self.prepare_output(obj.copy(), frame, s)
