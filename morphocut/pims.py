from morphocut.graph import Node, Output


@Output("image")
@Output("frame_number")
class VideoReader(Node):
    def __init__(self, path):
        super().__init__()

        self.path = path

        # TODO: Check pims availability like in:
        # from pandas.compat._optional import import_optional_dependency
        # from pandas.io.excel._xlrd import _XlrdReader

    def transform_stream(self, stream):
        for obj in stream:
            # TODO
            ...
            yield obj
