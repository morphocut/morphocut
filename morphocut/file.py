import os

from morphocut.graph import Node, Output


@Output("abs_path")
class Find(Node):
    """
    Read all image files under the specified directory.

    Args:
        image_root (str): Root path where images should be found.
        allowed_extensions (optional): List of allowed extensions (including the leading dot).

    Output:
        abs_path: Absolute path of the image file.
    """

    def __init__(self, image_root: str, allowed_extensions):
        super().__init__()

        self.image_root = image_root
        self.allowed_extensions = set(allowed_extensions)

    def transform_stream(self, stream):
        for obj in stream:
            image_root = self.prepare_input(obj, "image_root")
            for root, _, filenames in os.walk(image_root):
                for fn in filenames:
                    ext = os.path.splitext(fn)[1]

                    # Skip non-allowed extensions
                    if ext not in self.allowed_extensions:
                        continue

                    yield self.prepare_output(
                        {},
                        os.path.join(root, fn),
                    )
