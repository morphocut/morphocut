from skimage.exposure import rescale_intensity
from pprint import pprint
from morphocut.graph.scheduler import SimpleScheduler
import os

import numpy as np
from skimage import io

from morphocut.graph import Node, Output, Input

import_path = "/data-ssd/mschroeder/Datasets/generic_zooscan_peru_kosmos_2017"
archive_fn = "/tmp/kosmos.zip"


@Output("image")
@Output("meta")
class DirectoryReader(Node):
    """
    Read all image files under the specified directory.
    """

    def __init__(self, image_root, allowed_extensions=None):
        super().__init__()

        self.image_root = image_root

        if allowed_extensions is None:
            self.allowed_extensions = {".tif"}
        else:
            self.allowed_extensions = set(allowed_extensions)

    def transform_stream(self, stream):
        if stream:
            raise ValueError(
                "DirectoryReader is a source node and does not support ingoing streams!")

        for root, _, filenames in os.walk(self.image_root):
            rel_root = os.path.relpath(root, self.image_root)
            for fn in filenames:
                _, ext = os.path.splitext(fn)

                # Skip non-allowed extensions
                if ext not in self.allowed_extensions:
                    continue

                image_path = os.path.join(root, fn)
                img = io.imread(image_path)

                if img.ndim < 3:
                    img = img[:, :, np.newaxis]

                meta = {
                    "rel_path": os.path.join(rel_root, fn)
                }

                yield self.prepare_output({}, img, meta)


@Input("image")
@Output("rescaled")
class Rescale(Node):
    def __init__(self, dtype=None):
        super().__init__()
        self.dtype = dtype

        if dtype is not None:
            self.out_range = dtype
        else:
            self.out_range = "dtype"

    def transform(self, image):
        image = rescale_intensity(image, out_range=self.out_range)
        return image.astype(self.dtype, copy=False)


if __name__ == "__main__":
    dir_reader = DirectoryReader(import_path)
    rescale = Rescale("uint8")(dir_reader.image)

    pipeline = SimpleScheduler(rescale).to_pipeline()

    for x in pipeline.transform_stream([]):
        pprint(x)
        break
