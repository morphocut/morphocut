import os
from pprint import pprint

import numpy as np
import pandas as pd
import parse
from skimage import io
from skimage.exposure import rescale_intensity

from morphocut.graph import Input, Node, Output
from morphocut.graph.scheduler import SimpleScheduler

#import_path = "/data-ssd/mschroeder/Datasets/generic_zooscan_peru_kosmos_2017"
import_path = "/home/moi/Work/Datasets/generic_zooscan_peru_kosmos_2017"
archive_fn = "/tmp/kosmos.zip"


@Output("image")
@Output("rel_path")
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

                yield self.prepare_output({}, img, os.path.join(rel_root, fn))


@parse.with_pattern(".*")
def parse_greedystar(text):
    return text


EXTRA_TYPES = {
    "greedy": parse_greedystar
}


@Input("path")
@Output("meta")
class PathParser(Node):
    """
    Parse information from a path
    """

    def __init__(self, pattern, case_sensitive=False):
        super().__init__()

        self.pattern = parse.compile(
            pattern, extra_types=EXTRA_TYPES, case_sensitive=case_sensitive)

        print(self.pattern._match_re)

    def transform(self, path):
        return self.pattern.parse(path).named


@Input("meta")
class DumpMeta(Node):
    """
    Parse information from a path
    """

    def __init__(self, filename, fields=None, unique_col=None):
        super().__init__()

        if fields is None:
            fields = []

        self.fields = fields
        self.filename = filename
        self.unique_col = unique_col

    def transform_stream(self, stream):
        """Apply transform to every object in the stream.
        """

        result = []

        for obj in stream:
            meta = self.prepare_input(obj)["meta"]

            print(meta)

            row = {k: meta.get(k, None) for k in self.fields}

            result.append(row)

            yield obj

        result = pd.DataFrame(result)

        if self.unique_col is not None:
            result.drop_duplicates(subset=self.unique_col, inplace=True)

        result.to_csv(self.filename, index=False)


# @Input("meta")
# @Output("meta")
# class JoinMeta(Node):
#     """
#     Join information from a CSV file.
#     """

#     def __init__(self, ...):
#         super().__init__()

#     def transform(self, ...):
#         return ...


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
    dir_reader = DirectoryReader(os.path.join(import_path, "raw"))
    # Images are named <sampleid>/<anything>_<a|b>.tif
    # e.g. generic_Peru_20170226_slow_M1_dnet/Peru_20170226_M1_dnet_1_8_a.tif
    path_meta = PathParser(
        "generic_{sample_id}/{:greedy}_{ab}.tif")(dir_reader.rel_path)

    dump_meta = DumpMeta(
        os.path.join(import_path, "meta.csv"),
        ["sample_id", "object_lat", "object_lon", "object_date",
            "object_time", "object_depth_min", "object_depth_max"],
        unique_col="sample_id"
    )(path_meta)

    #rescale = Rescale("uint8")(dir_reader.image)

    pipeline = SimpleScheduler(dump_meta).to_pipeline()

    print(pipeline)

    for x in pipeline.transform_stream([]):
        print(x[dir_reader.rel_path], x[path_meta.meta])
