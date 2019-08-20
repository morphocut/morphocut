import csv
import io
import itertools
import operator
import os
import zipfile
from collections import namedtuple

import numpy as np
import pandas as pd
import parse
import PIL
import scipy.ndimage as ndi
import skimage
import skimage.io
import skimage.measure
import skimage.segmentation
from skimage.exposure import rescale_intensity
from tqdm import tqdm

from morphocut.graph import Input, Node, Output
from morphocut.graph.scheduler import SimpleScheduler

# import_path = "/data-ssd/mschroeder/Datasets/generic_zooscan_peru_kosmos_2017"
import_path = "/home/moi/Work/Datasets/generic_zooscan_peru_kosmos_2017"
archive_fn = "/tmp/kosmos.zip"


@Output("abs_path")
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

                yield self.prepare_output(
                    {},
                    os.path.join(root, fn),
                    os.path.join(rel_root, fn))


@Input("inp")
@Output("out")
class LambdaNode(Node):
    def __init__(self, clbl):
        super().__init__()
        self.clbl = clbl

    def transform(self, inp):
        return self.clbl(inp)


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

    def transform(self, path):
        return self.pattern.parse(path).named


@Input("meta")
class DumpMeta(Node):
    """
    Parse information from a path
    """

    def __init__(self, filename, fields=None, unique_col=None):
        super().__init__()

        self.fields = fields
        self.filename = filename
        self.unique_col = unique_col

    def transform_stream(self, stream):
        """Apply transform to every object in the stream.
        """

        result = []

        for obj in stream:
            meta = self.prepare_input(obj)["meta"]

            if self.fields is not None:
                row = {k: meta.get(k, None) for k in self.fields}
            else:
                row = meta

            result.append(row)

            yield obj

        result = pd.DataFrame(result)

        if self.unique_col is not None:
            result.drop_duplicates(subset=self.unique_col, inplace=True)

        result.to_csv(self.filename, index=False)


@Input("image")
class ImageStats(Node):
    """
    Parse information from a path
    """

    def __init__(self):
        super().__init__()

        self.min = np.inf
        self.max = -np.inf

    def transform(self, image):
        self.min = min(self.min, np.min(image))
        self.max = max(self.max, np.max(image))

    def after_stream(self):
        print("Value range:", self.min, self.max)


@Input("meta_in")
@Output("meta_out")
class JoinMeta(Node):
    """
    Join information from a CSV/TSV/Excel/... file.
    """

    def __init__(self, filename, on=None, fields=None):
        super().__init__()

        ext = os.path.splitext(filename)[1]

        self.on = on

        if ext in (".xls", ".xlsx"):
            dataframe = pd.read_excel(filename, usecols=fields)
        else:
            with open('example.csv', newline='') as csvfile:
                dialect = csv.Sniffer().sniff(csvfile.read(1024))
            dataframe = pd.read_csv(
                filename,
                dialect=dialect,
                usecols=fields)

        dataframe.set_index(self.on, inplace=True, verify_integrity=True)

        self.dataframe = dataframe

    def transform(self, meta_in):
        key = meta_in[self.on]

        row = self.dataframe.loc[key].to_dict()

        return {**meta_in, **row}


@Input("image")
@Output("mask")
class ThresholdConst(Node):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def transform(self, image):
        if image.ndim != 2:
            raise ValueError("image.ndim needs to be exactly 2.")

        mask = image <= self.threshold

        return mask


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


ROI = namedtuple("ROI", ["slice", "mask"])


@Input("mask")
@Output("roi")
class FindROI(Node):
    # TODO: Enlarge slice by linear padding
    def __init__(self, min_area=None, max_area=None):
        super().__init__()

        self.min_area = min_area
        self.max_area = max_area

    def transform_stream(self, stream):
        for obj in stream:
            mask = self.prepare_input(obj)["mask"]
            labels, nlabels = skimage.measure.label(mask, return_num=True)

            objects = ndi.find_objects(labels, nlabels)
            for i, sl in enumerate(objects):
                if sl is None:
                    continue

                roi_mask = labels[sl] == i+1

                area = np.sum(roi_mask)

                if self.min_area is not None and area < self.min_area:
                    continue

                if self.max_area is not None and area > self.max_area:
                    continue

                roi = ROI(sl, roi_mask)

                yield self.prepare_output(obj, roi)


@Input("image")
@Input("roi")
@Output("extracted_image")
class ExtractROI(Node):
    # TODO: Hide background using mask
    def transform(self, image, roi):
        return image[roi.slice]

# TODO: Draw object info


@Input("image")
@Input("meta")
class DumpToZip(Node):
    def __init__(self, archive_fn, image_fn, meta_fn="ecotaxa_export.tsv"):
        super().__init__()
        self.archive_fn = archive_fn
        self.image_fn = image_fn
        self.meta_fn = meta_fn
        self.image_ext = os.path.splitext(self.image_fn)[1]

    def transform_stream(self, stream):
        with zipfile.ZipFile(self.archive_fn, mode="w") as zf:
            dataframe = []
            for obj in stream:
                # TODO: Support multiple images
                inp = self.prepare_input(obj)

                pil_format = PIL.Image.registered_extensions()[self.image_ext]

                img = PIL.Image.fromarray(inp["image"])
                img_fp = io.BytesIO()
                img.save(img_fp, format=pil_format)

                arcname = self.image_fn.format(**inp["meta"])

                zf.writestr(arcname, img_fp.getvalue())

                dataframe.append({
                    **inp["meta"],
                    "img_file_name": arcname
                })

                yield obj

            dataframe = pd.DataFrame(dataframe)
            zf.writestr(
                "ecotaxa_export.tsv",
                dataframe.to_csv(sep='\t', encoding='utf-8', index=False))


@Input("meta_in")
@Input("_")
@Output("meta_out")
class GenerateObjectId(Node):
    def __init__(self, fmt, name="object_id"):
        super().__init__()
        self.fmt = fmt
        self.name = name

    def transform_stream(self, stream):
        for i, obj in enumerate(stream):
            meta_in = self.prepare_input(obj)["meta_in"]

            fields = {**meta_in, "i": i}
            name = self.fmt.format(**fields)

            yield self.prepare_output(obj, {**meta_in, self.name: name})


@Input("image")
@Input("meta")
class DumpImages(Node):
    def __init__(self, root, fmt):
        super().__init__()
        self.root = root
        self.fmt = fmt

    def transform_stream(self, stream):
        for dirname, group in itertools.groupby(self._gen_paths(stream), operator.itemgetter(0)):
            os.makedirs(dirname, exist_ok=True)
            for _, filename, image, obj in group:

                skimage.io.imsave(filename, image)

                yield obj

    def _gen_paths(self, stream):
        for obj in stream:
            inp = self.prepare_input(obj)

            filename = os.path.join(self.root, self.fmt.format(**inp["meta"]))

            dirname = os.path.dirname(filename)

            image = inp["image"]

            yield dirname, filename, image, obj


def chain(*nodes):
    def wrapper(port):
        for n in nodes:
            port = n(port)
        return port

    return wrapper


if __name__ == "__main__":
    dir_reader = DirectoryReader(os.path.join(import_path, "raw"))
    # Images are named <sampleid>/<anything>_<a|b>.tif
    # e.g. generic_Peru_20170226_slow_M1_dnet/Peru_20170226_M1_dnet_1_8_a.tif
    path_meta = PathParser(
        "generic_{sample_id}/{:greedy}_{sample_split:d}_{sample_nsplit:d}_{sample_subid}.tif")(dir_reader.rel_path)

    join_meta = JoinMeta(
        "/home/moi/Work/Datasets/generic_zooscan_peru_kosmos_2017/Morphocut_header_scans_peru_kosmos_2017.xlsx",
        "sample_id"
    )(path_meta)

    dump_meta = DumpMeta(
        os.path.join(import_path, "meta.csv"),
        unique_col="sample_id"
    )(join_meta)

    img_reader = LambdaNode(skimage.io.imread)(dir_reader.abs_path)

    img_ubyte = LambdaNode(skimage.img_as_ubyte)(img_reader)

    mask = chain(
        ThresholdConst(245),
        LambdaNode(skimage.segmentation.clear_border)
    )(img_ubyte)

    find_roi = FindROI(100)(mask)

    # Extract a vignette from the image
    extract_roi = ExtractROI()(img_ubyte, find_roi.roi)

    # It is not elegant to have this Node consume an arbitrary port to make it being scheduled the right time...
    gen_object_id = GenerateObjectId(
        "{sample_id}_{sample_split:d}_{sample_nsplit:d}_{sample_subid}_{i:d}")(join_meta, extract_roi)

    zip_dumper = DumpToZip(
        os.path.join(import_path, "export.zip"),
        "{object_id}.jpg")(extract_roi, gen_object_id)

    # Schedule and execute pipeline
    pipeline = SimpleScheduler(zip_dumper).to_pipeline()

    print(pipeline)

    stream = tqdm(pipeline.transform_stream([]))

    for x in stream:
        stream.set_description(x[gen_object_id.meta_out]["object_id"])
