import csv
import inspect
import io
import itertools
import operator
import os
import pprint
import typing as T
import zipfile
from collections import namedtuple
from typing import List, Optional, Type

import click
import docstring_parser
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

import pprint

from morphocut.graph import Input, Node, Output
from morphocut.graph.port import Port
from morphocut.graph.scheduler import SimpleScheduler

# import_path = "/data-ssd/mschroeder/Datasets/generic_zooscan_peru_kosmos_2017"
import_path = "/home/moi/Work/Datasets/generic_zooscan_peru_kosmos_2017"
archive_fn = "/tmp/kosmos.zip"


@Output("abs_path")
@Output("rel_path")
class DirectoryReader(Node):
    """
    Read all image files under the specified directory.

    Args:
        image_root (str): Root path where images should be found.
        allowed_extensions (optional): List of allowed image extensions (including the leading dot).
    """

    def __init__(self, image_root: str, allowed_extensions: Optional[List[str]] = None):
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

    def __str__(self):
        return "{}({})".format(self.__class__.__name__, self.clbl.__name__)


@parse.with_pattern(".*")
def parse_greedystar(text):
    return text


EXTRA_TYPES = {
    "greedy": parse_greedystar
}


@Input("input")
@Output("meta")
class PathParser(Node):
    """Parse information from a path.

    Args:
        pattern (str): The pattern to look for in the input.
        case_sensitive (bool): Match pattern with case.
    """

    def __init__(self, pattern: str, case_sensitive: bool = False):
        super().__init__()

        self.pattern = parse.compile(
            pattern, extra_types=EXTRA_TYPES, case_sensitive=case_sensitive)

    def transform(self, input):
        return self.pattern.parse(input).named


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

    def __init__(self, name=""):
        super().__init__()

        self.min = []
        self.max = []
        self.name = name

    def transform(self, image):
        self.min.append(np.min(image))
        self.max.append(np.max(image))

    def after_stream(self):
        print("### Range stats ({}) ###".format(self.name))
        mean_min = np.mean(self.min)
        mean_max = np.mean(self.max)
        print("Absolute: ", min(self.min), max(self.max))
        print("Average: ", mean_min, mean_max)


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
    def __init__(self, in_range='image', dtype=None):
        super().__init__()
        self.dtype = dtype

        self.in_range = in_range

        if dtype is not None:
            self.out_range = dtype
        else:
            self.out_range = "dtype"

    def transform(self, image):
        image = rescale_intensity(
            image, in_range=self.in_range, out_range=self.out_range)
        if self.dtype is not None:
            image = image.astype(self.dtype, copy=False)

        return image


@Input("mask")
@Input("image", required=False)
@Output("regionprops")
class FindRegions(Node):
    def __init__(self, min_area=None, max_area=None, padding=0):
        super().__init__()

        self.min_area = min_area
        self.max_area = max_area
        self.padding = padding

    @staticmethod
    def _enlarge_slice(slices, padding):
        return tuple(slice(max(0, s.start - padding), s.stop + padding) for s in slices)

    def transform_stream(self, stream):
        for obj in stream:
            inp = self.prepare_input(obj)
            mask = inp["mask"]
            image = inp["image"]

            labels, nlabels = skimage.measure.label(mask, return_num=True)

            objects = ndi.find_objects(labels, nlabels)
            for i, sl in enumerate(objects):
                if sl is None:
                    continue

                if self.padding:
                    sl = self._enlarge_slice(sl, self.padding)

                props = skimage.measure._regionprops._RegionProperties(
                    sl, i+1, labels, image, True, 'rc')

                if self.min_area is not None and props.area < self.min_area:
                    continue

                if self.max_area is not None and props.area > self.max_area:
                    continue

                yield self.prepare_output(obj, props)


@Input("image")
@Input("regionprops")
@Output("extracted_image")
class ExtractROI(Node):
    # TODO: Hide background using mask
    def transform(self, image, regionprops):
        return image[regionprops.slice]

# TODO: Draw object info


def regionprop2zooprocess(prop):
    """
    Calculate zooprocess features from skimage regionprops.

    Notes:
        - date/time specify the time of the sampling, not of the processing.
    """
    return {
        # width of the smallest rectangle enclosing the object
        'width': prop.bbox[3] - prop.bbox[1],
        # height of the smallest rectangle enclosing the object
        'height': prop.bbox[2] - prop.bbox[0],
        # X coordinates of the top left point of the smallest rectangle enclosing the object
        'bx': prop.bbox[1],
        # Y coordinates of the top left point of the smallest rectangle enclosing the object
        'by': prop.bbox[0],
        # circularity : (4∗π ∗Area)/Perim^2 a value of 1 indicates a perfect circle, a value approaching 0 indicates an increasingly elongated polygon
        'circ.': (4 * np.pi * prop.filled_area) / prop.perimeter**2,
        # Surface area of the object excluding holes, in square pixels (=Area*(1-(%area/100))
        'area_exc': prop.area,
        # Surface area of the object in square pixels
        'area': prop.filled_area,
        # Percentage of object’s surface area that is comprised of holes, defined as the background grey level
        '%area': 1 - (prop.area / prop.filled_area),
        # Primary axis of the best fitting ellipse for the object
        'major': prop.major_axis_length,
        # Secondary axis of the best fitting ellipse for the object
        'minor': prop.minor_axis_length,
        # Y position of the center of gravity of the object
        'y': prop.centroid[0],
        # X position of the center of gravity of the object
        'x': prop.centroid[1],
        # The area of the smallest polygon within which all points in the objet fit
        'convex_area': prop.convex_area,
        # Minimum grey value within the object (0 = black)
        'min': prop.min_intensity,
        # Maximum grey value within the object (255 = white)
        'max': prop.max_intensity,
        # Average grey value within the object ; sum of the grey values of all pixels in the object divided by the number of pixels
        'mean': prop.mean_intensity,
        # Integrated density. The sum of the grey values of the pixels in the object (i.e. = Area*Mean)
        'intden': prop.filled_area * prop.mean_intensity,
        # The length of the outside boundary of the object
        'perim.': prop.perimeter,
        # major/minor
        'elongation': np.divide(prop.major_axis_length, prop.minor_axis_length),
        # max-min
        'range': prop.max_intensity - prop.min_intensity,
        # perim/area_exc
        'perimareaexc': prop.perimeter / prop.area,
        # perim/major
        'perimmajor': prop.perimeter / prop.major_axis_length,
        # (4 ∗ π ∗ Area_exc)/perim 2
        'circex': np.divide(4 * np.pi * prop.area, prop.perimeter**2),
        # Angle between the primary axis and a line parallel to the x-axis of the image
        'angle': prop.orientation / np.pi * 180 + 90,
        # # X coordinate of the top left point of the image
        # 'xstart': data_object['raw_img']['meta']['xstart'],
        # # Y coordinate of the top left point of the image
        # 'ystart': data_object['raw_img']['meta']['ystart'],
        # Maximum feret diameter, i.e. the longest distance between any two points along the object boundary
        # 'feret': data_object['raw_img']['meta']['feret'],
        # feret/area_exc
        # 'feretareaexc': data_object['raw_img']['meta']['feret'] / property.area,
        # perim/feret
        # 'perimferet': property.perimeter / data_object['raw_img']['meta']['feret'],

        'bounding_box_area': prop.bbox_area,
        'eccentricity': prop.eccentricity,
        'equivalent_diameter': prop.equivalent_diameter,
        'euler_number': prop.euler_number,
        'extent': prop.extent,
        'local_centroid_col': prop.local_centroid[1],
        'local_centroid_row': prop.local_centroid[0],
        'solidity': prop.solidity,
    }


@Input("regionprops")
@Input("meta_in", required=False)
@Output("meta")
class CalculateZooProcessFeatures(Node):
    """Calculate descriptive features using skimage.measure.regionprops.
    """

    def __init__(self, prefix=None):
        super().__init__()
        self.prefix = prefix

    def transform(self, regionprops, meta_in):
        if meta_in is None:
            meta_in = {}

        features = regionprop2zooprocess(regionprops)

        if self.prefix is not None:
            features = {"{}{}".format(self.prefix, k)
                                      : v for k, v in features.items()}

        return {**meta_in, **features}


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


class StreamDebugger(Node):
    def transform_stream(self, stream):
        for obj in stream:
            pprint.pprint(obj)
            yield obj


class NodeRegistry:
    def __init__(self):
        self.nodes = set()

    def register(self, *nodes: List[Type[Node]]):
        self.nodes.update(nodes)

    def pipeline_factory(self, pipeline_spec):
        """Construct a pipeline according to the spec.
        """
        ...

    @staticmethod
    def _port_to_tuple(port: Port):
        return (
            None,
            inspect.cleandoc(port.help) if port.help else None
        )

    @staticmethod
    def _parse_docstr(obj):
        try:
            return docstring_parser.parse(obj.__doc__)
        except:
            print("Error parsing docstring of {}".format(obj.__name__))
            raise

    @staticmethod
    def _parse_arguments(node_cls: Type[Node]):
        # Use type annotations to determine the type.
        # Use the docstring for each argument.

        annotations = node_cls.__init__.__annotations__

        # Get docstring for each argument
        arg_desc = {
            p.arg_name: p.description
            for p in NodeRegistry._parse_docstr(node_cls).params
        }
        arg_desc.update({
            p.arg_name: p.description
            for p in NodeRegistry._parse_docstr(node_cls.__init__).params
        })

        return {
            k: (annotations[k], arg_desc[k])
            for k in annotations.keys() & arg_desc.keys()
        }

    @classmethod
    def _node_to_dict(cls, node_cls: Type[Node]):
        doc = cls._parse_docstr(node_cls)
        return {
            "name": node_cls.__name__,
            "short_description": doc.short_description,
            "long_description": doc.long_description,
            "inputs": {p.name: cls._port_to_tuple(p) for p in getattr(node_cls, "inputs", [])},
            "outputs": {p.name: cls._port_to_tuple(p) for p in getattr(node_cls, "outputs", [])},
            "options": cls._parse_arguments(node_cls),
        }

    def to_dict(self) -> dict:
        return {
            n.__name__: self._node_to_dict(n) for n in self.nodes
        }


if __name__ == "__main__":
    nreg = NodeRegistry()
    nreg.register(DirectoryReader)  # type: ignore
    pprint.pprint(nreg.to_dict())
    # abs_path, rel_path = DirectoryReader(os.path.join(import_path, "raw"))()
    # # Images are named <sampleid>/<anything>_<a|b>.tif
    # # e.g. generic_Peru_20170226_slow_M1_dnet/Peru_20170226_M1_dnet_1_8_a.tif

    # meta = chain(
    #     PathParser(
    #         "generic_{sample_id}/{:greedy}_{sample_split:d}_{sample_nsplit:d}_{sample_subid}.tif"),
    #     JoinMeta(
    #         "/home/moi/Work/Datasets/generic_zooscan_peru_kosmos_2017/Morphocut_header_scans_peru_kosmos_2017.xlsx",
    #         "sample_id"
    #     ),
    # )(rel_path)

    # DumpMeta(
    #     os.path.join(import_path, "meta.csv"),
    #     unique_col="sample_id"
    # )(meta)

    # img = chain(
    #     LambdaNode(skimage.io.imread),
    #     Rescale(in_range=(9252, 65278), dtype=np.uint8)
    # )(abs_path)

    # mask = chain(
    #     ThresholdConst(245),  # 245(ubyte) / 62965(uint16)
    #     LambdaNode(skimage.segmentation.clear_border),
    # )(img)

    # regionprops = FindRegions(100, padding=10)(mask, img)

    # # Extract a vignette from the image
    # vignette = ExtractROI()(img, regionprops)

    # # It is not elegant to have this Node consume an arbitrary port to make it being scheduled the right time...
    # meta = GenerateObjectId(
    #     "{sample_id}_{sample_split:d}_{sample_nsplit:d}_{sample_subid}_{i:d}")(meta, vignette)

    # meta = CalculateZooProcessFeatures("object_")(regionprops, meta)

    # zip_dumper = DumpToZip(
    #     os.path.join(import_path, "export.zip"),
    #     "{object_id}.jpg")(vignette, meta)

    # # Schedule and execute pipeline
    # pipeline = SimpleScheduler(zip_dumper).to_pipeline()

    # print(pipeline)

    # stream = tqdm(pipeline.transform_stream([]))

    # for x in stream:
    #     stream.set_description(x[meta]["object_id"])
    #     pass
