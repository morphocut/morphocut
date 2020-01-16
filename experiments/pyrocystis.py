"""Experiment on processing KOSMOS data using MorphoCut."""
import collections
import itertools
import os

import numpy as np
import skimage
import skimage.io
import skimage.measure
import skimage.segmentation
from skimage.util import img_as_ubyte

from morphocut import Call
from morphocut.contrib.ecotaxa import EcotaxaWriter
from morphocut.contrib.zooprocess import CalculateZooProcessFeatures
from morphocut.core import (
    Node,
    Output,
    Pipeline,
    ReturnOutputs,
    Stream,
    closing_if_closable,
)
from morphocut.file import Find
from morphocut.image import (
    ExtractROI,
    FindRegions,
    ImageReader,
    ImageWriter,
    RescaleIntensity,
    RGB2Gray,
)
from morphocut.stat import ExponentialSmoothing
from morphocut.str import Format
from morphocut.stream import TQDM, StreamBuffer, Enumerate

import_path = "/home/moi/Work/0-Datasets/Pyrocystis_noctiluca/RAW"
export_path = "/tmp/Pyrocystis_noctiluca"


@ReturnOutputs
@Output("agg_value")
class StreamRunningMedian(Node):
    """
    Calculate the median over a stream of objects.

    Uses the efficient approximation from:
        Mcfarlane, N. J. B., & Schofield, C. P. (1995).
        Segmentation and tracking of piglets in images.
        In Machine Vision and Applications (Vol. 8).
    """

    def __init__(self, value, n_init=10):
        super().__init__()
        self.value = value
        self.n_init = n_init
        self.median = None

    def transform_stream(self, stream: Stream) -> Stream:
        """Transform a stream."""

        with closing_if_closable(stream):
            # Initial approximation
            if self.median is None:
                objects = []
                values = []
                for obj in itertools.islice(stream, self.n_init):
                    value = self.prepare_input(obj, "value")
                    objects.append(obj)
                    values.append(value)

                self.median = np.median(values, axis=0)

                for obj in objects:
                    yield self.prepare_output(obj, self.median)

            # Process
            for obj in stream:
                value = self.prepare_input(obj, "value")

                # Update according to Mcfarlane & Schofield
                mask = value > self.median
                self.median[mask] += 1

                mask = value < self.median
                self.median[mask] -= 1

                yield self.prepare_output(obj, self.median)

        self.after_stream()


if __name__ == "__main__":
    print("Processing images under {}...".format(import_path))

    os.makedirs(export_path, exist_ok=True)

    with Pipeline() as p:
        abs_path = Find(import_path, [".jpg"], sort=True)

        name = Call(lambda p: os.path.splitext(os.path.basename(p))[0], abs_path)

        img = ImageReader(abs_path)

        flat_field = StreamRunningMedian(img, 10)

        img = img / flat_field

        img = RescaleIntensity(img, in_range=(0, 1.1), dtype="uint8")

        # export_fn = Call(os.path.join, export_path, Format("{}.png", name))
        # ImageWriter(export_fn, img, compress_level=1)

        TQDM(name)

        img_gray = RGB2Gray(img)

        threshold = 0.8  # Call(skimage.filters.threshold_otsu, img_gray)

        mask = img_gray < threshold

        img = Call(img_as_ubyte, img)
        img_gray = Call(img_as_ubyte, img_gray)

        # TODO: min_area
        regionprops = FindRegions(mask, img_gray, min_area=100, padding=10)

        # Extract a vignette from the image
        roi_orig = ExtractROI(img, regionprops, bg_color=255)
        roi_gray = ExtractROI(img_gray, regionprops, bg_color=255)

        # Call(lambda x: print(x.dtype), roi_orig)
        # Call(lambda x: print(x.dtype), roi_gray)

        i = Enumerate()
        object_id = Format("{name}_{i:d}", name=name, i=i)

        meta = CalculateZooProcessFeatures(regionprops, prefix="object_")
        meta["object_id"] = object_id

        orig_fn = Format("{object_id}.jpg", object_id=object_id)
        gray_fn = Format("{object_id}-gray.jpg", object_id=object_id)

        EcotaxaWriter(
            os.path.join(export_path, "pyrocystis.zip"),
            [(orig_fn, roi_orig), (gray_fn, roi_gray)],
            meta,
        )

        TQDM(object_id)

    p.run()
