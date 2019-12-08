"""Experiment on processing KOSMOS data using MorphoCut."""
import collections
import itertools
import os

import numpy as np
import skimage
import skimage.io
import skimage.measure
import skimage.segmentation

from morphocut import Call
from morphocut.core import (
    Node,
    Output,
    Pipeline,
    ReturnOutputs,
    Stream,
    closing_if_closable,
)
from morphocut.file import Find
from morphocut.image import ImageReader, ImageWriter, RescaleIntensity
from morphocut.stat import ExponentialSmoothing
from morphocut.stream import TQDM, StreamBuffer
from morphocut.str import Format

import_path = "/home/moi/Work/0-Datasets/Pyrocystis_noctiluca/RAW"
export_path = "/tmp/Pyrocystis_noctiluca"


@ReturnOutputs
@Output("agg_value")
class StreamWindowMedian(Node):
    """
    Calculate the median over stream objects using a sliding window.

    TODO: Use efficient online median calculation: https://github.com/suomela/median-filter
    """

    def __init__(self, value, window_size):
        super().__init__()
        self.value = value
        self.window_size = window_size
        self.queue = collections.deque(maxlen=self.window_size)

    def transform_stream(self, stream: Stream) -> Stream:
        """Transform a stream."""

        with closing_if_closable(stream):
            # Lead-in
            for obj in itertools.islice(stream, self.window_size):
                value = self.prepare_input(obj, "value")

                self.queue.append((obj, value))

            # Catch-up
            agg_value = self._calc_agg_value()
            for obj, _ in self.queue:
                yield self.prepare_output(obj, agg_value)

            # Process
            for obj in stream:
                value = self.prepare_input(obj, "value")
                self.queue.append((obj, value))
                yield self.prepare_output(obj, self._calc_agg_value())

        self.after_stream()

    def _calc_agg_value(self):
        return np.median([value for obj, value in self.queue], axis=0)


if __name__ == "__main__":
    print("Processing images under {}...".format(import_path))

    os.makedirs(export_path, exist_ok=True)

    with Pipeline() as p:
        # Images are named <sampleid>/<anything>_<a|b>.tif
        # e.g. generic_Peru_20170226_slow_M1_dnet/Peru_20170226_M1_dnet_1_8_a.tif
        abs_path = Find(import_path, [".jpg"])

        name = Call(lambda p: os.path.splitext(os.path.basename(p))[0], abs_path)

        TQDM(name)

        img = ImageReader(abs_path)

        flat_field = StreamWindowMedian(img, 10)

        img = img / flat_field

        img = RescaleIntensity(img, dtype="uint8")

        StreamBuffer(10)

        export_fn = Call(os.path.join, export_path, Format("{}.png", name))
        ImageWriter(export_fn, img)

    p.run()
