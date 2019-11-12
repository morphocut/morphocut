"""
Quickstart
==========
"""
import os.path

from morphocut import Call, Pipeline
from morphocut.file import Glob
from morphocut.image import FindRegions, ImageReader, ImageWriter
from morphocut.parallel import ParallelPipeline
from morphocut.str import Format
from morphocut.stream import Enumerate, FromIterable

with Pipeline() as p:
    base_path = FromIterable(["/path/a", "/path/b", "/path/c"])
    i = Enumerate()
    pattern = Call(os.path.join, base_path, "subpath/to/input/files/*.jpg")
    path = Glob(pattern)
    source_basename = Call(lambda x: os.path.splitext(os.path.basename(x))[0], path)

    with ParallelPipeline():
        image = ImageReader(path)
        mask = image < 128
        region = FindRegions(mask, image)
        roi_image = region.intensity_image

        output_fn = Format(
            "/path/to/output/{:d}-{}-{:d}.png", i, source_basename, region.label
        )

        ImageWriter(output_fn, roi_image)

p.run()
