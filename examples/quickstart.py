import os.path

from morphocut import LambdaNode, Pipeline
from morphocut.file import Glob
from morphocut.image import FindRegions, ImageReader, ImageWriter
from morphocut.stream import FromIterable, Enumerate
from morphocut.str import Format

with Pipeline() as p:
    base_path = FromIterable(["/path/a", "/path/b", "/path/c"])
    i = Enumerate()
    pattern = LambdaNode(os.path.join, base_path, "subpath/to/input/files/*.jpg")
    path = Glob(pattern)
    source_basename = LambdaNode(
        lambda x: os.path.splitext(os.path.basename(x))[0], path
    )

    image = ImageReader(path)
    mask = image < 128
    region = FindRegions(mask, image)
    roi_image = region.intensity_image

    output_fn = Format(
        "/path/to/output/{:d}-{}-{:d}.png", i, source_basename, region.label
    )

    ImageWriter(output_fn, roi_image)

p.run()
