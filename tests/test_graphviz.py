import pandas as pd
from networkx.drawing.nx_pydot import write_dot

from morphocut.core import Call, Node, Output, Pipeline
from morphocut.parallel import ParallelPipeline
from morphocut.image import ImageReader, FindRegions

from morphocut.stream import Unpack, Enumerate
from morphocut.file import Glob
from morphocut.str import Format
from morphocut.contrib.ecotaxa import EcotaxaWriter
from morphocut.contrib.zooprocess import CalculateZooProcessFeatures
import os.path

def test_graphviz():
    with Pipeline() as p:
        base_path = Unpack(["/path/a", "/path/b", "/path/c"])
        running_number = Enumerate()
        pattern = Call(os.path.join, base_path, "subpath/to/input/files/*.jpg")
        path = Glob(pattern)
        source_basename = Call(lambda x: os.path.splitext(os.path.basename(x))[0], path)

        with ParallelPipeline():
            image = ImageReader(path)
            mask = image < 128
            region = FindRegions(mask, image)

            # Extract just the object
            roi_image = region.intensity_image

            # An object is identified by its label
            roi_label = region.label

            # Calculate a filename for the ROI image:
            # "RUNNING_NUMBER-SOURCE_BASENAME-ROI_LABEL"
            roi_name = Format(
                "{:d}-{}-{:d}.jpg", running_number, source_basename, roi_label
            )

            meta = CalculateZooProcessFeatures(region, prefix="object_")
            # End of parallel execution

        # Store results
        EcotaxaWriter("archive.zip", (roi_name, roi_image), meta)

    p.to_dot("/tmp/pipeline.dot")
