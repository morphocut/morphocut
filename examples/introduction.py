import os.path

from morphocut import Call, Pipeline
from morphocut.contrib.ecotaxa import EcotaxaWriter
from morphocut.contrib.zooprocess import CalculateZooProcessFeatures
from morphocut.file import Glob
from morphocut.image import FindRegions, ImageReader
from morphocut.parallel import ParallelPipeline
from morphocut.str import Format
from morphocut.stream import Enumerate, Unpack

# First, a Pipeline is defined that contains all operations
# that should be carried out on the objects of the stream.
with Pipeline() as p:
    # Corresponds to `for base_path in ["/path/a", "/path/b", "/path/c"]:`
    base_path = Unpack(["/path/a", "/path/b", "/path/c"])

    # Number the objects in the stream
    running_number = Enumerate()

    # Call calls regular Python functions.
    # Here, a subpath is appended to base_path.
    pattern = Call(os.path.join, base_path, "subpath/to/input/files/*.jpg")

    # Corresponds to `for path in glob(pattern):`
    path = Glob(pattern)

    # Remove path and extension from the filename
    source_basename = Call(lambda x: os.path.splitext(os.path.basename(x))[0], path)

    with ParallelPipeline():
        # The following operations are distributed among multiple
        # worker processes to speed up the calculations.

        # Read the image
        image = ImageReader(path)

        # Do some thresholding
        mask = image < 128

        # Find regions in the image
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

# After the Pipeline was defined, it can be executed.
# A stream is created and transformed by the operations
# defined in the Pipeline.
p.run()
