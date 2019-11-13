import os.path

from morphocut import Call, Pipeline
from morphocut.file import Glob
from morphocut.image import FindRegions, ImageReader, ImageWriter
from morphocut.parallel import ParallelPipeline
from morphocut.str import Format
from morphocut.stream import Enumerate, Unpack

# First, a Pipeline is defined that contains all operations
# that should be carried out on the objects of the stream.
with Pipeline() as p:
    # Corresponds to `for base_path in ["/path/a", "/path/b", "/path/c"]:`
    base_path = Unpack(["/path/a", "/path/b", "/path/c"])

    # Number the objects in the stream
    i = Enumerate()

    # Call calls regular Python functions.
    # Here, a subpath is appended to base_path.
    pattern = Call(os.path.join, base_path, "subpath/to/input/files/*.jpg")

    # Corresponds to `for path in glob(pattern):`
    path = Glob(pattern)

    # Remove path and extension from the filename
    source_basename = Call(lambda x: os.path.splitext(os.path.basename(x))[0], path)

    # Execute parallelly:
    with ParallelPipeline():
        # Read the image
        image = ImageReader(path)

        # Do some thresholding
        mask = image < 128

        # Find regions in the image
        region = FindRegions(mask, image)

        # Extract just the object
        roi_image = region.intensity_image

        # Calculate an output filename for the ROI image
        output_fn = Format(
            "/path/to/output/{:d}-{}-{:d}.png", i, source_basename, region.label
        )

        # Write the image
        ImageWriter(output_fn, roi_image)

# In the end, the Pipeline is executed. A stream is created and transformed by the
# operations defined in the Pipeline.
p.run()
