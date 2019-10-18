from morphocut import Pipeline
from morphocut.io import ImageReader, ImageWriter

with Pipeline() as p:
    # TODO: Work on many images
    image = ImageReader("/path/to/input.png")
    ImageWriter("/path/to/output.png", image)

p.run()
