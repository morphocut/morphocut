from morphocut.graph import Pipeline
from morphocut.io import ImageWriter, ImageReader

with Pipeline() as p:
    # TODO: Work on many images
    image = ImageReader("/path/to/input.png")()
    ImageWriter("/path/to/output.png", image)


p.run()
