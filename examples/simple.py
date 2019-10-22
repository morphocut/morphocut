from morphocut import Pipeline
from morphocut.io import ImageReader, ImageWriter
from morphocut.file import Glob

with Pipeline() as p:
    path = Glob("/path/to/input/files")
    image = ImageReader(path)
    ImageWriter("/path/to/output.png", image)

p.run()
