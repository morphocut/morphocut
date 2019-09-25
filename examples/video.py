from morphocut.graph import Pipeline
from morphocut.pims import VideoReader
from morphocut.io import ImageWriter
from morphocut.str import Formatter

with Pipeline() as p:
    image, frame_number = VideoReader("/path/to/video.avi")()
    filename = Formatter("/output/path/frame_#{}.png", frame_number)()
    ImageWriter(filename, image)

p.run()
