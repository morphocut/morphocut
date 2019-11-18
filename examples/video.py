from morphocut import Pipeline
from morphocut.image import ImageWriter
from morphocut.pims import VideoReader
from morphocut.str import Format

with Pipeline() as p:
    frame = VideoReader("/path/to/video.avi")
    filename = Format("/output/path/frame_#{}.png", frame.frame_no)
    ImageWriter(filename, frame)

p.run()
