"""
Video example
=============
"""
from morphocut import Pipeline
from morphocut.io import ImageWriter
from morphocut.pims import VideoReader
from morphocut.str import Format

with Pipeline() as p:
    image, frame_number = VideoReader("/path/to/video.avi")
    filename = Format("/output/path/frame_#{}.png", frame_number)
    ImageWriter(filename, image)

p.run()
