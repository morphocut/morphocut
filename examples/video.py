"""Process video data using MorphoCut and store frames as individual frames."""

from morphocut import Pipeline
from morphocut.image import ImageWriter
from morphocut.pims import VideoReader
from morphocut.str import Format

with Pipeline() as p:
    # Read individual frames from a video file
    frame = VideoReader("/path/to/video.avi")

    # Format filename
    filename = Format("/output/path/frame_#{}.png", frame.frame_no)

    # Write individual frames as image files
    ImageWriter(filename, frame)

p.run()
