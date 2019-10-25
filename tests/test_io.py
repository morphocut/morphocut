import pytest
import skimage.io

from morphocut import Pipeline
from morphocut.file import Glob
from morphocut.io import ImageWriter, ImageReader


def test_ImageWriter(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "new.jpg"
    image = skimage.data.camera()
    with Pipeline() as pipeline:
        result = ImageWriter(p, image)

    stream = pipeline.transform_stream()
    pipeline.run()

def test_ImageReader():
    with Pipeline() as pipeline:
        path = Glob("/morphocut/tests/images/test_image_3.png")
        image = ImageReader(path)

    stream = pipeline.transform_stream()
    pipeline.run()