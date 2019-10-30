from morphocut import Pipeline
from morphocut.file import Find, Glob

import pytest

def test_Find(data_path):
    d = data_path / "images"
    with Pipeline() as pipeline:
        result = Find(d, [".png"])

    stream = pipeline.transform_stream()
    pipeline.run()

def test_Glob(data_path):
    d = data_path / "images/test_image_3.png"
    with Pipeline() as pipeline:
        result = Glob(d, True)

    stream = pipeline.transform_stream()
    pipeline.run()
