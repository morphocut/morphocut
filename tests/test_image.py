import numpy as np
import pytest
import skimage.io

from morphocut import Pipeline
from morphocut.file import Glob
from morphocut.image import FindRegions, Rescale, ThresholdConst, ImageWriter, ImageReader, ExtractROI, Gray2RGB, RGB2Gray


def test_ThresholdConst():
    image = skimage.data.camera()
    with Pipeline() as pipeline:
        result = ThresholdConst(image, 256)

    stream = pipeline.transform_stream()
    pipeline.run()

def test_Rescale():
    image = skimage.data.camera()
    with Pipeline() as pipeline:
        result = Rescale(image, in_range=(0, 200), dtype=np.uint8)

    stream = pipeline.transform_stream()
    pipeline.run()

def test_FindRegions():
    image = skimage.data.camera()
    with Pipeline() as pipeline:
        mask = ThresholdConst(image, 255)
        result = FindRegions(mask, image, 0, 100, padding=10)

    stream = pipeline.transform_stream()
    pipeline.run()

def test_ExtractROI():
    image = skimage.data.camera()
    with Pipeline() as pipeline:
        mask = ThresholdConst(image, 255)
        regions = FindRegions(mask, image, 0, 100, padding=10)
        result = ExtractROI(image, regions)

    stream = pipeline.transform_stream()
    pipeline.run()

def test_ImageWriter(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "new.jpg"
    image = skimage.data.camera()
    with Pipeline() as pipeline:
        result = ImageWriter(p, image)

    stream = pipeline.transform_stream()
    pipeline.run()

def test_ImageReader(data_path):
    d = data_path / "images/test_image_3.png"
    with Pipeline() as pipeline:
        image = ImageReader(d)

    stream = pipeline.transform_stream()
    pipeline.run()

def test_Gray2RGB():
    image = skimage.data.camera()
    with Pipeline() as pipeline:
        result = Gray2RGB(image)

    stream = pipeline.transform_stream()
    obj = next(stream)

    assert obj[result].ndim == 3
    assert obj[result].shape[-1] == 3

def test_RGB2Gray():
    image = skimage.data.astronaut()
    with Pipeline() as pipeline:
        result = RGB2Gray(image)

    stream = pipeline.transform_stream()
    obj = next(stream)

    assert obj[result].ndim == 2