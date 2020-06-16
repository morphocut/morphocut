import re

import numpy as np
import pytest
import skimage.io

from morphocut import Pipeline
from morphocut.file import Glob
from morphocut.image import (
    ExtractROI,
    FindRegions,
    Gray2RGB,
    ImageReader,
    ImageWriter,
    RescaleIntensity,
    RGB2Gray,
    ThresholdConst,
)
from morphocut.stream import Unpack


def test_ThresholdConst():
    images = [skimage.data.camera(), np.zeros((10, 10), np.uint8) + 255]
    with Pipeline() as pipeline:
        image = Unpack(images)
        mask = ThresholdConst(image, 255)

    objects = list(pipeline.transform_stream())

    assert not objects[1][mask].any()


def test_RescaleIntensity():
    image = skimage.data.camera()
    with Pipeline() as pipeline:
        result = RescaleIntensity(image, in_range=(0, 200), dtype=np.uint8)

    pipeline.run()


@pytest.mark.parametrize("warn_empty", [True, False, "foo"])
def test_FindRegions(warn_empty, recwarn):
    images = [skimage.data.camera(), np.zeros((10, 10), np.uint8) + 255]
    with Pipeline() as pipeline:
        image = Unpack(images)
        mask = ThresholdConst(image, 255)
        result = FindRegions(mask, image, 0, 100, padding=10, warn_empty=warn_empty)

    pipeline.run()

    if warn_empty:
        w = recwarn.pop(UserWarning)
        assert re.search(r"^(Image|foo) did not contain any objects.$", str(w.message))


def test_ExtractROI():
    with Pipeline() as pipeline:
        image = Unpack([skimage.data.camera()])
        mask = ThresholdConst(image, 255)
        regions = FindRegions(mask, image)
        ExtractROI(image, regions)
        ExtractROI(image, regions, 0.5)

    pipeline.run()


def test_ImageWriter(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "new.jpg"
    image = skimage.data.camera()
    with Pipeline() as pipeline:
        result = ImageWriter(p, image)

    pipeline.run()


def test_ImageReader(data_path):
    d = data_path / "images/test_image_3.png"
    with Pipeline() as pipeline:
        image = ImageReader(d)

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
