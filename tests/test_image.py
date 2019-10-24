from morphocut import Pipeline
from morphocut.image import ThresholdConst, Rescale, ImageWriter

import numpy as np
import skimage.io

import pytest

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

#TODO: Confused in fmt and meta. 
@pytest.mark.xfail(strict=True)
def test_ImageWriter():
    image_root = "/tests/images/"
    fmt = "{ext}"
    meta = ...
    image = skimage.data.camera()
    with Pipeline() as pipeline:
        result = ImageWriter(image_root, fmt, image, meta)

    stream = pipeline.transform_stream()
    pipeline.run()

#transform_stream method of FindRegion has an error because of which test fails
#Also ExtractROI class is dependent on FindRegion so couldn't write test of it either


'''
@pytest.mark.xfail(strict=True)
def test_FindRegions():
    image = skimage.data.camera()
    mask = ThresholdConst(image, 255)
    with Pipeline() as pipeline:
        result = FindRegions(mask, image, 0, 100, padding=10)

    stream = pipeline.transform_stream()
    pipeline.run()
'''

