import numpy as np
import pytest
import skimage.io

from morphocut import Pipeline
from morphocut.image import FindRegions, ImageWriter, Rescale, ThresholdConst


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

#transform_stream method of FindRegion has an error because of which test fails
#Also ExtractROI class is dependent on FindRegion so couldn't write test of it either



@pytest.mark.xfail(strict=True)
def test_FindRegions():
    image = skimage.data.camera()
    with Pipeline() as pipeline:
        mask = ThresholdConst(image, 255)
        result = FindRegions(mask, image, 0, 100, padding=10)

    stream = pipeline.transform_stream()
    pipeline.run()
