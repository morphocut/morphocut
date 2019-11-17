import pytest
import skimage.io

from morphocut import Pipeline
from morphocut.annotation import (
    DrawContours,
    DrawContoursOnParent,
)
from morphocut.image import ThresholdConst, FindRegions, ExtractROI


def test_DrawContours():
    image = skimage.data.camera()
    with Pipeline() as pipeline:
        mask = ThresholdConst(image, 255)
        result = DrawContours(image, mask, (0, 255, 0))

    stream = pipeline.transform_stream()
    pipeline.run()

def test_DrawContoursOnParent():
    image = skimage.data.camera()
    with Pipeline() as pipeline:
        mask = ThresholdConst(image, 255)
        regions = FindRegions(mask, image, 0, 100, padding=10)
        #extracted_images = ExtractROI(image, mask, regions)
        result = DrawContoursOnParent(image, mask, regions, (0, 255, 0))

    stream = pipeline.transform_stream()
    pipeline.run()

