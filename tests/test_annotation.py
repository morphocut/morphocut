import numpy as np
import pytest
import skimage.io

from morphocut import Pipeline
from morphocut.file import Glob
from morphocut.annotation import (
    DrawContours,
    DrawContoursOnParent,
)
from morphocut.image import ThresholdConst


def test_ThresholdConst():
    image = skimage.data.camera()
    with Pipeline() as pipeline:
        mask = ThresholdConst(image, 255)
        result = DrawContours(image, mask)

    stream = pipeline.transform_stream()
    pipeline.run()

