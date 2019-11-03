from morphocut import Pipeline
from morphocut.pipeline import Gray2RGB, RGB2Gray

import pytest
import numpy as np
import skimage.io

def test_Gray2RGB():
    image = skimage.data.camera()
    with Pipeline() as pipeline:
        result = Gray2RGB(image)

    stream = pipeline.transform_stream()
    pipeline.run()

def test_RGB2Gray():
    image = skimage.data.camera()
    with Pipeline() as pipeline:
        result = RGB2Gray(image)

    stream = pipeline.transform_stream()
    pipeline.run()