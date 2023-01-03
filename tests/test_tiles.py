import itertools
from morphocut.core import Call, Pipeline
from morphocut.tiles import TiledPipeline
import skimage.data
from morphocut.stream import Unpack
import numpy as np


def assert_(cond, msg):
    assert cond, msg


def assert_shape(arr, shape):
    assert arr.shape[: len(shape)] == shape, f"{arr.shape} vs. {shape}"


def test_TiledPipeline():
    images = [
        # skimage.data.camera(),
        skimage.data.coins()
    ]
    names = [
        # "camera",
        "coins"
    ]

    with Pipeline() as p:
        image = Unpack(images)
        mask = image > 128
        with TiledPipeline((128, 128), image, mask):
            image_copy = image.copy()
            Call(assert_shape, image, (128, 128))
            Call(assert_shape, mask, (128, 128))
            pass

        # Call(np.testing.assert_equal, image, image_copy)

    result = list(p.transform_stream())

    assert len(result) == len(images)

    for obj, image_, name_ in zip(result, images, names):
        # Check that the identity of the incoming variables was not changed
        assert obj[image] is image_, f"{name_} not identical"

        # Check that tiles are properly stitched
        np.testing.assert_equal(obj[image_copy], image_, err_msg=f"{name_} not equal")
