import pickle
import re

import numpy as np
import pytest
import skimage.io
from numpy.testing import assert_equal
from skimage.measure._regionprops import RegionProperties as RegionProperties_orig

from morphocut import Pipeline
from morphocut.image import (
    ExtractROI,
    FindRegions,
    Gray2RGB,
    ImageProperties,
    ImageReader,
    ImageWriter,
    RegionProperties,
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

    image = skimage.data.camera()
    mask = image > 128
    with Pipeline() as pipeline:
        result = FindRegions(mask, image, min_intensity=100)

        num_yielded_objects = 0
        for obj in pipeline.transform_stream():
            num_yielded_objects += 1
        assert num_yielded_objects > 0


def test_ExtractROI():
    with Pipeline() as pipeline:
        image = Unpack([skimage.data.camera()])
        mask = ThresholdConst(image, 255)
        regions = FindRegions(mask, image)
        ExtractROI(image, regions)
        ExtractROI(image, regions, 0.5)

    pipeline.run()


def test_ImageProperties():
    with Pipeline() as pipeline:
        image = Unpack([skimage.data.camera()])
        mask = ThresholdConst(image, 255)
        region = ImageProperties(mask, image)
        image2 = ExtractROI(image, region, 0)

    for obj in pipeline.transform_stream():
        assert_equal(obj[image], obj[image2])


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

    with Pipeline() as pipeline:
        image = ImageReader(d, pil_mode="L")
    pipeline.run()


@pytest.mark.parametrize("keep_dtype", [True, False])
def test_Gray2RGB(keep_dtype):
    image = skimage.data.camera()
    with Pipeline() as pipeline:
        result = Gray2RGB(image, keep_dtype)

    stream = pipeline.transform_stream()
    obj = next(stream)
    result = obj[result]

    assert result.ndim == 3
    assert result.shape[-1] == 3

    if keep_dtype:
        assert image.dtype == result.dtype


@pytest.mark.parametrize("keep_dtype", [True, False])
def test_RGB2Gray(keep_dtype):
    image = skimage.data.astronaut()
    with Pipeline() as pipeline:
        result = RGB2Gray(image, keep_dtype)

    stream = pipeline.transform_stream()
    obj = next(stream)
    result = obj[result]

    assert result.ndim == 2
    assert result.shape == image.shape[:-1]

    if keep_dtype:
        assert image.dtype == result.dtype

    # invalid 4D image
    invalid_image = np.random.rand(200, 200, 3, 3)
    with Pipeline() as pipeline:
        result = RGB2Gray(invalid_image, keep_dtype)

    stream = pipeline.transform_stream()

    with pytest.raises(ValueError, match="image.ndim != 3"):
        next(stream)


def regionproperties_to_dict(rprop):
    return {k: rprop[k] for k in rprop}


@pytest.mark.parametrize("intensity_image", [True, False])
@pytest.mark.parametrize("cache", [True, False])
def test_RegionProperties(intensity_image, cache):
    image = skimage.data.camera()

    cargs = (
        (slice(128, 256), slice(128, 256)),
        1,
        image < 128,
        image if intensity_image else None,
        cache,
    )

    rprops = RegionProperties(*cargs)
    rprops_orig = RegionProperties_orig(*cargs)

    np.testing.assert_equal(
        regionproperties_to_dict(rprops), regionproperties_to_dict(rprops_orig)
    )

    if intensity_image:
        assert rprops.intensity_image.base is None
