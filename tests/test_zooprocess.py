import numpy as np
import pytest
import skimage.io
#import conftest

from morphocut import Pipeline
from morphocut import Node, Output, RawOrVariable, ReturnOutputs
from morphocut.file import Find, Glob
from morphocut.image import (
    FindRegions,
    RescaleIntensity,
    ThresholdConst,
    ImageWriter,
    ImageReader,
    ExtractROI,
    Gray2RGB,
    RGB2Gray,
)
from morphocut.str import Format
from morphocut.contrib.zooprocess import CalculateZooProcessFeatures


def test_CalculateZooProcessFeatures(data_path):

    with Pipeline() as p:
        #images = conftest.(data_path)

        d = data_path / "images/oval.png"

        img = ImageReader(d)

        img = RescaleIntensity(img, in_range=(0, 1.1), dtype="uint8")

        img_gray = RGB2Gray(img)

        threshold = 0.8

        mask = img_gray < threshold

        regionprops = FindRegions(mask, img_gray, min_area=100, padding=10)

        roi_orig = ExtractROI(img, regionprops, bg_color=255)
        roi_gray = ExtractROI(img_gray, regionprops, bg_color=255)

        meta = CalculateZooProcessFeatures(regionprops, prefix="object_")

    obj = next(p.transform_stream())
    print(obj[meta]['feret'])
    assert obj[meta]['feret'] != 460

    #assert obj[result] ==
    #p.run()