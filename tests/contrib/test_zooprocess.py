import numpy as np
import pytest

from morphocut import Pipeline
from morphocut.contrib.zooprocess import CalculateZooProcessFeatures
from morphocut.image import FindRegions
from morphocut.stream import Unpack
from tests.helpers import BinaryBlobs, NoiseImage


@pytest.mark.parametrize("prefix", [None, "object_"])
def test_CalculateZooProcessFeatures(prefix):
    with Pipeline() as p:
        i = Unpack(range(10))
        mask = BinaryBlobs()
        image = NoiseImage(mask.shape)

        regionprops = FindRegions(mask, image)

        CalculateZooProcessFeatures(regionprops, prefix=prefix)

    p.run()
