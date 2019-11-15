import numpy as np

from morphocut import Pipeline
from morphocut.contrib.zooprocess import CalculateZooProcessFeatures
from morphocut.image import FindRegions
from morphocut.stream import Unpack
from tests.helpers import BinaryBlobs, NoiseImage


def test_CalculateZooProcessFeatures():
    with Pipeline() as p:
        i = Unpack(range(10))
        mask = BinaryBlobs()
        image = NoiseImage(mask.shape)

        regionprops = FindRegions(mask, image)

        CalculateZooProcessFeatures(regionprops)

    p.run()
