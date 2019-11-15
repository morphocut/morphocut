from morphocut.image import FindRegions
from morphocut import Call, Pipeline
from morphocut.contrib.zooprocess import CalculateZooProcessFeatures
from tests.helpers import BinaryBlobs, NoiseImage
import numpy as np
from morphocut.stream import Unpack


def test_CalculateZooProcessFeatures():
    with Pipeline() as p:
        i = Unpack(range(10))
        mask = BinaryBlobs()
        image = NoiseImage(mask.shape)

        regionprops = FindRegions(mask, image)

        CalculateZooProcessFeatures(regionprops)

    p.run()
