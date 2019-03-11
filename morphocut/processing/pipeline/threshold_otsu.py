from skimage import img_as_ubyte
from skimage.color import gray2rgb
from skimage.filters import threshold_otsu

import morphocut.processing.functional as proc
from morphocut.processing.pipeline import NodeBase, SimpleNodeBase


class ThresholdOtsu(SimpleNodeBase):
    def process(self, facet):
        image = facet["image"]

        thresh = threshold_otsu(image)
        mask = image < thresh

        return {
            "image": mask
        }
