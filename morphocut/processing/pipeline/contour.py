from skimage import img_as_ubyte
from skimage.color import gray2rgb
import cv2 as cv
import morphocut.processing.functional as proc
from morphocut.processing.pipeline import NodeBase, SimpleNodeBase


class ContourTransform(SimpleNodeBase):
    def process(self, obj):
        image = obj["facets"][self.input_facet]["image"]

        obj["facets"][self.output_facet] = {
            "image": gray2rgb(image)
        }

        return obj
