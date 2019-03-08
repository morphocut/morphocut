from skimage import img_as_ubyte
from skimage.color import gray2rgb
import cv2 as cv
import morphocut.processing.functional as proc
from morphocut.processing.pipeline import NodeBase, SimpleNodeBase


class ThresholdOtsu(SimpleNodeBase):
    def process(self, obj):
        image = obj["facets"][self.input_facet]["image"]

        # Segment foreground objects from background objects using thresholding with the otsu method
        _, mask = cv.threshold(
            image, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        mask = cv.bitwise_not(mask)

        obj["facets"][self.output_facet] = {
            "image": mask
        }

        return obj
