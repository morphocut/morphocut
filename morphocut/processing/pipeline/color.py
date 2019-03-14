import cv2 as cv
from morphocut.processing.pipeline import SimpleNodeBase


class Gray2BGR(SimpleNodeBase):
    def process(self, facet):
        image = facet["image"]

        return {
            "image": cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        }


class BGR2Gray(SimpleNodeBase):
    def process(self, facet):
        image = facet["image"]

        if len(image.shape) != 3:
            raise ValueError("image.shape != 3 in {!r}".format(self))

        try:
            converted = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        except cv.error:
            print(image.dtype, image.shape)
            raise

        return {
            "image": converted
        }
