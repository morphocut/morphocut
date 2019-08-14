from morphocut.pipeline import SimpleNodeBase
from skimage.color import rgb2gray, gray2rgb


class Gray2RGB(SimpleNodeBase):
    def process(self, facet):
        image = facet["image"]

        return {
            "image": gray2rgb(image)
        }


class RGB2Gray(SimpleNodeBase):
    def process(self, facet):
        image = facet["image"]

        if len(image.shape) != 3:
            raise ValueError("image.shape != 3 in {!r}".format(self))

        converted = rgb2gray(image)

        return {
            "image": converted
        }
