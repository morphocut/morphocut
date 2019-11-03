from morphocut import Node, Output
from morphocut.pipeline import SimpleNodeBase
from skimage.color import gray2rgb, rgb2gray


class Gray2RGB(Node):
    def __init__(self, image):
        self.image = image

    def transform(self, image):
        return gray2rgb(image)


class RGB2Gray(SimpleNodeBase):
    def __init__(self, image):
        self.image = image

    def transform(self, image):
        if len(image.shape) != 3:
            raise ValueError("image.shape != 3 in {!r}".format(self))

        return rgb2gray(image)