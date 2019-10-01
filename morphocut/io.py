import numpy as np
import PIL

from morphocut.graph import Node


class ImageReader(Node):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def transform_stream(self, stream):
        for obj in stream:
            path = self.prepare_input(obj, "path")

            image = np.array(PIL.Image.open(path))

            yield self.prepare_output(obj, image)


class ImageWriter(Node):
    def __init__(self, path, image):
        super().__init__()
        self.path = path
        self.image = image

    def transform_stream(self, stream):
        for obj in stream:
            path, image = self.prepare_input(obj, ("path", "image"))

            print(path)

            img = PIL.Image.fromarray(image)
            img.save(path)

            yield obj
