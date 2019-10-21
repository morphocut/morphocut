import numpy as np
import PIL

from morphocut import Node, RawOrVariable, ReturnOutputs


@ReturnOutputs
class ImageReader(Node):
    def __init__(self, fp: RawOrVariable):
        super().__init__()
        self.fp = fp

    def transform_stream(self, stream):
        for obj in stream:
            fp = self.prepare_input(obj, "fp")

            image = np.array(PIL.Image.open(fp))

            yield self.prepare_output(obj, image)


@ReturnOutputs
class ImageWriter(Node):
    def __init__(self, fp: RawOrVariable, image: RawOrVariable):
        super().__init__()
        self.fp = fp
        self.image = image

    def transform_stream(self, stream):
        for obj in stream:
            fp, image = self.prepare_input(obj, ("fp", "image"))

            print(fp)

            img = PIL.Image.fromarray(image)
            img.save(fp)

            yield obj
