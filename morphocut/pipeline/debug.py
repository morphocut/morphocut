
import os

from skimage import img_as_ubyte

import cv2 as cv
from morphocut.pipeline.base import NodeBase


class PrintFacettes(NodeBase):

    def __call__(self, input):
        for obj in input:
            print("object_id:", obj["object_id"])

            for k, facet in obj["facets"].items():
                print(k)
                if "data" in facet:
                    print("data", facet["data"])
                if "image" in facet:
                    image = facet["image"]
                    print("image", image.shape, image.dtype)

            yield obj


class DumpImages(NodeBase):
    def __init__(self, path, image_facet, img_ext=".jpg"):
        self.path = path
        self.image_facet = image_facet
        self.img_ext = img_ext

    def __call__(self, input):
        for obj in input:
            object_id = obj["object_id"]
            img = obj["facets"][self.image_facet]["image"]

            img_fn = os.path.join(
                self.path,
                "{}_{}{}".format(object_id, self.image_facet, self.img_ext))

            cv.imwrite(img_fn, img_as_ubyte(img))

            yield obj
