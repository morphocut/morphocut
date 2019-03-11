
from morphocut.processing.pipeline.base import NodeBase


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
