from morphocut.processing.pipeline import NodeBase
import cv2 as cv


class Importer(NodeBase):
    """
    An importing node. Imports images based on the filepath
    """

    def __call__(self, input=None):
        wp = input.__next__()
        while wp:
            img = cv.imread(wp)
            yield dict(src=img, object_id=wp.split('/')[-1].split('.')[1])
            wp = input.__next__()
