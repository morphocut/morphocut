import numpy as np
import math
import cv2

from morphocut.pipeline import NodeBase


class ObjectScale(NodeBase):
    '''
    Parameters
    ----------
    input_facets : list
        The facets onto which a scale should be drawn.
    output_facets : list
        The facets where the outputs should be saved. This corresponds to the input_facets element-wise.
    pixels_per_mm : int
        The number of pixels per milimeter in the image.
    scale_size : float
        The size of the scale that should be drawn in milimeter.
    '''

    def __init__(self, input_facets, output_facets, pixels_per_mm=200, scale_size=0.1):
        self.scale_size = scale_size
        self.pixels_per_mm = pixels_per_mm
        self.input_facets = input_facets
        self.output_facets = output_facets
        if not len(self.input_facets) == len(self.output_facets):
            raise ValueError(
                "The number of input facets and output facets must be equal.")

    def __call__(self, input=None):
        for obj in input:
            for i in range(len(self.input_facets)):
                input_facet = self.input_facets[i]
                output_facet = self.output_facets[i]

                obj["facets"][output_facet] = self.process(
                    obj["facets"][input_facet])
            yield obj

    def process(self, facet):
        image = facet["image"]

        width = image.shape[1]
        scale_x = 5

        scale_width = int(self.pixels_per_mm * self.scale_size)
        scale_y = -5

        complete_width = max(width, scale_x + scale_width + 1)

        # pad image if scale is too big
        if (complete_width > width):
            height = image.shape[0]
            pad_width = math.ceil((complete_width - width) / 2)
            pad_array = np.ones((height, pad_width, 3), dtype=np.int8) * 255
            pad_array_copy = np.ones(
                (height, pad_width, 3), dtype=np.int8) * 255
            image = np.append(pad_array_copy, image, axis=1)
            image = np.append(image, pad_array, axis=1)

        width = image.shape[1]

        pad_array = np.ones((31, width, 3), dtype=np.int8) * 255
        image = np.append(image, pad_array, axis=0)

        # draw scale line
        image[scale_y, scale_x:scale_x
              + scale_width] = np.zeros(3, dtype=np.int8)
        # draw scale line hooks
        image[scale_y - 3:scale_y, scale_x] = np.zeros(3, dtype=np.int8)
        image[scale_y - 3:scale_y, scale_x
              + (scale_width - 1)] = np.zeros(3, dtype=np.int8)
        # print(image)

        cv2.putText(image, '{}mm'.format(self.scale_size), (5, (image.shape[0] + scale_y) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.25, (0, 0, 0), lineType=cv2.LINE_AA)

        return {
            "image": image
        }
