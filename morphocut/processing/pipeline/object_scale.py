import numpy as np
import math
import cv2

from morphocut.processing.pipeline import SimpleNodeBase


class ObjectScale(SimpleNodeBase):
    def process(self, facet):
        image = facet["image"]

        width = image.shape[1]
        scale_x = 5
        # ~2400 pixels should be 14mm big. That means 0.17 pixels are one micrometer. Therefore 17 pixels are 100 micrometers = 0.1 milimeters
        scale_width = 17
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
        image[scale_y, scale_x:scale_x +
              scale_width] = np.zeros(3, dtype=np.int8)
        # draw scale line hooks
        image[scale_y - 3:scale_y, scale_x] = np.zeros(3, dtype=np.int8)
        image[scale_y - 3:scale_y, scale_x +
              scale_width] = np.zeros(3, dtype=np.int8)
        # print(image)

        cv2.putText(image, '0.1mm', (5, (image.shape[0] + scale_y) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.25, (0, 0, 0), lineType=cv2.LINE_AA)

        return {
            "image": image
        }
