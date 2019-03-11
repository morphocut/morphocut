
import numpy as np
from skimage import img_as_ubyte
from skimage.morphology import binary_dilation, disk

import cv2 as cv
from morphocut.processing.pipeline.base import NodeBase


class DrawContours(NodeBase):

    def __init__(self, image_facet, mask_facet, output_facet, dilate_rel=0, dilate_abs=0):
        self.image_facet = image_facet
        self.mask_facet = mask_facet
        self.output_facet = output_facet
        self.dilate_rel = dilate_rel
        self.dilate_abs = dilate_abs

    def __call__(self, input=None):
        for obj in input:
            img = obj["facets"][self.image_facet]["image"]
            mask = obj["facets"][self.mask_facet]["image"]

            # Calculate radius of the dilation
            area = np.sum(mask)
            radius = self.dilate_rel * np.sqrt(area) + self.dilate_abs

            # Dilate mask
            dilated_mask = binary_dilation(
                mask,
                disk(radius))

            # Draw contours
            _, contours, _ = cv.findContours(img_as_ubyte(dilated_mask), 1, 2)

            contour_image = img_as_ubyte(img, True)
            color = (0, 255, 0)

            cv.drawContours(contour_image, contours, -1, color, 1)

            obj["facets"][self.output_facet] = {
                "image": contour_image
            }

            yield obj


class FadeBackground(NodeBase):

    def __init__(self, image_facet, mask_facet, output_facet, alpha=0.5):
        self.image_facet = image_facet
        self.mask_facet = mask_facet
        self.output_facet = output_facet
        self.alpha = alpha

    def __call__(self, input=None):
        for obj in input:
            img = obj["facets"][self.image_facet]["image"]
            mask = obj["facets"][self.mask_facet]["image"]

            bg_color = np.max(img)
            bg_img = np.ones_like(img) * bg_color

            # Combine background and foreground
            result_img = self.alpha * bg_img + (1 - self.alpha) * img

            # Past foreground
            result_img[mask] = img[mask]

            obj["facets"][self.output_facet] = {
                "image": result_img
            }

            yield obj
