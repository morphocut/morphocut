
"""
Operations for annotating images.
"""
import numpy as np
from skimage import img_as_ubyte, img_as_float
from skimage.morphology import binary_dilation, disk

import cv2 as cv
from morphocut.pipeline.base import NodeBase


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


class DrawContoursOnParent(NodeBase):

    def __init__(self, image_facet, mask_facet, output_facet, dilate_rel=0, dilate_abs=0):
        self.image_facet = image_facet
        self.mask_facet = mask_facet
        self.output_facet = output_facet
        self.dilate_rel = dilate_rel
        self.dilate_abs = dilate_abs

    def __call__(self, input=None):
        for obj in input:
            parent, parent_slice = obj["parent"], obj["parent_slice"]

            try:
                parent_img = parent["facets"][self.output_facet]["image"]
            except KeyError:
                parent_img = img_as_ubyte(
                    parent["facets"][self.image_facet]["image"], force_copy=True)
                parent["facets"][self.output_facet] = {
                    "image": parent_img
                }

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

            color = (0, 255, 0)

            cv.drawContours(parent_img[parent_slice], contours, -1, color, 1)

            yield obj


class FadeBackground(NodeBase):
    """
    Fade the background around an object using its mask.

    Parameters:
        image_facet: This image gets transformed.
        mask_facet: This image contains the mask.
        output_facet: The result comes here.

        alpha: Amount of background reduction (0=full background, 1=no background)
        bg_color: Color of the background (0=black, 1=white). Can also be a tuple of shape (c).
    """

    def __init__(self, image_facet, mask_facet, output_facet, alpha=0.5, bg_color=1.0):
        self.image_facet = image_facet
        self.mask_facet = mask_facet
        self.output_facet = output_facet
        self.alpha = alpha
        self.bg_color = np.array(bg_color)

    def __call__(self, input=None):
        for obj in input:
            img = obj["facets"][self.image_facet]["image"]
            mask = obj["facets"][self.mask_facet]["image"]

            # convert img to float if needed
            # (because otherwise the following would not work)
            if not np.issubdtype(img.dtype, np.floating):
                img = img_as_float(img)

            # Combine background and foreground
            result_img = self.alpha * self.bg_color + (1 - self.alpha) * img

            # Paste foreground
            result_img[mask] = img[mask]

            obj["facets"][self.output_facet] = {
                "image": result_img
            }

            yield obj
