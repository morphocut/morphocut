import numpy as np
from skimage import img_as_float32
from skimage.exposure import rescale_intensity

import cv2 as cv
import morphocut.processing.functional as F
from morphocut.processing.pipeline import SimpleNodeBase


class VignetteCorrector(SimpleNodeBase):
    """
    A processing node. Removes the vignette effect from images.

    """

    def process(self, facet):
        """
        Take the image of the input facet and apply vignette correction
        """
        img = facet["image"]

        if len(img.shape) == 2:
            grey_img = img
        elif img.shape[-1] == 1:
            grey_img = img[:-1]
        else:
            grey_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        flat_image = F.calculate_flat_image(grey_img)

        # Add a dimension for multichannel input images
        if len(img.shape) == 3:
            flat_image = flat_image[:, :, np.newaxis]

        corrected_img = img / flat_image

        # TODO: img_as_float32 is required, because openCV cannot handle 64bit images, which is unfortunate
        corrected_img = img_as_float32(rescale_intensity(corrected_img))

        return {
            "image": corrected_img
        }
