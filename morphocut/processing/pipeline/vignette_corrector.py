import numpy as np
from skimage import img_as_float32
from skimage.exposure import rescale_intensity

import cv2 as cv
import morphocut.processing.functional as proc
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

        grey_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        flat_image = proc.calculate_flat_image(grey_img)
        corrected_img = img / flat_image[:, :, np.newaxis]

        # TODO: img_as_float32 is required, because openCV cannot handle 64bit images, which is unfortunate
        corrected_img = img_as_float32(rescale_intensity(corrected_img))

        return {
            "image": corrected_img
        }
