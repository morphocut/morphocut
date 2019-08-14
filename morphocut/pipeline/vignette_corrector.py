import numpy as np
from skimage import img_as_float32
from skimage.exposure import rescale_intensity
from skimage.color import rgb2gray

import morphocut.functional as F
from morphocut.pipeline import SimpleNodeBase


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
            grey_img = rgb2gray(img)

        flat_image = F.calculate_flat_image(grey_img)

        # Add a dimension for multichannel input images
        if len(img.shape) == 3:
            flat_image = flat_image[:, :, np.newaxis]

        corrected_img = img / flat_image

        return {
            "image": corrected_img
        }
