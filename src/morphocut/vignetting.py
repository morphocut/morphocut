from typing import Any, List, Mapping

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import binary_dilation, binary_erosion, dilation, disk

from morphocut import Node, Output, RawOrVariable, ReturnOutputs


@ReturnOutputs
@Output("result")
class VignettingCorrector(Node):
    """Remove the vignette effect from an image."""

    def __init__(self, image):
        super().__init__()
        self.image = image

    def transform(self, image: RawOrVariable):

        if len(image.shape) == 2:
            grey_img = image
        elif image.shape[-1] == 1:
            grey_img = image[:-1]
        else:
            grey_img = rgb2gray(image)

        flat_image = calculate_flat_image(grey_img)

        # Add a dimension for multichannel input images
        if len(image.shape) == 3:
            flat_image = flat_image[:, :, np.newaxis]

        corrected_img = image / flat_image

        return corrected_img


def calculate_flat_image(img: np.ndarray) -> np.ndarray:
    """
    Calculate a flat background image by removing dark objects and max-filtering.

    Parameters
    ==========
    img (ndarray of shape [h,w]): Graylevel image.
    """

    # Blur image to make subsequent dilation more robust
    # TODO: Optimal sigma
    img = gaussian(img, 3)

    # Find obvious objects and create a mask of valid background
    thr = threshold_otsu(img)
    valid_mask = img > thr

    # Shrink valid regions to avoid border regions of objects
    valid_mask = binary_erosion(valid_mask, disk(16))

    # Greyscale morphological dilation to remove dark structures
    # The larger the selem, the longer it takes
    img = dilation(img, disk(16))

    invalid_mask = np.bitwise_not(valid_mask)

    # Only keep valid borders of objects to avoid computational overhead in interpolation
    valid_mask &= binary_dilation(invalid_mask, disk(1))

    # Interpolate masked image

    # Fit interpolator with valid image parts
    coords = np.array(np.nonzero(valid_mask)).T
    values = img[valid_mask]
    interpolator = LinearNDInterpolator(coords, values)

    # Interpolate invalid regions
    img_filled = img.copy()
    coords_invalid = np.array(np.nonzero(invalid_mask)).T
    img_filled[invalid_mask] = interpolator(coords_invalid)

    # Fill regions where interpolation failed with original values
    mask = np.isnan(img_filled)
    img_filled[mask] = img[mask]

    # Finally smooth result
    img_filled = gaussian(img_filled, 64)

    return img_filled
