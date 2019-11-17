
"""
Operations for annotating images.
"""
import numpy as np
from skimage import img_as_ubyte, img_as_float
from skimage.color import gray2rgb, rgb2gray
from skimage.morphology import binary_dilation, disk
import skimage.exposure
import skimage.io
import skimage.measure
import skimage.segmentation

from morphocut import Node, Output, RawOrVariable, ReturnOutputs

@ReturnOutputs
@Output("contour")
class DrawContours(Node):
    """
    Dilate the image and make boundary around the dilated image and outline pixels in the background around 
    object boundaries.

    .. note::
        Uses the skimage library :py:func:`skimage.morphology.binary_dilation` and :py:func:`skimage.segmentation.find_boundaries`.

    Args:
        image (np.ndarray or Variable[np.ndarray]): An image to be contoured.
        mask (np.ndarray or Variable[np.ndarray]): Mask of an image.
        color (tuple): Value of RGB colors as a tuple.
        dilate_rel (int): Relative Dilation value.
        dilate_abs (int): Absolute Dilation value.

    Returns:
        Variable[np.ndarray]: Contoured image.
    """

    def __init__(self, image, mask, color, dilate_rel=0, dilate_abs=0):
        super().__init__()
        self.image = image
        self.mask = mask
        self.color = color
        self.dilate_rel = dilate_rel
        self.dilate_abs = dilate_abs

    def transform(self, image, mask):
        # Calculate radius of the dilation
        area = np.sum(mask)
        radius = self.dilate_rel * np.sqrt(area) + self.dilate_abs

        # Dilate mask
        dilated_mask = binary_dilation(
            mask,
            disk(radius))

        # Draw contours
        boundary_mask = skimage.segmentation.find_boundaries(dilated_mask, mode="outer")

        contour_image = skimage.color.gray2rgb(img_as_ubyte(image, True))
        #color = (0, 255, 0)

        contour_image[boundary_mask] = self.color

        return contour_image

@ReturnOutputs
@Output("parent_contour")
class DrawContoursOnParent(Node):
    """
    Dilate the image and make boundary around the dilated image and outline pixels in the background around 
    object boundaries.

    .. note::
        Uses the skimage library :py:func:`skimage.morphology.binary_dilation` and :py:func:`skimage.segmentation.find_boundaries`.

    Args:
        image (np.ndarray or Variable[np.ndarray]): An image to be contoured.
        mask (np.ndarray or Variable[np.ndarray]): Mask of an image.
        color (tuple): Value of RGB colors as a tuple.
        dilate_rel (int): Relative Dilation value.
        dilate_abs (int): Absolute Dilation value.

    Returns:
        Variable[np.ndarray]: Contoured image.
    """

    def __init__(self, image, mask, parent_slice, color, dilate_rel=0, dilate_abs=0):
        super().__init__()
        self.image = image
        self.mask = mask
        #self.output = output
        self.parent_slice = parent_slice
        self.color = color
        self.dilate_rel = dilate_rel
        self.dilate_abs = dilate_abs

    def transform_stream(self, stream):
        for obj in stream:
            parent = self.prepare_input(obj, ("parent"))

            try:
                parent_img = self.prepare_input(parent, ("output"))
            except KeyError:
                parent_img = skimage.color.gray2rgb(img_as_ubyte(
                    self.image, force_copy=True))
                self.prepare_output(parent, parent_img)

            #mask = self.prepare_input(obj, ("mask"))

            # Calculate radius of the dilation
            area = np.sum(self.mask)
            radius = self.dilate_rel * np.sqrt(area) + self.dilate_abs

            # Dilate mask
            dilated_mask = binary_dilation(
                self.mask,
                disk(radius))

            # Draw contours
            boundary_mask = skimage.segmentation.find_boundaries(dilated_mask, mode="outer")

            #color = (0, 255, 0)

            parent_img[self.parent_slice][boundary_mask] = self.color

            yield self.prepare_output(parent_img)
