"""
Operations for annotating images.
"""
import numpy as np
from skimage import img_as_ubyte
import skimage.color
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
    Draw the contours of an object using its mask.

    Args:
        image (np.ndarray or Variable): An image to be contoured.
        mask (np.ndarray or Variable): Binary mask of the object.
        color (tuple): Value of RGB colors as a tuple.
        dilate_rel (int): Relative Dilation value.
        dilate_abs (int): Absolute Dilation value.

    Returns:
        Variable[np.ndarray]: Contoured image.
    """

    def __init__(self, image, mask, color=None, dilate_rel=0, dilate_abs=0):
        super().__init__()
        self.image = image
        self.mask = mask
        self.color = color
        self.dilate_rel = dilate_rel
        self.dilate_abs = dilate_abs

    def transform(self, image, mask):

        if self.dilate_rel or self.dilate_abs:
            area = np.sum(mask)
            radius = self.dilate_rel * np.sqrt(area) + self.dilate_abs

            if mask.dtype == bool:
                mask = binary_dilation(mask, disk(radius))
            else:
                mask = skimage.segmentation.expand_labels(mask, radius)

        # FIXME: This returns bool
        boundary_mask = skimage.segmentation.find_boundaries(mask, mode="outer")

        if boundary_mask.dtype == bool:
            contour_image = skimage.color.gray2rgb(img_as_ubyte(image, True))
            contour_image[boundary_mask] = self.color
            return contour_image
        else:
            return skimage.color.label2rgb(
                boundary_mask, image, alpha=1.0, bg_label=0, bg_color=None
            )


@ReturnOutputs
@Output("parent_image_out")
class DrawContoursOnParent(Node):
    """
    Draw the contours of an object onto its parent image.

    Args:
        parent_image (np.ndarray or Variable): An image to be contoured.
        child_mask (np.ndarray or Variable): Mask of an image.
        parent_image_out (np.ndarray pr Variable): Image onto which the object will be drawn. This is passed
            by reference and will be altered in-place.
        parent_slice: Slice that locates the child object in the parent image.
        color (tuple): Value of RGB colors as a tuple.
        dilate_rel (int): Relative Dilation value.
        dilate_abs (int): Absolute Dilation value.

    .. warning::
        `parent_image_out` will be altered in-place. Pass a copy of the original image.

    Returns:
        Variable[np.ndarray]: Contoured image.
    """

    def __init__(
        self,
        parent_image,
        child_mask,
        parent_image_out,
        parent_slice,
        color,
        dilate_rel=0,
        dilate_abs=0,
    ):
        super().__init__()
        self.parent_image = parent_image
        self.child_mask = child_mask
        self.parent_image_out = parent_image_out
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
                parent_img = skimage.color.gray2rgb(
                    img_as_ubyte(self.parent_image, force_copy=True)
                )
                self.prepare_output(parent, parent_img)

            if self.dilate_rel or self.dilate_abs:
                area = np.sum(self.child_mask)
                radius = self.dilate_rel * np.sqrt(area) + self.dilate_abs

                dilated_mask = binary_dilation(self.child_mask, disk(radius))

            boundary_mask = skimage.segmentation.find_boundaries(
                dilated_mask, mode="outer"
            )

            parent_img[self.parent_slice][boundary_mask] = self.color

            yield self.prepare_output(self.parent_image_out, parent_img)


