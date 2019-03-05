from abc import abstractmethod, ABC
import math
import numpy as np

from skimage import img_as_ubyte, measure
from skimage.exposure import rescale_intensity
from skimage.morphology import binary_dilation, disk

import cv2 as cv
import morphocut.processing.functional as F

__all__ = ["ImageManipulator"]


class ImageManipulator(ABC):
    @abstractmethod
    def __call__(self, img=None, property=None, dimensions=None):  # pragma: no cover
        """
        Manipulate the image and return a dict containing the manipulation and its identifier
        """
        return None


class GreyImage(ImageManipulator):

    def __init__(self, key=None, img_rank=1):
        self.key = key or 'grey_img'
        self.img_rank = img_rank

    def __call__(self, img=None, property=None, dimensions=None):
        grey_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_dict = dict(
            image=grey_img,
            img_rank=self.img_rank
        )
        return self.key, img_dict


class ContourImage(ImageManipulator):

    def __init__(self, contour_distance_from_object=0, key=None, img_rank=1):
        self.key = key or 'contour_img'
        self.contour_distance_from_object = contour_distance_from_object
        self.img_rank = img_rank

    def __call__(self, img=None, property=None, dimensions=None):
        bordered_mask = cv.copyMakeBorder(img_as_ubyte(property.image), top=dimensions['border_left'],
                                          bottom=dimensions['border_right'], left=dimensions['border_top'], right=dimensions['border_bottom'], borderType=cv.BORDER_CONSTANT)

        dilated_mask = img_as_ubyte(binary_dilation(
            bordered_mask, selem=disk(radius=self.contour_distance_from_object * math.sqrt(property.area))))

        c_img, contours, hierarchy = cv.findContours(dilated_mask, 1, 2)
        contour_image = img.copy()
        cv.drawContours(contour_image, contours, -1, (0, 255, 0), 1)

        contours_masked = contour_image
        img_dict = dict(
            image=contours_masked,
            img_rank=self.img_rank
        )
        return self.key, img_dict


class WhiteBackgroundImage(ImageManipulator):

    def __init__(self, whiteness=0.5, key=None, img_rank=1):
        self.key = key or 'white_background_img'
        self.whiteness = whiteness
        self.img_rank = img_rank

    def __call__(self, img=None, property=None, dimensions=None):
        bordered_mask = cv.copyMakeBorder(img_as_ubyte(property.image), top=dimensions['border_left'],
                                          bottom=dimensions['border_right'], left=dimensions['border_top'], right=dimensions['border_bottom'], borderType=cv.BORDER_CONSTANT)

        bordered_mask = cv.cvtColor(bordered_mask, cv.COLOR_GRAY2BGR)

        white_image = cv.addWeighted(
            np.ones(img.shape, dtype=np.uint8) * 255, self.whiteness, img, (1 - self.whiteness), 0.0)
        selection = np.where((bordered_mask == [255, 255, 255]).all(axis=2))
        white_image[selection] = img[selection]

        img_dict = dict(
            image=white_image,
            img_rank=self.img_rank
        )
        return self.key, img_dict
