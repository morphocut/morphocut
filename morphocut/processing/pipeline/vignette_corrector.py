from morphocut.processing.pipeline import NodeBase
import cv2 as cv
import numpy as np
import argparse
import pandas as pd
import os
from parse import *
import sys
import morphocut.processing.functional as proc
from skimage.exposure import rescale_intensity
from skimage.morphology import binary_dilation, disk
from skimage import img_as_ubyte
import math


class VignetteCorrector(NodeBase):
    """
    A processing node. Removes the vignette effect from images.

    Input:

    {
        object_id: ...
        facets: {
            input_data: {
                meta: {filename: ...},
                image: <np.array of shape = [h,w,c]>
            }
        }
    }

    Output:

    {
        object_id: ...
        facets: {
            input_data: {
                meta: {filename: ...},
                image: <np.array of shape = [h,w,c]>
            }
            corrected_data: {
                image: <np.array of shape = [h,w,c]>
            }
        }
    }

    """

    def __call__(self, input=None):
        # print('vignette_corrector call ' + str(input))
        for data_object in input:
            data_object['facets']['corrected_data'] = dict(
                image=self.correct_vignette(
                    data_object['facets']['input_data']['image'])
            )
            yield data_object

    def correct_vignette(self, img):
        print('Correcting vignette...')
        grey_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        flat_image = proc.calculate_flat_image(grey_img)
        corrected_img = grey_img / flat_image
        corrected_img = rescale_intensity(corrected_img)
        corrected_img = img_as_ubyte(corrected_img)

        return corrected_img
