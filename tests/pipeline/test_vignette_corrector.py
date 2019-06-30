import glob
import os.path

import cv2

from morphocut.processing.pipeline.vignette_corrector import VignetteCorrector


def test_vignette_corrector_no_channel(image_fns):
    vignette_corrector = VignetteCorrector("input", "output")

    inp = (
        {
            "facets": {
                "input": {
                    "image": cv2.imread(img_fn, cv2.IMREAD_GRAYSCALE)
                }
            }
        }
        for img_fn in image_fns
    )

    for _ in vignette_corrector(inp):
        pass
