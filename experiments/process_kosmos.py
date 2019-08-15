import numpy as np
from skimage import io

from morphocut.pipeline import *
from morphocut.pipeline import SimpleNodeBase

import_path = "/data-ssd/mschroeder/Datasets/generic_zooscan_peru_kosmos_2017"
archive_fn = "/tmp/kosmos.zip"



class WhiteBalance(SimpleNodeBase):
    """
    White balance like in Gorsky, G. et al. (2010).

    TODO: This needs the reference disks.

    Gorsky, G. et al. (2010) ‘Digital zooplankton image analysis using the ZooScan integrated system’,
    Journal of Plankton Research. Oxford University Press, 32(3), pp. 285–303. doi: 10.1093/plankt/fbp124.
    """

    def process(self, facet):
        img = facet["image"]

        # Accept only gray-scale images
        assert img.shape[-1] == 1

        white_point = np.median(img) * 1.15

        corrected_img = img / flat_image

        return {
            "image": corrected_img
        }

class ThresholdFix(SimpleNodeBase):
    def __init__(self, input_facet, output_facet, threshold):
        super().__init__(input_facet, output_facet)

    def process(self, facet):
        image = facet["image"]

        thresh = threshold_otsu(image)
        mask = image < thresh

        return {
            "image": mask
        }


pipeline = Pipeline([
        DataLoader(import_path, output_facet="raw"),
        Progress("Loaded"),
        VignetteCorrector(input_facet="raw", output_facet="corrected"),
        ThresholdOtsu(input_facet="corrected", output_facet="mask"),
        ExtractRegions(
            mask_facet="mask",
            intensity_facet="corrected",
            image_facets=["corrected", "mask"],
            output_facet="roi",
            padding=10,
            min_area=30),
        FadeBackground(
            image_facet="gray",
            mask_facet="roi",
            output_facet="bg_white", alpha=1),
        DrawContours(
            image_facet="color",
            mask_facet="roi",
            output_facet="color_contours"),
        ObjectScale(input_facets=["color_contours"],
                    output_facets=["color_contours_scale"]),
        Exporter(
            archive_fn,
            data_facets=["roi"],
            img_facets=["bg_white", "color", "color_contours", "color_contours_scale"])
    ])

if __name__ == "__main__":
    image = io.imread("/data-ssd/mschroeder/Datasets/generic_zooscan_peru_kosmos_2017/generic_Peru_20170226_slow_M1_dnet/Peru_20170226_M1_dnet_1_8_a.tif")
