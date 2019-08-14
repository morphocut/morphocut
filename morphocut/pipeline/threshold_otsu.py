from skimage.filters import threshold_otsu

from morphocut.pipeline import SimpleNodeBase


class ThresholdOtsu(SimpleNodeBase):
    def process(self, facet):
        image = facet["image"]

        thresh = threshold_otsu(image)
        mask = image < thresh

        return {
            "image": mask
        }
