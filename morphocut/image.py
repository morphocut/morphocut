import itertools
import operator
import os
from typing import Any, List, Mapping

import numpy as np
import scipy.ndimage as ndi
import skimage.exposure
import skimage.io

from morphocut import Node, Output, RawOrVariable, ReturnOutputs


@ReturnOutputs
@Output("mask")
class ThresholdConst(Node):
    """Set the mask of image
    """

    def __init__(self, image: RawOrVariable, threshold: RawOrVariable):
        super().__init__()
        self.image = image
        self.threshold = threshold

    def transform(self, image):
        """Check if the image is 2 dimensional
        """
        if image.ndim != 2:
            raise ValueError("image.ndim needs to be exactly 2.")

        mask = image <= self.threshold

        return mask


@ReturnOutputs
@Output("rescaled")
class Rescale(Node):
    """Rescale the image
    """

    def __init__(self, image: RawOrVariable, in_range='image', dtype=None):
        super().__init__()

        self.image = image
        self.dtype = dtype
        self.in_range = in_range

        if dtype is not None:
            self.out_range = dtype
        else:
            self.out_range = "dtype"

    def transform(self, image):
        image = skimage.exposure.rescale_intensity(
            image, in_range=self.in_range, out_range=self.out_range
        )
        if self.dtype is not None:
            image = image.astype(self.dtype, copy=False)

        return image


@ReturnOutputs
class ImageWriter(Node):
    """Create a duplicate image, return its directory and filename
    """

    def __init__(self, root: str, fmt: str, image: RawOrVariable, meta: RawOrVariable[Mapping]):
        super().__init__()
        self.root = root
        self.fmt = fmt
        self.image = image
        self.meta = meta

    def transform_stream(self, stream):
        for dirname, group in itertools.groupby(
            self._gen_paths(stream), operator.itemgetter(0)
        ):
            os.makedirs(dirname, exist_ok=True)
            for _, filename, image, obj in group:

                skimage.io.imsave(filename, image)

                yield obj

    def _gen_paths(self, stream):
        for obj in stream:
            image, meta = self.prepare_input(obj, ("image", "meta"))

            filename = os.path.join(self.root, self.fmt.format(**meta))

            dirname = os.path.dirname(filename)

            yield dirname, filename, image, obj


@ReturnOutputs
@Output("regionprops")
class FindRegions(Node):
    """
    Find regions in a mask and calculate properties.

    TODO: Include reference to skimage.measure.regionsprops

    .. note::
        This Node creates multiple objects per incoming object.

    Example:
        .. code-block:: python
            mask = ...
            regionsprops = FindRegions(mask)

            # regionsprops: A skimage.measure.regionsprops object.

    """

    def __init__(
        self, mask: RawOrVariable, image: RawOrVariable = None, min_area=None, max_area=None, padding=0
    ):
        super().__init__()

        self.mask = mask
        self.image = image

        self.min_area = min_area
        self.max_area = max_area
        self.padding = padding

    @staticmethod
    def _enlarge_slice(slices, padding):
        return tuple(
            slice(max(0, s.start - padding), s.stop + padding) for s in slices
        )

    def transform_stream(self, stream):
        """Slice the image stream
        """
        for obj in stream:
            mask, image = self.prepare_input(obj, ("mask", "image"))

            labels, nlabels = skimage.measure.label(mask, return_num=True)

            objects = ndi.find_objects(labels, nlabels)
            for i, sl in enumerate(objects):
                if sl is None:
                    continue

                if self.padding:
                    sl = self._enlarge_slice(sl, self.padding)

                props = skimage.measure._regionprops._RegionProperties(
                    sl, i + 1, labels, image, True, 'rc'
                )

                if self.min_area is not None and props.area < self.min_area:
                    continue

                if self.max_area is not None and props.area > self.max_area:
                    continue

                yield self.prepare_output(obj.copy(), props)


@ReturnOutputs
@Output("extracted_image")
class ExtractROI(Node):
    """Return the extracted region/image
    """

    def __init__(self, image: RawOrVariable, regionprops: RawOrVariable):
        super().__init__()

        self.image = image
        self.regionprops = regionprops

    # TODO: Hide background using mask

    def transform(self, image, regionprops):
        return image[regionprops.slice]


@ReturnOutputs
class ImageStats(Node):
    """
    Parse information from a path
    """

    def __init__(self, image: RawOrVariable, name: str = ""):
        super().__init__()

        self.min = []  # type: List[Any]
        self.max = []  # type: List[Any]
        self.image = image
        self.name = name

    def transform(self, image):
        self.min.append(np.min(image))
        self.max.append(np.max(image))

    def after_stream(self):
        print("### Range stats ({}) ###".format(self.name))
        mean_min = np.mean(self.min)
        mean_max = np.mean(self.max)
        print("Absolute: ", min(self.min), max(self.max))
        print("Average: ", mean_min, mean_max)


image = skimage.data.camera()
with Pipeline() as pipeline:
    result = ThresholdConst(image, 256)

stream = pipeline.transform_stream()
result = [o[result] for o in stream]
#result.transform(image)
#obj = next(stream)

assert np.array_equal(result[0], np.ones((512,512), dtype=bool))