import itertools
import operator
import os
from typing import Any, List, Mapping

import numpy as np
import PIL
import scipy.ndimage as ndi
import skimage.exposure
import skimage.io
import skimage.measure
from skimage.color import gray2rgb, rgb2gray
from skimage.filters import threshold_otsu

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

    def __init__(
        self, image: RawOrVariable, in_range: RawOrVariable = "image", dtype=None
    ):
        super().__init__()

        self.image = image
        self.dtype = dtype
        self.in_range = in_range

        if dtype is not None:
            self.out_range = dtype
        else:
            self.out_range = "dtype"

    def transform(self, image, in_range):
        image = skimage.exposure.rescale_intensity(
            image, in_range=in_range, out_range=self.out_range
        )
        if self.dtype is not None:
            image = image.astype(self.dtype, copy=False)

        return image


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
        self,
        mask: RawOrVariable,
        image: RawOrVariable = None,
        min_area=None,
        max_area=None,
        padding=0,
    ):
        super().__init__()

        self.mask = mask
        self.image = image

        self.min_area = min_area
        self.max_area = max_area
        self.padding = padding

    @staticmethod
    def _enlarge_slice(slices, padding):
        return tuple(slice(max(0, s.start - padding), s.stop + padding) for s in slices)

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
                    sl, i + 1, labels, image, True
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


@ReturnOutputs
class ImageReader(Node):
    def __init__(self, fp: RawOrVariable):
        super().__init__()
        self.fp = fp

    def transform_stream(self, stream):
        for obj in stream:
            fp = self.prepare_input(obj, "fp")

            image = np.array(PIL.Image.open(fp))

            yield self.prepare_output(obj, image)


@ReturnOutputs
class ImageWriter(Node):
    def __init__(self, fp: RawOrVariable, image: RawOrVariable):
        super().__init__()
        self.fp = fp
        self.image = image

    def transform_stream(self, stream):
        for obj in stream:
            fp, image = self.prepare_input(obj, ("fp", "image"))

            img = PIL.Image.fromarray(image)
            img.save(fp)

            yield obj


@ReturnOutputs
@Output("image")
class Gray2RGB(Node):
    def __init__(self, image):
        super().__init__()
        self.image = image

    def transform(self, image):
        return gray2rgb(image)


@ReturnOutputs
@Output("image")
class RGB2Gray(Node):
    """
    Compute luminance of an RGB image using :py:func:`skimage.color.rgb2gray`.

    Returns:
        Variable[numpy.ndarray]: The luminance image:
            An array which is the same size as the input array,
            but with the channel dimension removed and dtype=float.
    """

    def __init__(self, image):
        super().__init__()
        self.image = image

    def transform(self, image):
        if len(image.shape) != 3:
            raise ValueError("image.shape != 3 in {!r}".format(self))

        return rgb2gray(image)


@ReturnOutputs
@Output("image")
class ThresholdOtsu(Node):
    def __init__(self, image):
        super().__init__()
        self.image = image

    def transform(self, image):
        thresh = threshold_otsu(image)
        mask = image < thresh

        return mask
