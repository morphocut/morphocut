from typing import Any, List

import numpy as np
import PIL
import scipy.ndimage as ndi
import skimage.exposure
import skimage.io
import skimage.measure
from skimage.color import gray2rgb, rgb2gray

from morphocut import Node, Output, RawOrVariable, ReturnOutputs, closing_if_closable


@ReturnOutputs
@Output("mask")
class ThresholdConst(Node):
    """
    Calculate a mask by applying a constant threshold.

    The result will be `image <= threshold`.

    Args:
        image (np.ndarray or Variable[np.ndarray]): Image for which the mask is to be calculated.
        threshold (Number or Variable[Number]): Threshold. Image intensities less than this will be `True` in the 
            result.
    """

    def __init__(self, image: RawOrVariable, threshold: RawOrVariable):
        super().__init__()
        self.image = image
        self.threshold = threshold

    def transform(self, image):
        if image.ndim != 2:
            raise ValueError("image.ndim needs to be exactly 2.")

        mask = image <= self.threshold

        return mask


@ReturnOutputs
@Output("rescaled")
class RescaleIntensity(Node):
    """
    Rescale the intensities of the image.

    .. note::
        Uses the skimage library :py:func:`skimage.exposure.rescale_intensity`.

    Args:
        image (np.ndarray or Variable[np.ndarray]): An image file to be rescaled.
        in_range ((str or 2-tuple) or (Variable[str or 2-tuple])): min/max as the intensity range.
        dtype (str or Variable[str]): min/max of the image's dtype as the intensity range.

    Returns:
        Variable[np.ndarray]: Image with intensities rescaled.
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
    |stream| Find regions in a mask and calculate properties.

    For more information see :py:func:`skimage.measure.regionprops`.

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
        with closing_if_closable(stream):
            for obj in stream:
                mask, image = self.prepare_input(obj, ("mask", "image"))

                labels, nlabels = skimage.measure.label(mask, return_num=True)

                objects = ndi.find_objects(labels, nlabels)
                for i, slices in enumerate(objects):
                    if slices is None:
                        continue

                    if self.padding:
                        slices = self._enlarge_slice(slices, self.padding)

                    props = skimage.measure._regionprops.RegionProperties(  # pylint: disable=protected-access
                        slices, i + 1, labels, image, True
                    )

                    if self.min_area is not None and props.area < self.min_area:
                        continue

                    if self.max_area is not None and props.area > self.max_area:
                        continue

                    yield self.prepare_output(obj.copy(), props)


@ReturnOutputs
@Output("extracted_image")
class ExtractROI(Node):
    """
    Extract part of an image using a :py:class:`RegionProperties <skimage.measure._regionprops.RegionProperties>` instance.

    To be used in conjunction with :py:class:`FindRegions`.

    Args:
        image (np.ndarray or Variable[np.ndarray]): Image from which regions are to be extracted.
        regionprops (RegionProperties or Variable[RegionProperties]): :py:class:`RegionProperties <skimage.measure._regionprops.RegionProperties>` instance returned by :py:class:`FindRegions`.
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
@Output("image")
class ImageReader(Node):
    """
    Read and open the image from a given path.

    Use Python Imaging Library `PIL`_ to open the file from a given path.

    .. _PIL: https://pillow.readthedocs.io/en/stable/

    Args:
        fp (RawOrVariable or Variable[RawOrVariable]): File path from where we want to open our image.
    """

    def __init__(self, fp: RawOrVariable):
        super().__init__()
        self.fp = fp

    def transform(self, fp):
        return np.array(PIL.Image.open(fp))


@ReturnOutputs
class ImageWriter(Node):
    """
    Write the image into the given directory path.

    Use Python Imaging Library `PIL`_ to save the image in a given path.

    .. _PIL: https://pillow.readthedocs.io/en/stable/

    Args:
        fp (RawOrVariable or Variable[RawOrVariable]): Path where we want to save our image.
        image (np.ndarray or Variable[np.ndarray]): Image which we want to save to a directory.
    """

    def __init__(self, fp: RawOrVariable, image: RawOrVariable):
        super().__init__()
        self.fp = fp
        self.image = image

    def transform(self, fp, image):
        img = PIL.Image.fromarray(image)
        img.save(fp)


@ReturnOutputs
@Output("image")
class Gray2RGB(Node):
    """
    Create an RGB representation of a gray-level image.

    Args:
        image (numpy.ndarray or Variable): Gray-level input image.

    Returns:
        Variable[numpy.ndarray]: The RGB image:
            An array which is the same size as the input array,
            but with a channel dimension appended.
    """

    def __init__(self, image: RawOrVariable[np.ndarray]):
        super().__init__()
        self.image = image

    def transform(self, image):
        return gray2rgb(image)


@ReturnOutputs
@Output("image")
class RGB2Gray(Node):
    """
    Compute luminance of an RGB image using :py:func:`skimage.color.rgb2gray`.

    Args:
        image (numpy.ndarray or Variable): The image in RGB format.

    Returns:
        Variable[numpy.ndarray]: The luminance image:
            An array which is the same size as the input array,
            but with the channel dimension removed and dtype=float.
    """

    def __init__(self, image: RawOrVariable[np.ndarray]):
        super().__init__()
        self.image = image

    def transform(self, image):
        if len(image.shape) != 3:
            raise ValueError("image.shape != 3 in {!r}".format(self))

        return rgb2gray(image)
