import warnings
from typing import Any, List, Optional

import numpy as np
import PIL
import scipy.ndimage as ndi
import skimage.exposure
import skimage.io
import skimage.measure
import skimage.measure._regionprops
from skimage.color import gray2rgb, rgb2gray
from skimage.util import dtype

from morphocut import Node, Output, RawOrVariable, ReturnOutputs, closing_if_closable


@ReturnOutputs
@Output("mask")
class ThresholdConst(Node):
    """
    Calculate a mask by applying a constant threshold.

    The result will be `image < threshold`.

    Args:
        image (np.ndarray or Variable): Image for which the mask is to be calculated.
        threshold (Number or Variable): Threshold. Image intensities less than this will be `True` in the
            result.

    Returns:
        Variable[np.ndarray]: Mask.
    """

    def __init__(self, image: RawOrVariable, threshold: RawOrVariable):
        super().__init__()
        self.image = image
        self.threshold = threshold

    def transform(self, image):
        if image.ndim != 2:
            raise ValueError("image.ndim needs to be exactly 2.")

        mask = image < self.threshold

        return mask


@ReturnOutputs
@Output("rescaled")
class RescaleIntensity(Node):
    """
    Rescale the intensities of the image.

    .. note::
        Uses the skimage library :py:func:`skimage.exposure.rescale_intensity`.

    Args:
        image (np.ndarray or Variable): An image file to be rescaled.
        in_range ((str or 2-tuple) or Variable): min/max as the intensity range.
        dtype (str or Variable): min/max of the image's dtype as the intensity range.

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


class RegionProperties(
    skimage.measure._regionprops.RegionProperties  # pylint: disable=protected-access
):
    """
    Like skimage.measure.RegionProperties but without storing the whole image.

    Please refer to `skimage.measure.regionprops` for more information
    on the available region properties.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._image = super().image

        if self._intensity_image is not None:
            self._image_intensity = super().image_intensity
        else:
            self._image_intensity = None

    @property
    def image(self):
        return self._image

    @property
    def intensity_image(self):
        if self._image_intensity is None:
            raise AttributeError("No intensity image specified.")
        return self._image_intensity


@ReturnOutputs
@Output("regionprops")
class FindRegions(Node):
    """
    |stream| Find regions in a mask and calculate properties.

    For more information see :py:func:`skimage.measure.regionprops`.

    .. note::
        This Node creates multiple objects per incoming object.

    Args:
        mask (np.ndarray or Variable): Mask of a given image.
        image (np.ndarray or Variable): An image whose mask we have to find region with.
        min_area (int): Minimum area of the region. If the area of our prop/region is
            smaller than our min_area then it will discard it.
        max_area (int): Maximum area of the region. If the area of our prop/region is
            bigger than our max_area then it will discard it.
        padding (int): Size of the slices/regions of our image.
        warn_empty (bool or str or Variable): Warn for empty images (default false).
            If a String is supplied, it is used as an identifier for the image.

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
        warn_empty=False,
    ):
        super().__init__()

        self.mask = mask
        self.image = image

        self.min_area = min_area
        self.max_area = max_area
        self.padding = padding
        self.warn_empty = warn_empty

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

                    props = RegionProperties(slices, i + 1, labels, image, True)

                    if self.min_area is not None and props.area < self.min_area:
                        continue

                    if self.max_area is not None and props.area > self.max_area:
                        continue

                    yield self.prepare_output(obj.copy(), props)

                if nlabels == 0:
                    if self.warn_empty is not False:
                        warn_empty = self.prepare_input(obj, "warn_empty")
                        if not isinstance(warn_empty, str):
                            warn_empty = "Image"
                        warnings.warn(f"{warn_empty} did not contain any objects.")


@ReturnOutputs
@Output("regionprops")
class ImageProperties(Node):
    """
    Calculate region properties for an image containing a single object.

    For more information see :py:func:`skimage.measure.regionprops`.

    Args:
        mask (np.ndarray or Variable): Mask of a given image.
        image (np.ndarray or Variable): An image whose mask we have to find region with.

    Example:
        .. code-block:: python

            image = ...
            mask = image < 128
            regionsprops = ImageProperties(mask, image)

            # regionsprops: A skimage.measure.regionsprops object.
    """

    def __init__(self, mask: RawOrVariable, image: RawOrVariable = None):
        super().__init__()

        self.mask = mask
        self.image = image

    def transform(self, mask: np.ndarray, image: np.ndarray):
        return RegionProperties(
            tuple(slice(0, s) for s in mask.shape),  # Whole mask
            True,  # Where mask == True
            mask,
            image,
            True,
        )


@ReturnOutputs
@Output("extracted_image")
class ExtractROI(Node):
    """
    Extract part of an image using a :py:class:`RegionProperties <skimage.measure._regionprops.RegionProperties>` instance.

    To be used in conjunction with :py:class:`FindRegions`.

    Args:
        image (np.ndarray or Variable): Image from which regions are to be extracted.
        regionprops (RegionProperties or Variable):
            :py:class:`RegionProperties <skimage.measure._regionprops.RegionProperties>`
            instance returned by :py:class:`FindRegions`.
        alpha: 1=Background completely reset to bg_color; 0 = Background fully visible.
        bg_color: Color for the background.
    """

    def __init__(
        self, image: RawOrVariable, regionprops: RawOrVariable, alpha=0, bg_color=0
    ):
        super().__init__()

        self.image = image
        self.regionprops = regionprops
        self.alpha = alpha
        self.bg_color = np.array(bg_color)

    def transform(self, image, regionprops):
        image = image[regionprops.slice]

        if self.alpha == 0:
            return image

        # Combine background and foreground
        result_img = (self.alpha * self.bg_color + (1 - self.alpha) * image).astype(
            image.dtype
        )

        # Paste foreground
        result_img[regionprops.image] = image[regionprops.image]

        return result_img


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
        fp (file or Variable): A filename (string), pathlib.Path object or file object.
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
        fp (file or Variable): A filename (string), pathlib.Path object or file object.
        image (np.ndarray or Variable): Image that is to be saved into a given directory.
        **kwargs: Arguments for :py:meth:`PIL.Image.Image.save`.
    """

    def __init__(self, fp: RawOrVariable, image: RawOrVariable, **kwargs):
        super().__init__()
        self.fp = fp
        self.image = image
        self.kwargs = kwargs

    def transform(self, fp, image):
        img = PIL.Image.fromarray(image)
        img.save(fp, **self.kwargs)


@ReturnOutputs
@Output("image")
class Gray2RGB(Node):
    """
    Create an RGB representation of a gray-level image.

    .. note::
        Uses the skimage library :py:func:`skimage.color.gray2rgb` to convert from Grayscale to RGB.

    Args:
        image (numpy.ndarray or Variable): Gray-level input image.

    Returns:
        Variable[numpy.ndarray]: The RGB image:
            An array which is the same size as the input array,
            but with a channel dimension appended.
    """

    def __init__(self, image: RawOrVariable[np.ndarray], keep_dtype=False):
        super().__init__()
        self.image = image
        self.keep_dtype = keep_dtype

    def transform(self, image):
        result = gray2rgb(image)

        if self.keep_dtype:
            result = dtype.convert(result, image.dtype)

        return result


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

    def __init__(self, image: RawOrVariable[np.ndarray], keep_dtype=False):
        super().__init__()
        self.image = image
        self.keep_dtype = keep_dtype

    def transform(self, image: np.ndarray):
        if len(image.shape) != 3:
            raise ValueError("image.shape != 3 in {!r}".format(self))

        result = rgb2gray(image)

        if self.keep_dtype:
            result = dtype.convert(result, image.dtype)

        return result
