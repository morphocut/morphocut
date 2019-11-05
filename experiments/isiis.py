import itertools
import os.path

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import skimage.exposure
from scipy.optimize import linear_sum_assignment
from skimage.filters import threshold_otsu
from timer_cm import Timer

from morphocut import (
    LambdaNode,
    Node,
    Output,
    Pipeline,
    RawOrVariable,
    ReturnOutputs,
    Variable,
)
from morphocut.ecotaxa import EcotaxaWriter
from morphocut.file import Glob
from morphocut.image import FindRegions, ImageWriter, Rescale, RGB2Gray, ThresholdOtsu
from morphocut.numpy import AsType
from morphocut.parallel import ParallelPipeline
from morphocut.pims import VideoReader
from morphocut.plot import Bar
from morphocut.str import Format
from morphocut.stream import TQDM, Filter, FilterVariables
from morphocut.zooprocess import CalculateZooProcessFeatures

# mypy: ignore-errors


@ReturnOutputs
@Output("out")
class ExponentialSmoothing(Node):
    def __init__(self, value, alpha):
        super().__init__()

        self.value = value
        self.alpha = alpha
        self.last_value = None

    def transform(self, value):
        if self.last_value is None:
            self.last_value = value
        else:
            self.last_value = self.alpha * value + (1 - self.alpha) * self.last_value

        return self.last_value


def bbox(mask):
    # a = np.where(mask)
    # return np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


class RegionProperties(skimage.measure._regionprops.RegionProperties):
    @property
    def intensity_image_unmasked(self):
        if self._intensity_image is None:
            raise AttributeError("No intensity image specified.")
        return self._intensity_image[self.slice]


@ReturnOutputs
@Output("regionprops")
class FindRegions(Node):
    """
    Find regions in a mask and calculate properties.

    This is an adapted version of :py:class:`morphocut.image.FindRegions`
    that merges objects that reach over the border of a frame.

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

    def _find_objects(self, l_image, image):
        objects = ndi.find_objects(l_image)
        for i, sl in enumerate(objects, 1):

            # Skip missing objects
            if sl is None:
                continue

            if self.padding:
                sl = self._enlarge_slice(sl, self.padding)

            props = RegionProperties(sl, i, l_image, image, True)

            yield props

    @staticmethod
    def _calc_correspondences(last_row, cur_row, min_iou=0.75):
        last_ul = np.unique(last_row)
        cur_ul = np.unique(cur_row)

        correspondence_matrix = np.zeros((len(last_ul), len(cur_ul)))
        for i, last_label in enumerate(last_ul):
            if last_label == 0:
                continue
            last_label_mask = last_row == last_label
            for j, cur_label in enumerate(cur_ul):
                if cur_label == 0:
                    continue
                cur_label_mask = cur_row == cur_label
                correspondence_matrix[i, j] = np.sum(
                    last_label_mask & cur_label_mask
                ) / np.sum(last_label_mask | cur_label_mask)

        ii, jj = linear_sum_assignment(-correspondence_matrix)

        for i, j in zip(ii, jj):
            iou = correspondence_matrix[i, j]
            if iou < min_iou:
                continue
            yield last_ul[i], cur_ul[j], iou

    def _find_overlapping_objects(self, last_lim, cur_lim, last_image, cur_image):
        """
        Find regions in the current and the last frame that correspond.

        Yield these regions and remove them from the last and current frame,
        so that the rest of the regions can be treated normally by ``_find_objects``.
        """
        last_row = last_lim[-1]
        cur_row = cur_lim[0]
        if np.sum(last_row) and np.sum(cur_row):
            for last_label, cur_label, iou in self._calc_correspondences(
                last_row, cur_row
            ):
                last_roi_mask = last_lim == last_label
                cur_roi_mask = cur_lim == cur_label

                if last_roi_mask.sum() < 10:
                    continue

                if cur_roi_mask.sum() < 10:
                    continue

                last_box = bbox(last_roi_mask)
                cur_box = bbox(cur_roi_mask)

                col_slice = slice(
                    min(last_box[2], cur_box[2]), max(last_box[3], cur_box[3]) + 1
                )

                last_slice = (slice(last_box[0], last_box[1] + 1), col_slice)
                last_slice = self._enlarge_slice(last_slice, self.padding)
                cur_slice = (slice(cur_box[0], cur_box[1] + 1), col_slice)
                cur_slice = self._enlarge_slice(cur_slice, self.padding)

                joint_image = np.concatenate(
                    (last_image[last_slice], cur_image[cur_slice])
                )

                last_roi_mask = last_lim[last_slice] == last_label
                cur_roi_mask = cur_lim[cur_slice] == cur_label
                joint_mask = np.concatenate((last_roi_mask, cur_roi_mask))

                # Delete region from last_lim and cur_lim
                last_lim[last_slice][last_roi_mask] = 0
                cur_lim[cur_slice][cur_roi_mask] = 0

                h, w = joint_mask.shape

                # Mark seam
                # joint_image[-cur_slice[0].stop, [0, 1, -2, -1]] = 0

                props = RegionProperties(
                    (slice(0, h), slice(0, w)),
                    last_label,
                    joint_mask * last_label,
                    joint_image,
                    True,
                )

                print("Overlapping object:", last_label)

                yield props

    def transform_stream(self, stream):
        last_lim = last_image = last_obj = None
        for cur_obj in stream:
            cur_mask, cur_image = self.prepare_input(cur_obj, ("mask", "image"))

            cur_lim, nlabels = skimage.measure.label(cur_mask, return_num=True)

            # Delete objects that touch left or right border
            border_labels = np.unique(cur_lim[:, [0, -1]])
            border_mask = np.isin(cur_lim, border_labels)
            cur_lim[border_mask] = 0

            if last_obj is not None:
                # Handle overlapping objects
                for prop in self._find_overlapping_objects(
                    last_lim, cur_lim, last_image, cur_image
                ):
                    if self.min_area is not None and prop.area < self.min_area:
                        continue

                    if self.max_area is not None and prop.area > self.max_area:
                        continue

                    yield self.prepare_output(last_obj.copy(), prop)

                # Handle regular objects
                for prop in self._find_objects(last_lim, last_image):
                    if self.min_area is not None and prop.area < self.min_area:
                        continue

                    if self.max_area is not None and prop.area > self.max_area:
                        continue

                    yield self.prepare_output(last_obj.copy(), prop)

            # Save last
            last_lim, last_image, last_obj = cur_lim, cur_image, cur_obj

        if last_obj:
            # Handle regular objects
            for prop in self._find_objects(last_lim, last_image):
                if self.min_area is not None and prop.area < self.min_area:
                    continue

                if self.max_area is not None and prop.area > self.max_area:
                    continue

                yield self.prepare_output(last_obj.copy(), prop)


WRITE_HISTOGRAMS = True
WRITE_FRAMES = True

with Pipeline() as pipeline:
    # Find files
    video_fn = Glob("/home/moi/Work/apeep_test/in/*.avi")
    video_basename = LambdaNode(
        lambda x: os.path.basename(os.path.splitext(x)[0]), video_fn
    )

    TQDM(Format("Reading files ({})...", video_basename))

    # Parallelize the processing of individual video files
    # The parallelization has to take place at this coarse level
    # to treat adjacent frames right (wrt. smoothing and overlapping objects).
    with ParallelPipeline(parent=pipeline):

        frame = VideoReader(video_fn)
        frame_no = frame.frame_no

        TQDM(Format("Reading frames ({})...", frame_no))

        frame = RGB2Gray(frame)

        ## Remove line background
        col_median = LambdaNode(np.median, frame, axis=0)
        # Average over multiple frames
        col_median = ExponentialSmoothing(col_median, 0.5)
        frame = frame - col_median

        ## Compute dynamic range of frames
        # Subsample frame for histogram computation
        frame_sml = frame[::4, ::4]
        range_ = LambdaNode(np.percentile, frame_sml, (0, 85))
        # Average over multiple frames
        range_ = ExponentialSmoothing(range_, 0.5)

        # Filter variables that need to be sent to worker processes
        FilterVariables(frame, range_, video_basename, frame_no)

        range_ = LambdaNode(tuple, range_)
        frame = Rescale(frame, in_range=range_, dtype="uint8")

        if WRITE_FRAMES:
            frame_fn = Format(
                "/tmp/apeep/{video_basename}-{frame_no}.jpg",
                video_basename=video_basename,
                frame_no=frame_no,
            )
            ImageWriter(frame_fn, frame)

        thresh = 200  # LambdaNode(threshold_otsu, frame)
        mask = frame < thresh

        hist = LambdaNode(lambda x: skimage.exposure.histogram(x)[0], frame)
        x = LambdaNode(lambda hist: np.arange(len(hist)), hist)

        if WRITE_HISTOGRAMS:
            hist_img = Bar(x, hist, vline=thresh)
            hist_fn = Format(
                "/tmp/apeep/{video_basename}-{frame_no}-hist.png",
                video_basename=video_basename,
                frame_no=frame_no,
            )
            ImageWriter(hist_fn, hist_img)

        region = FindRegions(mask, frame)

        Filter(lambda obj: obj[region].area > 100)

        object_mask = region.image
        object_image = region.intensity_image_unmasked
        object_no = region.label

        object_image_name = Format(
            "{video_basename}-{frame_no}-{object_no}",
            video_basename=video_basename,
            frame_no=frame_no,
            object_no=object_no,
        )

        object_mask_name = Format(
            "{video_basename}-{frame_no}-{object_no}-mask",
            video_basename=video_basename,
            frame_no=frame_no,
            object_no=object_no,
        )

        meta = CalculateZooProcessFeatures(region, prefix="object_")

        # Filter variables that need to be gathered from worker processes
        FilterVariables(
            object_mask, object_image, object_image_name, object_mask_name, meta
        )

    TQDM("Writing objects...")
    EcotaxaWriter(
        "/tmp/apeep_test.zip",
        (object_image, object_mask),
        (object_image_name, object_mask_name),
        meta,
    )


with Timer("Total time"):
    pipeline.run()
