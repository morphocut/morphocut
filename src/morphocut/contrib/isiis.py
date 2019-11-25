"""Nodes and helpers for the processing of ISIIS data."""

import numpy as np
import scipy.ndimage as ndi
import skimage.measure
from scipy.optimize import linear_sum_assignment
from morphocut import Node, Output, RawOrVariable, ReturnOutputs


def bbox(mask):
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

                # print("Overlapping object:", last_label)

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
