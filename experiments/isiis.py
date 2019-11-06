import contextlib
import os.path

import numpy as np
import skimage.exposure
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
from morphocut.contrib.isiis import FindRegions
from morphocut.ecotaxa import EcotaxaWriter
from morphocut.file import Glob
from morphocut.image import BinaryClosing, ImageWriter, Rescale, RGB2Gray, ThresholdOtsu
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


WRITE_HISTOGRAMS = False
WRITE_FRAMES = False
PARALLEL_FILE = False
CLOSING = True

# The processing pipeline is defined as follows:
# (It is not executed yet.)
with Pipeline() as pipeline:
    # Find files and put them into the stream
    video_fn = Glob("/home/moi/Work/apeep_test/in/*.avi")
    video_basename = LambdaNode(
        lambda x: os.path.basename(os.path.splitext(x)[0]), video_fn
    )

    TQDM(Format("Reading files ({})...", video_basename))

    # Parallelize the processing of individual video files
    # The parallelization has to take place at this coarse level
    # to treat adjacent frames right (wrt. smoothing and overlapping objects).
    if PARALLEL_FILE:
        parallel_context = ParallelPipeline(parent=True)
    else:
        parallel_context = contextlib.nullcontext()
    with parallel_context:
        # Put all the frames from each video_fn into the stream
        frame = VideoReader(video_fn)
        frame_no = frame.frame_no

        TQDM(Format("Reading frames ({})...", frame_no))

        # Convert frame to grayscale
        frame = RGB2Gray(frame)

        ## Remove stripe artifacts
        col_median = LambdaNode(np.median, frame, axis=0)
        # Average over multiple frames
        col_median = ExponentialSmoothing(col_median, 0.5)
        frame = frame - col_median

        ## Compute dynamic range of frames
        # Subsample frame for histogram computation
        # This should be much faster than skimage.transform.rescale.
        frame_sml = frame[::4, ::4]
        range_ = LambdaNode(np.percentile, frame_sml, (0, 85))
        # Average over multiple frames
        range_ = ExponentialSmoothing(range_, 0.5)

        # Convert range_ to tuple
        range_ = LambdaNode(tuple, range_)
        frame = Rescale(frame, in_range=range_, dtype="uint8")

        if WRITE_FRAMES:
            # Save pre-processed frames
            frame_fn = Format(
                "/tmp/apeep/{video_basename}-{frame_no}.jpg",
                video_basename=video_basename,
                frame_no=frame_no,
            )
            ImageWriter(frame_fn, frame)

        # A fixed threshold worked best so far
        thresh = 200  # LambdaNode(threshold_otsu, frame)

        if WRITE_HISTOGRAMS:
            # Save gray value histograms
            hist = LambdaNode(lambda x: skimage.exposure.histogram(x)[0], frame)
            x = LambdaNode(lambda hist: np.arange(len(hist)), hist)

            hist_img = Bar(x, hist, vline=thresh)
            hist_fn = Format(
                "/tmp/apeep/{video_basename}-{frame_no}-hist.png",
                video_basename=video_basename,
                frame_no=frame_no,
            )
            ImageWriter(hist_fn, hist_img)

        # Calculate a mask of objects
        mask = frame < thresh

        if CLOSING:
            # Remove small dark spots
            mask = BinaryClosing(mask, 2)

        # Find regions in the frame and put them into the stream
        region = FindRegions(mask, frame, min_area=100)

        # Extract some properties from a region
        object_mask = region.image
        object_image = region.intensity_image_unmasked
        object_no = region.label

        # Calculate filenames
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

        # Calculate (somewhat) ZooProcess-compatible features
        meta = CalculateZooProcessFeatures(region, prefix="object_")

        if PARALLEL_FILE:
            # Filter variables that need to be sent to the main process
            FilterVariables(
                object_mask, object_image, object_image_name, object_mask_name, meta
            )

    TQDM("Writing objects...")

    # Write objects and measurements into an EcoTaxa-compatible archive
    EcotaxaWriter(
        "/tmp/apeep_test.zip",
        (object_image, object_mask),
        (object_image_name, object_mask_name),
        meta,
    )

# Execute the pipeline
with Timer("Total time"):
    pipeline.run()
