import os.path

import matplotlib.pyplot as plt
import numpy as np
import skimage.exposure
from skimage.filters import threshold_otsu

from morphocut import LambdaNode, Node, Output, Pipeline, ReturnOutputs
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


with Pipeline() as pipeline:
    video_fn = Glob("/home/moi/Work/apeep_test/in/*.avi")
    video_basename = LambdaNode(
        lambda x: os.path.basename(os.path.splitext(x)[0]), video_fn
    )

    TQDM(Format("Reading files ({})...", video_basename))

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
    # FilterVariables(frame, range_, video_basename, frame_no)

    # with ParallelPipeline(queue_size=2, parent=pipeline):
    range_ = LambdaNode(tuple, range_)
    frame = Rescale(frame, in_range=range_, dtype="uint8")

    frame_fn = Format(
        "/tmp/apeep/{video_basename}-{frame_no}.jpg",
        video_basename=video_basename,
        frame_no=frame_no,
    )
    ImageWriter(frame_fn, frame)

    thresh = 200  # LambdaNode(threshold_otsu, frame)
    mask = frame < thresh

    hist = LambdaNode(lambda x: skimage.exposure.histogram(x)[0], frame)

    hist_img = Bar(np.arange(256), hist, vline=thresh)
    hist_fn = Format(
        "/tmp/apeep/{video_basename}-{frame_no}-hist.png",
        video_basename=video_basename,
        frame_no=frame_no,
    )
    ImageWriter(hist_fn, hist_img)

    region = FindRegions(mask, frame)

    Filter(lambda obj: obj[region].area > 100)

    object_image = frame[region.slice]
    object_no = region.label

    object_image_name = Format(
        "{video_basename}-{frame_no}-{object_no}",
        video_basename=video_basename,
        frame_no=frame_no,
        object_no=object_no,
    )

    meta = CalculateZooProcessFeatures(region, prefix="object_")

    # Filter variables that need to be gathered from worker processes
    # FilterVariables(object_image, object_image_name, meta)

    TQDM("Writing objects...")
    EcotaxaWriter("/tmp/apeep_test.zip", object_image, object_image_name, meta)


pipeline.run()
