import io
import itertools
from typing import TYPE_CHECKING

import numpy as np

from morphocut._optional import UnavailableObject
from morphocut.core import Node, Output, ReturnOutputs, Stream, closing_if_closable


def is_raspberrypi():
    """Check if the platform is a Raspberry Pi."""

    try:
        with io.open("/sys/firmware/devicetree/base/model", "r") as m:
            if "raspberry pi" in m.read().lower():
                return True
    except Exception:
        pass
    return False


if TYPE_CHECKING:
    import picamera.array as picamera_array
    import picamera.camera as picamera_camera
elif is_raspberrypi():
    try:
        import picamera.array as picamera_array
        import picamera.camera as picamera_camera
    except ImportError:
        msg = "See https://picamera.readthedocs.io/en/latest/install.html for installation instructions."
        picamera_array = UnavailableObject("picamera.array", msg)
        picamera_camera = UnavailableObject("picamera.camera", msg)
else:
    # picamera is installable and importable on systems other than the Raspberry Pi (using READTHEDOCS=True during installation),
    # but still not usable.
    msg = "This only works on a Raspberry Pi."
    picamera_array = UnavailableObject("picamera.array", msg)
    picamera_camera = UnavailableObject("picamera.camera", msg)


@ReturnOutputs
@Output("frame")
class PiCameraReader(Node):
    """
    |stream| Read frames from the Raspberry Pi's camera.

    Args:
        resolution (tuple (width, height), optional): The desired image resolution.
    """

    def __init__(self, **kwargs):
        super().__init__()

        kwargs.setdefault("resolution", picamera_camera.PiCameraMaxResolution)

        self.kwargs = kwargs

    def transform_stream(self, stream: Stream) -> Stream:
        with picamera_camera.PiCamera(**self.kwargs) as cam:
            resolution = picamera_array.raw_resolution(cam.resolution)

            with closing_if_closable(stream):
                for obj in stream:

                    # Capture continously, each time into a fresh buffer
                    for output in cam.capture_continuous(
                        (
                            np.empty(resolution[::-1] + (3,), dtype=np.uint8)
                            for _ in itertools.repeat(None)
                        ),
                        format="rgb",
                    ):
                        self.prepare_output(obj, output)

                        yield obj

        self.after_stream()
