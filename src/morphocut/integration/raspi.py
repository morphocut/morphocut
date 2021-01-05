import io

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


if is_raspberrypi():
    try:
        import picamera
        import picamera.array as picamera_array
    except ImportError:
        msg = "See https://picamera.readthedocs.io/en/latest/install.html for installation instructions."
        picamera = UnavailableObject("picamera", msg)
        picamera_array = UnavailableObject("picamera.array", msg)
else:
    # picamera is installable and importable on systems other than the Raspberry Pi (using READTHEDOCS=True during installation),
    # but still not usable.
    msg = "This only works on a Raspberry Pi."
    picamera = UnavailableObject("picamera", msg)
    picamera_array = UnavailableObject("picamera.array", msg)


@ReturnOutputs
@Output("frame")
class PiCameraReader(Node):
    """
    Read frames from the Raspberry Pi's camera.
    """

    def __init__(self, resolution=(1280, 720)):
        super().__init__()

        self._resolution = picamera_array.raw_resolution(*resolution)

    def transform_stream(self, stream: Stream) -> Stream:
        cam = picamera.PiCamera(resolution=self._resolution)
        with closing_if_closable(stream):
            for obj in stream:

                while True:
                    # Get image
                    output = np.empty(self._resolution[::-1] + (3,), dtype=np.uint8)
                    cam.capture(output, "rgb")

                    self.prepare_output(obj, output)

                    yield obj

        self.after_stream()
