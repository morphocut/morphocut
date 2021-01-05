import io
import logging
from multiprocessing.connection import Client
import time
from typing import Optional

import numpy as np
import PIL.Image

from morphocut.core import Node, Stream, Variable, closing_if_closable


def _pack_message(name: str, data: bytes):
    if "\n" in name:
        raise ValueError("Name must not contain \\n")

    return name.encode() + b"\n" + data


class MJPEGStreamer(Node):
    def __init__(
        self,
        image: Variable[np.ndarray],
        name: str,
        server_address="/tmp/mjpeg_streamer",
        max_fps: Optional[int] = None,
    ):
        super().__init__()

        self.image = image
        self.name = name
        self.server_address = server_address
        self.max_fps = max_fps

        self._logger = logging.getLogger("MJPEGStreamer")

    def transform_stream(self, stream: Stream) -> Stream:
        with closing_if_closable(stream), Client(self.server_address) as connection:
            time_sent = None
            for obj in stream:
                image = self.prepare_input(obj, "image")

                if (
                    self.max_fps is None
                    or time_sent is None
                    or (time.monotonic() - time_sent >= 1 / self.max_fps)
                ):

                    # Encode image and send to mjpeg server
                    image = PIL.Image.fromarray(image)
                    img_fp = io.BytesIO()
                    image.save(img_fp, format="jpeg")
                    buf = img_fp.getvalue()

                    self._logger.debug("Sending %s...", self.name)
                    connection.send_bytes(_pack_message(self.name, buf))

                    time_sent = time.monotonic()

                yield obj

        self.after_stream()
