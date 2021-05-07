"""

"""

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
    """
    Stream images via HTTP (e.g. to the Browser).

    This depends on a running streaming server.
    The server can be started with :code:`python -m morphocut.mjpeg_streamer.server` and then waits for connections.

    Args:
        image (np.ndarray or Variable): Image to be streamed.
        name (str): Name of the stream.
        server_address (optional): Connect to the streaming server at this address.
        max_fps (int, optional): Maximum frame rate for streaming.
            If set, some images will not be sent to the server to stay below the requested frame rate.

    Example:
        .. code-block:: python

            with Pipeline() as p:
                image = ...
                MJPEGStreamer(image, "image")
    """

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
        ""  # No documentation

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
