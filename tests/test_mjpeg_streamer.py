import http.client
import io
import logging
from multiprocessing.connection import Listener
import threading
import time

import PIL.Image
import pytest
import skimage.data

from morphocut import Pipeline
from morphocut.mjpeg_streamer import MJPEGStreamer
from morphocut.mjpeg_streamer.server import MJPEGServer
from morphocut.stream import Unpack
from tests.helpers import Sleep

import sys

# "unix" fails on Max OS (AF_UNIX path too long) and is unavailable on Windows.
input_families = ["unix", "inet"] if sys.platform.startswith("linux") else ["inet"]


@pytest.mark.slow
@pytest.mark.parametrize("input_family", input_families)
@pytest.mark.parametrize("server_max_fps", [None, 1, 10])
def test_mjpeg_streamer_server(tmp_path, input_family, server_max_fps):

    input_address = (
        str(tmp_path / "input") if input_family == "unix" else ("localhost", 0)
    )
    http_address = ("localhost", 0)

    server = MJPEGServer(input_address, http_address, max_fps=server_max_fps)

    input_address = server.input_address
    http_address = server.http_address

    print("input_address", input_address)
    print("http_address", http_address)

    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()

    with Pipeline() as p_failing:
        MJPEGStreamer(
            skimage.data.astronaut(),
            "invalid_name\n",
            input_address,
        )

    with pytest.raises(ValueError):
        p_failing.run()

    with Pipeline() as p:
        image = Unpack([skimage.data.astronaut() for _ in range(10)])
        Sleep(0.1)
        MJPEGStreamer(image, "image", input_address)

    conn = http.client.HTTPConnection(*http_address)
    conn.request("GET", "/image")
    response = conn.getresponse()

    pipeline_thread = threading.Thread(target=p.run)
    pipeline_thread.start()

    assert response.status == 200
    boundary = response.headers.get_boundary()
    assert boundary is not None

    delimiter = b"--" + boundary.encode()

    buffer = b""
    n_frames = 0
    while True:
        buffer += response.read(64)

        if delimiter in buffer:
            part, buffer = buffer.split(delimiter, 1)
            header_body = part.split(b"\r\n\r\n", 1)

            if len(header_body) == 1:
                continue

            body = io.BytesIO(header_body[1])
            image = PIL.Image.open(body)
            image.load()
            assert image.size == (512, 512)

            n_frames += 1
            print(f"Got frame {n_frames}")

            # Exit after one received image because if we are to slow to retrieve images,
            # the server throws them away and we never see them.
            break

    pipeline_thread.join()
    server.shutdown()
    server_thread.join()


@pytest.mark.slow
@pytest.mark.parametrize("max_fps", [1, 5, 10])
def test_mjpeg_streamer_fps(max_fps):
    listener = Listener(("localhost", 0))

    with Pipeline() as p:
        image = Unpack([skimage.data.astronaut() for _ in range(20)])
        Sleep(0.1)
        MJPEGStreamer(image, "image", listener.address, max_fps=max_fps)

    pipeline_thread = threading.Thread(target=p.run)
    pipeline_thread.start()

    conn = listener.accept()

    then = None
    fps_checked = False
    while True:
        try:
            buf = conn.recv_bytes()
            now = time.monotonic()
            if then is not None:
                fps = 1 / (now - then)
                print("fps:", fps)
                assert fps < max_fps
                fps_checked = True
            then = now
        except EOFError:
            break

    assert fps_checked