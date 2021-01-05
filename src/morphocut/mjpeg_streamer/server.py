from collections import defaultdict
from multiprocessing.connection import Connection, Listener

import http.server
import threading
from http import HTTPStatus
import time
from typing import Dict, List, Optional, Tuple, Union
import traceback
import logging

from numpy.lib.arraysetops import isin

logger = logging.getLogger("mjpeg_streamer.server")


def unpack_message(buf: bytes):
    name, data = buf.split(b"\n", maxsplit=1)
    return name.decode(), data


class Subscription:
    def __init__(self, publisher: "Publisher", channel: str):
        self.publisher = publisher
        self.channel = channel

        self._data_available = threading.Condition()
        self._data = None

    def recv(self, timeout=None):
        """
        Receive new data.
        """
        with self._data_available:
            if self._data_available.wait(timeout=timeout):
                return self._data
            return None

    def send(self, data):
        with self._data_available:
            self._data = data
            self._data_available.notify_all()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.publisher.unsubscribe(self)


class Publisher:
    def __init__(self):
        self.subscribers: Dict[str, List[Subscription]] = defaultdict(list)
        self.lock = threading.Lock()

    def subscribe(self, channel):
        with self.lock:
            subscription = Subscription(self, channel)
            self.subscribers[channel].append(subscription)

            return subscription

    def unsubscribe(self, subscription: Subscription):
        with self.lock:
            self.subscribers[subscription.channel].remove(subscription)

    def publish(self, channel, data):
        with self.lock:
            for sub in self.subscribers.get(channel, []):
                sub.send(data)


class _MJPEGRequestHandler(http.server.BaseHTTPRequestHandler):
    server: "_MJPEGHTTPServer"

    def do_GET(self):
        channel = self.path.lstrip("/")
        logger.info("HTTP request for %s from %s", channel, self.client_address)

        with self.server.publisher.subscribe(channel) as subscription:
            boundary = b"MJPEG_BOUNDARY"
            self.send_response(HTTPStatus.OK)
            self.send_header(
                "Content-type",
                f'multipart/x-mixed-replace; boundary="{boundary.decode()}"',
            )
            self.end_headers()
            self.wfile.flush()

            logger.debug("Sent initial headers.")

            while True:
                try:
                    start = time.monotonic()
                    self.wfile.write(b"\r\n")
                    self.wfile.write(b"--" + boundary + b"\r\n")

                    waiting = 0
                    while True:
                        data = subscription.recv(timeout=1)
                        if data is not None:
                            break
                        logger.debug(f"Waiting for {channel}...")
                        self.wfile.write(f"X-Waiting: {waiting}\r\n".encode())
                        waiting += 1

                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(b"\r\n")
                    self.wfile.write(data)

                    if self.server.max_fps is not None:
                        remain = start + 1 / self.server.max_fps - time.monotonic()
                        if remain > 0:
                            time.sleep(remain)

                    # end = time.monotonic()
                    # diff = end - start

                    # print(f"{1/diff:.2f} fps")
                except (BrokenPipeError, ConnectionResetError):
                    break

            logger.debug("Request closed.")


class _MJPEGHTTPServer(http.server.ThreadingHTTPServer):
    def __init__(self, server_address, publisher, max_fps):
        super().__init__(server_address, _MJPEGRequestHandler)
        self.publisher = publisher
        self.max_fps = max_fps


_Address = Union[str, Tuple[str, int]]


class MJPEGServer:
    def __init__(
        self,
        input_address: _Address,
        http_address: _Address,
        max_fps: Optional[int] = None,
    ):
        self._publisher = Publisher()
        self._http_server = _MJPEGHTTPServer(http_address, self._publisher, max_fps)
        self._listener = Listener(input_address)

    @property
    def input_address(self):
        return self._listener.address

    @property
    def http_address(self):
        return self._http_server.server_address

    def _listener_thread(self):
        try:
            with self._listener:
                while True:
                    logger.debug("Waiting for data source to connect...")
                    conn = self._listener.accept()
                    logger.info("Source connection accepted.")
                    receiver = threading.Thread(
                        target=self._receiver_thread, args=(conn,), daemon=True
                    )
                    receiver.start()
                    time.sleep(1)

        except:  # pragma: no cover
            self.shutdown()
            traceback.print_exc()

    def _receiver_thread(self, conn: Connection):
        with conn:
            while True:
                try:
                    buf = conn.recv_bytes()
                except EOFError:
                    break
                channel, data = unpack_message(buf)
                logger.debug("Received data from %s.", channel)
                self._publisher.publish(channel, data)
        logger.info("Source connection closed.")

    def serve_forever(self, poll_interval: float = 0.5) -> None:
        listener_thread = threading.Thread(target=self._listener_thread, daemon=True)
        listener_thread.start()

        self._http_server.serve_forever(poll_interval=poll_interval)

    def shutdown(self):
        self._http_server.shutdown()


if __name__ == "__main__":
    import argparse

    def parse_address(address: str):
        if address[0] == "/":
            # AF_UNIX
            return address
        if ":" in address:
            # AF_INET
            host, port = address.split(":", maxsplit=1)
            return host, int(port)
        raise ValueError(f"Unknown address format: {address!r}")

    def format_address(address):
        if isinstance(address, str):
            return address
        if isinstance(address, tuple):
            host, port = address
            return f"{host}:{port}"
        raise ValueError(f"Unknown address format: {address!r}")

    parser = argparse.ArgumentParser(
        description="Receive JPEG data and stream as MJPEG using HTTP multipart content."
    )
    parser.add_argument(
        "--input-address",
        dest="input_address",
        type=str,
        help="Listen for JPEG data on this address (default: %(default)s).",
        default="/tmp/mjpeg_streamer",
    )
    parser.add_argument(
        "--http-address",
        dest="http_address",
        type=str,
        help="Listen HTTP requests on this address (default: %(default)s).",
        default="localhost:8084",
    )
    parser.add_argument("--verbose", "-v", action="count", default=0)
    args = parser.parse_args()

    logging.basicConfig(level=max(0, logging.WARNING - args.verbose * 10))

    input_address = parse_address(args.input_address)
    http_address = parse_address(args.http_address)

    server = MJPEGServer(input_address, http_address)

    print(f"Listening on {format_address(server.input_address)} for input data.")
    print(
        f"Visit http://{format_address(server.http_address)}/<channel> to display the stream for the channel <channel>."
    )
    print("Press Ctrl-C to stop the server.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
        pass
