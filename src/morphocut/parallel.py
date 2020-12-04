import collections.abc
import enum
import multiprocessing
import queue
import sys
import threading
import traceback
from typing import Optional
import signal
from morphocut.core import Pipeline, Stream, StreamObject, closing_if_closable
import logging

QUEUE_POLL_INTERVAL = 1

# Store names for exit codes

_exitcode_to_signame = {}

for name, signum in list(signal.__dict__.items()):
    if name[:3] == "SIG" and "_" not in name:
        _exitcode_to_signame[-signum] = f"-{name}"

_logger = logging.getLogger(__name__)


class _Message:
    pass


class EndOfStream(_Message):
    pass


class _WorkerFinished(_Message):
    __slots__ = ["pid"]

    def __init__(self, pid):
        self.pid = pid


class _StrRepr(str):
    def __repr__(self):  # pylint: disable=invalid-repr-returned
        return self


class _WrappedException(_Message):
    """Wraps an exception plus traceback."""

    def __init__(self, where):
        exc_info = sys.exc_info()
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))
        self.where = where

    def reraise(self):
        """Reraise the wrapped exception in the current thread."""

        msg = "{} in {}:\n{}".format(self.exc_type.__name__, self.where, self.exc_msg)

        if self.exc_type == KeyError:
            msg = _StrRepr(msg)
        raise self.exc_type(msg) from None


class _Feeder(threading.Thread):
    """Feeder thread for the input queue."""

    def __init__(
        self,
        stream,
        input_queue: multiprocessing.Queue,
        stop_event: multiprocessing.Event,
    ):
        super().__init__(daemon=True)

        self._stream = stream
        self._input_queue = input_queue
        self._stop_event = stop_event
        self._exception = None

    def run(self):
        try:
            with closing_if_closable(self._stream):
                for i, obj in enumerate(self._stream):
                    if self._stop_event.is_set():
                        break

                    while not self._stop_event.is_set():
                        try:
                            self._input_queue.put(obj, timeout=QUEUE_POLL_INTERVAL)
                            _logger.debug(f"_Feeder put {i}:{obj} to queue.")
                            break
                        except queue.Full:
                            _logger.debug("_Feeder waiting for queue.")

            while not self._stop_event.is_set():
                try:
                    self._input_queue.put(EndOfStream(), timeout=QUEUE_POLL_INTERVAL)
                    break
                except queue.Full:
                    _logger.debug("_Feeder waiting for queue (EndOfStream).")
            _logger.debug(f"_Feeder put EndOfStream to queue.")
        except Exception as exc:
            # Stop everything immediately
            self._stop_event.set()
            self._input_queue.cancel_join_thread()
            _logger.info(f"Exception in _Feeder: {exc}")
            self._exception = _WrappedException(exc)

    @property
    def exception(self):
        return self._exception

    def stop(self):
        self._stop = True


class _Stop(Exception):
    pass


class WorkerDiedException(Exception):
    pass


class _Worker(multiprocessing.Process):
    """Worker process that executes Pipeline.transform_stream."""

    def __init__(
        self,
        pipeline: "ParallelPipeline",
        input_queue: multiprocessing.Queue,
        output_queue: multiprocessing.Queue,
        stop_event: multiprocessing.Event,
    ):
        super().__init__(daemon=True)

        self._pipeline = pipeline
        self._input_queue = input_queue
        self._output_queue = output_queue
        self._stop_event = stop_event
        self.running = multiprocessing.Event()
        self._finished = multiprocessing.Event()

    def start(self):
        multiprocessing.Process.start(self)

        watchdog = threading.Thread(target=self._watchdog, name=f"{self.name}-watchdog")
        watchdog.start()

    def _watchdog(self):
        try:
            while True:
                if self._finished.wait(QUEUE_POLL_INTERVAL):
                    break

                if not self.is_alive():
                    raise WorkerDiedException(
                        f"{self.name} died unexpectedly. Exit code: {_exitcode_to_signame.get(self.exitcode,self.exitcode)}"
                    )
        except Exception as exc:
            # Stop everything immediately
            self._stop_event.set()
            _logger.info(f"Exception in {self.name}._watchdog: {exc}")
            self._output_queue.put(_WrappedException(self.name), block=True)

    def run(self):
        _logger.debug(f"{self.name} started.")
        self.running.set()

        try:
            # Read from input queue
            stream = QueueIterator(
                self._input_queue, self._stop_event, name=f"QueueIterator({self.name})"
            )

            # Transform
            stream = Pipeline.transform_stream(self._pipeline, stream)

            # Write to output queue
            for i, obj in enumerate(stream):
                while True:
                    try:
                        self._output_queue.put(
                            obj, block=True, timeout=QUEUE_POLL_INTERVAL
                        )
                        break
                    except queue.Full:
                        if self._stop_event.is_set():
                            raise _Stop() from None
                        continue

            _logger.debug(f"{self.name} finished processing stream.")
            self._output_queue.put(_WorkerFinished(self.pid), block=True)
        except _Stop:
            pass
        except Exception as exc:
            # Stop everything immediately
            self._stop_event.set()
            _logger.info(f"Exception in {self.name}: {exc}")
            self._output_queue.put(_WrappedException(self.name), block=True)
        finally:
            _logger.debug(f"{self.name} announcing finished...")
            self._finished.set()
            _logger.debug(f"{self.name} done.")


class QueueIterator(collections.abc.Iterator):
    def __init__(
        self, queue: multiprocessing.Queue, stop_event: multiprocessing.Event, name=None
    ):
        self._queue = queue
        self._stop_event = stop_event
        self._name = name or self.__class__.__name__
        self._stop = False

    def __next__(self):
        "Return the next item from the queue. When exhausted, raise StopIteration"
        while not self._stop:
            try:
                obj = self._queue.get(block=True, timeout=QUEUE_POLL_INTERVAL)
                _logger.debug(f"{self._name} received object.")
            except queue.Empty:
                if self._stop_event.is_set():
                    raise _Stop() from None
                _logger.debug(f"{self._name} waiting for queue.")
                continue

            if isinstance(obj, EndOfStream):
                _logger.debug(f"{self._name} received EndOfStream.")
                # Put back signal so that other workers can receive it as well
                self._queue.put(obj, block=True)
                _logger.debug(f"{self._name} put EndOfStream back to queue.")
                raise StopIteration

            return obj

    def stop(self):
        self._stop = True


from typing import Dict, Any


class ParallelPipeline(Pipeline):
    """
    Parallel processing of the stream in multiple processes.

    Args:
        num_workers (int, optional): Number of worker processes.
            Default: Number of CPUs in the system.
        queue_size (int, optional): Upperbound limit on the number of items
            that can be placed in the input queue of each worker.
            If queue_size is less than or equal to zero, the queue size is infinite.
        multiprocessing_context (optional): Result of :py:func:`multiprocessing.get_context`.
        parent (:py:class:`~morphocut.core.Pipeline`):
            The parent pipeline.

    Note:
        The order in which objects are processed and returned is indeterminate.

    Note:
        :py:class:`~morphocut.parallel.ParallelPipeline` creates
        distinct copies of its nodes in each worker process that do not share state and are not accessible from
        or reflected in the main thread.

    Example:
        .. code-block:: python

            with Pipeline() as pipeline:
                # Regular sequential processing
                ...
                with ParallelPipeline():
                    # Parallelized processing in this block,
                    # work is distributed between cores.
                    ...

            pipeline.run()
    """

    def __init__(self, n_workers: Optional[int] = None):
        super().__init__()

        if n_workers is None:
            n_workers = multiprocessing.cpu_count()
        self._n_workers = n_workers

        self._input_queue = multiprocessing.Queue(self._n_workers)
        self._output_queue = multiprocessing.Queue(self._n_workers)
        self._feeder = None
        self._stop_event = multiprocessing.Event()
        self._workers: Dict[Any, _Worker] = {}

    def transform_stream(self, stream: Optional[Stream] = None) -> Stream:
        """
        Run the stream through all nodes and return it.

        Args:
            stream: A stream to transform.
                *This argument is solely to be used internally.*

        Returns:
            Stream: An iterable of stream objects.
        """
        if stream is None:
            stream = [StreamObject()]

        self._stop_event.clear()

        for _ in range(self._n_workers):
            w = _Worker(self, self._input_queue, self._output_queue, self._stop_event)
            w.start()
            if not w.running.wait(2):
                print(w)
                raise Exception(f"{w.name} did not enter running state.")
            self._workers[w.pid] = w
        _logger.debug(f"{len(self._workers)} workers started.")

        self._feeder = _Feeder(stream, self._input_queue, self._stop_event)
        self._feeder.start()
        _logger.debug("Feeder started.")

        stream = QueueIterator(
            self._output_queue, self._stop_event, name="QueueIterator(main)"
        )

        try:
            for obj in stream:
                if isinstance(obj, StreamObject):
                    yield obj
                elif isinstance(obj, _WorkerFinished):
                    w: _Worker = self._workers[obj.pid]
                    _logger.debug(f"Joining {w.name}...")
                    w.join(1)
                    w.terminate()
                    _logger.debug(f"Joined {w.name}.")
                    del self._workers[obj.pid]

                    _logger.debug(f"{len(self._workers)} workers remaining.")

                    if not self._workers:
                        _logger.debug("No workers left, exiting...")
                        break
                    continue
                elif isinstance(obj, _WrappedException):
                    obj.reraise()
                else:
                    raise Exception("Unknown message:", obj)
        except _Stop:
            if self._feeder.exception is not None:
                self._feeder.exception.reraise()
        finally:
            # Stop everything that is left
            self._stop_event.set()
