import enum
import multiprocessing
import multiprocessing.synchronize
import os
import queue
import sys
import threading
import traceback

from morphocut.core import Pipeline, closing_if_closable


class _Signal(enum.Enum):
    END = 0
    YIELD = 1


QUEUE_POLL_INTERVAL = 0.1

# TODO: Look at pytorch/torch/utils/data/_utils/worker.py for determining if the parent is dead


class StrRepr(str):
    def __repr__(self):
        return self


class ExceptionWrapper:
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
            msg = StrRepr(msg)
        raise self.exc_type(msg)


class _Stop(Exception):
    """Raised by _get_until_stop."""


def _put_until_stop(queue_, stop_event, obj):
    while True:
        if stop_event.is_set():
            return False
        try:
            queue_.put(obj, True, QUEUE_POLL_INTERVAL)
        except queue.Full:
            continue
        else:
            break
    return True


def _get_until_stop(queue_, stop_event):
    while True:
        if stop_event.is_set():
            raise _Stop()
        try:
            return queue_.get(True, QUEUE_POLL_INTERVAL)
        except queue.Empty:
            continue
        else:
            break


def _worker_loop(
    input_queue: multiprocessing.Queue,
    output_queue: multiprocessing.Queue,
    transform_stream,
    stop_event: multiprocessing.synchronize.Event,
):
    """
    Do the actual work.

    1. Get an object from the input queue.
    2. Transform this object, and
    3. Put all resulting objects onto the result queue.
    4. Put an additional _Signal.YIELD onto the result queue to signalize
       to the reader that it may switch to another worker.
    """
    try:
        while True:
            input_obj = _get_until_stop(input_queue, stop_event)

            if input_obj is _Signal.END:
                # Nothing is left to be done
                output_queue.put(_Signal.END)
                break

            try:
                # Transform object
                for output_obj in transform_stream([input_obj]):
                    if stop_event.is_set():
                        # If stop event is set, quit processing this object
                        break
                    # Put results onto the output_queue
                    output_queue.put(output_obj)
            except:  # pylint: disable=bare-except
                # Put exception to queue and quit processing
                output_queue.put(
                    ExceptionWrapper("worker process PID {}".format(os.getpid()))
                )
                break
            finally:
                # Signalize the collector to switch to the next worker
                output_queue.put(_Signal.YIELD)

    except KeyboardInterrupt:
        pass
    except _Stop:
        pass


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
        :py:class:`~morphocut.parallel.ParallelPipeline` creates
        distinct copies of its nodes in each worker thread that are not accessible
        from the main thread.

    Example:
        .. code-block:: python

            with Pipeline() as pipeline:
                # Regular sequential processing
                ...
                with ParallelPipeline():
                    # Parallelized processing in this block,
                    # work is distributed between all cores.
                    ...

            pipeline.run()
    """

    def __init__(
        self, num_workers=None, queue_size=2, multiprocessing_context=None, parent=None
    ):
        super().__init__(parent=parent)

        if num_workers is None:
            num_workers = multiprocessing.cpu_count()

        assert num_workers >= 1

        self.num_workers = num_workers
        self.queue_size = queue_size

        if multiprocessing_context is None:
            multiprocessing_context = multiprocessing
        self.multiprocessing_context = multiprocessing_context

    def transform_stream(self, stream):
        # Create queues and worker
        input_queues = []  # type: List[multiprocessing.Queue]
        output_queues = []  # type: List[multiprocessing.Queue]
        workers = []  # type: List[multiprocessing.Process]
        workers_running = []  # type: List[bool]
        stop_event = self.multiprocessing_context.Event()
        upstream_exception = []  # type: List[Exception]
        for i in range(self.num_workers):
            iqu = self.multiprocessing_context.Queue(self.queue_size)
            oqu = self.multiprocessing_context.Queue()

            w = self.multiprocessing_context.Process(
                target=_worker_loop,
                args=(iqu, oqu, super().transform_stream, stop_event),
            )
            w.daemon = True
            w.start()
            input_queues.append(iqu)
            output_queues.append(oqu)
            workers.append(w)
            workers_running.append(True)

        # Fill input queues in a thread
        def _queue_filler():
            try:
                with closing_if_closable(stream):
                    for i, obj in enumerate(stream):
                        # Send objects to workers in a round-robin fashion
                        worker_idx = i % self.num_workers

                        if not _put_until_stop(
                            input_queues[worker_idx], stop_event, obj
                        ):
                            return

                # Tell all workers to stop working
                for iqu in input_queues:
                    if not _put_until_stop(iqu, stop_event, _Signal.END):
                        return

            except Exception as exc:
                # Stop everything immediately
                stop_event.set()
                print("ParallelPipeline._queue_filler", exc)
                upstream_exception.append(ExceptionWrapper(exc))

        qf = threading.Thread(
            target=_queue_filler, name="ParallelPipeline._queue_filler"
        )
        qf.start()

        # Read output queues in the main thread
        try:
            while any(workers_running):
                for i, oqu in enumerate(output_queues):
                    if not workers_running[i]:
                        continue

                    while True:
                        output_object = _get_until_stop(oqu, stop_event)

                        if output_object is _Signal.END:
                            workers_running[i] = False
                            break

                        if output_object is _Signal.YIELD:
                            # Switch to the next worker
                            break

                        # Re-raise exceptions from workers and stop
                        if isinstance(output_object, ExceptionWrapper):
                            stop_event.set()
                            output_object.reraise()

                        yield output_object
        except (SystemExit, KeyboardInterrupt, GeneratorExit, Exception) as exc:
            print("Stopping workers due to {}...".format(type(exc).__name__))
            stop_event.set()
            raise
        finally:
            qf.join()

            for w in workers:
                w.join()

            if upstream_exception:
                upstream_exception[0].reraise()
