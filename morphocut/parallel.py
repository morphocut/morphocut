import enum
import multiprocessing
import multiprocessing.synchronize
import os
import sys
import threading
import traceback

from morphocut.core import Pipeline


class _Signal(enum.Enum):
    END = 0
    YIELD = 1


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
            input_obj = input_queue.get()

            if input_obj is _Signal.END:
                # Nothing is left to be done
                output_queue.put(_Signal.END)
                break

            if stop_event.is_set():
                # If stop event is set, continue until _Signal.END to empty the queue
                continue

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

            # Signalize the collector to switch to the next worker
            output_queue.put(_Signal.YIELD)

    except KeyboardInterrupt:
        pass


class ParallelPipeline(Pipeline):
    """
    Parallel processing of the stream in multiple processes.

    Args:
        num_workers (int, optional): Number of worker processes.
            Default: Number of CPUs in the system.
        parent (:py:class:`Pipeline <morphocut.Pipeline>`):
            The parent pipeline.

    Note:
        :py:class:`ParallelPipeline <morphocut.parallel.ParallelPipeline>` creates
        distinct copies of its nodes in each worker thread that are not accessible
        from the main thread.

    Example:
        .. code-block:: python

            with Pipeline() as pipeline:
                # Regular sequential processing
                ...
                with ParallelPipeline(parent=pipeline) as pp:
                    # Parallelized processing in this block,
                    # work is distributed between all cores.
                    ...

            pipeline.run()
    """

    def __init__(
        self, num_workers=None, queue_size=0, multiprocessing_context=None, parent=None
    ):
        super().__init__(parent=parent)

        if num_workers is None:
            num_workers = multiprocessing.cpu_count()

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
            for i, obj in enumerate(stream):
                if stop_event.is_set():
                    break

                # Send objects to workers in a round-robin fashion
                worker_idx = i % self.num_workers
                input_queues[worker_idx].put(obj)

            # Tell all workers to stop working
            for iqu in input_queues:
                iqu.put(_Signal.END)

        qf = threading.Thread(target=_queue_filler)
        qf.start()

        # Read output queues in the main thread
        try:
            while any(workers_running):
                for i, oqu in enumerate(output_queues):
                    if not workers_running[i]:
                        continue

                    while True:
                        output_object = oqu.get()

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
            # Anything, but most importantly GeneratorExit
            print("Stopping workers...")
            stop_event.set()
            raise
        finally:
            qf.join()

            for w in workers:
                w.join()
