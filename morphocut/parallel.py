import enum
import multiprocessing
import threading
import multiprocessing.synchronize

from morphocut.core import Pipeline


class _Signal(enum.Enum):
    END = 0
    YIELD = 1


# TODO: Look at pytorch/torch/utils/data/_utils/worker.py for determining if the parent is dead


def _worker_loop(
    input_queue: multiprocessing.Queue,
    output_queue: multiprocessing.Queue,
    transform_stream,
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

            try:
                # Transform object
                for output_obj in transform_stream([input_obj]):
                    # Put results onto the output_queue
                    output_queue.put(output_obj)
            except:
                # TODO: Put exception to queue
                raise

            # Signalize the collector to switch to the next worker
            output_queue.put(_Signal.YIELD)

    except KeyboardInterrupt:
        pass


class ParallelPipeline(Pipeline):
    def __init__(self, num_workers, multiprocessing_context=None, parent=None):
        super().__init__(parent=parent)
        self.num_workers = num_workers

        if multiprocessing_context is None:
            multiprocessing_context = multiprocessing
        self.multiprocessing_context = multiprocessing_context

    def transform_stream(self, stream):
        # Create queues and worker
        input_queues = []  # type: List[multiprocessing.Queue]
        output_queues = []  # type: List[multiprocessing.Queue]
        workers = []  # type: List[multiprocessing.Process]
        workers_running = []  # type: List[bool]
        for i in range(self.num_workers):
            iqu = self.multiprocessing_context.Queue()
            oqu = self.multiprocessing_context.Queue()

            w = self.multiprocessing_context.Process(
                target=_worker_loop, args=(iqu, oqu, super().transform_stream)
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
                # Send objects to workers in a round-robin fashion
                worker_idx = i % self.num_workers
                input_queues[worker_idx].put(obj)

            # Tell all workers to stop working
            for iqu in input_queues:
                iqu.put(_Signal.END)

        qf = threading.Thread(target=_queue_filler)
        qf.start()

        # Read output queues in the main thread
        while any(workers_running):
            for i, oqu in enumerate(output_queues):
                while True:
                    output_object = oqu.get()

                    if output_object is _Signal.END:
                        workers_running[i] = False
                        break

                    if output_object is _Signal.YIELD:
                        # Switch to the next worker
                        break

                    # TODO: Raise exceptions from workers

                    yield output_object
