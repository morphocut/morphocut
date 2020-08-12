from morphocut.core import Node
from morphocut.parallel import ParallelPipeline
import threading
import multiprocessing
import datetime
import queue
import enum


class _Signal(enum.Enum):
    STOP = 0


class Profiler:
    def __init__(self):
        self._orig_prepare_input = None
        self._orig_prepare_output = None

        # The use of a Queue allows events from child processes to be recorded.
        # TODO: The length of the queue is limited to 2^31-1.
        self._event_queue = multiprocessing.Queue()

        self._parallel_pipelines = set()

        print(self._event_queue._maxsize)

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, *_, **__):
        self.disable()

    def enable(self):
        self._orig_prepare_input = Node.prepare_input
        self._orig_prepare_output = Node.prepare_output

        def prepare_input(node, obj, *args, **kwargs):
            self._log_node_event(node, "in", id(obj))
            return self._orig_prepare_input(node, obj, *args, **kwargs)

        def prepare_output(node, obj, *args, **kwargs):
            self._log_node_event(node, "out", id(obj))
            return self._orig_prepare_output(node, obj, *args, **kwargs)

        Node.prepare_input = prepare_input
        Node.prepare_output = prepare_output

    def disable(self):
        if None in (self._orig_prepare_input, self._orig_prepare_output):
            return

        Node.prepare_input = self._orig_prepare_input
        Node.prepare_output = self._orig_prepare_output

    def _log_node_event(self, node: Node, evt, evt_data=None):
        self._event_queue.put(
            dict(
                pid=multiprocessing.current_process().pid,
                tid=threading.get_ident(),
                node_rank=node.rank,
                cls=node.__class__.__name__,
                datetime=datetime.datetime.now(datetime.timezone.utc),
                evt=evt,
                evt_data=evt_data,
            )
        )

    def collect(self):
        self._event_queue.put(_Signal.STOP)

        result = []

        while True:
            item = self._event_queue.get()
            if item == _Signal.STOP:
                break
            result.append(item)

        return result

