import concurrent.futures
import os
import threading
from typing import Callable, List, Optional, Tuple, Union
import queue

from morphocut.core import (
    Node,
    Pipeline,
    Stream,
    StreamObject,
    StreamTransformer,
    closing_if_closable,
    check_stream,
    resolve_variable,
)


class MergeNodesPipeline(Pipeline):
    """

    This reduces the nesting of generators and therefore the stack size.

    Args:
        drop_errors (bool, optional): Drop objects from the stream for which the transformations fail.
    """

    children: List[Node]

    def __init__(
        self,
        parent: Optional["Pipeline"] = None,
        on_error: Optional[Callable] = None,
        on_error_args: Optional[Tuple] = None,
    ):
        super().__init__(parent)

        if on_error_args is None:
            on_error_args = tuple()

        self.on_error = on_error
        self.on_error_args = on_error_args

        self.__child_names: Optional[Tuple[Tuple[Node, Tuple[str, ...]], ...]] = None

    def add_child(self, child: StreamTransformer):
        if (
            not isinstance(child, Node)
            or child.__class__.transform_stream != Node.transform_stream
        ):
            raise ValueError(
                f"{self.__class__.__name__} only accepts Node instances with default transform_stream."
            )

        return super().add_child(child)

    @property
    def _child_names(self) -> Tuple[Tuple[Node, Tuple[str, ...]], ...]:
        if self.__child_names is not None:
            return self.__child_names

        self.__child_names = tuple(
            (child, child._get_parameter_names()) for child in self.children
        )

        return self.__child_names

    def transform_object(self, obj: StreamObject) -> StreamObject:
        for child, names in self._child_names:
            parameters = child.prepare_input(obj, names)
            result = child.transform(*parameters)  # type: ignore
            child.prepare_output(obj, result)

        return obj

    def transform_stream(self, stream: Optional[Stream] = None) -> Stream:
        stream = check_stream(stream)

        with closing_if_closable(stream):
            for obj in stream:
                try:
                    # Try to transform object. If any child fails, directly continue with the next object
                    yield self.transform_object(obj)
                except Exception as exc:
                    if self.on_error is not None:
                        on_error_args = resolve_variable(obj, self.on_error_args)
                        self.on_error(exc, *on_error_args)
                        continue
                    raise


class AggregateErrorsPipeline(MergeNodesPipeline):
    def transform_stream(self, stream: Optional[Stream] = None) -> Stream:
        stream = check_stream(stream)

        buffer: List[StreamObject] = []
        errors = []
        first_exc: None | Exception = None

        with closing_if_closable(stream):
            for obj in stream:
                try:
                    # Try to transform object. If any child fails, save error and continue
                    buffer.append(self.transform_object(obj))
                except Exception as exc:
                    if first_exc is None:
                        first_exc = exc
                    if self.on_error is not None:
                        on_error_args = resolve_variable(obj, self.on_error_args)
                        errors.append(self.on_error(exc, *on_error_args))
                    else:
                        errors.append(f"{type(exc).__name__}: {exc}")

        if errors:
            assert isinstance(first_exc, Exception)

            print()
            for error in errors:
                print(error)
            print()
            raise first_exc

        yield from buffer


class DataParallelPipeline(MergeNodesPipeline):
    def __init__(
        self,
        parent: Optional["Pipeline"] = None,
        on_error: Optional[Callable] = None,
        on_error_args: Optional[Tuple] = None,
        executor: Union[None, int, concurrent.futures.Executor] = None,
    ):
        super().__init__(parent, on_error, on_error_args)

        if executor is None:
            executor = os.cpu_count() or 4

        self.executor = executor

    def transform_stream(self, stream: Optional[Stream] = None) -> Stream:
        stream = check_stream(stream)

        executor = (
            self.executor
            if isinstance(self.executor, concurrent.futures.Executor)
            else concurrent.futures.ThreadPoolExecutor(self.executor)
        )

        # TODO: This only works for ThreadPoolExecutor and ProcessPoolExecutor
        queue_size = executor._max_workers
        result_queue = queue.Queue(queue_size)

        def _queue_filler():
            with closing_if_closable(stream):
                try:
                    for obj in stream:
                        result_queue.put(
                            (obj, executor.submit(self.transform_object, obj))
                        )
                finally:
                    # Stop signal
                    result_queue.put(None)

        threading.Thread(target=_queue_filler, daemon=True).start()

        while True:
            obj_future = result_queue.get()

            if obj_future is None:
                return

            obj, future = obj_future

            try:
                yield future.result()
            except Exception as exc:
                if self.on_error is not None:
                    on_error_args = resolve_variable(obj, self.on_error_args)
                    self.on_error(exc, *on_error_args)
                    continue
                raise
