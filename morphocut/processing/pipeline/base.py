from abc import abstractmethod, ABC

__all__ = ["NodeBase"]


class NodeBase(ABC):
    @abstractmethod
    def __call__(self, input=None):  # pragma: no cover
        """
        Process the input stream
        """
        while False:
            yield None
