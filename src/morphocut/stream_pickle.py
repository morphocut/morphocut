"""
Pickle and unpickle StreamObjects that are derived from a base object.

For values in StreamObjects that are members of the base object, only a reference is stored.
This way, the memory footprint of a pickled object is minimized.
"""

import io
import pickle

from morphocut.core import StreamObject

__all__ = [
    "StreamObjectPickler",
    "StreamObjectUnpickler",
    "dump",
    "dumps",
    "load",
    "loads",
]


class StreamObjectPickler(pickle.Pickler):
    """
    Pickler for StreamObjects that are derived from a base object.

    Arguments:
        file: Binary file-like that has a write() method.
        base_object (StreamObject): Base object that is used as a template.
        **kwargs: Passed to the underlying :class:`pickle.Pickler`.
    """

    def __init__(self, file, base_object: StreamObject, **kwargs):
        super().__init__(file, **kwargs)
        self.base_object = base_object
        self._lut = {id(v): k for k, v in base_object.items()}

    def dump(self, obj: StreamObject) -> None:
        # Only dump keys that are not in base_object
        return super().dump({k: v for k, v in obj.items() if k not in self.base_object})

    def persistent_id(self, obj):
        # Replace an object with a reference to a key inside base_object
        try:
            key = self._lut[id(obj)]
            return key
        except KeyError:
            return None


def dump(obj, file, base_object: StreamObject, **kwargs) -> None:
    StreamObjectPickler(file, base_object, **kwargs).dump(obj)


def dumps(obj, base_object: StreamObject, **kwargs) -> bytes:
    f = io.BytesIO()
    StreamObjectPickler(f, base_object, **kwargs).dump(obj)
    return f.getvalue()


class StreamObjectUnpickler(pickle.Unpickler):
    """
    Unpickler for StreamObjects serialized using :class:`StreamObjectPickler`.

    Arguments:
        file: Binary file-like object with read() and readline() methods.
        base_object (StreamObject): Base object that is used as a template.
        **kwargs: Passed to the underlying :class:`pickle.Unpickler`.
    """

    def __init__(self, file, base_object: StreamObject, **kwargs):
        super().__init__(file, **kwargs)
        self.base_object = base_object

    def persistent_load(self, pid):
        return self.base_object[pid]

    def load(self) -> StreamObject:
        obj = self.base_object.copy()
        obj.update(super().load())
        return obj


def load(file, base_object: StreamObject, **kwargs) -> StreamObject:
    return StreamObjectUnpickler(file, base_object, **kwargs).load()


def loads(s: bytes, base_object: StreamObject, **kwargs) -> StreamObject:
    if isinstance(s, str):
        raise TypeError("Can't load pickle from unicode string")
    file = io.BytesIO(s)
    return StreamObjectUnpickler(file, base_object, **kwargs).load()
