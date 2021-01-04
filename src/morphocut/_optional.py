"""
Optional dependency checking.
"""

__all__ = ["UnavailableObjectError", "UnavailableObject", "check_optional_object"]

import sys


class UnavailableObjectError(Exception):
    pass


class UnavailableObject:
    """
    An unavailable object.
    """

    def __init__(self, name, msg=None):
        self.name = name
        self.msg = msg
        self._orig_exc = sys.exc_info()[1]

    def raise_(self, *_, **__):
        msg = f"{self.name} is unavailable."
        if self.msg is not None:
            msg = f"{msg}\n\n{self.msg}"

        raise UnavailableObjectError(msg) from self._orig_exc

    __call__ = raise_
    __getattr__ = raise_
    __getitem__ = raise_


def check_available(*objects):
    """
    Check the availability of an optional object.

    This is optional but can be used to have somthing fail early.
    """
    for obj in objects:
        if isinstance(obj, UnavailableObject):
            obj.raise_()