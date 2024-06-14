import sys


def exc_add_note(exc: BaseException, msg: str) -> None:
    if sys.version_info < (3, 11):
        exc.__notes__ = getattr(exc, "__notes__", []) + [msg]
    else:
        exc.add_note(msg)
