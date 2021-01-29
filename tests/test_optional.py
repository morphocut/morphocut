from morphocut._optional import (
    UnavailableObject,
    check_available,
    UnavailableObjectError,
)
import pytest


def test_OptionalObject():
    try:
        import foo_bar_baz
    except ImportError:
        foo_bar_baz = UnavailableObject("foo_bar_baz")

    with pytest.raises(UnavailableObjectError):
        check_available(foo_bar_baz)
