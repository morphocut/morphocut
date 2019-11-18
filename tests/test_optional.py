from morphocut._optional import import_optional_dependency
import pytest


def test_optional_hit():
    import_optional_dependency("morphocut")
    import_optional_dependency("skimage", min_version="0.16")


def test_optional_wrong_version():
    with pytest.raises(ImportError):
        import_optional_dependency("skimage", min_version="999")

    with pytest.warns(UserWarning, match="MorphoCut requires version.+"):
        import_optional_dependency("skimage", min_version="999", on_version="warn")


def test_optional_miss():
    with pytest.raises(ImportError):
        import_optional_dependency("foo_bar_baz")

    assert import_optional_dependency("foo_bar_baz", raise_on_missing=False) == None
