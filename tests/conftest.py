import pytest
import os.path
import glob


@pytest.fixture
def image_fns():
    image_glob = os.path.join(os.path.realpath(
        os.path.dirname(__file__)), "images", "*.png")

    image_fns = glob.glob(image_glob)

    assert len(image_fns) > 0

    return image_fns
