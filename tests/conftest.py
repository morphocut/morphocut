import glob
import pathlib

import pytest


@pytest.fixture(scope="session")
def data_path():
    return pathlib.Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def image_fns(data_path):
    image_glob = str(data_path / "images" / "*.png")

    image_fns = glob.glob(image_glob)

    assert len(image_fns) > 0

    return image_fns
