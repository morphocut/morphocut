import numpy as np
import pytest

from morphocut import Pipeline
from morphocut.integration.flowcam import FlowCamReader


def test_Find(data_path):
    with Pipeline() as pipeline:
        obj = FlowCamReader(
            data_path / "flowcam" / "flowcam_COMICS_DY086_20171118_MSC019_t0_4x.lst"
        )

    stream = pipeline.transform_stream()

    stream_obj = next(stream)

    with pytest.raises(AttributeError):
        stream_obj[obj].foo_bar

    assert isinstance(stream_obj[obj].image, np.ndarray)
    assert isinstance(stream_obj[obj].mask, np.ndarray)

    for _ in stream:
        pass
