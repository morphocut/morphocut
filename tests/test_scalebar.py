import numpy as np
import pytest
from morphocut import Node, Pipeline
from morphocut.stream import Unpack
from morphocut.scalebar import DrawScalebar, draw_scalebar


@pytest.fixture
def test_image():
    return np.zeros((100, 100), dtype=np.uint8)  # Grayscale image


@pytest.mark.parametrize("length_unit, px_per_unit", [
    (100, 1),
    (200, 2),
    (50, 0.5),
])
def test_draw_scalebar_function(length_unit, px_per_unit):
    result = draw_scalebar(length_unit, px_per_unit=px_per_unit)

    expected_width = int((length_unit * px_per_unit) + 20)  # 10 margin on each side

    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 32
    assert result.shape[1] == expected_width


def test_DrawScalebar_pipeline(test_image):
    length_unit = 100

    with Pipeline() as pipeline:
        Unpack([test_image])
        DrawScalebar(image=test_image, length_unit=length_unit)

    # Execute the pipeline
    objects = list(pipeline.transform_stream())

    # Check if the output image has the expected properties
    assert len(objects) == 1

    # Access the output data
    output_data = objects[0].data
    print(output_data.keys())

    # Retrieve one of the images (using the first key in the dict)
    result_image = output_data[next(iter(output_data.keys()))]

