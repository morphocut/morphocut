from morphocut.core import Pipeline, ReturnOutputs, Output
from morphocut.filters import MaxFilter, MedianFilter, _BaseFilter
from morphocut.stream import Unpack
import numpy as np
import pytest

@ReturnOutputs
@Output("response")
class IdFilter(_BaseFilter):
    def _update(self, value):
        return value

@pytest.mark.parametrize("centered", [True, False])
@pytest.mark.parametrize("filter_cls", [MaxFilter, MedianFilter, IdFilter])
def test_filter_scalar(filter_cls, centered):
    values = list(range(10))

    with Pipeline() as p:
        value = Unpack(values)
        response = filter_cls(value, size=5, centered=centered)

    responses = [obj[response] for obj in p.transform_stream()]

    assert len(responses) == len(values)

@pytest.mark.parametrize("filter_cls", [MaxFilter, MedianFilter])
def test_filter_scalar_centered_is_symmetric(filter_cls):
    values = list(range(10))
    values_r = values[::-1]

    print(values)
    print(values_r)

    with Pipeline() as p:
        value, value_r = Unpack(zip(values, values_r)).unpack(2)
        
        response = filter_cls(value, size=5, centered=True)
        response_r = filter_cls(value_r, size=5, centered=True)

    objs = list(p.transform_stream())
    responses = [obj[response] for obj in objs]
    responses_r = [obj[response_r] for obj in objs]

    assert responses == responses_r[::-1]

@pytest.mark.parametrize("filter_cls", [MaxFilter, MedianFilter])
def test_filter_numpy(filter_cls):
    values = np.arange(20)[:, np.newaxis] * np.ones((10))

    with Pipeline() as p:
        value = Unpack(values)
        response = filter_cls(value, size=3, centered=False)

    responses = [obj[response] for obj in p.transform_stream()]

    print(values)
    print(responses)

    assert len(responses) == len(values)