from morphocut.stitch import Region, Frame
import numpy as np
import pytest


@pytest.mark.parametrize("extra_shape", [tuple(), (3,)])
def test_Frame(extra_shape):
    shape = (10, 10) + extra_shape

    frame = Frame()
    ones = np.ones(shape)
    frame[:10, :10] = ones

    # Ranges need to be explicit
    with pytest.raises(ValueError, match="Can not infer stop from shape"):
        frame[:, :] = ones

    # Shapes of assigned arrays have to match
    with pytest.raises(ValueError, match="Mixed shape is not allowed"):
        frame[:10, :10] = ones[..., None]

    assert frame.shape == shape
    assert frame._shape == (None, None) + extra_shape

    regions = list(frame.iter_regions())
    assert len(regions) == 1

    # Make sure that all start/stop of the key are explicit
    assert regions[0].key == (slice(0, 10), slice(0, 10))

    # Convert to array
    frame_array = np.asarray(frame)
    assert frame.shape == frame_array.shape
    np.testing.assert_equal(frame_array, ones)

    # Requested region partially overlaps stored region
    region = frame[5:15, 5:15]
    assert region is not None
    assert region.key == (slice(5, 15), slice(5, 15))
    np.testing.assert_equal(region, np.pad(ones[:5, :5], ((5, 0), (5, 0))))  # type: ignore
