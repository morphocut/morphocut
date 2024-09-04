from morphocut.stitch import Region, Frame
import numpy as np
import pytest


@pytest.mark.parametrize("extra_shape", [tuple(), (3,)])
def test_Frame(extra_shape):
    shape = (10, 10) + extra_shape

    data = np.empty(shape, dtype=int)
    data[...] = np.arange(data.size).reshape(shape)

    frame = Frame()
    frame[5:15, 5:15] = data

    # Ranges need to be explicit
    # TODO: Actually, we can guess it from assigned data
    with pytest.raises(IndexError, match="Can not infer stop from shape"):
        frame[:, :] = data

    # Shapes of assigned arrays have to match
    with pytest.raises(ValueError, match="Mixed shape is not allowed"):
        frame[5:15, 5:15] = data[..., None]

    assert frame.shape == (15, 15) + extra_shape
    assert frame._shape == (None, None) + extra_shape

    regions = list(frame.iter_regions())
    assert len(regions) == 1

    # Make sure that all start/stop of the key are explicit
    assert regions[0].key == (slice(5, 15), slice(5, 15))

    # Convert to array
    frame_array = np.asarray(frame)
    assert frame.shape == frame_array.shape
    np.testing.assert_equal(frame_array[5:15, 5:15], data)

    # Requested region is partially outside of stored regions
    # Overlap to the right is truncated (like with np.ndarray)
    # Overlap to the left is padded
    region = frame[10:20, 0:15]
    assert region is not None
    assert region.key == (slice(10, 15), slice(0, 15))
    assert region.shape == (5, 15) + extra_shape
    np.testing.assert_equal(region, np.pad(data[5:, :], ((0, 0), (5, 0)) + ((0, 0),) * len(extra_shape)))  # type: ignore

    float_region = region.astype(float)
    assert float_region.key == region.key

    assert isinstance(region.sum(axis=0), np.ndarray)


@pytest.mark.parametrize("dtype", [int, float])
@pytest.mark.parametrize("fill_value", [0, 1])
def test_Frame_blend_samevalue(dtype, fill_value):
    # If the values in the overlapping areas are the same, all blending strategies should yield the same result
    frames = [
        Frame(dtype(fill_value), blend_strategy=blend_strategy)
        for blend_strategy in ["overwrite", "linear"]
    ]
    for f in frames:
        f[0:24, 0:24] = dtype(1.0)
        f[12:36, 12:36] = dtype(1.0)
        f[12:15, 12:15] = dtype(1.0)

    for f in frames[1:]:
        np.testing.assert_allclose(frames[0].asarray(), f.asarray())

    # If the regions don' overlap, all blending strategies should yield the same result
    frames = [
        Frame(0.0, blend_strategy=blend_strategy)
        for blend_strategy in ["overwrite", "linear"]
    ]
    for f in frames:
        f[0:24, 0:24] = 1.0
        f[24:48, 24:48] = 2.0

    for f in frames[1:]:
        np.testing.assert_allclose(frames[0].asarray(), f.asarray())
