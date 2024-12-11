from morphocut.core import Pipeline, Call
from morphocut.stitch import Region, Frame, Stitch
from morphocut.stream import Unpack
import numpy as np
import pytest


def test_Region():
    region1 = Region(np.array([[1, 2], [3, 4]]), (slice(0, 2), slice(0, 2)))

    region2 = Region(np.array([[1, 2], [3, 4]]), (slice(0, 2), slice(0, 2)))
    region2[:, 1] = np.array([5, 6])
    np.testing.assert_equal(region2, np.array([[1, 5], [3, 6]]))

    # Test with slice only
    subregion_slice_only = region1[0:2]
    np.testing.assert_equal(subregion_slice_only, np.array([[1, 2], [3, 4]]))
    assert subregion_slice_only.key == (slice(0, 2, 1), slice(0, 2, None))

    # Test with tuple of slices
    subregion = region1[0:2, 0:1]
    np.testing.assert_equal(subregion, np.array([[1], [3]]))
    assert subregion.key == (slice(0, 2, 1), slice(0, 1, 1))

    # Test with Ellipsis
    subregion_ellipsis = region1[...]
    np.testing.assert_equal(subregion_ellipsis, np.array([[1, 2], [3, 4]]))
    assert subregion_ellipsis.key == (slice(0, 2, 1), slice(0, 2, 1))

    # Test with Boolean indexing
    bool_idx = np.array([[True, False], [False, True]])
    subregion_bool = region1[bool_idx]
    np.testing.assert_equal(subregion_bool, np.array([1, 4]))

    # Test with None indexing
    subregion_none = region1[None, :]
    np.testing.assert_equal(subregion_none, np.array([[[1, 2], [3, 4]]]))

    # Test __array_ufunc__ which is called by numpy when using high-level functions like np.add, np.or, ...
    np.testing.assert_equal(np.add(region1, 1).array, np.add(region1.array, +1))  # type: ignore
    np.testing.assert_equal(np.add(region1, 1, out=np.empty_like(region1.array)).array, np.add(region1.array, +1))  # type: ignore
    assert np.add.at(region1.copy(), [[0, 0]], 1) is None
    a, b = np.divmod(region1, 2)
    np.testing.assert_equal(a.array, region1.array // 2)
    np.testing.assert_equal(b.array, region1.array % 2)


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

    assert frame.ndim == len(shape)
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

    with pytest.raises(
        IndexError, match="Invalid key: .* Only slice and tuple of slice are allowed."
    ):
        frame._validate_key(0)

    with pytest.raises(IndexError, match="Invalid key: .* step has to be 1 or None."):
        frame._validate_key((slice(0, 2, 2), slice(0, 2)))

    # Test with a key that has stop=None
    frame2 = Frame()
    with pytest.raises(IndexError, match="Can not infer stop from shape"):
        frame2[5:15, 5:] = data

    # Test with mixed key lengths
    frame3 = Frame(shape=(None, None) + extra_shape)
    with pytest.raises(ValueError, match="Mixed shape is not allowed"):
        frame3[5:15] = data

    # Test with a key that goes out of bounds for a fixed shape dimension
    frame4 = Frame(shape=(20, 20) + extra_shape)
    with pytest.raises(IndexError, match="Invalid key:.*Out of range"):
        frame4[15:25, 5:15] = data

    # Test with mixed dtype values
    data_float = data.astype(float)
    with pytest.raises(ValueError, match="Mixed dtype is not allowed"):
        frame[5:15, 5:15] = data_float

    # Test when there are no regions and empty_none is True
    frame5 = Frame()
    if not frame5._regions:
        assert frame5._shape is None


def assert_(cond):
    assert cond


@pytest.mark.parametrize(
    "input_stream_data",
    [
        {
            "group": 1,
            "inputs": [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])],
            "offset": (0, 0),
        },
        {
            "group": 2,
            "inputs": [
                np.array([[1, 2, 3], [4, 5, 6]]),
                np.array([[7, 8, 9], [10, 11, 12]]),
            ],
            "offset": (1, 1),
        },
    ],
)
def test_stitch_pipeline(input_stream_data):
    groupby_func = lambda x: x["group"]
    inputs = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    offset = (0, 0)
    fill_value = 0
    shape = None
    dtype = None
    empty_none = False

    input_stream = [input_stream_data]

    with Pipeline() as pipeline:
        a = Unpack(input_stream)
        stitch_node = Stitch(
            *inputs,
            groupby=groupby_func,
            offset=offset,
            fill_value=fill_value,
            shape=shape,
            dtype=dtype,
            empty_none=empty_none
        )
        Call(lambda a: assert_(isinstance(a, dict)), a)

    result = list(pipeline.transform_stream())
