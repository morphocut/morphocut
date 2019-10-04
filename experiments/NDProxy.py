"""
- `combine_index((None,), (0,)) == ()`
- `combine_index((None,), (None,)) == (None, None)`
- `combine_index((None, ), (slice(None), )) == (None, )`
- `combine_index((), (None, )) == (None, )`
- `combine_index((), right) == right`
- `combine_index((None,) + left, (0,) + right) == combine_index(left, right)`
- `combine_index((None,) + left, (slice(),) + right) == (None,) + combine_index(left, right)`
- `combine_index((n,) + left, right) == (n,) + combine_index(left, right)`
"""
import functools
import itertools
import pickle

import lazy_object_proxy.slots
import numpy as np
import pytest
from skimage.data import coins


def _make_factory(loader, id, index):
    return functools.partial(loader, id, index)


def _unpickle(func, *args):
    return NDProxy(func, *args)


def _slice_to_range(s):
    if isinstance(s, slice):
        start = s.start if s.start is not None else 1
        step = s.step if s.step is not None else 1

        return range(start, s.stop, step)
    if isinstance(s, int):
        return range(s, s + 1)


def _range_to_slice(r):
    if isinstance(r, range):
        return slice(r.start, r.stop, r.step)
    return r


class NDProxy(lazy_object_proxy.slots.Proxy):
    """A proxy around np.ndarray that knows how to load data.

    This allows the serialization of an array without serializing the actual data.
    """
    def __init__(self, loader, id=None, index=None, target=None):
        super().__init__(_make_factory(loader, id, index))
        object.__setattr__(self, '__index__', index)

        if target is not None:
            object.__setattr__(self, '__target__', target)

    def load(self):
        self.__wrapped__

    def unload(self):
        del self.__wrapped__

    def detatch(self):
        return self.__wrapped__

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index, )

        # TODO: Calculate index, given self_index is present
        self_index = object.__getattribute__(self, '__index__')
        if self_index is not None:
            index = combine_index(self_index, index)

        try:
            target = object.__getattribute__(self, '__target__')[index]
        except AttributeError:
            target = None

        f = self.__factory__
        return NDProxy(f.func, f.args[0], index, target)

    def __reduce__(self):
        f = self.__factory__
        return _unpickle, (f.func, *f.args)

    def __reduce_ex__(self, protocol):
        return self.__reduce__()


def loader(id, index):
    print("Loading({},{})...".format(id, index))
    result = coins()
    if index is not None:
        result = result[index]

    return result


def unpack_slice(slice):
    return slice.start or 0, slice.stop, slice.step or 1


def combine_index(left, right):
    """
    Args:
        left, right (tuple): Tuples of slices, indices or lists (as allowed by np.ndarray.__getitem__)
    """

    left = tuple(
        np.array(idx) if isinstance(idx, list) else idx for idx in left
    )
    right = tuple(
        np.array(idx) if isinstance(idx, list) else idx for idx in right
    )

    result = []

    i = 0
    j = 0

    while i < len(left) and j < len(right):
        # combine_index(left, (None, ) + right) == (None, ) + combine_index(left, right)
        if isinstance(right[j], type(None)):
            result.append(None)
            j += 1
            continue

        # combine_index((n,) + left, right) == (n,) + combine_index(left, right)
        if isinstance(left[i], int):
            result.append(left[i])
            i += 1
            continue
        # combine_index((None,) + ..., ...)
        if isinstance(left[i], type(None)):
            if isinstance(right[j], slice) and right[j] == slice(None):
                result.append(None)
                i += 1
                j += 1
                continue

            if isinstance(right[j], int):
                if right[j] != 0:
                    raise IndexError(
                        "Index {} is out of bounds for {}".format(
                            right[j], left[i]
                        )
                    )
                i += 1
                j += 1
                continue
            raise ValueError(
                "{} can not be combined with {}".format(left[i], right[j])
            )

        # combine_index((slice(None),) + ..., ...)
        if isinstance(left[i], slice) and left[i] == slice(None):
            result.append(right[j])
            i += 1
            j += 1
            continue

        # combine_index((slice(...),) + ..., ...)
        if isinstance(left[i], slice):
            lstart, lstop, lstep = unpack_slice(left[i])
            if isinstance(right[j], int):
                idx = lstart + lstep * right[j]
                if idx >= lstop:
                    raise IndexError(
                        "Index {} is out of bounds for {}".format(
                            right[j], left[i]
                        )
                    )
                result.append(idx)
                i += 1
                j += 1
                continue
            if isinstance(right[j], slice):
                rstart, rstop, rstep = unpack_slice(right[j])

                start = lstart + lstep * rstart
                stop = lstart + lstep * rstop if rstop is not None else lstop
                step = lstep * rstep

                if lstop is not None and stop > lstop:
                    stop = lstop

                result.append(slice(start, stop, step))
                i += 1
                j += 1
                continue
            try:
                # Raises AttributeError if not a numpy array
                right_type = right[j].dtype.type
            except AttributeError:
                pass
            else:
                right_ndim = np.ndim(right[j])
                if issubclass(right_type, np.integer):
                    idx = lstart + lstep * right[j]
                    if (idx >= lstop).any():
                        raise IndexError(
                            "Index {} is out of bounds for {}".format(
                                right[j], left[i]
                            )
                        )
                    result.append(idx)
                    i += 1
                    j += 1
                    continue

                if issubclass(right_type, np.bool_) and right_ndim == 1:
                    idx, = np.nonzero(right[j])
                    idx = lstart + lstep * idx
                    if (idx >= lstop).any():
                        raise IndexError(
                            "Index {} is out of bounds for {}".format(
                                right[j], left[i]
                            )
                        )
                    result.append(idx)
                    i += 1
                    j += 1
                    continue

        # combine_index(np.array([...]), ...)
        try:
            # Raises AttributeError if not a numpy array
            left_type = left[i].dtype.type
        except AttributeError:
            pass
        else:
            left_ndim = np.ndim(left[i])
            if issubclass(left_type, np.integer):
                result.append(left[i][right[j:j + left_ndim]])
                i += 1
                j += left_ndim
                continue

            if issubclass(left_type, np.bool_) and left_ndim == 1:
                idx, = np.nonzero(left[i])
                result.append(idx[right[j]])
                i += 1
                j += 1
                continue

        raise NotImplementedError(repr((left[i], right[j])))

    result.extend(left[i:])
    result.extend(right[j:])

    return tuple(result)


# left = NDProxy(loader)

# # a.load()
# print(left[11::2][0])

# # from scipy.ndimage import gaussian_filter

# # image = gaussian_filter(a, 1)

# # pickle.dumps(a)

test_indices = [
    (),
    (slice(None), ),
    (slice(1), ),
    (slice(1, 2), ),
    (slice(1, 3, 2), ),
    (0, ),
    (1, ),
    ([0, 1, 2], ),
    (np.array([0, 1, 2]), ),
    (np.array([True, True, False]), ),
    (None, ),
]

test_array = np.arange(27).reshape(3, 3, 3)


@pytest.mark.parametrize("left", test_indices)
@pytest.mark.parametrize("right", test_indices)
def test_recursive(left, right):
    native_index_error = False
    try:
        sub_native = test_array[left][right]
    except IndexError:
        native_index_error = True

    try:
        combined_index = combine_index(left, right)
        sub_combine = test_array[combined_index]
    except IndexError:
        # The occurence of an IndexError has to match the IndexError using native.
        assert native_index_error
    except ValueError:
        # A ValueError is ok for None in left
        if None in left:
            return
        raise
    else:
        # assert not native_index_error, "Native threw IndexError, combine didn't."
        if native_index_error:
            breakpoint()

        np.testing.assert_equal(sub_native, sub_combine)

        # if not np.all(sub_native == sub_combine):
        #     breakpoint()
