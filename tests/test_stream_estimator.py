import itertools
import random

from morphocut.utils import StreamEstimator


def test_StreamEstimator_no_estimate():
    est = StreamEstimator()

    stream_length = 10
    multiplier = 5

    n_remaining = []
    for i in range(stream_length):
        with est.consume(None) as incoming:
            for j in range(multiplier):
                n_remaining.append(incoming.emit())

    # No estimates are available
    assert n_remaining == [None for _ in range(stream_length * multiplier)]


def test_StreamEstimator_global_estimate():
    est = StreamEstimator()

    stream_length = 10
    multiplier = 5
    n_total = stream_length * multiplier

    n_remaining = []
    for i in range(stream_length):
        with est.consume(stream_length - i) as incoming:
            for j in range(multiplier):
                n_remaining.append(incoming.emit())

    # Estimates are available after first incoming object is fully processed
    assert n_remaining == [
        None if i < multiplier else n_total - i
        for i in range(stream_length * multiplier)
    ]

    # The last object has one remaining object (i.e. itself)
    assert n_remaining[-1] == 1


def test_StreamEstimator_full_estimate():
    est = StreamEstimator()

    stream_length = 10
    multiplier = 5
    n_total = stream_length * multiplier

    n_remaining = []
    for i in range(stream_length):
        with est.consume(stream_length - i, est_n_emit=multiplier) as incoming:
            for j in range(multiplier):
                n_remaining.append(incoming.emit())

    # Estimates are available
    assert n_remaining == [n_total - i for i in range(n_total)]

    # The last object has one remaining object (i.e. itself)
    assert n_remaining[-1] == 1


def test_StreamEstimator_non_deterministic_global_estimate():
    est = StreamEstimator()

    stream_length = 10

    n_remaining = []
    for i in range(stream_length):
        with est.consume(stream_length - i) as incoming:
            local_mult = random.randint(1, 10)
            print("local_mult", local_mult)
            print("rate", est.rate)
            for j in range(local_mult):
                n_remaining.append(incoming.emit())

    # The remaining hint of the last object is indeterminable


def test_StreamEstimator_non_deterministic_full_estimate():
    est = StreamEstimator()

    stream_length = 10

    n_remaining = []
    for i in range(stream_length):
        local_mult = random.randint(1, 10)
        with est.consume(stream_length - i, est_n_emit=local_mult) as incoming:
            for j in range(local_mult):
                n_remaining.append(incoming.emit())

    # The last object has one remaining object (i.e. itself)
    assert n_remaining[-1] == 1


def test_StreamEstimator_stacked():
    length0 = 10
    length1 = 5
    length2 = 3
    n_total = length0 * length1 * length2

    est0 = StreamEstimator()
    est1 = StreamEstimator()

    n_remaining = []
    for i in range(length0):
        with est0.consume(length0 - i, est_n_emit=length1) as incoming0:
            for j in range(length1):
                with est1.consume(incoming0.emit(), est_n_emit=length2) as incoming1:
                    for k in range(length2):
                        n_remaining.append(incoming1.emit())

    # Estimates are available
    assert n_remaining == [n_total - i for i in range(n_total)]

    # The last object has one remaining object (i.e. itself)
    assert n_remaining[-1] == 1


def test_StreamEstimator_downsample_global_estimate():
    est = StreamEstimator()

    stream_length = 10
    factor = 2
    n_total = stream_length // factor

    n_remaining = []
    for i in range(stream_length):
        with est.consume(stream_length - i) as incoming:
            if i % factor == 0:
                n_remaining.append(incoming.emit())

    # Estimates are available after first incoming object is fully processed
    assert n_remaining == [
        n_total - i if i > 0 else None for i in range(stream_length // factor)
    ]

    # The last object has one remaining object (i.e. itself)
    assert n_remaining[-1] == 1


def test_StreamEstimator_downsample_full_estimate():
    est = StreamEstimator()

    stream_length = 10
    factor = 2
    n_total = stream_length // factor

    n_remaining = []
    for i in range(stream_length):
        with est.consume(stream_length - i, est_n_emit=1 / factor) as incoming:
            if i % factor == 0:
                n_remaining.append(incoming.emit())

    # Estimates are available
    assert n_remaining == [n_total - i for i in range(stream_length // factor)]

    # The last object has one remaining object (i.e. itself)
    assert n_remaining[-1] == 1


def test_StreamEstimator_downsample_full_estimate_consume_multi():
    est = StreamEstimator()

    stream_length = 10
    factor = 2
    n_total = stream_length // factor

    stream = iter(range(stream_length))
    n_remaining_in = stream_length

    n_remaining = []
    while True:
        packed = list(itertools.islice(stream, factor))

        if not packed:
            break

        with est.consume(
            n_remaining_in, est_n_emit=1, n_consumed=len(packed)
        ) as incoming:
            n_remaining.append(incoming.emit())

        n_remaining_in -= len(packed)

    # Estimates are available
    assert n_remaining == [n_total - i for i in range(stream_length // factor)]

    # The last object has one remaining object (i.e. itself)
    assert n_remaining[-1] == 1
