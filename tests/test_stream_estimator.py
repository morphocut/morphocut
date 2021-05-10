import random

from morphocut.stream_estimator import StreamEstimator


def test_StreamEstimator_no_estimate():
    est = StreamEstimator()

    stream_length = 10
    multiplier = 5

    n_remaining = []
    for i in range(stream_length):
        with est.incoming_object(None):
            for j in range(multiplier):
                n_remaining.append(est.emit())

    # No estimates are available
    assert n_remaining == [None for _ in range(stream_length * multiplier)]


def test_StreamEstimator_global_estimate():
    est = StreamEstimator()

    stream_length = 10
    multiplier = 5
    n_total = stream_length * multiplier

    n_remaining = []
    for i in range(stream_length):
        with est.incoming_object(stream_length - i):
            for j in range(multiplier):
                n_remaining.append(est.emit())

    # Estimates are available after first incoming object is fully processed
    assert n_remaining == [
        None if i < multiplier else n_total - i
        for i in range(stream_length * multiplier)
    ]


def test_StreamEstimator_full_estimate():
    est = StreamEstimator()

    stream_length = 10
    multiplier = 5
    n_total = stream_length * multiplier

    n_remaining = []
    for i in range(stream_length):
        with est.incoming_object(stream_length - i, multiplier):
            for j in range(multiplier):
                n_remaining.append(est.emit())

    # Estimates are available
    assert n_remaining == [n_total - i for i in range(n_total)]


def test_StreamEstimator_non_deterministic_global_estimate():
    est = StreamEstimator()

    stream_length = 10

    n_remaining = []
    for i in range(stream_length):
        with est.incoming_object(stream_length - i):
            local_mult = random.randint(1, 10)
            print("local_mult", local_mult)
            print("global_estimate", est.global_estimate)
            for j in range(local_mult):
                n_remaining.append(est.emit())


def test_StreamEstimator_non_deterministic_full_estimate():
    est = StreamEstimator()

    stream_length = 10

    n_remaining = []
    for i in range(stream_length):
        local_mult = random.randint(1, 10)
        with est.incoming_object(stream_length - i, local_mult):
            for j in range(local_mult):
                n_remaining.append(est.emit())

    # Last estimate is 1
    assert n_remaining[-1] == [1]


def test_StreamEstimator_stacked():
    length0 = 10
    length1 = 5
    length2 = 3
    n_total = length0 * length1 * length2

    est0 = StreamEstimator()
    est1 = StreamEstimator()

    n_remaining = []
    for i in range(length0):
        with est0.incoming_object(length0 - i, length1):
            for j in range(length1):
                with est1.incoming_object(est0.emit(), length2):
                    for k in range(length2):
                        n_remaining.append(est1.emit())

    # Estimates are available
    assert n_remaining == [n_total - i for i in range(n_total)]
