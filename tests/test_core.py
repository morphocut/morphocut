import itertools
import operator

import pytest

from morphocut.core import (
    Call,
    Node,
    Output,
    Pipeline,
    ReturnOutputs,
    EmptyPipelineStackError,
)
from tests.helpers import Const


@ReturnOutputs
class TestNodeNoTransform(Node):
    pass


@ReturnOutputs
@Output("a")
@Output("b")
@Output("c")
class TestNode(Node):
    def __init__(self, a, b, c):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c

    def transform(self, a, b, c):
        return a, b, c


def test_Pipeline():
    with Pipeline() as p:
        a = Const("a")
        with Pipeline():
            b = Const("b")

    locals_hashes = set(v.hash for v in p.locals())

    # a is a local of p
    assert a.hash in locals_hashes
    # b is also a local of p
    assert b.hash in locals_hashes


def test_Node():
    # Assert that Node checks for the existence of a pipeline
    with pytest.raises(EmptyPipelineStackError):
        TestNode(1, 2, 3)

    # Assert that Node checks for the existance of transform
    with Pipeline() as pipeline:
        TestNodeNoTransform()

    with pytest.raises(AttributeError):
        pipeline.run()

    # Assert that parameters and outputs are passed as expected
    with Pipeline() as pipeline:
        a, b, c = TestNode(1, 2, 3)

    assert len(pipeline.children) == 1

    obj, *_ = list(pipeline.transform_stream())
    assert obj[a] == 1
    assert obj[b] == 2
    assert obj[c] == 3


def test_Call():
    def foo(bar, baz):
        return bar, baz

    class Bar:
        pass

    lambda_func = lambda: None

    with Pipeline() as pipeline:
        result = Call(foo, 1, 2)
    obj, *_ = list(pipeline.transform_stream())
    assert obj[result] == (1, 2)

    with Pipeline() as pipeline:
        # Test when clbl is a plain function for __str__ method
        call_obj1 = Call(foo, 1, 2)
    str_rep1 = str(call_obj1)
    assert "foo" in str_rep1


class _MatMullable:
    def __init__(self, value):
        self.value = value

    def __matmul__(self, other):
        if isinstance(other, _MatMullable):
            other = other.value
        return self.value * other

    def __rmatmul__(self, other):
        if isinstance(other, _MatMullable):
            other = other.value
        return other * self.value


@pytest.mark.parametrize(
    "inp",
    [
        (operator.add, 1, 2),
        (operator.truediv, 1, 2),
        (operator.floordiv, 5, 3),
        (operator.and_, 5, 3),
        (operator.xor, 5, 3),
        (operator.invert, 5),
        (operator.or_, 5, 3),
        (operator.pow, 5, 3),
        (operator.lshift, 5, 3),
        (operator.rshift, 5, 3),
        (operator.mod, 5, 3),
        (operator.mul, 5, 3),
        pytest.param(
            (operator.matmul, _MatMullable(5), _MatMullable(3)), marks=pytest.mark.xfail
        ),
        (operator.neg, 5),
        (operator.pos, 5),
        (operator.sub, 5, 3),
        (operator.lt, 5, 3),
        (operator.le, 5, 3),
        (operator.eq, 5, 3),
        (operator.ne, 5, 3),
        (operator.ge, 5, 3),
        (operator.gt, 5, 3),
        (operator.abs, -5),
    ],
)
def test_VariableOperations(inp):
    op, *args = inp

    result = op(*args)

    # print("{}={}".format(inp, result))

    for varidxs in set(
        frozenset(varidxs)
        for varidxs in list(itertools.product(range(len(args)), repeat=len(args) + 1))
    ):
        with Pipeline() as pipeline:
            args_ = tuple(Const(v) if i in varidxs else v for i, v in enumerate(args))
            result_ = op(*args_)

        obj = next(pipeline.transform_stream())

        assert obj[result_] == result


def test_VariableOperationsSpecial():
    class E:
        def __init__(self):
            self.a = 1

    with Pipeline() as pipeline:
        _1 = 1
        f_value = object()
        a = Const(False)
        b = Const(_1)
        c = Const(_1)
        d = Const([1, 2, 3])
        e = Const(E())
        f = Const(f_value)

        not_a = a.not_()  # True
        true_b = b.truth()  # True
        b_is_c = b.is_(c)  # True
        a_isnot_c = a.is_not(c)  # True
        b_in_d = b.in_(d)  # True
        d_contains_b = d.contains(b)  # True

        d1 = d[1]
        d1_3 = d[1:3]
        e_a = e.a

        d[0] = None
        del d[1]

        # Unset Variables. This should remove the value from the stream.
        f.delete()

    obj = next(pipeline.transform_stream())

    assert obj[not_a] == True
    assert obj[true_b] == True
    assert obj[b_is_c] == True
    assert obj[a_isnot_c] == True
    assert obj[b_in_d] == True
    assert obj[d_contains_b] == True

    assert obj[d1] == 2
    assert obj[d1_3] == [2, 3]
    assert obj[e_a] == 1
    assert obj[d] == [None, 3]
    assert obj[d1_3] == [2, 3]

    assert f_value not in obj.values()


def test_VariableCopy():

    with Pipeline() as pipeline:
        f_value = [1, 2, 3]
        f = Const(f_value)
        f_copy = f.copy()
        # Modify the original variable
        f[0] = None

    obj = next(pipeline.transform_stream())

    assert obj[f_copy] is not obj[f]
    assert obj[f_copy] == [1, 2, 3]
    assert obj[f] == [None, 2, 3]
