import pytest
import nd2py as nd


def test_replace1():
    x, y, z = nd.variables("x y z")
    sub_eq1 = (x + y) * z
    sub_eq2 = x * z + y * z
    parent = nd.Sin(sub_eq1)

    parent = parent.replace(sub_eq1, sub_eq2)
    assert parent.to_str() == "sin(x * z + y * z)"


def test_replace2():
    x, y, z = nd.variables("x y z")
    sub_eq1 = (x + y) * z

    parent = nd.Sin(sub_eq1) + sub_eq1

    sub_eq2 = x * z + y * z
    parent = parent.replace(sub_eq1, sub_eq2)
    assert parent.to_str() == "sin(x * z + y * z) + (x + y) * z"


def test_replace3():
    x = nd.Variable("x", nettype="edge")

    a = nd.aggr(x * x)
    b = nd.aggr(x + x)

    a = a.replace(a, b)
    assert a.to_str() == "aggr(x + x)"
