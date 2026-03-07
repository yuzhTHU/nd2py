import pytest
import nd2py as nd

x, y, z = nd.variables('x y z')

@pytest.mark.parametrize(
    "expected,nodes",
    [
        (x, [x]),
        (x + y, [nd.Add, x, y]),
        (x * y, [nd.Mul, x, y]),
        (x * nd.sin(y), [nd.Mul, x, nd.Sin, y]),
        (nd.Sin(x) * nd.sin(y), [nd.Mul, nd.Sin, x, nd.Sin, y]),
        (nd.Log(nd.Sin(x) * nd.sin(y)), [nd.Log, nd.Mul, nd.Sin, x, nd.Sin, y]),
        (x + y + z, [nd.Add, nd.Add, x, y, z]),
    ],
)
def test_iter_preorder(expected, nodes):
    assert str(nd.from_preorder(nodes)) == str(expected)
    