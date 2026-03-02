import pytest
import nd2py as nd

x, y, z = nd.variables('x y z')

@pytest.mark.parametrize(
    "expected,nodes",
    [
        (x, [x]),
        (x + y, [x, y, nd.Add]),
        (x * y, [x, y, nd.Mul]),
        (x * nd.sin(y), [x, y, nd.Sin, nd.Mul]),
        (nd.Sin(x) * nd.sin(y), [x, nd.Sin, y, nd.Sin, nd.Mul]),
        (nd.Log(nd.Sin(x) * nd.sin(y)), [x, nd.Sin, y, nd.Sin, nd.Mul, nd.Log]),
        (x + y + z, [x, y, nd.Add, z, nd.Add]),
    ],
)
def test_iter_postorder(expected, nodes):
    assert str(nd.from_postorder(nodes)) == str(expected)
