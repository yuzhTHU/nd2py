import pytest
import nd2py as nd

x = nd.Variable("x")
y = nd.Variable("y")
z = nd.Variable("z")
n = nd.Variable("n", nettype="node")
e = nd.Variable("e", nettype="edge")


@pytest.mark.parametrize(
    "node,flags,expected",
    [
        (x + y, {}, [x, y]),
        (x + y + z, {}, [x, y, z]),
        (2 * x + y + z, {"remove_coefficients": True}, [x, y, z]),
        (x + y - z, {"split_by_sub": True, "remove_coefficients": True}, [x, y, z]),
        ((x + y) * z, {"expand_mul": True}, [x * z, y * z]),
        ((x + y) * (x + z), {"expand_mul": True}, [x * z, y * z, x * x, y * x]),
        ((x + y) / z, {"expand_div": True}, [x / z, y / z]),
        (
            nd.aggr(x + y + z),
            {"expand_aggr": True},
            [nd.aggr(x), nd.aggr(y), nd.aggr(z)],
        ),
        (
            nd.aggr(x + e),
            {"expand_aggr": True},
            [nd.aggr(x), nd.aggr(e)],
        ),
        (
            nd.sour(x + y + z),
            {"expand_sour": True},
            [x, y, z],
        ),
        (
            nd.sour(x + n),
            {"expand_sour": True},
            [x, nd.sour(n)],
        ),
        (
            nd.targ(x + y + z),
            {"expand_targ": True},
            [x, y, z],
        ),
        (
            nd.targ(x + n),
            {"expand_targ": True},
            [x, nd.targ(n)],
        ),
        (
            nd.aggr(x + y) * z,
            {"expand_aggr": True, "expand_mul": True},
            [nd.aggr(x) * z, nd.aggr(y) * z],
        ),
        (
            nd.sour(x + n) * z,
            {"expand_sour": True, "expand_mul": True},
            [x * z, nd.sour(n) * z],
        ),
        (
            nd.targ(x + n) * z,
            {"expand_targ": True, "expand_mul": True},
            [x * z, nd.targ(n) * z],
        ),
    ],
)
def test_split_by_add(node, flags, expected):
    items = node.split_by_add(**flags)
    assert set(map(str, items)) == set(map(str, expected))
