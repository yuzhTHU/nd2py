import pytest
import nd2py as nd

s = nd.Variable("s", nettype="scalar")
n = nd.Variable("n", nettype="node")
e = nd.Variable("e", nettype="edge")


@pytest.mark.parametrize(
    "node,expected",
    [
        (s, "scalar"),
        (n, "node"),
        (e, "edge"),
        (nd.aggr(s), "node"),
        (nd.aggr(e), "node"),
        (nd.aggr(s).operands[0], "scalar"),
        (nd.aggr(e).operands[0], "edge"),
        (nd.rgga(s), "node"),
        (nd.rgga(e), "node"),
        (nd.rgga(s).operands[0], "scalar"),
        (nd.rgga(e).operands[0], "edge"),
        (nd.sour(s), "edge"),
        (nd.sour(n), "edge"),
        (nd.sour(s).operands[0], "scalar"),
        (nd.sour(n).operands[0], "node"),
        (nd.targ(s), "edge"),
        (nd.targ(n), "edge"),
        (nd.targ(s).operands[0], "scalar"),
        (nd.targ(n).operands[0], "node"),
        ((s + n), "node"),
        ((s + n).operands[0], "scalar"),
        ((s + n).operands[1], "node"),
        ((s + e), "edge"),
        ((s + e).operands[0], "scalar"),
        ((s + e).operands[1], "edge"),
        ((s + s), "scalar"),
        ((n + n), "node"),
        ((e + e), "edge"),
    ],
)
def test_nettype(node, expected):
    assert node.nettype == expected


@pytest.mark.parametrize(
    "node,expected",
    [
        (lambda: nd.aggr(n), (ValueError)),
        (lambda: nd.rgga(n), (ValueError)),
        (lambda: nd.sour(e), (ValueError)),
        (lambda: nd.targ(e), (ValueError)),
        (lambda: n + e, (ValueError)),
        (lambda: nd.Aggr(s, nettype="edge"), (ValueError)),
        (lambda: nd.Aggr(s, nettype="scalar"), (ValueError)),
        (lambda: nd.Aggr(e, nettype="edge"), (ValueError)),
        (lambda: nd.Aggr(e, nettype="scalar"), (ValueError)),
    ],
)
def test_nettype_error(node, expected):
    with pytest.raises(expected):
        node()
