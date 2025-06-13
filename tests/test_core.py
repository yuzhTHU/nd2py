import pytest
import nd2py as nd

x = nd.Variable("x")


@pytest.mark.parametrize(
    "node",
    [
        (x + x),
        (x - x),
        (x * x),
        (x / x),
    ],
)
def test_same_child(node):
    assert id(node.operands[0]) != id(node.operands[1])


@pytest.mark.parametrize(
    "node",
    [
        nd.Sin(nettype="scalar"),
        nd.Sin(nettype="node"),
        nd.Sin(nettype="edge"),
        nd.Cos(nettype="scalar"),
        nd.Cos(nettype="node"),
        nd.Cos(nettype="edge"),
        nd.Sin(nd.Variable("x", nettype="scalar")),
        nd.Sin(nd.Variable("x", nettype="node")),
        nd.Sin(nd.Variable("x", nettype="edge")),
        nd.Cos(nd.Variable("x", nettype="scalar")),
        nd.Cos(nd.Variable("x", nettype="node")),
        nd.Cos(nd.Variable("x", nettype="edge")),
        nd.Add(nettype="scalar"),
        nd.Add(nettype="node"),
        nd.Add(nettype="edge"),
        nd.Add(x, nettype="scalar"),
        nd.Add(x, x, nettype="scalar"),
    ]
)
def test_empty_parent(node):
    for op in node.operands:
        assert op.parent == node
