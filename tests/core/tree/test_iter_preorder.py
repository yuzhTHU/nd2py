import pytest
import nd2py as nd

x, y, z = nd.variables('x y z')

@pytest.mark.parametrize(
    "eqtree,expected",
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
def test_iter_preorder(eqtree, expected):
    preorder = list(eqtree.iter_preorder())
    assert len(preorder) == len(expected)
    for node, expected_node in zip(preorder, expected):
        if isinstance(expected_node, type):
            assert type(node) == expected_node, f"Expected {expected_node}, got {type(node)}"
        elif isinstance(expected_node, nd.Number):
            assert node == expected_node, f"Expected {expected_node.to_str(raw=True)}, got {node.to_str(raw=True)}"
        elif isinstance(expected_node, nd.Variable):
            assert node.name == expected_node.name, f"Expected {expected_node.to_str(raw=True)}, got {node.to_str(raw=True)}"
        else:
            assert node is expected_node, f"Expected {expected_node.to_str(raw=True)}, got {node.to_str(raw=True)}"
