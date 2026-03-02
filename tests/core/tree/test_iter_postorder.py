import pytest
import nd2py as nd

x, y, z = nd.variables('x y z')

@pytest.mark.parametrize(
    "eqtree,expected",
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
def test_iter_postorder(eqtree, expected):
    postorder = list(eqtree.iter_postorder())
    assert len(postorder) == len(expected)
    for node, expected_node in zip(postorder, expected):
        if isinstance(expected_node, type):
            assert type(node) == expected_node, f"Expected {expected_node}, got {type(node)}"
        elif isinstance(expected_node, nd.Number):
            assert node == expected_node, f"Expected {expected_node.to_str(raw=True)}, got {node.to_str(raw=True)}"
        elif isinstance(expected_node, nd.Variable):
            assert node.name == expected_node.name, f"Expected {expected_node.to_str(raw=True)}, got {node.to_str(raw=True)}"
        else:
            assert node is expected_node, f"Expected {expected_node.to_str(raw=True)}, got {node.to_str(raw=True)}"
