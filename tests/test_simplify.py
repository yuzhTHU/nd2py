import pytest
import numpy as np
import nd2py as nd

np.random.seed(42)

x, y, z = nd.variables('x y z', nettype='scalar')
n = nd.Variable('n', nettype='node')
e = nd.Variable('e', nettype='edge')
s = nd.Variable('s', nettype='scalar')

@pytest.mark.parametrize("node,flags,expected", [
    (nd.readout(s), {}, s),
    (nd.readout(n), {}, nd.readout(n)),
    (nd.readout(e), {}, nd.readout(e)),
    (nd.readout(nd.readout(e)), {}, nd.readout(e)),
    ((1 + 1 + e), {}, 2 + e),
    ((2 * n * 2), {}, 4 * n),
    ((1 + e + 1), {}, 2 + e),
    ((1 + e - 1), {}, e),
    ((1 + e - e - 1), {}, e - -e),
    ((2 * x / 2 * e), {}, 1 * x * e),
    ((nd.sin(nd.sin(x) * 2 + 2)), {'remove_nested_sin': True}, 2 + 2 * nd.sin(x)),
    ((nd.exp(nd.exp(x) * 2 + 2)), {'remove_nested_exp': True}, 2 + 2 * nd.exp(x)),
    (--x, {}, x),
    (nd.Inv(nd.Inv(x)), {}, x),
    # ((2 * x) ** 2, {}, 4 * x ** 2),
])
def test_simplify(node: nd.Symbol, flags: dict, expected: nd.Symbol):
    """
    Test the simplify function with various nodes and flags.
    """
    simplified = node.simplify(**flags)
    assert str(simplified) == str(expected), f"Expected {expected}, got {simplified}"
