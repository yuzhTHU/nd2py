import pytest
import numpy as np
import nd2py as nd

np.random.seed(42)

x, y, z = nd.variables('x y z')
vars = {'x': np.random.rand(100, 1),
        'y': np.random.rand(100, 1),
        'z': np.random.rand(100, 1)}

@pytest.mark.parametrize("fit_cls,node,vars", [
    (nd.BFGSFit, 2 * x, vars),
    (nd.BFGSFit, 2 * x + 2 * y, vars),
    (nd.BFGSFit, nd.sin(2 * x + 1), vars),
    (nd.BFGSFit, 0.1*nd.exp(-2*x) + 2.0, vars),
])
def test_bfgs_fit(fit_cls, node, vars):
    y = node.eval(vars)
    node2 = node.copy()
    for op in node2.iter_preorder():
        if isinstance(op, nd.Number):
            op.value = np.random.rand(*np.shape(op.value))
    bfgs_fit = fit_cls(node2)
    bfgs_fit.fit(vars, y)
    assert np.abs(node2.eval(vars) - y).max() < 1e-5, f'{node} != {node2}'
