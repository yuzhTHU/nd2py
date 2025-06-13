import pytest
import numpy as np
import nd2py as nd
import warnings

np.random.seed(42)

x, y, z = nd.variables('x y z', nettype='scalar')
vars = {'x': np.random.rand(100, 1), 'y': np.random.rand(100, 1), 'z': np.random.rand(100, 1)}

@pytest.mark.parametrize("gp_cls,flags,node,vars", [
    (nd.GP, dict(variables=[x,y], n_iter=30, random_state=42, binary=[nd.Add, nd.Mul], unary=[nd.Sin]), x*y, vars),
    (nd.GP, dict(variables=[x,y,z], n_iter=30, random_state=42, binary=[nd.Add, nd.Mul], unary=[nd.Sin]), (x+y)*z, vars),
    (nd.GP, dict(variables=[x,y,z], n_iter=30, random_state=42, binary=[nd.Add, nd.Mul], unary=[nd.Sin]), nd.sin(x+y)*z, vars),
    (nd.GP, dict(variables=[x], n_iter=30, random_state=42, binary=[nd.Add, nd.Mul], unary=[nd.Sin]), nd.sin(2*x), vars),
])
def test_gp(gp_cls, flags, node, vars):
    est = gp_cls(**flags)
    X = vars
    y = node.eval(vars)
    est.fit(X, y)
    assert np.abs(est.predict(X) - y).max() < 1e-10, f'{est.eqtree} != {node}'
