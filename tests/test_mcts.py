import pytest
import numpy as np
import nd2py as nd
import warnings

np.random.seed(42)

x, y, z = nd.variables('x y z', nettype='scalar')
vars = {'x': np.random.rand(100, 1), 'y': np.random.rand(100, 1), 'z': np.random.rand(100, 1)}

@pytest.mark.parametrize("flags,node,vars", [
    (dict(variables=[x,y],   n_iter=100, random_state=42, binary=[nd.Add, nd.Mul], unary=[nd.Sin], const_range=None), x+y,           vars),
    (dict(variables=[x,y],   n_iter=100, random_state=42, binary=[nd.Add, nd.Mul], unary=[nd.Sin], const_range=None), x*y,           vars),
    (dict(variables=[x,y,z], n_iter=100, random_state=42, binary=[nd.Add, nd.Mul], unary=[nd.Sin], const_range=None), (x+y)*z,       vars),
    (dict(variables=[x,y,z], n_iter=100, random_state=42, binary=[nd.Add, nd.Mul], unary=[nd.Sin], const_range=None), nd.sin(x)+y, vars),
    (dict(variables=[x],     n_iter=100, random_state=42, binary=[nd.Add, nd.Mul], unary=[nd.Sin], const_range=None), nd.sin(2*x),   vars),
])
def test_mcts(flags, node, vars):
    est = nd.MCTS(**flags)
    X = vars
    y = node.eval(vars)
    est.fit(X, y)
    assert np.abs(est.predict(X) - y).max() < 1e-10, f'{est.eqtree} != {node}'
