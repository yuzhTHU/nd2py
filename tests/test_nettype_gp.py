import pytest
import numpy as np
import nd2py as nd

np.random.seed(42)

def test_nettype_gp():
    num_nodes = 10
    num_edges = 40
    edge_list = np.random.randint(0, num_nodes, (2, num_edges))
    x = nd.Variable('x', nettype='node')
    y = nd.Variable('y', nettype='edge')
    vars = {'x': np.random.rand(100, num_nodes), 'y': np.random.rand(100, num_edges)}
    node = x + nd.aggr(y * nd.sour(x))

    est = nd.GP(variables=[x,y],
                n_iter=30, 
                random_state=42, 
                binary=[nd.Add, nd.Mul], 
                unary=[nd.Aggr, nd.Sour],
                edge_list=edge_list,
                num_nodes=num_nodes,
                nettype='node')
    X = vars
    y = node.eval(vars, edge_list=edge_list, num_nodes=num_nodes)
    est.fit(X, y)
    assert np.abs(est.predict(X) - y).max() < 1e-8, f'{est.eqtree} != {node}'
