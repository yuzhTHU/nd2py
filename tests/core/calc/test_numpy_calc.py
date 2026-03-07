import pytest
import numpy as np
import nd2py as nd

@pytest.mark.parametrize("calc_cls,node,flags,expected", [
    (nd.NumpyCalc, nd.Number(3.14), {}, 3.14),
    (nd.NumpyCalc, nd.Variable("x"), dict(vars={'x':[1,2,3]}), [1,2,3]),
    (nd.NumpyCalc, nd.Variable("x") + 1, dict(vars={'x':[1,2,3]}), [2,3,4]),
    (nd.NumpyCalc, 2 * nd.Variable("x"), dict(vars={'x':[1,2,3]}), [2,4,6]),
    (nd.NumpyCalc, nd.Aggr(nd.Variable("x")), dict(vars={'x':1}, edge_list=[[0,0,0], [1,1,3]]), [0,2,0,1]),
    (nd.NumpyCalc, nd.Aggr(nd.Variable("x", nettype='edge')), dict(vars={'x':[1,2,2]}, edge_list=[[0,0,0], [1,1,3]]), [0,3,0,2]),
    (nd.NumpyCalc, nd.Aggr(nd.Variable("x")), dict(vars={'x':[[1],[2]]}, edge_list=[[0,0,0], [1,1,3]]), [[0,2,0,1],[0,4,0,2]]),
])
def test_calc_number(calc_cls, node, flags, expected):
    calc = calc_cls()
    output = calc(node, **flags)
    if isinstance(output, np.ndarray): 
        output = output.tolist()
    assert output == expected
