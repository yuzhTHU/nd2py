import pytest
import nd2py as nd

x, y = nd.variables('x y')

@pytest.mark.parametrize("fold_cls,node,vars,expected", [
    (nd.FoldConstant, nd.Number(1.0) * nd.Constant(2.0), {}, nd.Number(1.0) * nd.Constant(2.0)),
    (nd.FoldConstant, nd.Number(1.0) * x, {'x':2.0}, nd.Number(1.0) * nd.Constant(2.0)),
    (nd.FoldConstant, nd.Number(1.0) * (x + x), {'x':2.0}, nd.Number(1.0) * nd.Constant(4.0)),
    (nd.FoldConstant, nd.Number(1.0) * (x + x) + nd.Number(1.0), {'x':2.0}, nd.Number(1.0) * nd.Constant(4.0) + nd.Number(1.0)),
])
def test_fold_constant(fold_cls, node, vars, expected):
    node2 = fold_cls()(node, vars)
    assert str(node2) == str(expected)
