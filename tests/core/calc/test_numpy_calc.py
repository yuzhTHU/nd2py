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


def test_calc_returns_subexpression_exceptions():
    x = nd.Variable("x")
    expression = nd.Log(x) + nd.Inv(x)

    result, exceptions = expression.eval(
        {"x": np.array([-1.0, 0.0, 1.0])}, return_exceptions=True
    )

    np.testing.assert_equal(result, np.array([np.nan, np.nan, 1.0]))
    assert exceptions == [
        "Subexpression log(x) produced non-finite values: 33.33% NaN, 0.00% Inf, 33.33% NegInf",
        "Subexpression 1 / x produced non-finite values: 0.00% NaN, 33.33% Inf, 0.00% NegInf",
        "Subexpression log(x) + 1 / x produced non-finite values: 66.67% NaN, 0.00% Inf, 0.00% NegInf",
    ]


@pytest.mark.parametrize(
    "expression,vars,expected",
    [
        (
            nd.Variable("x"),
            {"x": np.array([np.nan, np.inf, -np.inf, 0.0])},
            "Subexpression x produced non-finite values: 25.00% NaN, 25.00% Inf, 25.00% NegInf",
        ),
        (
            nd.Number(np.inf),
            {},
            "Subexpression inf produced non-finite values: 0.00% NaN, 100.00% Inf, 0.00% NegInf",
        ),
    ],
)
def test_calc_records_leaf_exceptions(expression, vars, expected):
    _, exceptions = expression.eval(vars, return_exceptions=True)
    assert exceptions == [expected]


def test_calc_default_return_value_is_unchanged():
    result = nd.Variable("x").eval({"x": [1.0, 2.0]})
    np.testing.assert_equal(result, np.array([1.0, 2.0]))
