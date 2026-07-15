import numpy as np
import pytest

import nd2py as nd


@pytest.fixture
def grouped_data():
    return {
        "s": np.array(["group1", "group1", "group2", "group2", "group3", "group3"]),
        "x": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        "y": np.array([1.0, 2.0, 6.0, 8.0, 15.0, 18.0]),
    }


def test_eval_binds_labels_in_first_seen_order(grouped_data):
    s = nd.Variable("s")
    parameter = nd.GroupedParameter(s, default=2.0)

    result = parameter.eval(grouped_data)

    assert parameter.group_labels == ["group1", "group2", "group3"]
    assert parameter.value_dict == {"group1": 2.0, "group2": 2.0, "group3": 2.0}
    np.testing.assert_array_equal(result, np.full(6, 2.0))


@pytest.mark.parametrize("fold_constant", [False, True])
def test_bfgs_fits_grouped_parameters(grouped_data, fold_constant):
    s = nd.Variable("s")
    x = nd.Variable("x")
    expression = nd.GroupedParameter(s, default=1.0) * x

    fit = nd.BFGSFit(expression, fold_constant=fold_constant)
    fit.fit(grouped_data, grouped_data["y"])

    parameter = fit.expression.operands[0]
    assert parameter.value_dict == pytest.approx(
        {"group1": 1.0, "group2": 2.0, "group3": 3.0}, abs=1e-5
    )
    assert np.mean((fit.predict(grouped_data) - grouped_data["y"]) ** 2) < 1e-10


def test_copy_preserves_independent_parameter_state(grouped_data):
    parameter = nd.GroupedParameter(
        nd.Variable("s"), value={"group1": 1.0}, default=3.0, fitable=False
    )
    copied = parameter.copy()
    copied.bind(["group2"])
    copied.value[0] = 10.0

    assert parameter.value_dict == {"group1": 1.0}
    assert copied.value_dict == {"group1": 10.0, "group2": 3.0}
    assert copied.fitable is False


def test_string_formats():
    parameter = nd.GroupedParameter(
        nd.Variable("s"), value={"group1": 1.0, "group2": 2.0}
    )

    assert parameter.to_str() == "alpha[s]"
    assert nd.parse(parameter.to_str(raw=True)).value_dict == parameter.value_dict


def test_multiple_grouped_parameters_are_numbered_by_expression_order():
    s = nd.Variable("s")
    x = nd.Variable("x")
    expression = (
        nd.GroupedParameter(s, value={"group1": 1.0}) * x
        + nd.GroupedParameter(s, value={"group1": 2.0})
    )

    assert expression.to_str() == "alpha^(1)[s] * x + alpha^(2)[s]"
    assert expression.to_str(latex=True) == (
        r"\alpha^{(1)}_{s} \times x + \alpha^{(2)}_{s}"
    )
    assert expression.to_str(grouped_parameter_symbol="theta") == (
        "theta^(1)[s] * x + theta^(2)[s]"
    )


def test_rejects_non_variable_operand():
    with pytest.raises(TypeError, match="Variable"):
        nd.GroupedParameter(nd.Number(1.0))
