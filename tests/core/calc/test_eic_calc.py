import numpy as np
import pytest

import nd2py as nd


def test_leaf_eic_is_zero():
    x = nd.Variable("x")
    assert nd.EICCalc()(x, {"x": [1.0, 2.0, 3.0]}) == 0.0
    assert x.eval_eic({"x": [1.0, 2.0, 3.0]}) == 0.0


def test_add_uses_analytic_partial_derivatives():
    x = nd.Variable("x")
    values = np.array([1.0, 2.0, 3.0])
    expected_s2 = 1 + (values / (values + 1)) ** 2 + (1 / (values + 1)) ** 2
    expected = 0.5 * np.log10(np.mean(expected_s2))

    assert x.__add__(1).eval_eic({"x": values}) == pytest.approx(expected)


def test_generic_visit_uses_finite_difference_partial_derivatives():
    x = nd.Variable("x")
    # Max has no analytic derivative visitor and therefore exercises generic_visit.
    eic = nd.Max(x, 1).eval_eic({"x": np.array([2.0, 3.0])})
    assert eic == pytest.approx(0.5 * np.log10(2), rel=1e-6)


def test_high_local_eic_is_reported():
    x = nd.Variable("x")
    expression = x - x

    eic, exceptions = expression.eval_eic(
        {"x": np.array([1.0, 2.0])}, return_exceptions=True
    )

    assert np.isinf(eic)
    assert exceptions == [
        "The subexpression x - x exhibits elevated local numerical sensitivity, "
        "corresponding to an estimated loss of inf decimal digits of precision."
    ]


def test_zero_sensitivity_does_not_multiply_infinite_child_variance():
    x = nd.Variable("x")
    expression = nd.Exp(x - x)

    eic, exceptions = expression.eval_eic(
        {"x": np.array([990.0, 1000.0, 1010.0])}, return_exceptions=True
    )

    assert np.isinf(eic)
    assert len(exceptions) == 1
    assert "x - x" in exceptions[0]
    assert all("nan" not in message.lower() for message in exceptions)


def test_zero_multiplier_masks_unstable_child_locally():
    x = nd.Variable("x")
    unstable = nd.Exp(1.001 * x - x)
    expression = 0 * unstable + x

    eic, exceptions = expression.eval_eic(
        {"x": np.array([990.0, 1000.0, 1010.0])}, return_exceptions=True
    )

    assert eic > 3
    assert len(exceptions) == 2
    assert "1.001 * x - x" in exceptions[0]
    assert "exp(1.001 * x - x)" in exceptions[1]
    assert all("0 * exp" not in message for message in exceptions)
    assert all("nan" not in message.lower() for message in exceptions)


def test_exception_threshold_is_configurable():
    x = nd.Variable("x")
    eic, exceptions = (x + 1).eval_eic(
        {"x": np.array([1.0, 2.0, 3.0])},
        return_exceptions=True,
        exception_threshold=0.05,
    )
    assert eic > 0.05
    assert len(exceptions) == 1
    assert exceptions[0].startswith(
        "The subexpression x + 1 exhibits elevated local numerical sensitivity"
    )


def test_perturbation_must_be_positive():
    with pytest.raises(ValueError, match="perturbation must be positive"):
        nd.Number(1).eval_eic(perturbation=0)
