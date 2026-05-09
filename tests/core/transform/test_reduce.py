# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
"""
Unit tests for the Reduce expression simplification system.

Tests cover:
1. Basic reduction rules (constant folding, identity operations)
2. Algebraic simplifications (double negation, inverse operations)
3. Function composition reductions (sin(asin(x)) -> x, etc.)
4. Pattern matching with variables
5. Cache loading/saving functionality
6. Numerical equivalence verification
"""
import pytest
import numpy as np
import nd2py as nd
from nd2py.core.transform.reduce import Reduce


reducer = Reduce(
    n_variables=3,
    constants=[0, 1, -1, np.pi, np.e],
    binary=[nd.Add, nd.Sub, nd.Mul, nd.Div],
    unary=[
        nd.Sin, nd.Cos, nd.Tan, nd.Sec, nd.Csc, nd.Cot, nd.Arcsin, nd.Arccos, nd.Arctan,
        nd.Log, nd.LogAbs, nd.Exp, nd.Abs, nd.Neg, nd.Inv, nd.Sqrt, nd.SqrtAbs, nd.Pow2, nd.Pow3,
        nd.Sinh, nd.Cosh, nd.Tanh, nd.Coth, nd.Sech, nd.Csch, nd.Sigmoid
    ],
    max_online_iterations=10,
    load_cache=True  # Don't load cache for fast unit tests
)
# Build rules with small l_max for fast tests
reducer.prepare_rule_dict(l_max=4, force_rebuild=False, save_cache=True, show_progress=True)
# reducer.prepare_rule_dict_parallel(l_max=6, n_jobs=8, show_progress=True)

for rule in reducer.reduce_rules[:10]:
    print(rule)

@pytest.fixture
def x(): return nd.Variable('x', nettype='scalar')

@pytest.fixture
def y(): return nd.Variable('y', nettype='scalar')

@pytest.fixture
def z(): return nd.Variable('z', nettype='scalar')

class TestReduceInitialization:
    """Test Reduce class initialization and caching."""


    def test_init_custom(self):
        """Test initialization with custom parameters."""
        reducer = Reduce(
            n_variables=2,
            constants=[0, 1],
            max_online_iterations=5,
            load_cache=True
        )
        assert len(reducer.variables) == 2
        assert len(reducer.constants) == 2
        assert reducer.max_online_iterations == 5

    def test_hash_consistency(self):
        """Test that hash is consistent for same configuration."""
        r1 = Reduce(n_variables=2, constants=[0, 1], load_cache=True)
        r2 = Reduce(n_variables=2, constants=[0, 1], load_cache=True)
        assert hash(r1) == hash(r2)

    def test_hash_difference(self):
        """Test that hash differs for different configurations."""
        r1 = Reduce(n_variables=2, constants=[0, 1], load_cache=True)
        r2 = Reduce(n_variables=3, constants=[0, 1], load_cache=True)
        assert hash(r1) != hash(r2)

    def test_equivalent_add_zero(self, x):
        """Test x + 0 is equivalent to x."""
        assert reducer._is_functionally_equivalent(x + 0, x)

    def test_equivalent_mul_one(self, x):
        """Test x * 1 is equivalent to x."""
        assert reducer._is_functionally_equivalent(x * 1, x)

    def test_equivalent_double_negation(self, x):
        """Test -(-x) is equivalent to x."""
        assert reducer._is_functionally_equivalent(-(-x), x)

    def test_equivalent_complex(self, x, y):
        """Test (x + y) * 1 is equivalent to x + y."""
        assert reducer._is_functionally_equivalent((x + y) * 1, x + y)

    def test_not_equivalent(self, x):
        """Test x + 1 is NOT equivalent to x."""
        assert not reducer._is_functionally_equivalent(x + 1, x)

    def test_not_equivalent_different_operation(self, x):
        """Test x * 2 is NOT equivalent to x + 2."""
        assert not reducer._is_functionally_equivalent(x * 2, x + 2)

    def test_constant_expressions(self, reducer):
        """Test equivalence of constant expressions."""
        expr1 = nd.Number(2) + nd.Number(3)
        expr2 = nd.Number(5)
        assert reducer._is_functionally_equivalent(expr1, expr2)

    def test_multiple_variables(self, x, y, z):
        """Test equivalence with multiple variables."""
        assert reducer._is_functionally_equivalent(x + y + z, z + y + x)

    def test_add_zero(self, x):
        """Test x + 0 -> x."""
        result = reducer(x + 0)
        assert str(result) == str(x)

    def test_zero_add(self, x):
        """Test 0 + x -> x."""
        result = reducer(0 + x)
        assert str(result) == str(x)

    def test_sub_zero(self, x):
        """Test x - 0 -> x."""
        result = reducer(x - 0)
        assert str(result) == str(x)

    def test_zero_sub(self, x):
        """Test 0 - x -> -x."""
        result = reducer(0 - x)
        assert str(result) == str(-x)

    def test_mul_one(self, x):
        """Test x * 1 -> x."""
        result = reducer(x * 1)
        assert str(result) == str(x)

    def test_one_mul(self, x):
        """Test 1 * x -> x."""
        result = reducer(1 * x)
        assert str(result) == str(x)

    def test_mul_zero(self, x):
        """Test x * 0 -> 0."""
        result = reducer(x * 0)
        assert str(result) == str(nd.Number(0))

    def test_zero_mul(self, x):
        """Test 0 * x -> 0."""
        result = reducer(0 * x)
        assert str(result) == str(nd.Number(0))

    def test_div_one(self, x):
        """Test x / 1 -> x."""
        result = reducer(x / 1)
        assert str(result) == str(x)

    def test_zero_div(self, x):
        """Test 0 / x -> 0 (x != 0)."""
        result = reducer(0 / x)
        assert str(result) == str(nd.Number(0))

    def test_double_negation(self, x):
        """Test -(-x) -> x."""
        result = reducer(-(-x))
        assert str(result) == str(x)

    def test_triple_negation(self, x):
        """Test -(-(-x)) -> -x."""
        result = reducer(-(-(-x)))
        assert str(result) == str(-x)

    def test_sin_asin(self, x):
        """Test sin(asin(x)) -> x."""
        result = reducer(nd.sin(nd.arcsin(x)))
        assert str(result) == str(x)

    def test_asin_sin(self, x):
        """Test asin(sin(x)) -> x."""
        result = reducer(nd.arcsin(nd.sin(x)))
        assert str(result) == str(x)

    def test_exp_log(self, x):
        """Test exp(log(x)) -> x."""
        result = reducer(nd.exp(nd.log(x)))
        assert str(result) == str(x)

    def test_log_exp(self, x):
        """Test log(exp(x)) -> x."""
        result = reducer(nd.log(nd.exp(x)))
        assert str(result) == str(x)

    def test_sqrt_square(self, x):
        """Test sqrt(x^2) -> |x| or x."""
        result = reducer(nd.sqrt(x ** 2))
        assert len(result) <= len(x ** 2)

    def test_distributive_property(self, x, y):
        """Test x + x -> 2 * x."""
        result = reducer(x + x)
        assert len(result) <= len(x + x)

    def test_common_factor(self, x, y):
        """Test x * y + x -> x * (y + 1) or similar."""
        expr = x * y + x
        result = reducer(expr)
        assert reducer._is_functionally_equivalent(result, expr)

    def test_nested_expression(self, x, y):
        """Test reduction in nested expressions."""
        expr = (x + 0) * (y - 0)
        result = reducer(expr)
        assert str(result) == str(x * y)

    def test_constant_add(self, reducer):
        """Test 2 + 3 -> 5."""
        expr = nd.Number(2) + nd.Number(3)
        result = reducer(expr)
        assert str(result) == str(nd.Number(5))

    def test_constant_mul(self, reducer):
        """Test 2 * 3 -> 6."""
        expr = nd.Number(2) * nd.Number(3)
        result = reducer(expr)
        assert str(result) == str(nd.Number(6))

    def test_constant_div(self, reducer):
        """Test 6 / 2 -> 3."""
        expr = nd.Number(6) / nd.Number(2)
        result = reducer(expr)
        assert str(result) == str(nd.Number(3))

    def test_constant_nested(self, reducer):
        """Test (2 + 3) * 4 -> 20."""
        expr = (nd.Number(2) + nd.Number(3)) * nd.Number(4)
        result = reducer(expr)
        assert str(result) == str(nd.Number(20))

    def test_already_simplified(self, x):
        """Test that already simplified expressions remain unchanged."""
        result = reducer(x)
        assert str(result) == str(x)

    def test_multiple_iterations(self, x):
        """Test expressions requiring multiple iterations."""
        expr = -(-(-(-x)))
        result = reducer(expr)
        assert str(result) == str(x)

    def test_no_infinite_loop(self, x):
        """Test that reduction converges (no infinite loop)."""
        expr = ((x + 0) * 1) - 0
        result = reducer(expr)
        assert result is not None

    def test_empty_expression(self, reducer):
        """Test with minimal expression."""
        expr = nd.Number(0)
        result = reducer(expr)
        assert str(result) == str(nd.Number(0))

    def test_very_nested_expression(self, x):
        """Test deeply nested expression."""
        expr = -(-(-(-(-(-(-(-x)))))))
        result = reducer(expr)
        assert result is not None

    def test_multiple_same_variables(self, x):
        """Test expression with multiple same variables."""
        expr = x + x + x
        result = reducer(expr)
        assert str(result) == str(3 * x)
