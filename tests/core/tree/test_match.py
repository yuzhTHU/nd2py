import pytest
import nd2py as nd
from nd2py.core.tree import match


class TestMatch:
    """Tests for the pattern match functionality."""

    @pytest.fixture
    def x(self):
        return nd.Variable('x')

    @pytest.fixture
    def y(self):
        return nd.Variable('y')

    @pytest.fixture
    def z(self):
        return nd.Variable('z')

    @pytest.fixture
    def a(self):
        return nd.Variable('a')

    @pytest.fixture
    def b(self):
        return nd.Variable('b')

    @pytest.fixture
    def c(self):
        return nd.Variable('c')

    def _assert_bindings_equal(self, result, expected):
        """Helper to compare bindings by string representation."""
        assert result is not None, "Expected a match but got None"
        assert set(result.keys()) == set(expected.keys())
        for key in result.keys():
            assert str(result[key]) == str(expected[key])

    def test_variable_matches_simple_expression(self, a, x):
        """Test that a variable matches a simple expression."""
        result = match(a, x)
        self._assert_bindings_equal(result, {'a': x})

    def test_variable_matches_complex_expression(self, a, x):
        """Test that a variable matches a complex expression."""
        result = match(a, 2*x+1)
        self._assert_bindings_equal(result, {'a': 2*x+1})

    def test_sin_pattern_matches(self, a, x):
        """Test that sin(a) matches sin(x)."""
        result = match(nd.sin(a), nd.sin(x))
        self._assert_bindings_equal(result, {'a': x})

    def test_sin_pattern_matches_complex_target(self, a, x):
        """Test that sin(a) matches sin(x*2+1)."""
        expr = nd.sin(2*x+1)
        result = match(nd.sin(a), expr)
        self._assert_bindings_equal(result, {'a': 2*x+1})

    def test_same_variable_same_subexpression(self, a, x):
        """Test that a+a matches sin(x)+sin(x)."""
        result = match(a+a, nd.sin(x)+nd.sin(x))
        self._assert_bindings_equal(result, {'a': nd.sin(x)})

    def test_same_variable_different_subexpression_fails(self, a, x):
        """Test that a+a does NOT match sin(x)+cos(x)."""
        result = match(a+a, nd.sin(x) + nd.cos(x))
        assert result is None

    def test_different_variables_same_subexpression(self, a, b, x):
        """Test that a+b matches sin(x)+sin(x)."""
        result = match(a+b, nd.sin(x)+nd.sin(x))
        self._assert_bindings_equal(result, {'a': nd.sin(x), 'b': nd.sin(x)})

    def test_different_variables_different_subexpression(self, a, b, x):
        """Test that a+b matches sin(x)+cos(x)."""
        result = match(a+b, nd.sin(x) + nd.cos(x))
        self._assert_bindings_equal(result, {'a': nd.sin(x), 'b': nd.cos(x)})

    def test_nested_pattern(self, a, b, x, y):
        """Test that sin(a+b) matches sin(x+y)."""
        result = match(nd.sin(a+b), nd.sin(x+y))
        self._assert_bindings_equal(result, {'a': x, 'b': y})

    def test_exp_pattern(self, a, b, x, y):
        """Test that exp(a)+b matches exp(x)+y."""
        result = match(nd.exp(a) + b, nd.exp(x) + y)
        self._assert_bindings_equal(result, {'a': x, 'b': y})

    def test_no_commutativity(self, a, b, x):
        """Test that exp(a)+b does NOT match sin(x)+exp(x)."""
        result = match(nd.exp(a) + b, nd.sin(x) + nd.exp(x))
        assert result is None

    def test_different_operator_types_fail(self, a, x):
        """Test that sin(a) does NOT match cos(x)."""
        result = match(nd.sin(a), nd.cos(x))
        assert result is None

    def test_different_operand_count_fails(self, a, b, x):
        """Test that a+b does NOT match x (different operand count)."""
        result = match(a+b, x)
        assert result is None

    def test_number_matches_same_number(self, a, x):
        """Test that a+2 matches x+2."""
        result = match(a+2, x+2)
        self._assert_bindings_equal(result, {'a': x})

    def test_number_matches_different_number_fails(self, a, x):
        """Test that a+2 does NOT match x+3."""
        result = match(a+2, x+3)
        assert result is None

    def test_mul_pattern(self, a, b, x, y):
        """Test that a*b matches x*y."""
        result = match(a*b, x*y)
        self._assert_bindings_equal(result, {'a': x, 'b': y})

    def test_nested_unary_operators(self, a, x):
        """Test that sin(cos(a)) matches sin(cos(x))."""
        result = match(nd.sin(nd.cos(a)), nd.sin(nd.cos(x)))
        self._assert_bindings_equal(result, {'a': x})

    def test_nested_unary_operators_fail(self, a, x):
        """Test that sin(cos(a)) does NOT match cos(sin(x))."""
        result = match(nd.sin(nd.cos(a)), nd.cos(nd.sin(x)))
        assert result is None

    def test_multiple_occurrences_same_variable(self, a, x):
        """Test that sin(a)+cos(a) matches sin(x)+cos(x)."""
        result = match(nd.sin(a) + nd.cos(a), nd.sin(x) + nd.cos(x))
        self._assert_bindings_equal(result, {'a': x})

    def test_multiple_occurrences_same_variable_fail(self, a, x, y):
        """Test that sin(a)+cos(a) does NOT match sin(x)+cos(y)."""
        # Note: sin(a)+cos(a) will match sin(x)+cos(x) but not sin(x)+cos(y)
        # because 'a' must map to the same thing in both places
        result = match(nd.sin(a) + nd.cos(a), nd.sin(x) + nd.cos(y))
        assert result is None

    def test_identity_operator_pattern(self, a, x):
        """Test that Identity(a) matches Identity(x)."""
        result = match(nd.Identity(a), nd.Identity(x))
        self._assert_bindings_equal(result, {'a': x})

    def test_complex_nested_pattern(self, a, b, c, x, y, z):
        """Test a complex nested pattern."""
        result = match(a * nd.sin(b) + nd.exp(c), x * nd.sin(y) + nd.exp(z))
        self._assert_bindings_equal(result, {'a': x, 'b': y, 'c': z})

    def test_div_pattern(self, a, b, x, y):
        """Test that a/b matches x/y."""
        result = match(a / b, x / y)
        self._assert_bindings_equal(result, {'a': x, 'b': y})

    def test_pow_pattern(self, a, b, x, y):
        """Test that Pow(a,b) matches Pow(x,y)."""
        result = match(a ** b, x ** y)
        self._assert_bindings_equal(result, {'a': x, 'b': y})

    def test_pattern_longer_than_target_fails(self, a, b, c, x, y):
        """Test that a+b+c does NOT match x+y (pattern too long)."""
        result = match((a+b) + c, x + y)
        assert result is None

    def test_variable_in_target_not_in_pattern(self, a, x, y):
        """Test that matching works when target has variables not in pattern."""
        # Pattern 'a' should match any expression including sin(x+y)
        result = match(a, nd.sin(x + y))
        self._assert_bindings_equal(result, {'a': nd.sin(x + y)})

    def test_empty_match(self, x):
        """Test matching with no variables (exact match)."""
        result = match(nd.sin(2), nd.sin(2))
        assert result == {}

    def test_exact_match_different_values_fail(self, x):
        """Test that exact structure but different values fails."""
        result = match(nd.sin(2), nd.sin(3))
        assert result is None

    def test_subtraction_pattern(self, a, b, x, y):
        """Test that a-b matches x-y."""
        result = match(nd.Sub(a, b), nd.Sub(x, y))
        self._assert_bindings_equal(result, {'a': x, 'b': y})

    def test_abs_pattern(self, a, x):
        """Test that abs(a) matches abs(x)."""
        result = match(nd.Abs(a), nd.Abs(x))
        self._assert_bindings_equal(result, {'a': x})

    def test_sqrt_pattern(self, a, x):
        """Test that sqrt(a) matches sqrt(x)."""
        result = match(nd.Sqrt(a), nd.Sqrt(x))
        self._assert_bindings_equal(result, {'a': x})

    def test_log_pattern(self, a, x):
        """Test that log(a) matches log(x)."""
        result = match(nd.Log(a), nd.Log(x))
        self._assert_bindings_equal(result, {'a': x})
