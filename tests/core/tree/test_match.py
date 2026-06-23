import pytest
import nd2py as nd


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
        result = x.match(a)
        self._assert_bindings_equal(result, {'a': x})

    def test_variable_matches_complex_expression(self, a, x):
        """Test that a variable matches a complex expression."""
        result = (2*x+1).match(a)
        self._assert_bindings_equal(result, {'a': 2*x+1})

    def test_sin_pattern_matches(self, a, x):
        """Test that sin(a) matches sin(x)."""
        result = nd.sin(x).match(nd.sin(a))
        self._assert_bindings_equal(result, {'a': x})

    def test_sin_pattern_matches_complex_target(self, a, x):
        """Test that sin(a) matches sin(x*2+1)."""
        result = nd.sin(2*x+1).match(nd.sin(a))
        self._assert_bindings_equal(result, {'a': 2*x+1})

    def test_same_variable_same_subexpression(self, a, x):
        """Test that a+a matches sin(x)+sin(x)."""
        result = (nd.sin(x)+nd.sin(x)).match(a+a)
        self._assert_bindings_equal(result, {'a': nd.sin(x)})

    def test_same_variable_different_subexpression_fails(self, a, x):
        """Test that a+a does NOT match sin(x)+cos(x)."""
        result = (nd.sin(x) + nd.cos(x)).match(a+a)
        assert result is None

    def test_different_variables_same_subexpression(self, a, b, x):
        """Test that a+b matches sin(x)+sin(x)."""
        result = (nd.sin(x)+nd.sin(x)).match(a+b)
        self._assert_bindings_equal(result, {'a': nd.sin(x), 'b': nd.sin(x)})

    def test_different_variables_different_subexpression(self, a, b, x):
        """Test that a+b matches sin(x)+cos(x)."""
        result = (nd.sin(x) + nd.cos(x)).match(a+b)
        self._assert_bindings_equal(result, {'a': nd.sin(x), 'b': nd.cos(x)})

    def test_nested_pattern(self, a, b, x, y):
        """Test that sin(a+b) matches sin(x+y)."""
        result = (nd.sin(x+y)).match(nd.sin(a+b))
        self._assert_bindings_equal(result, {'a': x, 'b': y})

    def test_exp_pattern(self, a, b, x, y):
        """Test that exp(a)+b matches exp(x)+y."""
        result = (nd.exp(x) + y).match(nd.exp(a) + b)
        self._assert_bindings_equal(result, {'a': x, 'b': y})

    def test_no_commutativity(self, a, b, x):
        """Test that exp(a)+b does NOT match sin(x)+exp(x)."""
        result = (nd.sin(x) + nd.exp(x)).match(nd.exp(a) + b)
        assert result is None

    def test_different_operator_types_fail(self, a, x):
        """Test that sin(a) does NOT match cos(x)."""
        result = (nd.cos(x)).match(nd.sin(a))
        assert result is None

    def test_different_operand_count_fails(self, a, b, x):
        """Test that a+b does NOT match x (different operand count)."""
        result = (x).match(a+b)
        assert result is None

    def test_number_matches_same_number(self, a, x):
        """Test that a+2 matches x+2."""
        result = (x+2).match(a+2)
        self._assert_bindings_equal(result, {'a': x})

    def test_number_matches_different_number_fails(self, a, x):
        """Test that a+2 does NOT match x+3."""
        result = (x+3).match(a+2)
        assert result is None

    def test_mul_pattern(self, a, b, x, y):
        """Test that a*b matches x*y."""
        result = (x*y).match(a*b)
        self._assert_bindings_equal(result, {'a': x, 'b': y})

    def test_nested_unary_operators(self, a, x):
        """Test that sin(cos(a)) matches sin(cos(x))."""
        result = (nd.sin(nd.cos(x))).match(nd.sin(nd.cos(a)))
        self._assert_bindings_equal(result, {'a': x})

    def test_nested_unary_operators_fail(self, a, x):
        """Test that sin(cos(a)) does NOT match cos(sin(x))."""
        result = (nd.cos(nd.sin(x))).match(nd.sin(nd.cos(a)))
        assert result is None

    def test_multiple_occurrences_same_variable(self, a, x):
        """Test that sin(a)+cos(a) matches sin(x)+cos(x)."""
        result = (nd.sin(x) + nd.cos(x)).match(nd.sin(a) + nd.cos(a))
        self._assert_bindings_equal(result, {'a': x})

    def test_multiple_occurrences_same_variable_fail(self, a, x, y):
        """Test that sin(a)+cos(a) does NOT match sin(x)+cos(y)."""
        # Note: sin(a)+cos(a) will match sin(x)+cos(x) but not sin(x)+cos(y)
        # because 'a' must map to the same thing in both places
        result = (nd.sin(x) + nd.cos(y)).match(nd.sin(a) + nd.cos(a))
        assert result is None

    def test_identity_operator_pattern(self, a, x):
        """Test that Identity(a) matches Identity(x)."""
        result = (nd.Identity(x)).match(nd.Identity(a))
        self._assert_bindings_equal(result, {'a': x})

    def test_complex_nested_pattern(self, a, b, c, x, y, z):
        """Test a complex nested pattern."""
        result = (x * nd.sin(y) + nd.exp(z)).match(a * nd.sin(b) + nd.exp(c))
        self._assert_bindings_equal(result, {'a': x, 'b': y, 'c': z})

    def test_div_pattern(self, a, b, x, y):
        """Test that a/b matches x/y."""
        result = (x / y).match(a / b)
        self._assert_bindings_equal(result, {'a': x, 'b': y})

    def test_pow_pattern(self, a, b, x, y):
        """Test that Pow(a,b) matches Pow(x,y)."""
        result = (x ** y).match(a ** b)
        self._assert_bindings_equal(result, {'a': x, 'b': y})

    def test_pattern_longer_than_target_fails(self, a, b, c, x, y):
        """Test that a+b+c does NOT match x+y (pattern too long)."""
        result = (x + y).match(a + b + c)
        assert result is None

    def test_variable_in_target_not_in_pattern(self, a, x, y):
        """Test that matching works when target has variables not in pattern."""
        # Pattern 'a' should match any expression including sin(x+y)
        result = (nd.sin(x + y)).match(a)
        self._assert_bindings_equal(result, {'a': nd.sin(x + y)})

    def test_empty_match(self, x):
        """Test matching with no variables (exact match)."""
        result = (nd.sin(2)).match(nd.sin(2))
        assert result == {}

    def test_exact_match_different_values_fail(self, x):
        """Test that exact structure but different values fails."""
        result = (nd.sin(2)).match(nd.sin(3))
        assert result is None

    def test_subtraction_pattern(self, a, b, x, y):
        """Test that a-b matches x-y."""
        result = (nd.Sub(x, y)).match(nd.Sub(a, b))
        self._assert_bindings_equal(result, {'a': x, 'b': y})

    def test_abs_pattern(self, a, x):
        """Test that abs(a) matches abs(x)."""
        result = (nd.Abs(x)).match(nd.Abs(a))
        self._assert_bindings_equal(result, {'a': x})

    def test_sqrt_pattern(self, a, x):
        """Test that sqrt(a) matches sqrt(x)."""
        result = (nd.Sqrt(x)).match(nd.Sqrt(a))
        self._assert_bindings_equal(result, {'a': x})

    def test_log_pattern(self, a, x):
        """Test that log(a) matches log(x)."""
        result = (nd.Log(x)).match(nd.Log(a))
        self._assert_bindings_equal(result, {'a': x})
