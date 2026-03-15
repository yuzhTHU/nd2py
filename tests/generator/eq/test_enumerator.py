import pytest
import nd2py as nd
from nd2py.generator.eq import Enumerator


class TestEnumerator:
    """Tests for the Enumerator class."""

    @pytest.fixture
    def x(self):
        return nd.Variable('x')

    @pytest.fixture
    def y(self):
        return nd.Variable('y')

    @pytest.fixture
    def c(self):
        return nd.Number(2.0)

    def test_length_1_single_leaf(self, x):
        """Test enumeration with a single leaf."""
        enumerator = Enumerator(leafs=[x], binary=[], unary=[])
        eqs = enumerator(length=1)
        assert set(map(str, eqs)) == {'x'}

    def test_length_1_leafs_only(self, x, y):
        """Test enumeration with length=1 (only leaf nodes)."""
        enumerator = Enumerator(leafs=[x, y], binary=[], unary=[])
        eqs = enumerator(length=1)
        assert set(map(str, eqs)) == {'x', 'y'}

    def test_length_2_unary(self, x):
        """Test enumeration with length=2 (one unary operator)."""
        enumerator = Enumerator(leafs=[x], binary=[], unary=[nd.Sin, nd.Cos])
        eqs = enumerator(length=2)
        assert set(map(str, eqs)) == {'sin(x)', 'cos(x)'}

    def test_length_3_binary_single_leaf(self, x):
        """Test enumeration with single leaf and binary operators."""
        enumerator = Enumerator(leafs=[x], binary=[nd.Add, nd.Mul], unary=[])
        eqs = enumerator(length=3)
        assert set(map(str, eqs)) == {'x + x', 'x * x'}

    def test_length_3_binary(self, x, y):
        """Test enumeration with length=3 (one binary operator)."""
        enumerator = Enumerator(leafs=[x, y], binary=[nd.Add, nd.Mul], unary=[])
        eqs = enumerator(length=3)
        assert set(map(str, eqs)) == {
            'x + x', 'x + y', 'y + x', 'y + y',
            'x * x', 'x * y', 'y * x', 'y * y',
        }

    def test_length_4_mixed(self, x, y):
        """Test enumeration with length=4 (mixed unary and binary)."""
        enumerator = Enumerator(leafs=[x, y], binary=[nd.Add], unary=[nd.Sin])
        eqs = enumerator(length=4)
        assert set(map(str, eqs)) == {
            'sin(sin(sin(x)))', 'sin(sin(sin(y)))',
            'sin(x + x)', 'sin(x + y)', 'sin(y + x)', 'sin(y + y)',
            'x + sin(x)', 'x + sin(y)', 'y + sin(x)', 'y + sin(y)',
            'sin(x) + x', 'sin(x) + y', 'sin(y) + x', 'sin(y) + y',
        }

    def test_length_5_binary_only(self, x):
        """Test enumeration with length=5 (two binary operators)."""
        enumerator = Enumerator(leafs=[x], binary=[nd.Add, nd.Mul], unary=[])
        eqs = enumerator(length=5)
        assert set(map(str, eqs)) == {
            'x + x + x', 'x + x * x',
            'x * x + x', 'x * x * x',
            '(x + x) * x', 'x * (x + x)',
        }

    def test_with_number_leaf(self, x, c):
        """Test enumeration with Number as leaf."""
        enumerator = Enumerator(leafs=[x, c], binary=[nd.Add], unary=[])
        eqs = enumerator(length=3)
        assert set(map(str, eqs)) == {'x + x', 'x + 2', '2 + x', '2 + 2'}

    def test_max_results_limit(self, x, y):
        """Test that max_results limits the number of returned equations."""
        enumerator = Enumerator(leafs=[x, y], binary=[nd.Add, nd.Sub, nd.Mul, nd.Div], unary=[])
        eqs = list(enumerator(length=3, max_results=5))  # Convert generator to list
        # Just verify we get exactly 5 results
        assert len(eqs) == 5
        # All results should be from the expected set
        expected = {'x + x', 'x + y', 'y + x', 'y + y', 'x - x', 'x - y', 'y - x', 'y - y', 'x * x', 'x * y', 'y * x', 'y * y', 'x / x', 'x / y', 'y / x', 'y / y'}
        assert set(map(str, eqs)).issubset(expected)

    def test_empty_binary_unary(self, x):
        """Test enumeration with no operators (only length=1 should work)."""
        enumerator = Enumerator(leafs=[x], binary=[], unary=[])
        eqs = enumerator(length=1)
        assert set(map(str, eqs)) == {'x'}

        eqs = enumerator(length=2)
        assert set(map(str, eqs)) == set()  # length > 1 with no operators should return empty

    def test_length_0(self, x):
        """Test enumeration with length=0 (should return empty)."""
        enumerator = Enumerator(leafs=[x], binary=[], unary=[])
        eqs = enumerator(length=0)
        assert set(map(str, eqs)) == set()

    def test_tree_structure_independence(self, x, y):
        """Test that returned trees are independent (modifying one doesn't affect others)."""
        enumerator = Enumerator(leafs=[x, y], binary=[nd.Add], unary=[])
        eqs = list(enumerator(length=3))  # Convert generator to list for indexing
        eq1 = eqs[0]  # x + x
        eq2 = eqs[1]  # x + y
        eq1.operands[0] = nd.Number(999)  # Modify eq1
        assert str(eq2) == 'x + y'  # eq2 should not be affected

    def test_complex_unary_stack(self, x):
        """Test stacking multiple unary operators."""
        enumerator = Enumerator(leafs=[x], binary=[], unary=[nd.Sin, nd.Cos])
        eqs = list(enumerator(length=4))  # Convert generator to list for iteration
        assert set(map(str, eqs)) == {
            'sin(sin(sin(x)))', 'sin(sin(cos(x)))', 'sin(cos(sin(x)))', 'sin(cos(cos(x)))',
            'cos(sin(sin(x)))', 'cos(sin(cos(x)))', 'cos(cos(sin(x)))', 'cos(cos(cos(x)))',
        }
        for eq in eqs:
            # All should have 3 nested unary ops
            assert len(eq) == 4

    def test_mixed_unary_binary_depth(self, x, y):
        """Test mixed unary and binary operators at various depths."""
        enumerator = Enumerator(leafs=[x, y], binary=[nd.Add], unary=[nd.Sin])
        eqs = list(enumerator(length=5))  # Convert generator to list for multiple iterations
        # Verify some expected results are present (subset check for large result set)
        eq_strings = set(map(str, eqs))
        expected_subset = {
            'sin(sin(sin(sin(x))))', 'sin(sin(sin(sin(y))))',
            'x + x + x', 'x + x + y', 'x + y + x', 'x + y + y',
            'y + x + x', 'y + x + y', 'y + y + x', 'y + y + y',
            'sin(x) + sin(x)', 'sin(x) + sin(y)', 'sin(y) + sin(x)', 'sin(y) + sin(y)',
        }
        assert expected_subset.issubset(eq_strings)

    def test_network_operators(self):
        """Test with network-specific operators."""
        x = nd.Variable('x', nettype='node')
        y = nd.Variable('y', nettype='edge')
        enumerator = Enumerator(leafs=[x, y], binary=[nd.Add], unary=[nd.Aggr])
        eqs = enumerator(length=4)
        assert set(map(str, eqs)) == {'aggr(y + y)', 'x + aggr(y)', 'aggr(y) + x'}
