# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations
from typing import Optional, Dict, TYPE_CHECKING, Tuple, Generator
from ..base_visitor import Visitor, yield_nothing

if TYPE_CHECKING:
    from ..symbols import Symbol, Variable, Number, Empty
    _YieldType = Tuple[Symbol, tuple, dict]  # (child_node, args, kwargs)
    _SendType = bool  # Result from child match
    _ReturnType = bool
    _Type = Generator[_YieldType, _SendType, _ReturnType]


__all__ = ["MatchVisitor"]


class Match(Visitor):
    """
    A visitor that matches a pattern tree against a target tree.

    Uses an iterative approach with explicit stack to avoid recursion depth issues.
    """

    def __call__(self, pattern: Symbol, target: Symbol) -> Dict[str, Symbol] | None:
        """
        Match pattern against target.

        Args:
            pattern: The pattern expression containing variables.
            target: The target expression to match against.

        Returns:
            A dictionary mapping variable names to matched subexpressions,
            or None if the pattern does not match.
        """
        bindings: Dict[str, Symbol] = {}
        # Pass target and bindings as positional args
        result = super().__call__(pattern, target, bindings)
        if result:
            return bindings
        return None

    def generic_visit(self, pattern: Symbol, target: Symbol, bindings: Dict[str, Symbol]) -> _Type:
        """
        Default visitor for operator nodes.

        Matches operator types and recursively matches operands.
        """
        # Check type match
        if type(pattern) is not type(target):
            return False

        # Check number of operands
        if pattern.n_operands != target.n_operands:
            return False

        # Recursively match all operands
        for p_op, t_op in zip(pattern.operands, target.operands):
            # Yield child pattern/target pair for matching
            # Must yield (child, args, kwargs) tuple
            match_result = yield (p_op, (t_op, bindings), {})
            if not match_result:
                return False

        return True

    def visit_Variable(self, pattern: Variable, target: Symbol, bindings: Dict[str, Symbol]) -> _Type:
        """
        Visitor for Variable nodes.

        A variable can match any subexpression, subject to consistency constraints.
        """
        yield from yield_nothing()
        var_name = pattern.name

        if var_name not in bindings:
            # New variable - bind it to the target
            # Check for cyclic binding (variable cannot bind to expression containing itself)
            if any(node.name == var_name for node in target.iter_preorder() if type(node).__name__ == 'Variable'):
                return False
            else:
                bindings[var_name] = target
                return True
        else:
            # Existing binding - must match exactly
            bound_target = bindings[var_name]
            return _trees_equal(bound_target, target)

    def visit_Number(self, pattern: Number, target: Symbol, bindings: Dict[str, Symbol]) -> _Type:
        """
        Visitor for Number nodes.

        Numbers must match exactly by value.
        """
        yield from yield_nothing()
        return _numbers_equal(pattern, target)

    def visit_Empty(self, pattern: Empty, target: Symbol, bindings: Dict[str, Symbol]) -> _Type:
        """
        Visitor for Empty nodes.

        Empty matches anything.
        """
        yield from yield_nothing()
        return True


def _trees_equal(t1: Symbol, t2: Symbol) -> bool:
    """
    Check if two trees have identical structure and values.

    Uses iterative approach to avoid recursion depth issues.
    """
    # Use explicit stack for iterative comparison
    stack = [(t1, t2)]

    while stack:
        node1, node2 = stack.pop()

        # Check type
        if type(node1) is not type(node2):
            return False

        # Check if both are variables - must have same name
        if type(node1).__name__ == 'Variable' and type(node2).__name__ == 'Variable':
            if node1.name != node2.name:
                return False
            continue

        # Check if both are numbers
        if type(node1).__name__ == 'Number' and type(node2).__name__ == 'Number':
            if not _numbers_equal(node1, node2):
                return False
            continue

        # Check number of operands
        if node1.n_operands != node2.n_operands:
            return False

        # Add operand pairs to stack for comparison
        for op1, op2 in zip(node1.operands, node2.operands):
            stack.append((op1, op2))

    return True


def _numbers_equal(n1: Symbol, n2: Symbol) -> bool:
    """Check if two number symbols have equal values."""
    import numpy as np

    if not hasattr(n1, 'value') or not hasattr(n2, 'value'):
        return type(n1) is type(n2)

    try:
        val1 = n1.value
        val2 = n2.value
        if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            return val1.shape == val2.shape and np.allclose(val1, val2)
        return val1 == val2
    except (TypeError, ValueError):
        return False
