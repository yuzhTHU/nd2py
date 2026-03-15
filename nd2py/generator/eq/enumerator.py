# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations
from itertools import islice
from typing import List, Type, Optional, Generator, Iterator
from ...core.symbols import Symbol, Variable, Number, Empty
from ...core.symbols.operands import Add, Sub, Mul, Div


class Enumerator:
    """
    Enumerate all possible expression trees of a given length.

    This generator exhaustively constructs all valid expression trees
    by combining given leaf nodes, binary operators, and unary operators
    to reach the target tree size (measured by number of nodes).

    Example:
        >>> from nd2py import Variable
        >>> from nd2py.generator.eq import Enumerator
        >>> x = Variable('x')
        >>> enumerator = Enumerator(leafs=[x], binary=[Add, Mul], unary=[])
        >>> for eq in enumerator(length=3): print(eq)
        x + x
        x * x
    """

    def __init__(
        self,
        leafs: List[Variable | Number],
        binary: List[Type[Symbol]],
        unary: List[Type[Symbol]],
    ):
        """
        Initialize the enumerator.

        Args:
            leafs: List of leaf symbols (Variable, Number instances).
            binary: List of binary operator classes (e.g., Add, Sub, Mul, Div).
            unary: List of unary operator classes (e.g., Sin, Cos, Log, Exp).
        """
        self.leafs = leafs
        self.binary = binary
        self.unary = unary

    def __call__(
        self,
        length: int,
        max_results: Optional[int] = None,
    ) -> Generator[Symbol, None, None]:
        """
        Generate all possible expressions of the specified length.

        Args:
            length: Target tree length (total number of nodes in the expression tree).
            max_results: Maximum number of results to return. If None,
                generate all possible expressions (may be very large).

        Yields:
            Symbol: Expression trees of the specified length.
        """
        if length < 1:
            return

        cache: dict[int, list] = {}
        gen = self._build_trees(length, cache)

        if max_results is None:
            yield from gen
        else:
            yield from islice(gen, max_results)

    def _build_trees(self, length: int, cache: dict[int, list]) -> Generator[Symbol, None, None]:
        """
        Build all expression trees of a given length using memoization.

        Args:
            length: Target tree length.
            cache: Memoization cache storing previously computed trees.

        Yields:
            Symbol: Expression trees of the specified length.
        """
        # Return cached results if available
        if length in cache:
            yield from cache[length]
            return

        trees: list = []

        # Base case: length = 1, only leaf nodes
        if length == 1:
            for leaf in self.leafs:
                tree = leaf.copy()
                trees.append(tree)
                yield tree
            cache[length] = trees

        # Case 1: Unary operators (consume 1 length, child has length-1)
        if self.unary and length >= 2:
            for child in self._build_trees(length - 1, cache):
                for op in self.unary:
                    try:
                        tree = op(child.copy())
                        trees.append(tree)
                        yield tree
                    except Exception:
                        pass

        # Case 2: Binary operators (consume 1 length, split remaining between two children)
        if self.binary and length >= 3:
            # Binary op takes 1 node, remaining (length-1) nodes split between 2 children
            # Each child needs at least length 1, so: left_len + right_len = length - 1
            for left_len in range(1, length - 1):
                right_len = length - 1 - left_len
                for left in self._build_trees(left_len, cache):
                    for right in self._build_trees(right_len, cache):
                        for op in self.binary:
                            try:
                                tree = op(left.copy(), right.copy())
                                trees.append(tree)
                                yield tree
                            except Exception:
                                pass

        cache[length] = trees
