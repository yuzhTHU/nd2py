# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations
from itertools import islice
from functools import lru_cache
from typing import List, Type, Optional, Generator
from ...core.symbols import Symbol, Variable, Number


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
        self.cache: dict[int, list] = {}

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

        gen = self._build_trees(length)

        if max_results is None:
            yield from gen
        else:
            yield from islice(gen, max_results)

    def _build_trees(self, length: int) -> Generator[Symbol, None, None]:
        """
        Build all expression trees of a given length using memoization.

        Args:
            length: Target tree length.

        Yields:
            Symbol: Expression trees of the specified length.
        """
        # Return cached results if available
        if length in self.cache:
            yield from self.cache[length]
            return

        trees: list = []

        # Base case: length = 1, only leaf nodes
        if length == 1:
            for node in self.leafs:
                trees.append(tree := node.copy())
                yield tree

        # Case 1: Unary operators (consume 1 length, child has length-1)
        if self.unary and length >= 2:
            for child in self._build_trees(length - 1):
                for node in self.unary:
                    try:
                        trees.append(tree := node(child.copy()))
                        yield tree
                    except Exception:
                        pass

        # Case 2: Binary operators (consume 1 length, split remaining between two children)
        if self.binary and length >= 3:
            # left_len + right_len = length - 1, each child needs at least length 1
            for left_len in range(1, length - 1):
                right_len = length - 1 - left_len
                for left in self._build_trees(left_len):
                    for right in self._build_trees(right_len):
                        for node in self.binary:
                            try:
                                trees.append(tree := node(left.copy(), right.copy()))
                                yield tree
                            except Exception:
                                pass

        self.cache[length] = trees
    
    def estimate_total(self, length: int) -> int:
        """ 估计长度为 length 的公式总数 """
        return self._estimate_total(len(self.binary), len(self.unary), len(self.leafs), length)

    @staticmethod
    def _estimate_total(binary_num: int, unary_num: int, leaf_num: int, length: int) -> int:
        """
        递归计算长度为 L 的抽象语法树 (AST) 的可能种类数。
        
        Args:
            binary_num: 二元内部节点 (Binary inner nodes) 的种类数
            unary_num: 一元内部节点 (Unary inner nodes) 的种类数
            leaf_num: 叶子节点 (Leaf nodes) 的种类数
            length: 树包含的总节点数
        """
        @lru_cache(maxsize=None)
        def dp(current_L: int) -> int:
            # 边界情况：没有节点无法成树
            if current_L <= 0:
                return 0
            # 基础情况：长度为 1，只能是叶子节点
            if current_L == 1:
                return leaf_num
            total = 0
            # 1. 一元节点作为根节点的情况
            total += unary_num * dp(current_L - 1) # 根节点有 unary_num 种选择，其唯一的子树长度为 current_L - 1
            # 2. 二元节点作为根节点的情况
            if current_L >= 3:
                binary_combinations = 0 # 根节点有 binary_num 种选择，剩下的 current_L - 1 个节点分配给左右两棵子树
                for left_size in range(1, current_L - 1):
                    right_size = current_L - 1 - left_size
                    binary_combinations += dp(left_size) * dp(right_size)
                total += binary_num * binary_combinations
            return total
        
        return dp(length)
