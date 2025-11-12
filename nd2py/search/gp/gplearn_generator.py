import logging
import numpy as np
from typing import Tuple, Optional, List, Dict, Literal, Set
from numpy.random import RandomState, default_rng
from ...core.symbols import *

__all__ = ["GPLearnGenerator"]
_logger = logging.getLogger(__name__)


class GPLearnGenerator:
    def __init__(
        self,
        variables: List[Variable],
        binary: List[Symbol] = [Add, Sub, Mul, Div, Max, Min],
        unary: List[Symbol] = [Sqrt, Log, Abs, Neg, Inv, Sin, Cos, Tan],
        full_prob: float = 0.5,
        depth_range: Tuple[int, int] = (2, 6),
        const_range: Tuple[float, float] = None,
        rng: RandomState = None,
        edge_list: Tuple[List[int], List[int]] = None,
        num_nodes: int = None,
        scalar_number_only=True,
        **kwargs,
    ):
        self.binary = binary
        self.unary = unary
        self.symbols = self.binary + self.unary
        self.variables = variables
        self.full_prob = full_prob
        self.depth_range = depth_range
        self.const_range = const_range
        self.scalar_number_only = scalar_number_only
        self._rng = rng or default_rng()

        if num_nodes is None and edge_list is not None:
            num_nodes = np.reshape(edge_list, (-1,)).max() + 1
        self.num_nodes = num_nodes
        self.edge_list = edge_list

        if any(kwargs):
            _logger.warning(f"Unused arguments: {kwargs}")

    def generate_node(self, nettype: Set[Literal["node", "edge", "scalar"]]) -> Symbol:
        symbol_nettypes = []
        for sym in self.symbols:
            for nt in sorted(nettype & sym.nettype_range()):
                symbol_nettypes.append((sym, nt))
        symbol, nettype = self._rng.choice(symbol_nettypes)
        node = symbol(nettype=nettype)
        return node

    def generate_leaf(
        self, nettype: Set[Literal["node", "edge", "scalar"]]
    ) -> Number | Variable:
        leafs = [var for var in self.variables if var.nettype in nettype]

        if self.const_range is not None:
            const_range = self.const_range
        elif len(leafs) == 0:
            const_range = (-1, 1)
        else:
            const_range = None

        if const_range is not None:
            if nettype == "scalar" or self.scalar_number_only or self._rng.integers(2):
                number = Number(self._rng.uniform(*const_range), nettype="scalar")
            elif nettype == "node":
                number = Number(
                    self._rng.uniform(*const_range, (self.num_nodes,)), nettype="node"
                )
            elif nettype == "edge":
                number = Number(
                    self._rng.uniform(*const_range, (len(self.edge_list[0]),)),
                    nettype="edge",
                )
            else:
                raise ValueError(
                    f"Unknown nettype: {nettype}. Supported types are 'scalar', 'node', and 'edge'."
                )
            leafs = leafs + [number]

        return self._rng.choice(leafs)

    def generate_eqtree(
        self, nettype: Set[Literal["node", "edge", "scalar"]]
    ) -> Symbol:
        if isinstance(nettype, str):
            nettype = {nettype}

        full_tree = self._rng.random() < self.full_prob
        max_depth = self._rng.integers(*self.depth_range)
        op_prob = (
            1.0
            if full_tree
            else len(self.symbols) / (len(self.variables) + len(self.symbols))
        )

        # Start a eqtree with a function to avoid degenerative eqtrees
        eqtree = self.generate_node(nettype)
        empty_nodes_and_depth = [(i, 1) for i in eqtree.operands]

        while empty_nodes_and_depth:
            empty_node, depth = empty_nodes_and_depth.pop(0)
            if (depth < max_depth) and (self._rng.random() < op_prob):
                node = self.generate_node(empty_node.replaceable_nettype())
                eqtree.replace(empty_node, node)
                empty_nodes_and_depth.extend([(i, depth + 1) for i in node.operands])
            else:  # Variable or Number
                leaf = self.generate_leaf(empty_node.replaceable_nettype())
                eqtree.replace(empty_node, leaf)
        return eqtree
