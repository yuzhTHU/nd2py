# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, List, Dict, Literal, Set, TYPE_CHECKING
from numpy.random import RandomState, default_rng
from ... import core
if TYPE_CHECKING:
    from ...core.nettype import NetType
    from ...core.symbols import *


class GPLearnGenerator:
    def __init__(
        self,
        variables: List[Variable],
        binary: List[str|Symbol] = [core.Add, core.Sub, core.Mul, core.Div],
        unary: List[str|Symbol] = [core.Sqrt, core.Log, core.Abs, core.Neg, core.Inv, core.Sin, core.Cos, core.Tan],
        full_prob: float = 0.5,
        depth_range: Tuple[int, int] = (2, 6),
        const_range: Tuple[float, float] = None,
        rng: RandomState = None,
        edge_list: Tuple[List[int], List[int]] = None,
        num_nodes: int = None,
        scalar_number_only=True,
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

    def generate_node(self, nettypes: Set[NetType]) -> Symbol:
        symbols = [sym for sym in self.symbols if (nettypes & sym.nettype_range)]
        symbol = self._rng.choice(symbols)
        node = symbol() # 不用指定 nettype, 由 .infer_nettype() 自行推断即可
        return node

    def generate_leaf(self, nettypes: Set[NetType]) -> Number | Variable:
        leafs = [var for var in self.variables if var.nettype in nettypes]

        if self.const_range is not None:
            const_range = self.const_range
        elif len(leafs) == 0:
            const_range = (-1, 1)
        else:
            const_range = None

        if const_range is not None:
            if "scalar" in nettypes:
                values = self._rng.uniform(*const_range)
                number = core.Number(values, nettype="scalar")
                leafs = leafs + [number]
            if "node" in nettypes and not self.scalar_number_only:
                values = self._rng.uniform(*const_range, (self.num_nodes,))
                number = core.Number(values, nettype="node")
                leafs = leafs + [number]
            if "edge" in nettypes and not self.scalar_number_only:
                values = self._rng.uniform(*const_range, (len(self.edge_list[0]),))
                number = core.Number(values, nettype="edge")
                leafs = leafs + [number]

        return self._rng.choice([var for var in leafs if var.nettype in nettypes])

    def sample(
        self, 
        nettypes: Set[NetType] = "scalar", 
        assign_root_nettypes=True
    ) -> Symbol:
        if isinstance(nettypes, str):
            nettypes = {nettypes}

        full_tree = self._rng.random() < self.full_prob
        max_depth = self._rng.integers(*self.depth_range)
        op_prob = 1.0 if full_tree else len(self.symbols) / (len(self.variables) + len(self.symbols))

        # Start a eqtree with a function to avoid degenerative eqtrees
        eqtree = self.generate_node(nettypes)
        eqtree.nettype = nettypes & eqtree.nettype_range
        empty_nodes_and_depth = [(i, 1) for i in eqtree.operands]

        while empty_nodes_and_depth:
            empty_node, depth = empty_nodes_and_depth.pop(0)
            if (depth < max_depth) and (self._rng.random() < op_prob):
                node = self.generate_node(empty_node.possible_nettypes)
                eqtree.replace(empty_node, node)
                empty_nodes_and_depth.extend([(i, depth + 1) for i in node.operands])
            else:  # Variable or Number
                leaf = self.generate_leaf(empty_node.possible_nettypes)
                eqtree.replace(empty_node, leaf)

        if not assign_root_nettypes:
            eqtree.nettype = None 
        return eqtree
