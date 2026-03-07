# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations
from typing import Generator, Tuple, Dict, List, Type, TYPE_CHECKING
from ..base_visitor import Visitor, yield_nothing
if TYPE_CHECKING:
    from ..symbols import *
    _YieldType = Tuple[Symbol|Type[Symbol], List[Symbol|Type[Symbol]], Dict]  # (node, args, kwargs)
    _SendType = Symbol
    _ReturnType = List[Symbol|Type[Symbol]]
    _Type = Generator[_YieldType, _SendType, _ReturnType]


class FromPreorder(Visitor):
    def __call__(self, nodes: List[Symbol | Type[Symbol]], **kwargs) -> Symbol:
        """Construct a Symbol tree from a list of Symbols in preorder traversal order."""
        if len(nodes) == 0:
            raise ValueError
        eqtree, remaining_nodes = super().__call__(nodes[0], iter(nodes[1:]), **kwargs)
        if len(remaining_nodes := list(remaining_nodes)) > 0:
            raise ValueError(f"Not all nodes were consumed. Remaining nodes: {remaining_nodes}")
        return eqtree

    def generic_visit(self, node: Symbol, following_nodes, **kwargs) -> _Type:
        yield from yield_nothing()
        if isinstance(node, type):
            node = node()
        for idx, empty in enumerate(node.operands):
            next_node = next(following_nodes)
            operand, following_nodes = yield (next_node, (following_nodes,), kwargs)
            node = node.replace(empty, operand)
        return node, following_nodes

def from_preorder(nodes: List[Symbol | Type[Symbol]], **kwargs) -> Symbol:
    """Construct a Symbol tree from a list of Symbols in preorder traversal order.""" 
    return FromPreorder()(nodes, **kwargs)