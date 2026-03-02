# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations
from typing import List, Type, TYPE_CHECKING
if TYPE_CHECKING:
    from ..symbols import *

class FromPostorder:
    def __call__(self, nodes: List[Symbol | Type[Symbol]], **kwargs) -> Symbol:
        """Construct a Symbol tree from a list of Symbols in postorder traversal order.""" 

        stack = []
        for node in nodes:
            if isinstance(node, type):
                node = node()
            if len(stack) < node.n_operands:
                raise ValueError(f"Not enough operands on stack for node {node}. Stack: {stack}")
            for empty in node.operands[::-1]:
                operand = stack.pop(-1)
                node = node.replace(empty, operand)
            stack.append(node)
        if len(stack) != 1:
            raise ValueError(f"Expected exactly one root node after processing, but got {len(stack)} nodes. Stack: {stack}")
        return stack[0]
    
def from_postorder(nodes: List[Symbol | Type[Symbol]], **kwargs) -> Symbol:
    """Construct a Symbol tree from a list of Symbols in postorder traversal order.""" 
    return FromPostorder()(nodes, **kwargs)